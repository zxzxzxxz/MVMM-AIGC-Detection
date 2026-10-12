import argparse
import csv
import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_paths import DATASET_PATHS
from main import RealFakeDataset, resolve_device, set_random_seed, validate, write_run_outputs, format_duration, write_ap_header, write_acc0_header
from models import get_model


def extract_features(model, loader, device):
    feature_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.extract_features(images)
            feature_list.append(features.cpu())
            label_list.append(labels.float().view(-1, 1).cpu())
    return torch.cat(feature_list, dim=0), torch.cat(label_list, dim=0)


def evaluate_feature_head(head, features, labels, device, batch_size):
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sigmoid = torch.nn.Sigmoid()

    y_true = []
    y_pred = []
    with torch.no_grad():
        head.eval()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            logits = head(batch_features)
            probs = sigmoid(logits).view(-1).cpu().tolist()
            y_pred.extend(probs)
            y_true.extend(batch_labels.view(-1).cpu().tolist())

    y_true = torch.tensor(y_true).numpy()
    y_pred = torch.tensor(y_pred).numpy()
    acc = ((y_pred > 0.5) == y_true).mean()
    return acc


def save_training_curve(log_rows, output_dir):
    epochs = [row["epoch"] for row in log_rows]
    train_loss = [row["train_loss"] for row in log_rows]
    val_acc = [row["val_acc_percent"] for row in log_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    plt.plot(epochs, val_acc, marker="s", label="val acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("No-VIB Ablation Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=200)
    plt.close()


def write_result_text_files(result_folder, split_key, ap, auc_score, r_acc0, f_acc0, acc0):
    ap_path = os.path.join(result_folder, "ap.txt")
    acc_path = os.path.join(result_folder, "acc0.txt")
    with open(ap_path, "a", encoding="utf-8") as f:
        f.write(f"{split_key}: {round(ap * 100, 2)}  {round(auc_score, 2)}\n")
    with open(acc_path, "a", encoding="utf-8") as f:
        f.write(f"{split_key}: {round(r_acc0 * 100, 2)}  {round(f_acc0 * 100, 2)}  {round(acc0 * 100, 2)}\n")


def main():
    run_start_ts = time.time()
    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--arch", default="CLIP:ViT-L/14")
    parser.add_argument("--classifier_head", default="dropout_linear", choices=["linear", "dropout_linear"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--feature_batch_size", type=int, default=8)
    parser.add_argument("--head_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train_max_sample", type=int, default=500)
    parser.add_argument("--val_max_sample", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--result_folder", default=r".\results\ablation_no_vib_eval_500")
    parser.add_argument("--save_ckpt", default=r".\models\checkpoints\ablation_no_vib_dropout_linear.pth")
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = resolve_device(args.device)

    if os.path.exists(args.result_folder):
        shutil.rmtree(args.result_folder)
    os.makedirs(args.result_folder)
    save_dir = os.path.dirname(os.path.abspath(args.save_ckpt))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    dataset_train = DATASET_PATHS[0]
    dataset_val = DATASET_PATHS[1]

    train_dataset = RealFakeDataset(
        dataset_train["real_path"],
        dataset_train["fake_path"],
        dataset_train["data_mode"],
        args.train_max_sample,
        args.arch,
    )
    val_dataset = RealFakeDataset(
        dataset_val["real_path"],
        dataset_val["fake_path"],
        dataset_val["data_mode"],
        args.val_max_sample,
        args.arch,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(args.arch, classifier_head=args.classifier_head, disable_vib=True)
    model.to(device)

    print("Extracting CLIP features for No-VIB ablation...")
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)

    head = model.decode.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_feature_dataset = TensorDataset(train_features, train_labels)
    train_feature_loader = DataLoader(train_feature_dataset, batch_size=args.head_batch_size, shuffle=True)

    best_val_acc = -1.0
    best_state = None
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        running_loss = 0.0
        total_samples = 0
        for batch_features, batch_labels in train_feature_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size_now = batch_features.size(0)
            running_loss += loss.item() * batch_size_now
            total_samples += batch_size_now

        avg_train_loss = running_loss / max(total_samples, 1)
        val_acc = evaluate_feature_head(head, val_features, val_labels, device, args.head_batch_size)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_acc_percent": round(val_acc * 100, 2),
            }
        )
        print(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.6f}, val_acc={val_acc * 100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    head.load_state_dict(best_state)
    model.decode.load_state_dict(best_state)
    model.eval()

    ckpt_payload = {
        "model": model.state_dict(),
        "meta": {
            "arch": args.arch,
            "classifier_head": args.classifier_head,
            "disable_vib": True,
            "seed": args.seed,
            "train_max_sample": args.train_max_sample,
            "val_max_sample": args.val_max_sample,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }
    torch.save(ckpt_payload, args.save_ckpt)

    with open(os.path.join(args.result_folder, "ap.txt"), "w", encoding="utf-8-sig") as f:
        write_ap_header(f)
    with open(os.path.join(args.result_folder, "acc0.txt"), "w", encoding="utf-8-sig") as f:
        write_acc0_header(f)

    metrics_summary = {}
    eval_specs = [
        ("GenImage-SDV1.4-train", train_dataset),
        ("GenImage-SDV1.4-val", val_dataset),
    ]
    for split_key, dataset in eval_specs:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ap, r_acc0, f_acc0, acc0, _r_acc_eval, _f_acc_eval, _acc_eval, auc_score, avg_latent_uncertainty = validate(
            model,
            loader,
            device,
            eval_threshold=0.5,
        )
        metrics_summary[split_key] = {
            "ap_percent": round(ap * 100, 2),
            "auc": round(auc_score, 4),
            "acc_at_0_5": {
                "real_percent": round(r_acc0 * 100, 2),
                "fake_percent": round(f_acc0 * 100, 2),
                "overall_percent": round(acc0 * 100, 2),
            },
            "latent_uncertainty": round(float(avg_latent_uncertainty), 6),
            "num_images": len(dataset),
        }
        write_result_text_files(args.result_folder, split_key, ap, auc_score, r_acc0, f_acc0, acc0)

    run_info = {
        "ckpt": os.path.abspath(args.save_ckpt),
        "arch": args.arch,
        "classifier_head": args.classifier_head,
        "disable_vib": True,
        "run_started_at": run_started_at,
        "batch_size": args.batch_size,
        "feature_batch_size": args.feature_batch_size,
        "head_batch_size": args.head_batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "seed": args.seed,
        "eval_threshold": None,
        "train_max_sample": args.train_max_sample,
        "val_max_sample": args.val_max_sample,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "result_folder": os.path.abspath(args.result_folder),
        "save_ckpt": os.path.abspath(args.save_ckpt),
        "datasets": [dataset_train, dataset_val],
    }
    run_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_duration_seconds = round(time.time() - run_start_ts, 3)
    run_info["run_finished_at"] = run_finished_at
    run_info["run_duration_seconds"] = run_duration_seconds
    run_info["run_duration_hms"] = format_duration(run_duration_seconds)
    write_run_outputs(args.result_folder, run_info, metrics_summary)

    with open(os.path.join(args.result_folder, "training_log.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc_percent"])
        writer.writeheader()
        writer.writerows(log_rows)

    save_training_curve(log_rows, args.result_folder)

    print("No-VIB 消融训练完成。")
    print("模型权重已保存到：%s" % os.path.abspath(args.save_ckpt))
    print("实验结果已保存到：%s" % os.path.abspath(args.result_folder))


if __name__ == "__main__":
    main()
