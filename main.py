import argparse
import csv
import os
import json
import time
from datetime import datetime
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image
import pickle
from io import BytesIO
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def resolve_device(device_name):
    # 统一处理 cpu / cuda / auto 三种写法。
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return torch.device(device_name)


def set_random_seed(seed):
    # 固定随机种子，尽量让多次运行结果更稳定。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset_key(real_path, fake_path, data_mode, dataset_key=None):
    # 如果手动传了 --dataset_key，就直接用；
    # 否则程序自动根据路径拼一个名字。
    if dataset_key:
        return dataset_key

    real_name = os.path.basename(os.path.normpath(real_path)) if real_path else "real"
    fake_name = os.path.basename(os.path.normpath(fake_path)) if fake_path else "fake"

    if data_mode == "wang2020":
        root_name = os.path.basename(os.path.normpath(real_path)) if real_path else "wang2020"
        return f"ForenSynths-{root_name}"

    if real_name == fake_name:
        return f"{data_mode}-{real_name}"
    return f"{data_mode}-{real_name}-vs-{fake_name}"


def format_duration(seconds):
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_run_outputs(result_folder, run_info, metrics):
    metrics_with_guide = {
        "__field_meanings__": {
            "ap_percent": "\u0041\u0050\uff0c\u5e73\u5747\u7cbe\u5ea6\u3002\u5b83\u7efc\u5408\u8003\u8651\u4e0d\u540c\u5224\u5b9a\u9608\u503c\u4e0b\u7684\u67e5\u51c6\u7387\u548c\u53ec\u56de\u7387\uff0c\u8d8a\u9ad8\u8bf4\u660e\u6a21\u578b\u628a\u771f\u56fe\u548c\u751f\u6210\u56fe\u6392\u5f00\u7684\u80fd\u529b\u8d8a\u5f3a\u3002",
            "auc": "\u0041\u0055\u0043\uff0c\u0052\u004f\u0043 \u66f2\u7ebf\u4e0b\u9762\u79ef\u3002\u5b83\u53cd\u6620\u6a21\u578b\u6574\u4f53\u533a\u5206\u4e24\u7c7b\u6837\u672c\u7684\u80fd\u529b\uff0c\u8d8a\u63a5\u8fd1 \u0031 \u8d8a\u597d\uff0c\u0030\u002e\u0035 \u9644\u8fd1\u53ef\u7406\u89e3\u4e3a\u63a5\u8fd1\u968f\u673a\u731c\u6d4b\u3002",
            "acc_result": "\u672c\u6b21\u6700\u7ec8\u91c7\u7528\u7684\u51c6\u786e\u7387\u7ed3\u679c\u3002",
            "acc_result.real_percent": "\u771f\u56fe\u8bc6\u522b\u6b63\u786e\u7387\uff0c\u5373\u771f\u5b9e\u56fe\u50cf\u88ab\u6b63\u786e\u5224\u4e3a\u771f\u56fe\u7684\u6bd4\u4f8b\u3002",
            "acc_result.fake_percent": "\u5047\u56fe\u8bc6\u522b\u6b63\u786e\u7387\uff0c\u5373 \u0041\u0049 \u751f\u6210\u56fe\u50cf\u88ab\u6b63\u786e\u5224\u4e3a\u5047\u56fe\u7684\u6bd4\u4f8b\u3002",
            "acc_result.overall_percent": "\u603b\u51c6\u786e\u7387\uff0c\u5373\u5168\u90e8\u6d4b\u8bd5\u56fe\u50cf\u4e2d\u5224\u5bf9\u7684\u6bd4\u4f8b\u3002",
            "acc_at_0_5": "\u5982\u679c\u8be5\u5b57\u6bb5\u5b58\u5728\uff0c\u8868\u793a\u9608\u503c\u53d6 \u0030\u002e\u0035 \u65f6\u7684\u5bf9\u7167\u51c6\u786e\u7387\u7ed3\u679c\u3002",
            "acc_at_best_threshold": "\u5982\u679c\u8be5\u5b57\u6bb5\u5b58\u5728\uff0c\u8868\u793a\u65e7\u7ed3\u679c\u91cc\u6309\u6700\u4f73\u9608\u503c\u7edf\u8ba1\u7684\u53c2\u8003\u7ed3\u679c\uff0c\u4e0d\u4f5c\u4e3a\u5f53\u524d\u7edf\u4e00\u5c55\u793a\u53e3\u5f84\u3002",
            "latent_uncertainty": "\u6f5c\u53d8\u91cf\u4e0d\u786e\u5b9a\u6027\uff0c\u6570\u503c\u8d8a\u5927\u8868\u793a\u6a21\u578b\u5185\u90e8\u6f5c\u8868\u793a\u6ce2\u52a8\u8d8a\u660e\u663e\u3002\u8fd9\u4e2a\u91cf\u5728\u5e26 \u0056\u0049\u0042 \u7684\u6a21\u578b\u91cc\u66f4\u6709\u89e3\u91ca\u610f\u4e49\u3002",
            "num_images": "\u8be5\u6570\u636e\u5212\u5206\u5b9e\u9645\u53c2\u4e0e\u672c\u6b21\u8bc4\u4f30\u7684\u56fe\u7247\u603b\u6570\u3002"
        }
    }
    metrics_with_guide.update(metrics)

    with open(os.path.join(result_folder, "run_config.json"), "w", encoding="utf-8-sig") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    with open(os.path.join(result_folder, "metrics_summary.json"), "w", encoding="utf-8-sig") as f:
        json.dump(metrics_with_guide, f, indent=2, ensure_ascii=False)


def write_ap_header(file_obj):
    file_obj.write("# \u672c\u6587\u4ef6\u8bb0\u5f55 AP \u548c AUC \u4e24\u4e2a\u6307\u6807\u3002\n")
    file_obj.write("# AP = Average Precision\uff0c\u5e73\u5747\u7cbe\u5ea6\u3002\u5b83\u4e0d\u53ea\u770b\u4e00\u4e2a\u9608\u503c\uff0c\u800c\u662f\u7efc\u5408\u8003\u5bdf\u6a21\u578b\u628a\u771f\u56fe\u548c\u751f\u6210\u56fe\u6392\u5f00\u7684\u80fd\u529b\uff0c\u8d8a\u9ad8\u8d8a\u597d\u3002\n")
    file_obj.write("# AUC = ROC \u66f2\u7ebf\u4e0b\u9762\u79ef\u3002\u5b83\u8868\u793a\u6a21\u578b\u6574\u4f53\u533a\u5206\u4e24\u7c7b\u6837\u672c\u7684\u80fd\u529b\uff0c\u8d8a\u63a5\u8fd1 1 \u8d8a\u597d\uff0c0.5 \u9644\u8fd1\u53ef\u7406\u89e3\u4e3a\u63a5\u8fd1\u968f\u673a\u731c\u6d4b\u3002\n")
    file_obj.write("# \u6bcf\u884c\u683c\u5f0f\uff1a\u6570\u636e\u96c6\u540d: AP(%)  AUC\n")


def write_acc0_header(file_obj):
    file_obj.write("# \u672c\u6587\u4ef6\u8bb0\u5f55\u672c\u6b21\u8bc4\u4f30\u91c7\u7528\u7684\u5206\u7c7b\u51c6\u786e\u7387\u7ed3\u679c\u3002\n")
    file_obj.write("# real_acc_percent = \u771f\u56fe\u8bc6\u522b\u6b63\u786e\u7387\u3002\n")
    file_obj.write("# fake_acc_percent = \u5047\u56fe\u8bc6\u522b\u6b63\u786e\u7387\u3002\n")
    file_obj.write("# overall_acc_percent = \u603b\u51c6\u786e\u7387\u3002\n")
    file_obj.write("# \u6bcf\u884c\u683c\u5f0f\uff1a\u6570\u636e\u96c6\u540d: \u771f\u56fe\u51c6\u786e\u7387(%)  \u5047\u56fe\u51c6\u786e\u7387(%)  \u603b\u51c6\u786e\u7387(%)\n")


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)


def resize_downsample(img, scale):
    if scale is None:
        return img
    if scale <= 0 or scale >= 1:
        raise ValueError("--resize_scale should be in the interval (0, 1).")
    resampling = getattr(Image, "Resampling", Image)
    width, height = img.size
    down_width = max(1, int(round(width * scale)))
    down_height = max(1, int(round(height * scale)))
    img = img.resize((down_width, down_height), resample=resampling.BICUBIC)
    return img.resize((width, height), resample=resampling.BICUBIC)



def calculate_acc(y_true, y_pred, thres):
    # 三个准确率分别表示:
    # r_acc: 真实图识别率
    # f_acc: 假图识别率
    # acc:   总准确率
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def build_acc_result(y_true, y_pred, threshold):
    real_acc, fake_acc, overall_acc = calculate_acc(y_true, y_pred, threshold)
    return {
        "threshold": float(threshold),
        "real_percent": round(real_acc * 100, 4),
        "fake_percent": round(fake_acc * 100, 4),
        "overall_percent": round(overall_acc * 100, 4),
        "balanced_percent": round((real_acc + fake_acc) * 50, 4),
        "predicted_fake_percent": round(float(np.mean(y_pred > threshold)) * 100, 4),
    }


def parse_prior_grid(value):
    priors = []
    for item in value.split(","):
        prior = float(item.strip())
        if not 0.0 < prior < 1.0:
            raise ValueError("Every --prior_grid value must be strictly between 0 and 1.")
        priors.append(prior)
    return sorted(set(priors))


def calculate_prior_sensitivity(y_true, y_pred, priors):
    rows = []
    actual_fake_prior = float(np.mean(y_true == 1))
    for assumed_prior in priors:
        threshold = float(np.quantile(y_pred, 1.0 - assumed_prior))
        row = build_acc_result(y_true, y_pred, threshold)
        row.update({
            "actual_fake_prior": actual_fake_prior,
            "assumed_fake_prior": float(assumed_prior),
            "prior_error": float(assumed_prior - actual_fake_prior),
        })
        rows.append(row)
    return rows


def safe_filename(value):
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def write_score_csv(path, y_true, y_pred, latent_uncertainty):
    with open(path, "w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["sample_index", "label", "score", "prediction_at_0_5", "latent_uncertainty"])
        for index, (label, score, uncertainty) in enumerate(zip(y_true, y_pred, latent_uncertainty)):
            writer.writerow([index, int(label), format(float(score), ".10f"), int(score > 0.5), format(float(uncertainty), ".10f")])


def write_prior_sensitivity_csv(path, rows):
    fieldnames = [
        "actual_fake_prior", "assumed_fake_prior", "prior_error", "threshold",
        "real_percent", "fake_percent", "overall_percent", "balanced_percent",
        "predicted_fake_percent",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate(model, loader, device, eval_threshold=0.5, threshold_mode="fixed", target_fake_prior=0.5,
             prior_grid=None, timing_warmup_batches=2):
    # validate 是项目的“评估函数”。
    # 输出: AP / AUC / 准确率

    if prior_grid is None:
        prior_grid = [0.5]

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    inference_seconds = 0.0
    timed_images = 0
    timed_views = 0
    evaluation_wall_start = time.perf_counter()

    with torch.no_grad():
        y_true, y_pred = [], []
        latent_uncertainty_list = []
        print ("Length of dataset: %d" %(len(loader)))
        for batch_index, (img, label) in enumerate(loader):
            logical_batch_size = img.size(0)
            num_views = img.size(1) if img.dim() == 5 else 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_start = time.perf_counter()
            if img.dim() == 5:
                batch_size, num_views = img.size(0), img.size(1)
                in_tens = img.view(batch_size * num_views, *img.shape[2:]).to(device)
                (mu, std), logit = model(in_tens)
                probs = logit.sigmoid().view(batch_size, num_views, -1).mean(dim=1).flatten()
                view_uncertainty = std.mean(dim=1).view(batch_size, num_views).mean(dim=1)
                latent_uncertainty_list.extend(view_uncertainty.tolist())
            else:
                in_tens = img.to(device)
                (mu, std), logit = model(in_tens)
                probs = logit.sigmoid().flatten()
                latent_uncertainty_list.extend(std.mean(dim=1).tolist())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_seconds = time.perf_counter() - batch_start
            if batch_index >= timing_warmup_batches:
                inference_seconds += batch_seconds
                timed_images += logical_batch_size
                timed_views += logical_batch_size * num_views
            y_pred.extend(probs.tolist())

            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    evaluation_wall_seconds = time.perf_counter() - evaluation_wall_start
    latent_uncertainty_array = np.array(latent_uncertainty_list)


    # AP 反映排序质量。
    ap = average_precision_score(y_true, y_pred)

    # 0.5 是最常见的默认分类阈值。
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    auc_score = roc_auc_score(y_true, y_pred)

    target_prior_threshold = float(np.quantile(y_pred, 1.0 - target_fake_prior))
    selected_threshold = target_prior_threshold if threshold_mode == "target_prior" else eval_threshold

    efficiency = {
        "timing_warmup_batches": int(timing_warmup_batches),
        "timed_images": int(timed_images),
        "timed_views": int(timed_views),
        "inference_seconds": round(inference_seconds, 6),
        "logical_images_per_second": round(timed_images / inference_seconds, 4) if inference_seconds else None,
        "model_views_per_second": round(timed_views / inference_seconds, 4) if inference_seconds else None,
        "latency_ms_per_logical_image": round(inference_seconds * 1000 / timed_images, 4) if timed_images else None,
        "evaluation_wall_seconds": round(evaluation_wall_seconds, 6),
        "end_to_end_images_per_second": round(len(y_true) / evaluation_wall_seconds, 4) if evaluation_wall_seconds else None,
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 4) if device.type == "cuda" else None,
    }
    return {
        "ap": float(ap),
        "auc": float(auc_score),
        "acc_at_0_5": build_acc_result(y_true, y_pred, 0.5),
        "acc_at_fixed_threshold": build_acc_result(y_true, y_pred, eval_threshold),
        "acc_at_target_prior": build_acc_result(y_true, y_pred, target_prior_threshold),
        "selected_acc": build_acc_result(y_true, y_pred, selected_threshold),
        "target_prior_threshold": target_prior_threshold,
        "prior_sensitivity": calculate_prior_sensitivity(y_true, y_pred, prior_grid),
        "efficiency": efficiency,
        "average_latent_uncertainty": float(latent_uncertainty_array.mean()),
        "y_true": y_true,
        "y_pred": y_pred,
        "latent_uncertainty": latent_uncertainty_array,
    }







def recursively_read(rootdir, must_contain, exts=["png", "jpg", "jpeg", "bmp"]):
    # 从目录里递归收集图片文件路径。
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            ext = os.path.splitext(file)[1].lstrip(".").lower()
            if (ext in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    # 支持两种数据来源:
    # 1. 直接给文件夹
    # 2. 给 pickle 文件，里面提前存好图片路径列表
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list





class RealFakeDataset(Dataset):
    def __init__(self,  real_path,
                        fake_path,
                        data_mode,
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None,
                        resize_scale=None,
                        tta_views="center"):

        # data_mode:
        # - wang2020: ForenSynths / wang2020 那类目录命名规则
        # - manually: 真实图和假图目录都明确给出
        assert data_mode in ["wang2020", "ours", "manually"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.resize_scale = resize_scale
        self.tta_views = tta_views

        # = = = = = = data path = = = = = = = = = #
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # 给每张图片建立标签:
        # 真实图 -> 0
        # 生成图 -> 1

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        # 不同骨干网络对应不同的归一化均值和方差。
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            self.normalize_transform,
        ])
        self.view_crop = transforms.FiveCrop(224)
        self.view_resize = transforms.Resize(224)

    def read_path(self, real_path, fake_path, data_mode, max_sample):

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)


        if max_sample is not None:
            # 为了对比公平，这里会对真假图各抽相同数量。
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = min(len(real_list), len(fake_list))
                print("not enough images, max_sample falling to %d" % max_sample)
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]
        else:
            balanced_count = min(len(real_list), len(fake_list))
            if len(real_list) != len(fake_list):
                print("real/fake counts differ, auto-balancing to %d images per class" % balanced_count)
                random.shuffle(real_list)
                random.shuffle(fake_list)
                real_list = real_list[0:balanced_count]
                fake_list = fake_list[0:balanced_count]

        assert len(real_list) == len(fake_list)

        return real_list, fake_list



    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        # Dataset 每次返回:
        # 1. 处理好的图像张量
        # 2. 对应标签(0=真实图, 1=生成图)
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)
        if self.resize_scale is not None:
            img = resize_downsample(img, self.resize_scale)

        if self.tta_views in ["five", "ten"]:
            if min(img.size) < 224:
                img = self.view_resize(img)
            crops = list(self.view_crop(img))
            if self.tta_views == "ten":
                crops += [crop.transpose(Image.FLIP_LEFT_RIGHT) for crop in crops]
            img = torch.stack([self.normalize_transform(crop) for crop in crops])
        else:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    run_start_ts = time.time()
    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # main.py 的功能:
    # 加载一个权重，然后在一个或多个数据集上做测试评估。
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default='', help='wang2020 or ours')
    parser.add_argument('--dataset_key', type=str, default=None, help='custom display name for the current dataset when using manual paths')
    parser.add_argument('--max_sample', type=int, default=100, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--classifier_head', type=str, default='linear', choices=['linear', 'dropout_linear'])
    parser.add_argument('--disable_vib', action='store_true', help='use the no-VIB ablation model structure')
    parser.add_argument('--uncertainty_weighting', action='store_true', help='use VIB uncertainty weighting')
    parser.add_argument('--uncertainty_weight_scale', type=float, default=1.0, help='scale factor for VIB uncertainty weighting')
    parser.add_argument('--uncertainty_weight_mode', type=str, default='residual', choices=['residual', 'direct'], help='residual keeps mu and weights z-mu; direct weights the whole latent z')
    parser.add_argument('--vib_inference_mode', type=str, default='sample', choices=['sample', 'mean', 'marginal'], help='VIB inference rule: stochastic sample, posterior mean, or analytic marginal logit')

    parser.add_argument('--result_folder', type=str, default='', help='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto', help="auto, cpu, cuda, cuda:0 ...")
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--eval_threshold', type=float, default=None, help='fixed evaluation threshold; default uses 0.5')
    parser.add_argument('--threshold_mode', type=str, default='fixed', choices=['fixed', 'target_prior'], help='fixed uses eval_threshold; target_prior sets threshold from target score quantile')
    parser.add_argument('--target_fake_prior', type=float, default=0.5, help='expected fake-image ratio for target_prior threshold mode')
    parser.add_argument('--prior_grid', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9', help='comma-separated assumed fake priors for sensitivity analysis')
    parser.add_argument('--save_scores', action='store_true', help='save one CSV row per evaluated sample')
    parser.add_argument('--timing_warmup_batches', type=int, default=2, help='exclude this many initial batches from inference timing')
    parser.add_argument('--tta_views', type=str, default='center', choices=['center', 'five', 'ten'], help='test-time crop views to average')

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")
    parser.add_argument('--resize_scale', type=float, default=None, help="0.75, 0.5, 0.25. Downsample and upsample before evaluation. Not apply if None")


    opt = parser.parse_args()
    prior_grid = parse_prior_grid(opt.prior_grid)
    if not 0.0 < opt.target_fake_prior < 1.0:
        raise ValueError("--target_fake_prior must be strictly between 0 and 1.")
    if opt.timing_warmup_batches < 0:
        raise ValueError("--timing_warmup_batches cannot be negative.")
    set_random_seed(opt.seed)
    device = resolve_device(opt.device)


    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    # 把 checkpoint 权重加载进去
    if opt.disable_vib and opt.uncertainty_weighting:
        raise ValueError("--uncertainty_weighting requires VIB, so it cannot be used with --disable_vib.")

    model = get_model(
        opt.arch,
        classifier_head=opt.classifier_head,
        disable_vib=opt.disable_vib,
        uncertainty_weighting=opt.uncertainty_weighting,
        uncertainty_weight_scale=opt.uncertainty_weight_scale,
        uncertainty_weight_mode=opt.uncertainty_weight_mode,
        vib_inference_mode=opt.vib_inference_mode,
    )
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    # print(state_dict.keys())
    model_= state_dict['model']
    # print(model_dice.keys())
    load_result = model.load_state_dict(model_, strict=False)
    if load_result.missing_keys:
        print("Warning: some module parameters were not found in the checkpoint and kept their initialized values.")
        print("Missing keys:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Unexpected keys:", load_result.unexpected_keys)

    print ("Model loaded..")
    model.eval()
    model.to(device)
    print("Running on device: %s" % device)

    # 如果没有手动指定测试集，就读取 dataset_paths.py 里的默认测试集。
    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [
            dict(
                real_path=opt.real_path,
                fake_path=opt.fake_path,
                data_mode=opt.data_mode,
                key=build_dataset_key(opt.real_path, opt.fake_path, opt.data_mode, opt.dataset_key),
            )
        ]

    run_info = {
        "ckpt": os.path.abspath(opt.ckpt),
        "arch": opt.arch,
        "classifier_head": opt.classifier_head,
        "disable_vib": opt.disable_vib,
        "uncertainty_weighting": opt.uncertainty_weighting,
        "uncertainty_weight_scale": opt.uncertainty_weight_scale,
        "uncertainty_weight_mode": opt.uncertainty_weight_mode,
        "vib_inference_mode": opt.vib_inference_mode,
        "run_started_at": run_started_at,
        "batch_size": opt.batch_size,
        "num_workers": opt.num_workers,
        "device": str(device),
        "max_sample": opt.max_sample,
        "eval_threshold": opt.eval_threshold,
        "threshold_mode": opt.threshold_mode,
        "target_fake_prior": opt.target_fake_prior,
        "prior_grid": prior_grid,
        "save_scores": opt.save_scores,
        "timing_warmup_batches": opt.timing_warmup_batches,
        "tta_views": opt.tta_views,
        "jpeg_quality": opt.jpeg_quality,
        "gaussian_sigma": opt.gaussian_sigma,
        "resize_scale": opt.resize_scale,
        "seed": opt.seed,
        "result_folder": os.path.abspath(opt.result_folder),
        "datasets": dataset_paths,
    }
    metrics_summary = {}

    with open(os.path.join(opt.result_folder, 'ap.txt'), 'w', encoding='utf-8-sig') as f:
        write_ap_header(f)

    with open(os.path.join(opt.result_folder, 'acc0.txt'), 'w', encoding='utf-8-sig') as f:
        write_acc0_header(f)

    effective_eval_threshold = 0.5 if opt.eval_threshold is None else float(opt.eval_threshold)

    # 允许一次命令评估多个数据划分。
    for dataset_path in (dataset_paths):

        dataset = RealFakeDataset(  dataset_path['real_path'],
                                    dataset_path['fake_path'],
                                    dataset_path['data_mode'],
                                    opt.max_sample,
                                    opt.arch,
                                    jpeg_quality=opt.jpeg_quality,
                                    gaussian_sigma=opt.gaussian_sigma,
                                    resize_scale=opt.resize_scale,
                                    tta_views=opt.tta_views)

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        evaluation = validate(
            model,
            loader,
            device,
            eval_threshold=effective_eval_threshold,
            threshold_mode=opt.threshold_mode,
            target_fake_prior=opt.target_fake_prior,
            prior_grid=prior_grid,
            timing_warmup_batches=opt.timing_warmup_batches)
        selected_acc = evaluation["selected_acc"]
        metrics_summary[dataset_path['key']] = {
            "ap_percent": round(evaluation["ap"] * 100, 4),
            "auc": round(evaluation["auc"], 6),
            "acc_result": selected_acc,
            "acc_at_0_5": evaluation["acc_at_0_5"],
            "acc_at_fixed_threshold": evaluation["acc_at_fixed_threshold"],
            "acc_at_target_prior": evaluation["acc_at_target_prior"],
            "target_prior_threshold": round(evaluation["target_prior_threshold"], 10),
            "prior_sensitivity": evaluation["prior_sensitivity"],
            "efficiency": evaluation["efficiency"],
            "latent_uncertainty": round(evaluation["average_latent_uncertainty"], 6),
            "num_images": len(dataset),
        }

        file_stem = safe_filename(dataset_path['key'])
        if opt.save_scores:
            write_score_csv(
                os.path.join(opt.result_folder, file_stem + "_scores.csv"),
                evaluation["y_true"],
                evaluation["y_pred"],
                evaluation["latent_uncertainty"],
            )
        write_prior_sensitivity_csv(
            os.path.join(opt.result_folder, file_stem + "_prior_sensitivity.csv"),
            evaluation["prior_sensitivity"],
        )

        with open( os.path.join(opt.result_folder,'ap.txt'), 'a', encoding='utf-8-sig') as f:
            f.write(dataset_path['key']+': ' + str(round(evaluation["ap"]*100, 2))+'  '+str(round(evaluation["auc"], 4))+'\n' )

        with open( os.path.join(opt.result_folder,'acc0.txt'), 'a', encoding='utf-8-sig') as f:
            f.write(dataset_path['key']+': ' + str(round(selected_acc["real_percent"], 2))+'  '+str(round(selected_acc["fake_percent"], 2))+'  '+str(round(selected_acc["overall_percent"], 2))+'\n' )

    run_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_duration_seconds = round(time.time() - run_start_ts, 3)
    run_info["run_finished_at"] = run_finished_at
    run_info["run_duration_seconds"] = run_duration_seconds
    run_info["run_duration_hms"] = format_duration(run_duration_seconds)

    write_run_outputs(opt.result_folder, run_info, metrics_summary)
    print("实验完成，结果已保存到：%s" % os.path.abspath(opt.result_folder))
