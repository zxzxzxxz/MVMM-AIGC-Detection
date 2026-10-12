# MVMM-AIGC-Detection

基于 CLIP 图像特征与变分信息瓶颈（VIB）的 AI 生成图像检测项目，包含多视角评估、边缘化推理、阈值校准、鲁棒性测试及 No-VIB 消融实验。

## 1. 安装环境

在项目根目录执行（PowerShell 示例）：

```powershell
git clone https://github.com/zxzxzxxz/MVMM-AIGC-Detection.git
cd MVMM-AIGC-Detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

使用 Python 3.9 或 3.10，安装与 CPU / CUDA 环境匹配的 PyTorch 和 torchvision。`--device auto` 自动选择运行设备。

## 2. 准备数据和权重

自行准备真实图像、生成图像和检测模型权重。例如：

```text
datasets/GenImage/Stable Diffusion V1.4/
├── train/
│   ├── nature/    # 真实图像
│   └── ai/        # 生成图像
└── val/
    ├── nature/
    └── ai/
models/checkpoints/
├── sd_v14.pth
└── progan.pth
```

检测权重需单独准备，包含 `model` 状态字典，并与模型结构匹配。首次运行时，程序会下载 CLIP 骨干权重到用户目录的 `.cache/clip/`。

在 `dataset_paths.py` 中填写训练集和验证集路径，或通过命令行指定真实图与生成图目录。

## 3. 运行评估

下面以 `CLIP:ViT-L/14` 和 SD v1.4 检测权重为例。每类最多抽取 500 张图片：

```powershell
python main.py --arch "CLIP:ViT-L/14" --ckpt "models/checkpoints/sd_v14.pth" --real_path "datasets/GenImage/Stable Diffusion V1.4/val/nature" --fake_path "datasets/GenImage/Stable Diffusion V1.4/val/ai" --data_mode manually --max_sample 500 --batch_size 8 --device auto --result_folder "results/example_baseline" --save_scores
```

在上述命令后追加参数，可切换评估方式：

| 功能 | 参数 |
| --- | --- |
| 五视角 + 边缘化推理 | `--tta_views five --vib_inference_mode marginal` |
| 目标先验阈值校准 | `--threshold_mode target_prior --target_fake_prior 0.5` |
| JPEG 压缩测试 | `--jpeg_quality 80` |
| 高斯模糊测试 | `--gaussian_sigma 1` |
| 缩放测试 | `--resize_scale 0.5` |
| 固定随机种子 | `--seed 2026` |

默认使用中心裁剪、随机 VIB 采样和 0.5 分类阈值。校准阈值来自目标图像分数的分位数，不使用目标标签，但依赖假图比例的假设；数据加载器会平衡真假图数量。当前 CLIP-VIB 层按 ViT-L/14 的 768 维特征构建，建议使用示例骨干。

**每次运行请使用独立的 `--result_folder`：程序会删除并重建指定的已有结果目录。**

## 4. 查看结果

结果保存到指定目录，主要包括：

- `run_config.json`：运行参数、数据路径和时间记录。
- `metrics_summary.json`：AP、AUC、真假图准确率、总体准确率、潜变量不确定性和效率统计。
- `ap.txt` / `acc0.txt`：简要指标。
- 先验敏感性 CSV：不同假图比例假设下的指标；使用 `--save_scores` 后，额外保存逐样本分数 CSV。

真实图标签为 `0`，生成图标签为 `1`，分数越高越倾向生成图。AP 与准确率以百分比输出，AUC 在主评估结果中为 0–1。

## 5. 消融实验

配置好 `dataset_paths.py` 的训练集和验证集后，可训练冻结 CLIP 骨干的 No-VIB 分类头：

```powershell
python train_no_vib.py --epochs 20 --device auto --result_folder "results/example_no_vib" --save_ckpt "models/checkpoints/no_vib.pth"
```

之后使用 `main.py`，传入该权重并追加 `--disable_vib --classifier_head dropout_linear` 进行评估。项目当前提供 No-VIB 训练脚本，VIB 检测模型需要已有权重。

## 代码结构

```text
main.py                       # 检测与评估入口
dataset_paths.py              # 默认数据路径
train_no_vib.py               # No-VIB 消融训练
models/                       # CLIP、VIB 和分类头实现
networks/、options/            # 辅助网络与参数模块
requirements.txt              # Python 依赖
```

## 来源说明

第三方代码的许可见对应源码中的版权声明。
