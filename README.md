# TCM-SliceAI YOLOv8 训练与消融实验

当前已完成数据集规范化、Ultralytics YOLOv8 Baseline 训练入口，以及 5 个实质性改进方向：CBAM 注意力、BiFPN Neck、Focal Loss、GhostConv 轻量化骨干、Decoupled Head 解耦检测头。

## 已完成内容

- 创建 YOLO 标准目录：`dataset/images/train`、`dataset/images/val`、`dataset/labels/train`、`dataset/labels/val`
- 固定 15 类饮片顺序，禁止自动重排类别
- 检查 `classes.txt` 类别顺序
- 检查图片缺失标注、孤立标注、空标注、YOLO 标注格式、类别编号范围、归一化 bbox
- 自动生成 `dataset/data.yaml`
- 提供 YOLOv8 Baseline 训练脚本，CUDA 可用时优先 GPU，否则回退 CPU
- 训练输出目录固定为 `runs/baseline`
- 新增 CBAM 模块与 YOLOv8n-CBAM 结构，默认不影响 Baseline
- 新增 BiFPN Neck，可替换原 YOLOv8 PAN-FPN
- 新增 Focal Loss，可替换 YOLOv8 原始分类 BCE 损失
- 新增 GhostConv 轻量化骨干，替换部分 Backbone 下采样 Conv
- 新增 Decoupled Head，显式拆分分类分支与回归分支
- 通过 `enable_cbam`、`enable_bifpn`、`enable_focal_loss`、`enable_ghostconv`、`enable_decoupled_head` 配置切换训练路径和损失函数

## 固定类别顺序

```text
0 zexie
1 niuxi
2 gaoliangjiang
3 mudanpi
4 yuzhu
5 baizhi
6 baishao
7 dazao
8 danshen
9 gancao
10 baixianpi
11 baihe
12 sangzhi
13 jiegeng
14 banlangen
```

## 使用方式

推荐创建项目虚拟环境并安装依赖：

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.12.1+cu126 torchvision==0.27.1+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本机已验证 CUDA 环境：`torch 2.12.1+cu126` 可识别 `NVIDIA GeForce RTX 4060 Laptop GPU`。如果机器没有 NVIDIA GPU，训练脚本会自动回退 CPU。

检查原始数据：

```bash
.\.venv\Scripts\python.exe scripts/check_dataset.py --images data/images --labels data/labels
```

清洗原始数据，检测模糊、无效、重复图片：

```bash
.\.venv\Scripts\python.exe scripts/clean_dataset.py --images data/images --labels data/labels --output data/cleaned
```

规范化数据并生成 `data.yaml`：

```bash
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --images data/images --labels data/labels --output dataset --mode copy
```

生成离线增强数据集：

```bash
.\.venv\Scripts\python.exe scripts/augment_dataset.py --dataset-root dataset --output dataset_augmented --copies 1 --mosaic-count 200 --mixup-count 200
```

训练 Baseline：

```bash
.\.venv\Scripts\python.exe train.py --data dataset/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

也可以使用配置文件训练 Baseline：

```bash
.\.venv\Scripts\python.exe train.py --config configs/baseline.yaml
```

训练 CBAM 版本：

```bash
.\.venv\Scripts\python.exe train.py --config configs/cbam.yaml
```

训练 BiFPN 版本：

```bash
.\.venv\Scripts\python.exe train.py --config configs/bifpn.yaml
```

训练 CBAM + BiFPN 版本：

```bash
.\.venv\Scripts\python.exe train.py --config configs/cbam_bifpn.yaml
```

训练 CBAM + BiFPN + Focal Loss 版本：

```bash
.\.venv\Scripts\python.exe train.py --config configs/cbam_bifpn_focal.yaml
```

训练 GhostConv 轻量化骨干版本：

```bash
.\.venv\Scripts\python.exe train.py --config configs/ghostconv.yaml
```

训练 Decoupled Head 版本：

```bash
.\.venv\Scripts\python.exe train.py --config configs/decoupled_head.yaml
```

训练当前 FullModel：

```bash
.\.venv\Scripts\python.exe train.py --config configs/full_model.yaml
```

当前 FullModel 等同于已实现模块全集：GhostConv + CBAM + BiFPN + Decoupled Head + Focal Loss。

也可以在命令行显式开启 CBAM：

```bash
.\.venv\Scripts\python.exe train.py --enable-cbam true --model models/yolov8n_cbam.yaml --name cbam
```

命令行也支持显式开启 BiFPN：

```bash
.\.venv\Scripts\python.exe train.py --enable-bifpn true --name bifpn
```

命令行也支持显式开启 Focal Loss：

```bash
.\.venv\Scripts\python.exe train.py --enable-cbam true --enable-bifpn true --enable-focal-loss true --focal-loss-type soft_focal --focal-gamma 1.0 --focal-alpha none --name cbam_bifpn_focal
```

命令行也支持显式开启 GhostConv 和 Decoupled Head：

```bash
.\.venv\Scripts\python.exe train.py --enable-ghostconv true --name ghostconv
.\.venv\Scripts\python.exe train.py --enable-decoupled-head true --name decoupled_head
.\.venv\Scripts\python.exe train.py --enable-cbam true --enable-bifpn true --enable-ghostconv true --enable-decoupled-head true --enable-focal-loss true --name full_model
```

也可以直接运行：

```bash
.\.venv\Scripts\python.exe scripts/train_baseline.py
```

训练脚本会自动检测 `torch.cuda.is_available()`：可用时使用 GPU `0`，不可用时使用 CPU，并自动调整默认 batch size。Baseline 训练默认输出到 `runs/baseline`，CBAM 输出到 `runs/cbam`，BiFPN 输出到 `runs/bifpn`，CBAM+BiFPN 输出到 `runs/cbam_bifpn`，CBAM+BiFPN+Focal 输出到 `runs/cbam_bifpn_focal`，FullModel 输出到 `runs/full_model`。

## 消融实验

一键训练并验证 5 组实验：

```bash
.\.venv\Scripts\python.exe scripts/ablation.py
```

默认实验顺序：

```text
Baseline
Baseline+CBAM
Baseline+CBAM+BiFPN
Baseline+CBAM+BiFPN+Focal
FullModel (GhostConv+CBAM+BiFPN+DecoupledHead+Focal)
```

消融实验输出保存到 `reports/ablation`：

- `summary.csv`：Precision、Recall、mAP50、mAP50-95、FPS 汇总
- `summary.xlsx`：Excel 汇总表
- `history.csv`：每轮训练指标
- `plots/loss_curve.png`：Loss 曲线
- `plots/map_curve.png`：mAP 曲线
- `plots/pr_curve.png`：PR 曲线
- `runs/`：各实验训练输出
- `val/`：各实验验证输出

本机 CUDA 正式消融训练使用增强数据集和较稳的 batch 设置：

```bash
.\.venv\Scripts\python.exe scripts/ablation.py --data dataset_augmented\data.yaml --epochs 100 --imgsz 640 --batch 8 --workers 0 --device auto
```

正式结果摘要：

| 实验 | mAP50 | mAP50-95 | FPS |
| --- | ---: | ---: | ---: |
| Baseline | 0.94706 | 0.74452 | 144.91 |
| Baseline+CBAM | 0.92953 | 0.72314 | 148.42 |
| Baseline+CBAM+BiFPN | 0.92321 | 0.71355 | 167.94 |
| Baseline+CBAM+BiFPN+Focal | 0.61593 | 0.47776 | 95.43 |
| FullModel | 0.52885 | 0.40597 | 60.29 |

当前精度交付权重建议使用 `reports/ablation/runs/baseline/weights/best.pt`；结构改进展示可使用 `reports/ablation/runs/baseline_cbam_bifpn/weights/best.pt`。Focal Loss 与 FullModel 已跑通，但在当前参数下欠收敛，后续需要继续调参。

只跑指定实验：

```bash
.\.venv\Scripts\python.exe scripts/ablation.py --experiments baseline,cbam_bifpn_focal
```

快速冒烟验证：

```bash
.\.venv\Scripts\python.exe scripts/ablation.py --epochs 1 --imgsz 128 --batch 2 --workers 0 --experiments baseline
```

复用已有训练权重并重新导出验证报告：

```bash
.\.venv\Scripts\python.exe scripts/ablation.py --skip-train
```

## 评估、测速与部署

验证模型并导出指标与混淆矩阵：

```bash
.\.venv\Scripts\python.exe evaluate.py --weights runs/baseline/weights/best.pt --data dataset/data.yaml
```

测试 CPU/GPU 推理速度：

```bash
.\.venv\Scripts\python.exe benchmark.py --weights runs/baseline/weights/best.pt --devices auto,cpu
```

导出 `.pt`、`.onnx`、`.torchscript`、OpenVINO、NCNN：

```bash
.\.venv\Scripts\python.exe export.py --weights runs/baseline/weights/best.pt --output exports/pi
```

打包 Raspberry Pi 部署目录：

```bash
.\.venv\Scripts\python.exe deploy_pi.py --weights runs/baseline/weights/best.pt --output dist/raspberry_pi --zip
```

## 当前数据检查结果

- 原始数据：1049 张图片，1048 个标注文件
- 发现 1 张图片缺失标注：`2026-06-05_10_48_20_101.jpg`
- 发现 84 个标注文件包含 15-28 的越界类别编号
- 规范化数据集：964 对有效样本，其中训练集 771，对验证集 193
- 规范化后的 `dataset/` 严格检查通过，阻断问题为 0
