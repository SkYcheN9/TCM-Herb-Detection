# TCM-SliceAI YOLOv8 训练与消融实验

当前已完成数据集规范化、离线数据增强、Ultralytics YOLOv8 Baseline 训练入口，以及 5 个实质性改进方向：CBAM 注意力、BiFPN Neck、Focal Loss、GhostConv 轻量化骨干、Decoupled Head 解耦检测头。最终 11 组公平消融实验均已完成，网页端和桌面端默认部署 `Baseline+CBAM+BiFPN`，树莓派 5 无算力棒端默认部署 `Baseline+GhostConv`。

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
.\.venv\Scripts\python.exe scripts/augment_dataset.py --dataset-root dataset --output dataset_augmented --copies 2 --mosaic-count 250 --mixup-count 250
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

训练 CBAM + BiFPN + GhostConv 候选版本：

```bash
.\.venv\Scripts\python.exe train.py --config configs/cbam_bifpn_ghost.yaml
```

训练 CBAM + BiFPN + GhostConv + Decoupled Head 候选版本：

```bash
.\.venv\Scripts\python.exe train.py --config configs/cbam_bifpn_ghost_decoupled.yaml
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

一键训练并验证默认实验组：

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

正式 11 组实验由两部分组成：

```bash
# 9 组主消融
.\.venv\Scripts\python.exe scripts/ablation.py --experiments extended --data dataset_augmented\data.yaml --epochs 150 --imgsz 640 --batch 16 --workers 8 --device 0 --init pretrained --pretrained-weights yolov8n.pt

# 2 组 GhostConv 组合候选
.\.venv\Scripts\python.exe scripts/ablation.py --experiments candidate --output reports/ablation_candidate --data dataset_augmented\data.yaml --epochs 150 --imgsz 640 --batch 16 --workers 8 --device 0 --init pretrained --pretrained-weights yolov8n.pt
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

本机 CUDA 快速复现实验可使用增强数据集和较稳的 batch 设置：

```bash
.\.venv\Scripts\python.exe scripts/ablation.py --data dataset_augmented\data.yaml --epochs 100 --imgsz 640 --batch 8 --workers 0 --device auto
```

最终 11 组实验结果摘要：

| 实验 | mAP50 | mAP50-95 | FPS |
| --- | ---: | ---: | ---: |
| Baseline+CBAM | 0.99227 | 0.80125 | 242.11 |
| Baseline+CBAM+BiFPN | 0.99162 | 0.80076 | 302.73 |
| Baseline+GhostConv | 0.98915 | 0.79822 | 306.11 |
| Baseline | 0.99118 | 0.79532 | 304.50 |
| Baseline+DecoupledHead | 0.99133 | 0.79500 | 235.72 |
| Baseline+CBAM+BiFPN+GhostConv+DecoupledHead | 0.99026 | 0.79446 | 280.49 |
| Baseline+BiFPN | 0.99251 | 0.79234 | 299.28 |
| Baseline+CBAM+BiFPN+GhostConv | 0.98853 | 0.79027 | 287.42 |
| Baseline+CBAM+BiFPN+Focal | 0.99027 | 0.78909 | 198.90 |
| FullModel | 0.99012 | 0.78472 | 226.29 |
| Baseline+Focal | 0.98906 | 0.78438 | 235.14 |

最终部署选型：

- 网页端/桌面端默认使用 `Baseline+CBAM+BiFPN`：mAP50-95 为 0.80076，速度 302.73 FPS，精度几乎追平最高精度模型且速度更好。
- 最高精度参考模型为 `Baseline+CBAM`：mAP50-95 为 0.80125。
- 树莓派 5 8GB 无算力棒默认使用 `Baseline+GhostConv`：mAP50-95 为 0.79822，速度 306.11 FPS，更适合 CPU/OpenVINO 轻量部署。
- Focal Loss 与 FullModel 作为负向消融结论保留，不作为部署模型。

完整表格见 `docs/FINAL_MODEL_SELECTION.md`。

只跑指定实验：

```bash
.\.venv\Scripts\python.exe scripts/ablation.py --experiments baseline,cbam_bifpn_focal
```

只补跑 GhostConv 组合候选实验：

```bash
.\.venv\Scripts\python.exe scripts/ablation.py --experiments candidate --output reports/ablation_candidate --data dataset_augmented\data.yaml --epochs 150 --imgsz 640 --batch 16 --workers 8 --device 0 --init pretrained --pretrained-weights yolov8n.pt
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
.\.venv\Scripts\python.exe export.py --output exports/pi
```

打包 Raspberry Pi 部署目录：

```bash
.\.venv\Scripts\python.exe deploy_pi.py --output dist/raspberry_pi --zip
```

## 当前数据检查结果

- 人工清洗后原始数据：1049 张图片，1049 个标注文件
- 当前 `data/` 严格检查通过：1049 对有效样本，阻断问题为 0
- 规范化数据集：1049 对有效样本，其中训练集 838，对验证集 211
- 增强数据集：训练集 3014 张，验证集 211 张，总计 3225 对样本
- `dataset/` 与 `dataset_augmented/` 均严格检查通过，阻断问题为 0
- 每次人工修改 `data/` 后，需要重新运行 `prepare_dataset.py` 和 `augment_dataset.py`，再使用 `dataset_augmented/data.yaml` 训练
