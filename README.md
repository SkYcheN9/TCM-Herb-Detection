# TCM-SliceAI Phase 1 + CBAM

当前已完成数据集规范化、Ultralytics YOLOv8 Baseline 训练入口，以及可选 CBAM 注意力模块。不包含 BiFPN、Focal Loss、GhostConv 等其他改进。

## 已完成内容

- 创建 YOLO 标准目录：`dataset/images/train`、`dataset/images/val`、`dataset/labels/train`、`dataset/labels/val`
- 固定 15 类饮片顺序，禁止自动重排类别
- 检查 `classes.txt` 类别顺序
- 检查图片缺失标注、孤立标注、空标注、YOLO 标注格式、类别编号范围、归一化 bbox
- 自动生成 `dataset/data.yaml`
- 提供 YOLOv8 Baseline 训练脚本，CUDA 可用时优先 GPU，否则回退 CPU
- 训练输出目录固定为 `runs/baseline`
- 新增 CBAM 模块与 YOLOv8n-CBAM 结构，默认不影响 Baseline
- 通过 `enable_cbam` 配置切换 Baseline/CBAM 训练路径

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

规范化数据并生成 `data.yaml`：

```bash
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --images data/images --labels data/labels --output dataset --mode copy
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

也可以在命令行显式开启 CBAM：

```bash
.\.venv\Scripts\python.exe train.py --enable-cbam true --model models/yolov8n_cbam.yaml --name cbam
```

也可以直接运行：

```bash
.\.venv\Scripts\python.exe scripts/train_baseline.py
```

训练脚本会自动检测 `torch.cuda.is_available()`：可用时使用 GPU `0`，不可用时使用 CPU，并自动调整默认 batch size。CBAM 训练默认输出到 `runs/cbam`，Baseline 训练默认输出到 `runs/baseline`。

## 当前数据检查结果

- 原始数据：1049 张图片，1048 个标注文件
- 发现 1 张图片缺失标注：`2026-06-05_10_48_20_101.jpg`
- 发现 84 个标注文件包含 15-28 的越界类别编号
- 规范化数据集：964 对有效样本，其中训练集 771，对验证集 193
- 规范化后的 `dataset/` 严格检查通过，阻断问题为 0
