# Phase 1 项目报告

## 1. 项目背景

本项目面向中医药饮片智能检测与识别场景，目标是在规范化饮片图像数据集的基础上，构建一个基于 Ultralytics YOLOv8 的多类别目标检测模型，实现 15 类中药饮片的定位、识别与计数能力。

当前阶段已完成数据集规范化、YOLOv8 Baseline 训练、CBAM 注意力模块、BiFPN Neck、Focal Loss 分类损失替换、GhostConv 轻量化骨干、Decoupled Head 解耦检测头，以及自动化消融实验脚本。Baseline 仍保持官方 YOLOv8n 路径，所有改进模块均通过配置开关启用。

## 2. 数据集规模

| 项目 | 数量/状态 |
| --- | ---: |
| 原始图片 | 1049 |
| 原始标注文件 | 1048 |
| 缺失标注图片 | 1 |
| 含越界类别编号的标注文件 | 84 |
| 规范化后有效样本 | 964 |
| 训练集样本 | 771 |
| 验证集样本 | 193 |
| 类别数 | 15 |
| 规范化后阻断问题 | 0 |

规范化后的数据集采用 YOLO 标准目录结构，配置文件为 `dataset/data.yaml`。类别顺序固定，不允许训练或检查工具自动重排类别。

各类别目标数量如下：

| ID | 类别 | 中文名 | 目标数 |
| ---: | --- | --- | ---: |
| 0 | `zexie` | 泽泻 | 331 |
| 1 | `niuxi` | 牛膝 | 322 |
| 2 | `gaoliangjiang` | 高良姜 | 326 |
| 3 | `mudanpi` | 牡丹皮 | 307 |
| 4 | `yuzhu` | 玉竹 | 304 |
| 5 | `baizhi` | 白芷 | 315 |
| 6 | `baishao` | 白芍 | 296 |
| 7 | `dazao` | 大枣 | 314 |
| 8 | `danshen` | 丹参 | 310 |
| 9 | `gancao` | 甘草 | 324 |
| 10 | `baixianpi` | 白鲜皮 | 321 |
| 11 | `baihe` | 百合 | 324 |
| 12 | `sangzhi` | 桑枝 | 312 |
| 13 | `jiegeng` | 桔梗 | 303 |
| 14 | `banlangen` | 板蓝根 | 285 |

## 3. 15 类饮片

固定类别顺序如下：

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

该顺序已经写入 `dataset/classes.txt` 与 `dataset/data.yaml`，并由数据集检查工具自动校验。

## 4. Baseline 结果

Baseline 使用 `yolov8n.pt`，训练配置如下：

| 项目 | 配置 |
| --- | --- |
| 模型 | YOLOv8n |
| 配置文件 | `configs/baseline.yaml` |
| 训练轮数 | 100 |
| 图像尺寸 | 640 |
| Batch size | 16 |
| 设备 | CUDA:0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 输出目录 | `runs/baseline` |

Baseline 最优验证结果如下：

| 指标 | 数值 |
| --- | ---: |
| Best epoch | 51 |
| Precision | 0.94926 |
| Recall | 0.91324 |
| mAP50 | 0.94042 |
| mAP50-95 | 0.73898 |

Baseline 最后一轮结果如下：

| 指标 | 数值 |
| --- | ---: |
| Epoch | 100 |
| Precision | 0.94537 |
| Recall | 0.90156 |
| mAP50 | 0.91438 |
| mAP50-95 | 0.71305 |

## 5. CBAM 结果

CBAM 阶段新增 `models/modules/cbam.py`，并提供 `models/yolov8n_cbam.yaml` 与 `configs/cbam.yaml`。CBAM 插入在 Backbone 的 C2f 后，保持特征尺寸不变，不影响 Baseline 默认路径。

当前本地已完成 CBAM 结构构建与 1 epoch CUDA smoke 训练，结果如下。该结果仅用于验证训练链路可运行，不应与 100 epoch Baseline 作为正式精度对比。

| 项目 | 数值 |
| --- | ---: |
| Epoch | 1 |
| Image size | 128 |
| Batch size | 2 |
| Precision | 0.00078 |
| Recall | 0.00194 |
| mAP50 | 0.00003 |
| mAP50-95 | 0.00000 |

## 6. BiFPN 结果

BiFPN 阶段新增 `models/modules/bifpn.py`，通过 `enable_bifpn` 控制是否将 YOLOv8 原始 PAN-FPN 替换为 BiFPN Neck。BiFPN 使用可学习融合权重，并在前向过程中自动归一化。

当前本地有两类 BiFPN smoke 记录：

| 实验 | Epoch | Image size | Batch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BiFPN-only smoke | 1 | 128 | 2 | 0.00247 | 0.14045 | 0.00133 | 0.00031 |
| CBAM+BiFPN smoke | 1 | 128 | 2 | 0.13675 | 0.01247 | 0.00057 | 0.00014 |

以上结果仅证明结构可训练、可验证，正式结论需要按统一训练轮数和图像尺寸重新完成消融实验。

## 7. Focal 结果

Focal Loss 阶段新增 `models/losses/focal_loss.py`，只替换 YOLOv8 检测训练中的分类损失项；box loss、DFL、TaskAlignedAssigner 和模型结构保持 Ultralytics 原流程。

默认配置如下：

| 配置项 | 数值 |
| --- | --- |
| `enable_focal_loss` | `true` |
| `focal_gamma` | `2.0` |
| `focal_alpha` | `0.25` |

CBAM + BiFPN + Focal Loss 的 1 epoch CUDA smoke 结果如下：

| 项目 | 数值 |
| --- | ---: |
| Epoch | 1 |
| Image size | 128 |
| Batch size | 2 |
| Precision | 0.00128 |
| Recall | 0.06261 |
| mAP50 | 0.00087 |
| mAP50-95 | 0.00031 |

该结果仅用于验证 Focal Loss 能兼容 CBAM 与 BiFPN 训练流程。

## 8. 消融实验结果

项目已新增 `scripts/ablation.py`，支持自动运行并验证以下实验：

```text
Baseline
Baseline+CBAM
Baseline+CBAM+BiFPN
Baseline+CBAM+BiFPN+Focal
FullModel
```

其中 FullModel 当前定义为：

```text
GhostConv + CBAM + BiFPN + Decoupled Head + Focal Loss
```

自动导出内容包括：

- `reports/ablation/summary.csv`
- `reports/ablation/summary.xlsx`
- `reports/ablation/history.csv`
- `reports/ablation/plots/pr_curve.png`
- `reports/ablation/plots/loss_curve.png`
- `reports/ablation/plots/map_curve.png`

当前 `reports/ablation` 中已有一次 Baseline 端到端 smoke 验证，用于确认自动训练、验证、FPS 统计、CSV/XLSX 导出与曲线绘制流程可用：

| 实验 | Precision | Recall | mAP50 | mAP50-95 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline smoke | 0.03801 | 0.53964 | 0.05290 | 0.03347 | 169.40 |

正式消融实验建议使用统一配置完整运行：

```bash
.\.venv\Scripts\python.exe scripts\ablation.py --epochs 100 --imgsz 640 --batch 16 --workers 4
```

完整训练完成后，`reports/ablation/summary.csv` 与 `summary.xlsx` 将作为正式消融实验结果表。

## 9. 最终模型结构

当前阶段的 FullModel 定义为已实现模块全集：

```text
YOLOv8n Backbone
  + GhostConv replacing selected downsampling Conv layers
  + CBAM attention after backbone C2f blocks
  + SPPF
  + BiFPN Neck replacing PAN-FPN
  + DecoupledDetect head with separated regression/classification towers
  + Focal Loss for classification branch during training
```

对应配置与模型文件：

| 文件 | 作用 |
| --- | --- |
| `configs/full_model.yaml` | FullModel 训练配置 |
| `models/yolov8n_full.yaml` | GhostConv + CBAM + BiFPN + Decoupled Head 模型结构 |
| `models/yolov8n_ghost.yaml` | GhostConv 轻量化骨干单独结构 |
| `models/yolov8n_decoupled.yaml` | Decoupled Head 单独结构 |
| `models/modules/cbam.py` | CBAM 注意力模块 |
| `models/modules/bifpn.py` | BiFPN 加权融合模块 |
| `models/modules/decoupled_head.py` | Decoupled Head 解耦检测头 |
| `models/losses/focal_loss.py` | Focal Loss 分类损失替换 |

FullModel 已完成 1 epoch CUDA smoke 训练验证，日志显示使用 `CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU)`，可以正常完成训练、验证和权重保存。该 smoke 结果只用于验证链路，正式指标仍需按统一 100 epoch 消融实验生成。

## 10. 下一阶段规划

1. 完整运行 5 组消融实验，使用统一 `epochs=100`、`imgsz=640` 和相同数据划分，生成正式 `summary.csv`、`summary.xlsx` 与曲线图。
2. 基于正式消融结果选择最终模型权重，并补充 PR 曲线、Loss 曲线、mAP 曲线、混淆矩阵等报告素材。
3. 对 GhostConv 与 Decoupled Head 的独立实验进行补充训练，评估参数量、推理速度、收敛速度和精度变化。
4. 使用 `evaluate.py`、`benchmark.py`、`export.py` 补齐评估、测速和模型导出结果。
5. 开发或完善摄像头实时检测与自动计数功能。
6. 推进 Web 端与 Raspberry Pi 部署，验证 CPU/GPU/边缘设备运行能力。
7. 整理最终实验报告，统一呈现 Baseline、各改进模块和 FullModel 的对比结果。
