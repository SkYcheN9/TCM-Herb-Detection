# Phase 1 项目报告

## 1. 项目背景

本项目面向中医药饮片智能检测与识别场景，目标是在规范化饮片图像数据集的基础上，构建基于 Ultralytics YOLOv8 的多类别目标检测模型，实现 15 类中药饮片的定位、识别、置信度输出与计数能力。

当前阶段已完成数据集规范化、离线数据增强、YOLOv8 Baseline 训练、CBAM 注意力模块、BiFPN Neck、Focal Loss 分类损失替换、GhostConv 轻量化骨干、Decoupled Head 解耦检测头，以及自动化消融实验。Baseline 保持官方 `yolov8n.pt` 路径，所有改进模块均通过配置开关启用，不破坏原始训练流程。

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
| 训练/验证比例 | 约 8:2 |
| 类别数 | 15 |
| 规范化后阻断问题 | 0 |

规范化后的数据集采用 YOLO 标准目录结构，配置文件为 `dataset/data.yaml`。类别顺序固定，不允许训练或检查工具自动重排类别。

本次正式消融训练统一使用增强数据集 `dataset_augmented/data.yaml`。增强集由 771 张原始训练样本、771 张 Albumentations 增强样本、200 张 Mosaic 样本和 200 张 MixUp 样本组成，训练集共 1942 张；验证集保持原始 193 张不增强，以保证各模型在同一验证集上公平比较。

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
| 训练数据 | `dataset_augmented/data.yaml` |
| 训练轮数 | 100 |
| 图像尺寸 | 640 |
| Batch size | 8 |
| Workers | 0 |
| 设备 | CUDA:0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 输出目录 | `reports/ablation/runs/baseline` |

Baseline 正式验证结果如下：

| 指标 | 数值 |
| --- | ---: |
| Best epoch | 50 |
| Precision | 0.96646 |
| Recall | 0.90696 |
| mAP50 | 0.94706 |
| mAP50-95 | 0.74452 |
| FPS | 144.91 |

Baseline 最后一轮训练日志结果如下：

| 指标 | 数值 |
| --- | ---: |
| Epoch | 100 |
| mAP50 | 0.93528 |
| mAP50-95 | 0.73228 |

## 5. CBAM 结果

CBAM 阶段新增 `models/modules/cbam.py`，并提供 `models/yolov8n_cbam.yaml` 与 `configs/cbam.yaml`。CBAM 插入在 Backbone 的 C2f 后，保持特征尺寸不变，不影响 Baseline 默认路径。

CBAM 正式验证结果如下：

| 指标 | 数值 |
| --- | ---: |
| Best epoch | 87 |
| Precision | 0.94672 |
| Recall | 0.86021 |
| mAP50 | 0.92953 |
| mAP50-95 | 0.72314 |
| FPS | 148.42 |

CBAM 在本次增强数据集上训练稳定，但相较 Baseline 精度略低；其价值主要体现在验证了注意力模块可无缝接入 YOLOv8，并为后续组合结构提供可切换实现。

## 6. BiFPN 结果

BiFPN 阶段新增 `models/modules/bifpn.py`，通过 `enable_bifpn` 控制是否将 YOLOv8 原始 PAN-FPN 替换为 BiFPN Neck。BiFPN 使用可学习融合权重，并在前向过程中自动归一化。

Baseline+CBAM+BiFPN 正式验证结果如下：

| 指标 | 数值 |
| --- | ---: |
| Best epoch | 98 |
| Precision | 0.91858 |
| Recall | 0.86675 |
| mAP50 | 0.92321 |
| mAP50-95 | 0.71355 |
| FPS | 167.94 |

BiFPN 的精度略低于 Baseline 与 CBAM，但推理 FPS 是本次五组中最高，说明加权融合 Neck 在当前实现中有一定速度优势。

## 7. Focal 结果

Focal Loss 阶段新增 `models/losses/focal_loss.py`，只替换 YOLOv8 检测训练中的分类损失项；box loss、DFL、TaskAlignedAssigner 和模型结构保持 Ultralytics 原流程。

默认配置如下：

| 配置项 | 数值 |
| --- | --- |
| `enable_focal_loss` | `true` |
| `focal_gamma` | `2.0` |
| `focal_alpha` | `0.25` |

CBAM+BiFPN+Focal Loss 正式验证结果如下：

| 指标 | 数值 |
| --- | ---: |
| Best epoch | 98 |
| Precision | 0.50033 |
| Recall | 0.68291 |
| mAP50 | 0.61593 |
| mAP50-95 | 0.47776 |
| FPS | 95.43 |

Focal Loss 与 CBAM+BiFPN 的组合训练流程稳定，但在当前 `gamma=2.0`、`alpha=0.25` 配置和 100 epoch 训练条件下明显欠收敛，精度低于原始分类损失版本。因此后续推荐继续保留可配置实现，但默认最终权重不采用该组合。

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

本次正式消融实验已在本机 CUDA 环境完成，统一使用 `epochs=100`、`imgsz=640`、`batch=8`、`workers=0`、`device=auto` 与增强数据集 `dataset_augmented/data.yaml`。

正式运行命令如下：

```bash
.\.venv\Scripts\python.exe scripts\ablation.py --data dataset_augmented\data.yaml --epochs 100 --imgsz 640 --batch 8 --workers 0 --device auto
```

自动导出内容包括：

- `reports/ablation/summary.csv`
- `reports/ablation/summary.xlsx`
- `reports/ablation/history.csv`
- `reports/ablation/plots/pr_curve.png`
- `reports/ablation/plots/loss_curve.png`
- `reports/ablation/plots/map_curve.png`
- `reports/ablation/val/`

正式汇总结果如下：

| 实验 | Precision | Recall | mAP50 | mAP50-95 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.96646 | 0.90696 | 0.94706 | 0.74452 | 144.91 |
| Baseline+CBAM | 0.94672 | 0.86021 | 0.92953 | 0.72314 | 148.42 |
| Baseline+CBAM+BiFPN | 0.91858 | 0.86675 | 0.92321 | 0.71355 | 167.94 |
| Baseline+CBAM+BiFPN+Focal | 0.50033 | 0.68291 | 0.61593 | 0.47776 | 95.43 |
| FullModel | 0.43324 | 0.65441 | 0.52885 | 0.40597 | 60.29 |

消融结论：当前增强数据集与 100 epoch 训练设置下，Baseline 是精度最优模型；CBAM 与 CBAM+BiFPN 均保持 0.92 以上 mAP50，证明改进结构已可训练、可切换、可复现；Focal Loss 与 FullModel 组合存在欠收敛，需要下一阶段进一步调参或采用预训练/更长训练策略。

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

FullModel 已完成 100 epoch CUDA 正式训练，训练、验证和权重保存均正常。正式验证结果如下：

| 指标 | 数值 |
| --- | ---: |
| Best epoch | 93 |
| Precision | 0.43324 |
| Recall | 0.65441 |
| mAP50 | 0.52885 |
| mAP50-95 | 0.40597 |
| FPS | 60.29 |

由于 FullModel 同时叠加 GhostConv、CBAM、BiFPN、Decoupled Head 与 Focal Loss，结构变化较大；在当前训练轮数和 Focal 参数下并未达到 Baseline 精度。当前推荐的精度交付权重为 `reports/ablation/runs/baseline/weights/best.pt`，推荐的改进结构展示权重为 `reports/ablation/runs/baseline_cbam_bifpn/weights/best.pt`。

## 10. 下一阶段规划

1. 围绕 Focal Loss 与 FullModel 欠收敛问题继续实验，重点尝试更小 `gamma`、不同 `alpha`、预训练权重迁移、更长训练轮数和学习率策略。
2. 对 GhostConv 与 Decoupled Head 进行独立消融，分离判断轻量化骨干与解耦检测头对精度和速度的影响。
3. 以 Baseline best 权重作为当前精度交付模型，以 CBAM+BiFPN best 权重作为结构改进展示模型，继续补充混淆矩阵和类别级 AP 分析。
4. 使用 `benchmark.py` 在 CPU/GPU 上复测 FPS，并在树莓派或边缘设备上完成实机测速。
5. 使用 `export.py` 导出 ONNX/OpenVINO/NCNN，并验证部署后检测框、类别标签、置信度和计数输出一致性。
6. 补充 RTSP 独立接口与 UI 入口，完善视频流场景验收能力。
7. 整理最终实验报告，重点呈现“未改进 vs 改进后”的真实对比、有效改进与无效组合分析。
