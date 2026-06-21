# Phase 1 项目报告

## 1. 项目背景

本项目面向中医药饮片智能检测与识别场景，目标是在规范化饮片图像数据集的基础上，构建基于 Ultralytics YOLOv8 的多类别目标检测模型，实现 15 类中药饮片的定位、识别、置信度输出与自动计数能力。

当前已完成数据集规范化、离线数据增强、YOLOv8 Baseline 训练、CBAM 注意力模块、BiFPN Neck、Focal Loss 分类损失替换、GhostConv 轻量化骨干、Decoupled Head 解耦检测头，以及 11 组公平消融实验。所有改进模块均通过配置开关启用，不破坏原始 Baseline 训练流程。

## 2. 数据集规模

| 项目 | 数量/状态 |
| --- | ---: |
| 原始图片 | 1049 |
| 原始标注文件 | 1049 |
| 有效样本 | 1049 |
| 训练集样本 | 838 |
| 验证集样本 | 211 |
| 训练/验证比例 | 约 8:2 |
| 类别数 | 15 |
| 数据集阻断问题 | 0 |

规范化后的数据集采用 YOLO 标准目录结构，配置文件为 `dataset/data.yaml`。类别顺序固定，不允许训练或检查工具自动重排类别。

正式消融训练统一使用增强数据集 `dataset_augmented/data.yaml`。增强集训练部分包含原始训练样本、Albumentations 增强样本、Mosaic 样本、MixUp 样本、HSV 扰动和随机裁剪结果；训练集共 3014 张，验证集保持 211 张不增强，总计 3225 对样本，以保证各模型在同一验证集上公平比较。

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

## 4. 改进模块

| 改进方向 | 作用 | 代码入口 |
| --- | --- | --- |
| CBAM 注意力 | 强化通道和空间注意力表达 | `models/modules/cbam.py` |
| BiFPN Neck | 使用可学习权重进行双向特征融合 | `models/modules/bifpn.py` |
| Focal Loss | 替换分类 BCE，分析难易样本不均衡影响 | `models/losses/focal_loss.py` |
| GhostConv | 降低部分 Backbone 卷积计算量 | `models/yolov8n_ghost.yaml` |
| Decoupled Head | 分离分类与回归分支 | `models/modules/decoupled_head.py` |

FullModel 定义为：

```text
GhostConv + CBAM + BiFPN + Decoupled Head + Focal Loss
```

## 5. 公平消融设置

最终消融实验使用相同训练集、验证集、图像尺寸、训练轮数和预训练迁移策略：

| 项目 | 配置 |
| --- | --- |
| 训练数据 | `dataset_augmented/data.yaml` |
| 预训练权重 | `yolov8n.pt` |
| 训练轮数 | 150 |
| 图像尺寸 | 640 |
| Batch size | 16 |
| Workers | 8 |
| 设备 | CUDA:0 |
| Focal 类型 | `soft_focal` |
| Focal gamma | 1.0 |
| Focal alpha | none |

主消融命令：

```bash
python scripts/ablation.py --experiments extended --data dataset_augmented/data.yaml --epochs 150 --imgsz 640 --batch 16 --workers 8 --device 0 --init pretrained --pretrained-weights yolov8n.pt --focal-loss-type soft_focal --focal-gamma 1.0 --focal-alpha none
```

候选组合补跑命令：

```bash
python scripts/ablation.py --experiments candidate --output reports/ablation_candidate --data dataset_augmented/data.yaml --epochs 150 --imgsz 640 --batch 16 --workers 8 --device 0 --init pretrained --pretrained-weights yolov8n.pt
```

## 6. 最终消融结果

| 排名 | 模型 | Precision | Recall | mAP50 | mAP50-95 | FPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline+CBAM | 0.99425 | 0.99366 | 0.99227 | 0.80125 | 242.11 |
| 2 | Baseline+CBAM+BiFPN | 0.99161 | 0.99051 | 0.99162 | 0.80076 | 302.73 |
| 3 | Baseline+GhostConv | 0.98803 | 0.98988 | 0.98915 | 0.79822 | 306.11 |
| 4 | Baseline | 0.99332 | 0.99495 | 0.99118 | 0.79532 | 304.50 |
| 5 | Baseline+DecoupledHead | 0.99246 | 0.99455 | 0.99133 | 0.79500 | 235.72 |
| 6 | Baseline+CBAM+BiFPN+GhostConv+DecoupledHead | 0.97997 | 0.98383 | 0.99026 | 0.79446 | 280.49 |
| 7 | Baseline+BiFPN | 0.99197 | 0.99362 | 0.99251 | 0.79234 | 299.28 |
| 8 | Baseline+CBAM+BiFPN+GhostConv | 0.98294 | 0.98299 | 0.98853 | 0.79027 | 287.42 |
| 9 | Baseline+CBAM+BiFPN+Focal | 0.98482 | 0.98622 | 0.99027 | 0.78909 | 198.90 |
| 10 | FullModel | 0.97765 | 0.97535 | 0.99012 | 0.78472 | 226.29 |
| 11 | Baseline+Focal | 0.98285 | 0.98031 | 0.98906 | 0.78438 | 235.14 |

## 7. 结果分析

`Baseline+CBAM` 获得最高 mAP50-95，为 0.80125，说明注意力机制能够提升饮片局部纹理和形态特征表达。

`Baseline+CBAM+BiFPN` 的 mAP50-95 为 0.80076，仅比最高精度模型低 0.00049，但 FPS 从 242.11 提升到 302.73，因此更适合作为网页端和桌面端默认部署模型。

`Baseline+GhostConv` 的 mAP50-95 为 0.79822，FPS 为 306.11，是 11 个模型中速度最高的轻量化模型，因此适合作为 Raspberry Pi 5 8GB 无算力棒端默认部署模型。

Focal Loss 相关模型和 FullModel 未取得最佳结果。`Baseline+Focal`、`Baseline+CBAM+BiFPN+Focal` 与 `FullModel` 的 mAP50-95 均低于 CBAM/CBAM+BiFPN/GhostConv，说明当前数据集规模和类别分布下，Focal Loss 的难易样本重加权反而削弱了 YOLOv8 原有分类损失与任务分配机制的稳定性。因此 Focal Loss 应作为已完成的负向消融结论保留，而不作为最终部署方案。

## 8. 最终部署选型

| 部署端 | 推荐模型 | 原因 |
| --- | --- | --- |
| Web 端 | Baseline+CBAM+BiFPN | 精度接近最高，速度明显更优 |
| PC 桌面端 | Baseline+CBAM+BiFPN | 适合本地图片、视频、摄像头检测与药材计数 |
| Raspberry Pi 5 8GB 无算力棒 | Baseline+GhostConv | 速度最高，更适合 CPU/OpenVINO 轻量部署 |
| 论文最高精度参考 | Baseline+CBAM | mAP50-95 最高 |

完整选型说明见 `docs/FINAL_MODEL_SELECTION.md`。

## 9. 自动导出内容

消融脚本自动导出：

- `summary.csv` / `summary.xlsx`
- `history.csv`
- `plots/loss_curve.png`
- `plots/map_curve.png`
- `plots/pr_curve.png`
- 各实验 `runs/*/weights/best.pt`
- Ultralytics 验证图表、PR 曲线和混淆矩阵

项目部署侧已补充 Web、FastAPI、PySide6 桌面端与 Raspberry Pi 导出脚本，并在检测接口和界面中支持总数统计、英文类别计数和中文药材计数。
