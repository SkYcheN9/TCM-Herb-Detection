# FullModel 结构与 Diff 说明

## 改进前 Baseline

```mermaid
flowchart LR
    A["Input Image"] --> B["YOLOv8n Backbone"]
    B --> C["SPPF"]
    C --> D["PAN-FPN Neck"]
    D --> E["YOLO Detect Head"]
    E --> F["bbox / class / confidence"]
```

## 改进后 FullModel

```mermaid
flowchart LR
    A["Input Image"] --> B["GhostConv Backbone"]
    B --> C["CBAM Attention"]
    C --> D["SPPF"]
    D --> E["BiFPN Neck"]
    E --> F["DecoupledDetect Head"]
    F --> G["bbox / class / confidence"]
    H["Focal Loss"] -. "training cls loss" .-> F
```

## 模块 Diff

| 改进方向 | Baseline | FullModel | 代码入口 |
| --- | --- | --- | --- |
| 骨干网络优化 | 标准 Conv 下采样 | 部分 Backbone 下采样 Conv 替换为 GhostConv | `models/yolov8n_ghost.yaml`、`models/yolov8n_full.yaml` |
| 注意力机制 | 无显式注意力 | Backbone C2f 后插入 CBAM | `models/modules/cbam.py` |
| 特征融合 | PAN-FPN + Concat | BiFPN 加权双向融合 | `models/modules/bifpn.py` |
| 损失函数 | YOLOv8 分类 BCE | Focal Loss 分类项 | `models/losses/focal_loss.py` |
| 检测头 | YOLO Detect | DecoupledDetect 分类/回归解耦 | `models/modules/decoupled_head.py` |

## 配置开关

```yaml
enable_ghostconv: true
enable_cbam: true
enable_bifpn: true
enable_decoupled_head: true
enable_focal_loss: true
```

Baseline 保持 `configs/baseline.yaml` 与 `yolov8n.pt`，不启用上述结构改动。

## 正式训练结论

本次 5 组 CUDA 消融训练已完成，统一使用 `dataset_augmented/data.yaml`、`epochs=100`、`imgsz=640`、`batch=8`、`workers=0`。

| 实验 | mAP50 | mAP50-95 | FPS |
| --- | ---: | ---: | ---: |
| Baseline | 0.94706 | 0.74452 | 144.91 |
| Baseline+CBAM | 0.92953 | 0.72314 | 148.42 |
| Baseline+CBAM+BiFPN | 0.92321 | 0.71355 | 167.94 |
| Baseline+CBAM+BiFPN+Focal | 0.61593 | 0.47776 | 95.43 |
| FullModel | 0.52885 | 0.40597 | 60.29 |

当前精度交付推荐使用 Baseline best 权重；结构改进展示推荐使用 CBAM+BiFPN best 权重。FullModel 已跑通完整训练链路，但在当前 Focal 参数和训练轮数下欠收敛，后续需要继续调参或引入更长训练策略。
