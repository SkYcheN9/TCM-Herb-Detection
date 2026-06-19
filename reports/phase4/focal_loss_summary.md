# Focal Loss 说明

本阶段只替换 YOLOv8 检测训练中的分类损失项，box loss、DFL、TaskAlignedAssigner 和模型结构保持原 Ultralytics 流程。

## 配置项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `enable_focal_loss` | `false` | 是否启用 Focal Loss |
| `focal_gamma` | `2.0` | Focal Loss 聚焦系数 |
| `focal_alpha` | `0.25` | 类别平衡系数；可设为 `none` 禁用 alpha weighting |

## 切换方式

原始 YOLOv8 分类 BCE：

```yaml
enable_focal_loss: false
```

Focal Loss：

```yaml
enable_focal_loss: true
focal_gamma: 2.0
focal_alpha: 0.25
```

CBAM + BiFPN + Focal Loss：

```bash
.\.venv\Scripts\python.exe train.py --config configs/cbam_bifpn_focal.yaml
```

## 实现说明

- `models/losses/focal_loss.py` 新增 `FocalClassificationLoss`。
- `FocalV8DetectionLoss` 包装 Ultralytics 原始 `v8DetectionLoss`，只替换分类分支。
- `scripts/trainers.py` 注入 Focal Loss 配置，避免把自定义参数直接传给 Ultralytics 配置校验。
- `enable_focal_loss: false` 时仍调用原始 `DetectionModel.init_criterion()`，保留原始 Loss 可切换。

## 兼容性

- Focal Loss 不依赖 Neck 或 Backbone，因此兼容 Baseline、CBAM、BiFPN、CBAM+BiFPN。
- 已提供 `configs/cbam_bifpn_focal.yaml` 作为组合训练入口。
