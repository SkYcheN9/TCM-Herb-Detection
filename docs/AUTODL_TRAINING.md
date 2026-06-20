# AutoDL 公平消融训练说明

## 当前修正

本阶段修正两处影响消融公平性的训练问题：

1. 改进模型可通过 `--init pretrained` 从 `yolov8n.pt` 迁移同形状权重，新增模块保持随机初始化。
2. Focal Loss 默认改为兼容 YOLOv8 软标签的 `soft_focal`，旧实现保留为 `legacy_focal` 仅用于复现实验。

## 推荐正式命令

默认 5 组实验：

```bash
python scripts/ablation.py \
  --data dataset_augmented/data.yaml \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --init pretrained \
  --pretrained-weights yolov8n.pt \
  --focal-loss-type soft_focal \
  --focal-gamma 1.0 \
  --focal-alpha none
```

完整单模块 + 组合消融：

```bash
python scripts/ablation.py \
  --experiments extended \
  --data dataset_augmented/data.yaml \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --init pretrained \
  --pretrained-weights yolov8n.pt \
  --focal-loss-type soft_focal \
  --focal-gamma 1.0 \
  --focal-alpha none
```

## Focal Loss 小网格

建议先在 `cbam_bifpn_focal` 上筛选损失参数：

```bash
python scripts/ablation.py --experiments cbam_bifpn_focal --data dataset_augmented/data.yaml --epochs 150 --imgsz 640 --batch 16 --workers 8 --device 0 --init pretrained --focal-loss-type soft_focal --focal-gamma 0.5 --focal-alpha none
python scripts/ablation.py --experiments cbam_bifpn_focal --data dataset_augmented/data.yaml --epochs 150 --imgsz 640 --batch 16 --workers 8 --device 0 --init pretrained --focal-loss-type soft_focal --focal-gamma 1.0 --focal-alpha none
python scripts/ablation.py --experiments cbam_bifpn_focal --data dataset_augmented/data.yaml --epochs 150 --imgsz 640 --batch 16 --workers 8 --device 0 --init pretrained --focal-loss-type varifocal --focal-gamma 1.5 --focal-alpha 0.75
```

旧结果复现：

```bash
python scripts/ablation.py --experiments cbam_bifpn_focal --data dataset_augmented/data.yaml --epochs 100 --imgsz 640 --batch 8 --workers 0 --device 0 --focal-loss-type legacy_focal --focal-gamma 2.0 --focal-alpha 0.25
```

训练完成后回传 `reports/ablation/summary.csv`、`summary.xlsx`、`history.csv`、`plots/` 和各实验 `weights/best.pt` 即可更新最终报告。
