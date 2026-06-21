# AutoDL 公平消融训练说明

## 当前修正

本阶段修正两处影响消融公平性的训练问题：

1. 改进模型可通过 `--init pretrained` 从 `yolov8n.pt` 迁移同形状权重，新增模块保持随机初始化。
2. Focal Loss 默认改为兼容 YOLOv8 软标签的 `soft_focal`，旧实现保留为 `legacy_focal` 仅用于复现实验。

## 推荐正式命令

默认核心实验：

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

完整单模块 + 组合消融共 9 组：

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

## 候选组合补跑

已完成 9 组消融后，若要验证 GhostConv 是否能继续提升 `CBAM+BiFPN`，只补跑下面 2 组即可：

- `Baseline+CBAM+BiFPN+GhostConv`
- `Baseline+CBAM+BiFPN+GhostConv+DecoupledHead`

这两组不启用 Focal Loss，用于判断 GhostConv 和 Decoupled Head 在当前最佳结构上的叠加价值。

```bash
python scripts/ablation.py \
  --experiments candidate \
  --output reports/ablation_candidate \
  --data dataset_augmented/data.yaml \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --init pretrained \
  --pretrained-weights yolov8n.pt
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

最终训练已完成后，模型选型以 `docs/FINAL_MODEL_SELECTION.md` 为准：

- 网页端/桌面端默认部署 `Baseline+CBAM+BiFPN`
- 最高精度参考模型为 `Baseline+CBAM`
- 树莓派 5 无算力棒端默认部署 `Baseline+GhostConv`
- Focal Loss 与 FullModel 作为负向消融结论保留

若重新训练，完成后回传 `reports/ablation/summary.csv`、`summary.xlsx`、`history.csv`、`plots/` 和各实验 `weights/best.pt` 即可更新最终报告。
