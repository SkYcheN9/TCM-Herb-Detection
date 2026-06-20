# Experiment Pipeline

This note is for the final experiment round. It does not add model features; it only standardizes fair ablation, pretrained-transfer training, Focal Loss parameter search, and paper-ready result export.

## 1. Fair Ablation

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

Outputs:

- `reports/ablation/summary.csv`
- `reports/ablation/summary.xlsx`
- `reports/ablation/history.csv`
- `reports/ablation/plots/`
- `reports/ablation/runs/*/weights/best.pt`

## 2. Focal Parameter Search

```bash
python scripts/focal_search.py \
  --output reports/focal_search \
  --experiments cbam_bifpn_focal \
  --data dataset_augmented/data.yaml \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --init pretrained \
  --pretrained-weights yolov8n.pt \
  --loss-types soft_focal,varifocal \
  --gammas 0.5,1.0,1.5,2.0 \
  --alphas none,0.25,0.5,0.75
```

Outputs:

- `reports/focal_search/summary.csv`
- `reports/focal_search/summary.xlsx`
- One nested ablation report directory for each Focal parameter combination.

## 3. Paper Result Export

```bash
python scripts/paper_results.py \
  --ablation-summary reports/ablation/summary.csv \
  --focal-summary reports/focal_search/summary.csv \
  --output reports/final_experiments
```

Outputs:

- `reports/final_experiments/paper_table.csv`
- `reports/final_experiments/paper_results.md`

## Interpretation Rule

If the improved model is lower than Baseline, report it directly. The current analysis should focus on fair causes: pretrained initialization mismatch, partial transfer coverage in modified structures, small-data optimization difficulty, and Focal settings that may over-suppress classification gradients.
