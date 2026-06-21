# Experiment Pipeline

This note is for the final experiment round. It does not add model features; it only standardizes fair ablation, pretrained-transfer training, Focal Loss parameter search, and paper-ready result export.

## 1. Fair Ablation

After changing files under `data/`, rebuild the normalized and augmented datasets first:

```bash
python scripts/check_dataset.py --images data/images --labels data/labels --strict
python scripts/prepare_dataset.py --images data/images --labels data/labels --output dataset --mode copy
python scripts/augment_dataset.py --dataset-root dataset --output dataset_augmented --copies 2 --mosaic-count 250 --mixup-count 250
python scripts/check_dataset.py --dataset-root dataset_augmented --strict
```

Current cleaned dataset:

- Raw cleaned samples: 1049 image/label pairs
- Normalized split: 838 train pairs and 211 validation pairs
- Augmented training set: 838 originals + 1676 Albumentations + 250 Mosaic + 250 MixUp = 3014 train images
- Augmented validation set: 211 original validation images
- Augmented total: 3225 image/label pairs

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

## 2. Candidate Combination Check

After the 9-model ablation, run only the clean GhostConv combinations below if `Baseline+CBAM+BiFPN` is the current best structural model:

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

The candidate suite contains:

- `Baseline+CBAM+BiFPN+GhostConv`
- `Baseline+CBAM+BiFPN+GhostConv+DecoupledHead`

It intentionally disables Focal Loss so the result is not confounded by the weaker loss branch.

## 3. Focal Parameter Search

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

## 4. Paper Result Export

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
