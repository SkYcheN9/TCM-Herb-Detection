"""Focal Loss integration for Ultralytics YOLOv8 detection training."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalClassificationLoss(nn.Module):
    """Soft-label focal loss for YOLOv8 quality-aware classification targets."""

    def __init__(self, gamma: float = 1.0, alpha: float | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be >= 0")
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("focal alpha must be in [0, 1] or None")
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return element-wise focal loss that keeps YOLOv8 soft targets intact."""

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        hard_targets = targets.gt(0).to(dtype=targets.dtype)
        p_t = hard_targets * prob + (1 - hard_targets) * (1 - prob)
        focal_factor = (1 - p_t).clamp(min=0).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = hard_targets * self.alpha + (1 - hard_targets) * (1 - self.alpha)
            focal_factor = alpha_t * focal_factor
        return bce_loss * focal_factor


class LegacyFocalClassificationLoss(FocalClassificationLoss):
    """Original project focal loss kept only for exact experiment reproduction."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25) -> None:
        super().__init__(gamma=gamma, alpha=alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the legacy element-wise focal loss over raw soft targets."""

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = targets * prob + (1 - targets) * (1 - prob)
        focal_factor = (1 - p_t).clamp(min=0).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_factor = alpha_t * focal_factor
        return bce_loss * focal_factor


class VarifocalClassificationLoss(nn.Module):
    """Varifocal loss for YOLOv8 soft quality targets."""

    def __init__(self, gamma: float = 1.5, alpha: float | None = 0.75) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("varifocal gamma must be >= 0")
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("varifocal alpha must be in [0, 1] or None")
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return element-wise varifocal loss."""

        prob = torch.sigmoid(logits)
        hard_targets = targets.gt(0).to(dtype=targets.dtype)
        negative_weight = prob.pow(self.gamma) * (1 - hard_targets)
        if self.alpha is not None:
            negative_weight = self.alpha * negative_weight
        weight = targets * hard_targets + negative_weight
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return bce_loss * weight


class FocalV8DetectionLoss:
    """YOLOv8 detection loss with only the classification term replaced."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        from ultralytics.utils.loss import v8DetectionLoss

        self.base_loss = v8DetectionLoss(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        loss_type = str(getattr(model, "focal_loss_type", "soft_focal")).lower()
        gamma = float(getattr(model, "focal_gamma", 1.0))
        alpha_value = getattr(model, "focal_alpha", None)
        alpha = None if alpha_value is None or str(alpha_value).lower() == "none" else float(alpha_value)
        self.loss_type = loss_type
        self.focal = build_classification_loss(loss_type, gamma=gamma, alpha=alpha)

    def __getattr__(self, name: str):
        """Delegate Ultralytics loss attributes to the wrapped v8 loss."""

        if name in {"base_loss", "focal"}:
            raise AttributeError(name)
        return getattr(self.base_loss, name)

    def get_assigned_targets_and_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple:
        """Calculate YOLOv8 losses with focal classification loss."""

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        from ultralytics.utils.tal import make_anchors

        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        cls_loss = self.focal(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            cls_loss *= self.class_weights
        loss[1] = cls_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    def parse_output(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""

        return self.base_loss.parse_output(preds)

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate total detection loss and detached loss components."""

        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple:
        """Calculate loss using the wrapped assignment and bbox loss logic."""

        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


def register_focal_loss() -> None:
    """Patch Ultralytics DetectionModel to use focal classification loss."""

    import ultralytics.nn.tasks as tasks

    if getattr(tasks.DetectionModel.init_criterion, "_tcm_focal_patched", False):
        return

    original_init_criterion = tasks.DetectionModel.init_criterion

    def init_criterion(self):
        if getattr(self, "enable_focal_loss", False):
            return FocalV8DetectionLoss(self)
        return original_init_criterion(self)

    init_criterion._tcm_focal_patched = True
    init_criterion._tcm_original_init_criterion = original_init_criterion
    tasks.DetectionModel.init_criterion = init_criterion


def build_classification_loss(
    loss_type: str,
    gamma: float,
    alpha: float | None,
) -> nn.Module:
    """Build the configured classification-loss replacement."""

    normalized = loss_type.lower()
    if normalized in {"soft_focal", "focal"}:
        return FocalClassificationLoss(gamma=gamma, alpha=alpha)
    if normalized == "legacy_focal":
        return LegacyFocalClassificationLoss(gamma=gamma, alpha=alpha)
    if normalized == "varifocal":
        return VarifocalClassificationLoss(gamma=gamma, alpha=alpha)
    raise ValueError(
        "Unknown focal loss type "
        f"{loss_type!r}; expected soft_focal, legacy_focal, or varifocal"
    )
