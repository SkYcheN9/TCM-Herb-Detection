from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.losses import (
    FocalClassificationLoss,
    FocalV8DetectionLoss,
    VarifocalClassificationLoss,
    register_focal_loss,
)
from models.modules import register_ultralytics_modules
from scripts.trainers import build_focal_trainer


class FocalLossTestCase(unittest.TestCase):
    def test_focal_loss_matches_bce_when_gamma_zero_and_alpha_disabled(self) -> None:
        logits = torch.tensor([[-1.5, 0.0, 2.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 1.0, 1.0]])
        focal = FocalClassificationLoss(gamma=0.0, alpha=None)

        focal_loss = focal(logits, targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_loss.mean().backward()

        self.assertTrue(torch.allclose(focal_loss, bce_loss))
        self.assertIsNotNone(logits.grad)

    def test_focal_loss_uses_hard_sign_for_soft_targets(self) -> None:
        logits = torch.tensor([[0.25, -0.5]], requires_grad=True)
        targets = torch.tensor([[0.6, 0.0]])
        focal = FocalClassificationLoss(gamma=1.0, alpha=None)

        focal_loss = focal(logits, targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        expected_factor = torch.stack((1 - prob[0, 0], prob[0, 1])).view(1, 2)

        self.assertTrue(torch.allclose(focal_loss, bce_loss * expected_factor))

    def test_varifocal_loss_accepts_soft_targets(self) -> None:
        logits = torch.tensor([[0.25, -0.5]], requires_grad=True)
        targets = torch.tensor([[0.6, 0.0]])
        loss_fn = VarifocalClassificationLoss(gamma=1.5, alpha=0.75)

        loss = loss_fn(logits, targets)
        loss.mean().backward()

        self.assertEqual(loss.shape, targets.shape)
        self.assertIsNotNone(logits.grad)

    def test_register_focal_loss_keeps_original_loss_switchable(self) -> None:
        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True)
        register_focal_loss()
        model = YOLO(str(ROOT / "models" / "yolov8n_cbam_bifpn.yaml")).model
        model.args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )
        model.enable_focal_loss = False

        criterion = model.init_criterion()

        self.assertIsInstance(criterion, v8DetectionLoss)

    def test_register_focal_loss_replaces_classification_loss(self) -> None:
        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True)
        register_focal_loss()
        model = YOLO(str(ROOT / "models" / "yolov8n_cbam_bifpn.yaml")).model
        model.args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )
        model.enable_focal_loss = True
        model.focal_loss_type = "soft_focal"
        model.focal_gamma = 1.5
        model.focal_alpha = 0.35

        criterion = model.init_criterion()

        self.assertIsInstance(criterion, FocalV8DetectionLoss)
        self.assertEqual(criterion.loss_type, "soft_focal")
        self.assertEqual(criterion.focal.gamma, 1.5)
        self.assertEqual(criterion.focal.alpha, 0.35)

    def test_focal_trainer_carries_configured_parameters(self) -> None:
        trainer_class = build_focal_trainer(
            enable_focal_loss=True,
            focal_gamma=2.25,
            focal_alpha=None,
            focal_loss_type="varifocal",
        )

        self.assertTrue(trainer_class.enable_focal_loss)
        self.assertEqual(trainer_class.focal_loss_type, "varifocal")
        self.assertEqual(trainer_class.focal_gamma, 2.25)
        self.assertIsNone(trainer_class.focal_alpha)


if __name__ == "__main__":
    unittest.main()
