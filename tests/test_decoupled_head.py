from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.modules import BiFPNFusion, CBAM, DecoupledDetect, register_ultralytics_modules


class DecoupledHeadTestCase(unittest.TestCase):
    def test_decoupled_model_yaml_builds(self) -> None:
        register_ultralytics_modules(enable_cbam=False, enable_bifpn=False, enable_decoupled_head=True)

        model = YOLO(str(ROOT / "models" / "yolov8n_decoupled.yaml"))
        detect_layers = [layer for layer in model.model.model if isinstance(layer, DecoupledDetect)]

        self.assertEqual(len(detect_layers), 1)
        self.assertEqual(model.model.stride.tolist(), [8.0, 16.0, 32.0])

    def test_full_model_yaml_combines_all_structural_modules(self) -> None:
        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True, enable_decoupled_head=True)

        model = YOLO(str(ROOT / "models" / "yolov8n_full.yaml"))
        cbam_layers = [layer for layer in model.model.model if isinstance(layer, CBAM)]
        bifpn_layers = [layer for layer in model.model.model if isinstance(layer, BiFPNFusion)]
        detect_layers = [layer for layer in model.model.model if isinstance(layer, DecoupledDetect)]

        self.assertEqual(len(cbam_layers), 4)
        self.assertEqual(len(bifpn_layers), 4)
        self.assertEqual(len(detect_layers), 1)
        self.assertEqual(model.model.stride.tolist(), [8.0, 16.0, 32.0])

    def test_cbam_bifpn_ghost_decoupled_yaml_builds(self) -> None:
        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True, enable_decoupled_head=True)

        model = YOLO(str(ROOT / "models" / "yolov8n_cbam_bifpn_ghost_decoupled.yaml"))
        cbam_layers = [layer for layer in model.model.model if isinstance(layer, CBAM)]
        bifpn_layers = [layer for layer in model.model.model if isinstance(layer, BiFPNFusion)]
        detect_layers = [layer for layer in model.model.model if isinstance(layer, DecoupledDetect)]

        self.assertEqual(len(cbam_layers), 4)
        self.assertEqual(len(bifpn_layers), 4)
        self.assertEqual(len(detect_layers), 1)
        self.assertEqual(model.model.stride.tolist(), [8.0, 16.0, 32.0])

    def test_decoupled_head_returns_training_dict(self) -> None:
        head = DecoupledDetect(nc=15, head_channels=64, ch=(32, 64, 128))
        head.train()
        features = [
            torch.randn(2, 32, 16, 16),
            torch.randn(2, 64, 8, 8),
            torch.randn(2, 128, 4, 4),
        ]

        output = head(features)

        self.assertIn("boxes", output)
        self.assertIn("scores", output)
        self.assertIn("feats", output)
        self.assertEqual(output["scores"].shape[1], 15)


if __name__ == "__main__":
    unittest.main()
