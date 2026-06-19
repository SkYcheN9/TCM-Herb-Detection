from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.modules import BiFPNFusion, CBAM, register_ultralytics_modules


class BiFPNTestCase(unittest.TestCase):
    def test_bifpn_fusion_normalizes_weights_and_preserves_shape(self) -> None:
        module = BiFPNFusion(num_inputs=3, channels=32)
        inputs = [
            torch.randn(2, 32, 20, 20, requires_grad=True),
            torch.randn(2, 32, 20, 20, requires_grad=True),
            torch.randn(2, 32, 20, 20, requires_grad=True),
        ]

        y = module(inputs)
        y.mean().backward()
        weights = module.normalized_weights()

        self.assertEqual(y.shape, inputs[0].shape)
        self.assertAlmostEqual(float(weights.sum().detach()), 1.0, places=3)
        self.assertIsNotNone(module.weights.grad)

    def test_bifpn_model_yaml_replaces_pan_fpn(self) -> None:
        register_ultralytics_modules(enable_cbam=False, enable_bifpn=True)

        model = YOLO(str(ROOT / "models" / "yolov8n_bifpn.yaml"))
        bifpn_layers = [layer for layer in model.model.model if isinstance(layer, BiFPNFusion)]

        self.assertEqual(len(bifpn_layers), 4)
        self.assertFalse(any(layer.type.endswith("Concat") for layer in model.model.model))
        self.assertEqual(model.model.stride.tolist(), [8.0, 16.0, 32.0])

    def test_cbam_bifpn_model_yaml_builds(self) -> None:
        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True)

        model = YOLO(str(ROOT / "models" / "yolov8n_cbam_bifpn.yaml"))
        bifpn_layers = [layer for layer in model.model.model if isinstance(layer, BiFPNFusion)]
        cbam_layers = [layer for layer in model.model.model if isinstance(layer, CBAM)]

        self.assertEqual(len(bifpn_layers), 4)
        self.assertEqual(len(cbam_layers), 4)
        self.assertEqual(model.model.stride.tolist(), [8.0, 16.0, 32.0])


if __name__ == "__main__":
    unittest.main()
