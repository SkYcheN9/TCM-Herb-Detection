from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.modules.cbam import CBAM, register_ultralytics_modules


class CbamTestCase(unittest.TestCase):
    def test_cbam_preserves_shape_and_backpropagates(self) -> None:
        module = CBAM(reduction=8, kernel_size=7)
        x = torch.randn(2, 32, 20, 20, requires_grad=True)

        y = module(x)
        y.mean().backward()

        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(x.grad)

    def test_cbam_model_yaml_builds_with_ultralytics(self) -> None:
        register_ultralytics_modules()

        model = YOLO(str(ROOT / "models" / "yolov8n_cbam.yaml"))
        cbam_layers = [layer for layer in model.model.model if isinstance(layer, CBAM)]

        self.assertEqual(len(cbam_layers), 4)
        self.assertTrue(all(layer.channel_attention is not None for layer in cbam_layers))
        self.assertGreater(sum(p.numel() for layer in cbam_layers for p in layer.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
