from __future__ import annotations

from pathlib import Path
import sys
import unittest

from ultralytics import YOLO
from ultralytics.nn.modules.conv import GhostConv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.modules import BiFPNFusion, CBAM, register_ultralytics_modules


class GhostConvTestCase(unittest.TestCase):
    def test_ghostconv_model_yaml_builds(self) -> None:
        model = YOLO(str(ROOT / "models" / "yolov8n_ghost.yaml"))
        ghost_layers = [layer for layer in model.model.model if isinstance(layer, GhostConv)]

        self.assertEqual(len(ghost_layers), 4)
        self.assertEqual(model.model.stride.tolist(), [8.0, 16.0, 32.0])

    def test_ghostconv_backbone_is_lighter_than_baseline_yaml(self) -> None:
        baseline = YOLO("yolov8n.yaml")
        ghost = YOLO(str(ROOT / "models" / "yolov8n_ghost.yaml"))

        baseline_params = sum(parameter.numel() for parameter in baseline.model.parameters())
        ghost_params = sum(parameter.numel() for parameter in ghost.model.parameters())

        self.assertLess(ghost_params, baseline_params)

    def test_cbam_bifpn_ghost_model_yaml_builds(self) -> None:
        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True, enable_decoupled_head=False)

        model = YOLO(str(ROOT / "models" / "yolov8n_cbam_bifpn_ghost.yaml"))
        ghost_layers = [layer for layer in model.model.model if isinstance(layer, GhostConv)]
        cbam_layers = [layer for layer in model.model.model if isinstance(layer, CBAM)]
        bifpn_layers = [layer for layer in model.model.model if isinstance(layer, BiFPNFusion)]

        self.assertEqual(len(ghost_layers), 4)
        self.assertEqual(len(cbam_layers), 4)
        self.assertEqual(len(bifpn_layers), 4)
        self.assertEqual(model.model.stride.tolist(), [8.0, 16.0, 32.0])


if __name__ == "__main__":
    unittest.main()
