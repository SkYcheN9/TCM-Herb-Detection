from __future__ import annotations

from pathlib import Path
import sys
import unittest

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.modules import register_ultralytics_modules
from scripts.pretrained import transfer_pretrained_weights


class PretrainedTransferTestCase(unittest.TestCase):
    def test_transfer_pretrained_weights_reports_copied_tensors(self) -> None:
        register_ultralytics_modules(enable_cbam=True, enable_bifpn=False)
        target = YOLO(str(ROOT / "models" / "yolov8n_cbam.yaml")).model

        report = transfer_pretrained_weights(target, "yolov8n.pt")

        self.assertGreater(report.transferred_tensors, 0)
        self.assertGreater(report.total_target_tensors, report.transferred_tensors)
        self.assertIn("Transferred", report.summary())

    def test_bifpn_transfer_leaves_new_fusion_layers_random(self) -> None:
        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True)
        target = YOLO(str(ROOT / "models" / "yolov8n_cbam_bifpn.yaml")).model

        report = transfer_pretrained_weights(target, "yolov8n.pt")

        self.assertGreater(report.transferred_tensors, 0)
        self.assertGreater(report.skipped_tensors, 0)


if __name__ == "__main__":
    unittest.main()
