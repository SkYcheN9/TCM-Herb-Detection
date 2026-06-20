from __future__ import annotations

import argparse
from pathlib import Path
import unittest

from scripts.focal_search import build_ablation_command, search_key


class FocalSearchTestCase(unittest.TestCase):
    def test_search_key_is_filesystem_friendly(self) -> None:
        self.assertEqual(search_key("soft_focal", "1.5", "none"), "soft_focal_g1p5_anone")

    def test_build_ablation_command_includes_focal_parameters(self) -> None:
        args = argparse.Namespace(
            output="reports/focal_search",
            experiments="cbam_bifpn_focal",
            data="dataset/data.yaml",
            init="pretrained",
            pretrained_weights="yolov8n.pt",
            epochs=150,
            imgsz=640,
            batch=16,
            workers=8,
            device="0",
            skip_train=False,
            skip_val=False,
        )

        command = build_ablation_command(args, "soft_focal_g1p0_anone", "soft_focal", "1.0", "none")

        self.assertIn("scripts/ablation.py", command)
        self.assertIn("--output", command)
        self.assertIn(str(Path("reports/focal_search") / "soft_focal_g1p0_anone"), command)
        self.assertIn("--focal-loss-type", command)
        self.assertIn("soft_focal", command)
        self.assertIn("--focal-alpha", command)
        self.assertIn("none", command)
        self.assertIn("--init", command)
        self.assertIn("pretrained", command)


if __name__ == "__main__":
    unittest.main()
