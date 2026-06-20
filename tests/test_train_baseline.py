from __future__ import annotations

import unittest

from scripts.train_baseline import scratch_model_path


class TrainBaselineTestCase(unittest.TestCase):
    def test_scratch_model_path_replaces_known_pt_weights_with_yaml(self) -> None:
        self.assertEqual(scratch_model_path("yolov8n.pt"), "yolov8n.yaml")
        self.assertEqual(scratch_model_path("models/yolov8n_cbam.yaml"), "models/yolov8n_cbam.yaml")


if __name__ == "__main__":
    unittest.main()
