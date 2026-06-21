from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_slice_ai.dataset import write_data_yaml


class DatasetHelpersTestCase(unittest.TestCase):
    def test_write_data_yaml_uses_portable_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)

            data_yaml = write_data_yaml(dataset_root)
            content = data_yaml.read_text(encoding="utf-8")

        self.assertNotIn("path:", content)
        self.assertIn("train: images/train", content)
        self.assertIn("val: images/val", content)
        self.assertIn("0: zexie", content)


if __name__ == "__main__":
    unittest.main()
