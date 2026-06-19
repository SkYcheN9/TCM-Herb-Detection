from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import numpy as np

from scripts.ablation import (
    EXPERIMENTS,
    SUMMARY_FIELDS,
    metric_float,
    precision_recall_curve,
    selected_experiments,
    summarize_metrics,
    train_command,
    write_xlsx,
)


class AblationTestCase(unittest.TestCase):
    def test_selected_experiments_keeps_canonical_order(self) -> None:
        selected = selected_experiments("cbam_bifpn,baseline")

        self.assertEqual([experiment.key for experiment in selected], ["baseline", "cbam_bifpn"])

    def test_train_command_overrides_common_training_options(self) -> None:
        args = argparse.Namespace(
            data="dataset/data.yaml",
            epochs=2,
            imgsz=320,
            batch=4,
            workers=0,
            device="0",
        )
        command = train_command(args, EXPERIMENTS[0], Path("reports/ablation/runs"))

        self.assertIn("train.py", command)
        self.assertIn("configs/baseline.yaml", command)
        self.assertIn("--epochs", command)
        self.assertIn("2", command)
        self.assertIn("--device", command)
        self.assertIn("0", command)

    def test_summarize_metrics_exports_requested_metrics_and_fps(self) -> None:
        metrics = SimpleNamespace(
            speed={"preprocess": 1.0, "inference": 8.0, "postprocess": 1.0},
            box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.6, map=0.5),
        )

        row = summarize_metrics(EXPERIMENTS[0], Path("best.pt"), metrics)

        self.assertEqual(row["Precision"], 0.8)
        self.assertEqual(row["Recall"], 0.7)
        self.assertEqual(row["mAP50"], 0.6)
        self.assertEqual(row["mAP50-95"], 0.5)
        self.assertEqual(row["FPS"], 100.0)

    def test_metric_float_ignores_invalid_values(self) -> None:
        self.assertEqual(metric_float({"x": "0.25"}, "x"), 0.25)
        self.assertIsNone(metric_float({"x": ""}, "x"))
        self.assertIsNone(metric_float({"x": ["bad"]}, "x"))

    def test_precision_recall_curve_averages_class_curves(self) -> None:
        metrics = SimpleNamespace(
            curves_results=[
                [
                    np.array([0.0, 0.5, 1.0]),
                    np.array([[1.0, 0.7, 0.2], [0.8, 0.6, 0.4]]),
                    "Recall",
                    "Precision",
                ]
            ]
        )

        curve = precision_recall_curve(metrics)

        self.assertIsNotNone(curve)
        recall, precision = curve
        self.assertTrue(np.allclose(recall, [0.0, 0.5, 1.0]))
        self.assertTrue(np.allclose(precision, [0.9, 0.65, 0.3]))

    def test_write_xlsx_creates_valid_workbook_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.xlsx"
            write_xlsx(
                path,
                [{"Experiment": "Baseline", "mAP50": 0.5, "FPS": 33.3}],
                ["Experiment", "mAP50", "FPS"],
            )

            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")

        self.assertIn("[Content_Types].xml", names)
        self.assertIn("xl/workbook.xml", names)
        self.assertIn("Baseline", sheet)
        self.assertIn("<v>0.5</v>", sheet)


if __name__ == "__main__":
    unittest.main()
