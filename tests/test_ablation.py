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
    experiment_metadata,
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

    def test_default_experiments_excludes_extended_single_module_runs(self) -> None:
        selected = selected_experiments("default")

        self.assertEqual(
            [experiment.key for experiment in selected],
            ["baseline", "cbam", "cbam_bifpn", "cbam_bifpn_focal", "full_model"],
        )

    def test_extended_experiments_includes_single_module_runs(self) -> None:
        selected = selected_experiments("extended")
        keys = [experiment.key for experiment in selected]

        self.assertIn("bifpn", keys)
        self.assertIn("focal", keys)
        self.assertIn("ghostconv", keys)
        self.assertIn("decoupled_head", keys)
        self.assertEqual(len(keys), 9)
        self.assertNotIn("cbam_bifpn_ghost", keys)

    def test_candidate_experiments_only_selects_clean_combinations(self) -> None:
        selected = selected_experiments("candidate")

        self.assertEqual(
            [experiment.key for experiment in selected],
            ["cbam_bifpn_ghost", "cbam_bifpn_ghost_decoupled"],
        )
        self.assertTrue(all(not experiment.enable_focal_loss for experiment in selected))

    def test_train_command_overrides_common_training_options(self) -> None:
        args = argparse.Namespace(
            data="dataset/data.yaml",
            epochs=2,
            imgsz=320,
            batch=4,
            workers=0,
            device="0",
            init="pretrained",
            pretrained_weights="yolov8n.pt",
            focal_loss_type=None,
            focal_gamma=None,
            focal_alpha=None,
        )
        command = train_command(args, EXPERIMENTS[0], Path("reports/ablation/runs"))

        self.assertIn("train.py", command)
        self.assertIn("configs/baseline.yaml", command)
        self.assertIn("--epochs", command)
        self.assertIn("2", command)
        self.assertIn("--device", command)
        self.assertIn("0", command)
        self.assertIn("--init", command)
        self.assertIn("pretrained", command)

    def test_train_command_forwards_focal_options_only_for_focal_experiments(self) -> None:
        args = argparse.Namespace(
            data="dataset/data.yaml",
            epochs=None,
            imgsz=None,
            batch=None,
            workers=None,
            device=None,
            init=None,
            pretrained_weights=None,
            focal_loss_type="varifocal",
            focal_gamma=1.5,
            focal_alpha="0.75",
        )
        focal_experiment = next(experiment for experiment in EXPERIMENTS if experiment.key == "focal")
        baseline_command = train_command(args, EXPERIMENTS[0], Path("reports/ablation/runs"))
        focal_command = train_command(args, focal_experiment, Path("reports/ablation/runs"))

        self.assertNotIn("--focal-loss-type", baseline_command)
        self.assertIn("--focal-loss-type", focal_command)
        self.assertIn("varifocal", focal_command)

    def test_experiment_metadata_records_fairness_controls(self) -> None:
        args = argparse.Namespace(
            data="dataset/data.yaml",
            epochs=150,
            imgsz=640,
            batch=16,
            workers=8,
            device="0",
            init="pretrained",
            pretrained_weights="yolov8n.pt",
            focal_loss_type="soft_focal",
            focal_gamma=1.0,
            focal_alpha="none",
        )
        focal_experiment = next(experiment for experiment in EXPERIMENTS if experiment.key == "focal")

        metadata = experiment_metadata(args, focal_experiment)

        self.assertTrue(metadata["FocalLoss"])
        self.assertEqual(metadata["Init"], "pretrained")
        self.assertEqual(metadata["PretrainedWeights"], "yolov8n.pt")
        self.assertEqual(metadata["FocalLossType"], "soft_focal")
        self.assertEqual(metadata["FocalGamma"], 1.0)

    def test_summarize_metrics_exports_requested_metrics_and_fps(self) -> None:
        args = argparse.Namespace(
            data="dataset/data.yaml",
            epochs=None,
            imgsz=None,
            batch=None,
            workers=None,
            device=None,
            init=None,
            pretrained_weights=None,
            focal_loss_type=None,
            focal_gamma=None,
            focal_alpha=None,
        )
        metrics = SimpleNamespace(
            speed={"preprocess": 1.0, "inference": 8.0, "postprocess": 1.0},
            box=SimpleNamespace(mp=0.8, mr=0.7, map50=0.6, map=0.5),
        )

        row = summarize_metrics(args, EXPERIMENTS[0], Path("best.pt"), metrics)

        self.assertEqual(row["Precision"], 0.8)
        self.assertEqual(row["Recall"], 0.7)
        self.assertEqual(row["mAP50"], 0.6)
        self.assertEqual(row["mAP50-95"], 0.5)
        self.assertEqual(row["FPS"], 100.0)
        self.assertIn("Init", SUMMARY_FIELDS)
        self.assertEqual(row["CBAM"], False)

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
