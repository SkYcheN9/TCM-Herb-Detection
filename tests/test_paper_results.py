from __future__ import annotations

import unittest

from scripts.paper_results import add_deltas, best_row, generate_report, project_fields


class PaperResultsTestCase(unittest.TestCase):
    def test_add_deltas_uses_baseline(self) -> None:
        rows = [
            {"Experiment": "Baseline", "mAP50": "0.90", "mAP50-95": "0.70", "FPS": "100"},
            {"Experiment": "Improved", "mAP50": "0.85", "mAP50-95": "0.65", "FPS": "120"},
        ]

        enriched = add_deltas(rows)

        self.assertAlmostEqual(enriched[0]["Delta_mAP50-95"], 0.0)
        self.assertAlmostEqual(enriched[1]["Delta_mAP50"], -0.05)
        self.assertAlmostEqual(enriched[1]["Delta_FPS"], 20.0)

    def test_best_row_selects_highest_metric(self) -> None:
        rows = [
            {"Experiment": "A", "mAP50-95": "0.40"},
            {"Experiment": "B", "mAP50-95": "0.50"},
        ]

        self.assertEqual(best_row(rows, "mAP50-95")["Experiment"], "B")

    def test_generate_report_includes_diagnosis_and_focal_section(self) -> None:
        rows = [
            {"Experiment": "Baseline", "mAP50": "0.90", "mAP50-95": "0.70", "FPS": "100"},
            {"Experiment": "FullModel", "mAP50": "0.50", "mAP50-95": "0.40", "FPS": "60"},
        ]
        focal_rows = [
            {
                "SearchKey": "soft_focal_g1p0_anone",
                "LossType": "soft_focal",
                "Gamma": "1.0",
                "Alpha": "none",
                "mAP50": "0.88",
                "mAP50-95": "0.68",
                "Precision": "0.90",
                "Recall": "0.80",
                "FPS": "95",
            }
        ]

        report = generate_report("Test", rows, focal_rows)

        self.assertIn("公平消融结果", report)
        self.assertIn("改进模型低于 Baseline", report)
        self.assertIn("soft_focal_g1p0_anone", report)

    def test_project_fields_drops_extra_summary_columns(self) -> None:
        rows = [{"Experiment": "Baseline", "mAP50": "0.9", "extra": "drop"}]

        projected = project_fields(rows, ["Experiment", "mAP50"])

        self.assertEqual(projected, [{"Experiment": "Baseline", "mAP50": "0.9"}])


if __name__ == "__main__":
    unittest.main()
