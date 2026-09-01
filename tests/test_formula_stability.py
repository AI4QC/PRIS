import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formula_stability import (  # noqa: E402
    paired_group_stability,
    summarize_fixed_commitment,
)


class PairedGroupStabilityTests(unittest.TestCase):
    def test_fixed_commitment_summary_keeps_fold_directions(self):
        existing = [
            {"0.30": {"group_equal_accuracy": 0.50, "coverage": 0.31}},
            {"0.30": {"group_equal_accuracy": 0.60, "coverage": 0.29}},
        ]
        additive = [
            {"0.30": {"group_equal_accuracy": 0.55, "coverage": 0.30}},
            {"0.30": {"group_equal_accuracy": 0.59, "coverage": 0.28}},
        ]

        summary = summarize_fixed_commitment(
            existing,
            additive,
            targets=("0.30",),
        )["0.30"]

        self.assertAlmostEqual(summary["group_equal_accuracy_delta"], 0.02)
        self.assertAlmostEqual(summary["fold_deltas"][0], 0.05)
        self.assertAlmostEqual(summary["fold_deltas"][1], -0.01)
        self.assertTrue(summary["fold_direction_gate"])

    def test_reports_paired_delta_and_single_group_influence(self):
        result = paired_group_stability(
            {"a": 0.0, "b": 0.5, "c": 1.0},
            {"a": 1.0, "b": 0.5, "c": 1.0},
            n_bootstrap=500,
            seed=17,
        )

        self.assertAlmostEqual(result["mean_group_accuracy_delta"], 1 / 3)
        self.assertEqual(result["n_groups"], 3)
        self.assertEqual(result["n_positive_groups"], 1)
        self.assertEqual(result["n_negative_groups"], 0)
        self.assertAlmostEqual(
            result["max_leave_one_group_out_mean_change"],
            1 / 3,
        )
        self.assertEqual(len(result["paired_group_bootstrap_ci"]), 2)

    def test_requires_identical_group_sets(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_group_stability(
                {"a": 0.5},
                {"b": 0.5},
                n_bootstrap=10,
            )


if __name__ == "__main__":
    unittest.main()
