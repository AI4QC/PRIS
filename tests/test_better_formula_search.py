import pathlib
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from better_formula_search import (  # noqa: E402
    evaluate_fixed_thresholds,
    fit_sparse_pair_model,
    fixed_confidence_thresholds,
    inner_oof_confidence_thresholds,
    outer_fold_direction_gate,
)


class FixedAbstentionTests(unittest.TestCase):
    def test_inner_oof_thresholds_cover_each_training_group_once(self):
        rows = []
        for group in range(9):
            rows.extend(
                [
                    {"rk": f"g{group}", "e_hull": 0.0, "x": 0.0},
                    {"rk": f"g{group}", "e_hull": 1.0, "x": 1.0 + group / 10},
                ]
            )
        thresholds, audit = inner_oof_confidence_thresholds(
            pd.DataFrame(rows),
            feature_columns=["x"],
            l1_c=1.0,
            max_terms=1,
            n_folds=3,
            seed=41,
            min_gap=0.0,
            coverages=(1.0, 0.3, 0.1),
        )

        self.assertEqual(set(thresholds), {"1.00", "0.30", "0.10"})
        self.assertEqual(audit["n_folds"], 3)
        self.assertEqual(audit["n_oof_groups"], 9)
        self.assertGreater(audit["n_oof_pairs"], 0)

    def test_outer_direction_gate_limits_the_single_negative_fold(self):
        self.assertTrue(
            outer_fold_direction_gate([0.03, 0.02, 0.01, 0.04, 0.005])
        )
        self.assertTrue(
            outer_fold_direction_gate([0.03, 0.02, 0.01, 0.04, -0.009])
        )
        self.assertFalse(
            outer_fold_direction_gate([0.03, 0.02, 0.01, 0.04, -0.011])
        )
        self.assertFalse(
            outer_fold_direction_gate([0.03, 0.02, 0.01, -0.001, -0.002])
        )

    def test_numeric_thresholds_are_frozen_from_training_scores(self):
        thresholds = fixed_confidence_thresholds(
            np.array([1.0, 2.0, 3.0, 4.0]),
            coverages=(1.0, 0.5),
        )
        self.assertEqual(thresholds["1.00"], 0.0)
        self.assertEqual(thresholds["0.50"], 3.0)
        evaluated = evaluate_fixed_thresholds(
            scores=np.array([2.5, 3.0, 4.0]),
            labels=np.array([1, 1, 0]),
            groups=np.array(["a", "a", "b"]),
            thresholds=thresholds,
        )
        self.assertAlmostEqual(evaluated["0.50"]["coverage"], 2 / 3)
        self.assertAlmostEqual(evaluated["0.50"]["group_equal_accuracy"], 0.5)
        self.assertEqual(evaluated["0.50"]["threshold"], 3.0)


class SparseFormulaTests(unittest.TestCase):
    def test_selector_is_genuinely_l1_and_can_zero_all_terms(self):
        rows = []
        for group in range(8):
            rows.extend(
                [
                    {"rk": f"g{group}", "e_hull": 0.0, "x": 0.0},
                    {"rk": f"g{group}", "e_hull": 1.0, "x": 1.0},
                ]
            )
        with self.assertRaisesRegex(ValueError, "empty formula"):
            fit_sparse_pair_model(
                pd.DataFrame(rows),
                feature_columns=["x"],
                l1_c=1e-8,
                max_terms=1,
            )

    def test_sparse_fit_keeps_at_most_k_terms_and_learns_direction(self):
        rows = []
        for group in range(8):
            rows.extend(
                [
                    {
                        "rk": f"g{group}",
                        "e_hull": 0.0,
                        "x": 0.0,
                        "noise": float(group % 2),
                    },
                    {
                        "rk": f"g{group}",
                        "e_hull": 1.0,
                        "x": 2.0,
                        "noise": float((group + 1) % 2),
                    },
                ]
            )
        model = fit_sparse_pair_model(
            pd.DataFrame(rows),
            feature_columns=["x", "noise"],
            l1_c=1.0,
            max_terms=1,
        )
        self.assertLessEqual(len(model.feature_names), 1)
        self.assertEqual(model.feature_names, ("x",))
        self.assertLess(model.coefficients[0], 0.0)


if __name__ == "__main__":
    unittest.main()
