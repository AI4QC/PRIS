import pathlib
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from better_law_search import LawCandidate  # noqa: E402
from better_law_stability import run_leave_one_kind_out  # noqa: E402


class LeaveOneKindOutSearchTests(unittest.TestCase):
    def test_search_builder_only_receives_nonheld_discovery_kinds(self):
        real = pd.DataFrame(
            {
                "source_id": ["r1", "r2", "r3"],
                "x": [0.1, 0.2, 0.3],
                "bvloc_x": [0.1, 0.2, 0.3],
            }
        )
        calibration_real = pd.DataFrame(
            {
                "source_id": ["c1", "c2"],
                "x": [0.15, 0.25],
                "bvloc_x": [0.15, 0.25],
            }
        )
        bad = pd.DataFrame(
            {
                "sid": ["a1", "a2", "b1", "b2"],
                "parent": ["p1", "p2", "p3", "p4"],
                "kind": ["S1", "S1", "S2", "S2"],
                "x": [1.0, 1.1, 1.2, 1.3],
                "bvloc_x": [1.0, 1.1, 1.2, 1.3],
            }
        )
        calibration_bad = pd.DataFrame(
            {
                "sid": ["ca", "cb"],
                "parent": ["cp1", "cp2"],
                "kind": ["S1", "S2"],
                "x": [1.4, 1.5],
                "bvloc_x": [1.4, 1.5],
            }
        )
        seen_training_kinds = []

        def candidate_builder(fit_real, fit_bad, **_):
            seen_training_kinds.append(tuple(sorted(set(fit_bad["kind"]))))

            def candidate(feature, origin):
                return LawCandidate(
                    description=f"{feature} <= 0.5",
                    feature=feature,
                    family="one-sided",
                    origin=origin,
                    side="hi",
                    thresholds=(0.5,),
                    real_mask=fit_real[feature].to_numpy() <= 0.5,
                    bad_mask=fit_bad[feature].to_numpy() <= 0.5,
                    real_coverage=1.0,
                    bad_coverage=1.0,
                )

            old = candidate("x", "existing")
            new = candidate("bvloc_x", "bvloc-p1")
            return {
                "existing_loop": [old],
                "additive_bvloc_loop": [old, new],
            }, {
                "old_candidates": 1,
                "combined_candidates": 2,
            }

        report = run_leave_one_kind_out(
            real,
            calibration_real,
            bad,
            calibration_bad,
            floor=1.0,
            min_coverage=0.9,
            width=4,
            max_rules=2,
            max_guard_targets=10,
            candidate_builder=candidate_builder,
        )

        self.assertEqual(seen_training_kinds, [("S2",), ("S1",)])
        self.assertFalse(report["protocol"]["lockbox_access"])
        self.assertEqual(set(report["folds"]), {"S1", "S2"})
        for held_kind, fold in report["folds"].items():
            self.assertNotIn(held_kind, fold["training_kinds"])
            self.assertEqual(fold["held_discovery_count"], 2)
            self.assertEqual(fold["held_calibration_count"], 1)
            self.assertEqual(
                fold["variants"]["existing_loop"]["held_discovery"]["rejection"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
