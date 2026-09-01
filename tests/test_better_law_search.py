import pathlib
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from better_law_search import (  # noqa: E402
    LawCandidate,
    apply_candidate,
    apply_serialized_law_set,
    build_band_candidates,
    build_candidate_sets,
    build_guarded_candidates,
    build_one_sided_candidates,
    evaluate_masks,
    historical_rule_masks,
    law_preliminary_gate,
    leave_one_kind_out_frames,
    pareto_beam,
    selected_rule_coverage,
)


class CandidateConstructionTests(unittest.TestCase):
    def test_candidate_sets_keep_existing_pool_and_add_new_family(self):
        real = pd.DataFrame(
            {
                "old_signal": np.linspace(0.0, 1.0, 100),
                "bvloc_an_absolute_mismatch_max": np.linspace(
                    0.0, 1.0, 100
                ),
            }
        )
        bad = pd.DataFrame(
            {
                "old_signal": np.linspace(0.0, 2.0, 100),
                "bvloc_an_absolute_mismatch_max": np.linspace(
                    0.0, 3.0, 100
                ),
            }
        )

        pools, counts = build_candidate_sets(
            real,
            bad,
            min_coverage=0.9,
            max_guard_targets=10,
        )

        existing = pools["existing_loop"]
        additive = pools["additive_bvloc_loop"]
        self.assertTrue(existing)
        self.assertGreater(len(additive), len(existing))
        self.assertEqual(
            {_candidate.description for _candidate in existing},
            {
                _candidate.description
                for _candidate in additive
                if _candidate.origin == "existing"
            },
        )
        self.assertGreater(counts["new_one_sided_candidates"], 0)
        self.assertTrue(
            any(candidate.origin == "bvloc-p1" for candidate in additive)
        )

    def test_thresholds_are_fit_on_real_rows_and_missing_values_abstain(self):
        real = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, np.nan]})
        bad = pd.DataFrame({"x": [-10.0, 1.5, 10.0, np.nan]})
        candidates = build_one_sided_candidates(
            real,
            bad,
            ["x"],
            alphas=(0.25,),
            min_coverage=0.75,
            min_rejection=0.0,
            min_real_satisfaction=0.0,
            origin="test",
        )
        self.assertEqual({candidate.side for candidate in candidates}, {"hi", "lo"})
        upper = next(candidate for candidate in candidates if candidate.side == "hi")
        self.assertAlmostEqual(upper.thresholds[0], 2.25)
        self.assertTrue(upper.real_mask[-1])
        self.assertTrue(upper.bad_mask[-1])

    def test_central_band_rejects_both_extreme_directions(self):
        real = pd.DataFrame({"x": np.arange(10, dtype=float)})
        bad = pd.DataFrame({"x": [-100.0, 5.0, 100.0]})
        candidates = build_band_candidates(
            real,
            bad,
            ["x"],
            central_coverages=(0.8,),
            min_coverage=0.9,
            min_rejection=0.0,
            min_real_satisfaction=0.0,
            origin="test",
        )
        self.assertEqual(len(candidates), 1)
        band = candidates[0]
        self.assertEqual(band.family, "band")
        self.assertAlmostEqual(band.real_mask.mean(), 0.8)
        self.assertAlmostEqual(1 - band.bad_mask.mean(), 2 / 3)

    def test_guard_is_not_applicable_when_false_or_missing(self):
        candidate = LawCandidate(
            description="if g then x <= 1",
            feature="x",
            family="guarded-one-sided",
            origin="test",
            side="hi",
            thresholds=(1.0,),
            real_mask=np.array([], dtype=bool),
            bad_mask=np.array([], dtype=bool),
            real_coverage=1.0,
            bad_coverage=1.0,
            guard_feature="g",
            guard_side="hi",
            guard_threshold=0.5,
        )
        frame = pd.DataFrame(
            {
                "x": [2.0, 2.0, 2.0, np.nan],
                "g": [1.0, 0.0, np.nan, 1.0],
            }
        )
        np.testing.assert_array_equal(
            apply_candidate(frame, candidate),
            np.array([False, True, True, True]),
        )
        np.testing.assert_array_equal(
            apply_serialized_law_set(
                frame,
                [
                    {
                        "description": "if g then x <= 1",
                        "feature": "x",
                        "family": "guarded-one-sided",
                        "origin": "test",
                        "side": "hi",
                        "thresholds": [1.0],
                        "real_coverage": 1.0,
                        "bad_coverage": 1.0,
                        "guard_feature": "g",
                        "guard_side": "hi",
                        "guard_threshold": 0.5,
                    }
                ],
            ),
            np.array([False, True, True, True]),
        )

    def test_guard_builder_can_rescue_a_context_specific_target(self):
        real = pd.DataFrame({"g": [0.0, 0.0, 1.0, 1.0]})
        bad = pd.DataFrame({"g": [0.0, 1.0, 1.0, 1.0]})
        target = LawCandidate(
            description="target",
            feature="x",
            family="one-sided",
            origin="test",
            side="hi",
            thresholds=(1.0,),
            real_mask=np.array([False, True, True, True]),
            bad_mask=np.array([False, False, True, True]),
            real_coverage=1.0,
            bad_coverage=1.0,
        )
        guarded = build_guarded_candidates(
            real,
            bad,
            [target],
            guard_columns=["g"],
            guard_quantiles=(0.5,),
            min_real_satisfaction=0.9,
            min_rejection=0.2,
            max_targets=10,
        )
        self.assertTrue(guarded)
        self.assertTrue(
            any(
                candidate.real_mask.mean() == 1.0
                and 1 - candidate.bad_mask.mean() >= 0.25
                for candidate in guarded
            )
        )


class MetricAndBeamTests(unittest.TestCase):
    def test_success_gate_requires_coverage_and_no_materialized_lockbox_rows(self):
        common = {
            "new_descriptor_selected": True,
            "additive_satisfaction": 0.98,
            "base_satisfaction": 0.98,
            "rejection_delta": 0.03,
            "min_kind_rejection_delta": 0.04,
            "worst_anion_delta": -0.005,
            "real_coverage": 0.95,
            "bad_coverage": 0.95,
        }
        self.assertTrue(
            law_preliminary_gate(
                **common,
                source_tables_materialized_lockbox_rows=False,
            )
        )
        self.assertFalse(
            law_preliminary_gate(
                **{**common, "bad_coverage": 0.89},
                source_tables_materialized_lockbox_rows=False,
            )
        )
        self.assertFalse(
            law_preliminary_gate(
                **common,
                source_tables_materialized_lockbox_rows=True,
            )
        )

    def test_selected_rule_coverage_includes_target_and_guard_features(self):
        real = pd.DataFrame(
            {
                "x": [1.0, np.nan, 2.0, 3.0],
                "g": [1.0, 2.0, np.nan, 4.0],
            }
        )
        bad = pd.DataFrame(
            {
                "x": [1.0, 2.0, np.nan, np.nan],
                "g": [1.0, 2.0, 3.0, 4.0],
            }
        )
        rules = [{"feature": "x", "guard_feature": "g"}]

        coverage = selected_rule_coverage(real, bad, rules)

        self.assertEqual(coverage["features"], ["g", "x"])
        self.assertAlmostEqual(coverage["real_min"], 0.75)
        self.assertAlmostEqual(coverage["bad_min"], 0.5)

    def test_leave_one_kind_out_frames_never_expose_held_kind_to_search(self):
        discovery_bad = pd.DataFrame(
            {
                "sid": ["a1", "a2", "b1", "b2", "c1"],
                "kind": ["S1", "S1", "S2", "S2", "S3"],
            }
        )
        calibration_bad = pd.DataFrame(
            {
                "sid": ["ca", "cb", "cc", "cd"],
                "kind": ["S1", "S2", "S3", "S4"],
            }
        )

        folds = list(
            leave_one_kind_out_frames(discovery_bad, calibration_bad)
        )

        self.assertEqual([fold[0] for fold in folds], ["S1", "S2", "S3"])
        for held_kind, training, held_discovery, held_calibration in folds:
            self.assertNotIn(held_kind, set(training["kind"]))
            self.assertEqual(set(held_discovery["kind"]), {held_kind})
            self.assertEqual(set(held_calibration["kind"]), {held_kind})
            self.assertEqual(
                len(training) + len(held_discovery),
                len(discovery_bad),
            )
        self.assertEqual(
            len(folds[-1][3]),
            1,
            "calibration-only kinds must not create a fitted fold",
        )

    def test_metrics_report_pooled_group_equal_and_kind_rejection(self):
        metrics = evaluate_masks(
            real_mask=np.array([True, True, True, False]),
            bad_mask=np.array([False, True, True, True]),
            bad_groups=np.array(["small", "large", "large", "large"]),
            bad_kinds=np.array(["S1", "S1", "S2", "S2"]),
        )
        self.assertAlmostEqual(metrics["satisfaction"], 0.75)
        self.assertAlmostEqual(metrics["rejection"], 0.25)
        self.assertAlmostEqual(metrics["group_equal_rejection"], 0.5)
        self.assertAlmostEqual(metrics["by_kind"]["S1"], 0.5)
        self.assertAlmostEqual(metrics["by_kind"]["S2"], 0.0)

    def test_beam_combines_complementary_candidates_under_floor(self):
        candidates = [
            LawCandidate(
                description="a",
                feature="a",
                family="one-sided",
                origin="test",
                side="hi",
                thresholds=(1.0,),
                real_mask=np.array([True, True, True, False]),
                bad_mask=np.array([False, True, True, True]),
                real_coverage=1.0,
                bad_coverage=1.0,
            ),
            LawCandidate(
                description="b",
                feature="b",
                family="one-sided",
                origin="test",
                side="hi",
                thresholds=(1.0,),
                real_mask=np.array([True, True, False, True]),
                bad_mask=np.array([True, False, True, True]),
                real_coverage=1.0,
                bad_coverage=1.0,
            ),
        ]
        result = pareto_beam(
            candidates,
            real_size=4,
            bad_size=4,
            bad_kinds=np.array(["S1", "S2", "S1", "S2"]),
            satisfaction_floor=0.5,
            max_rules=2,
            width=4,
            min_gain=0.0,
        )
        self.assertEqual(result.indices, (0, 1))
        self.assertAlmostEqual(result.real_mask.mean(), 0.5)
        self.assertAlmostEqual(1 - result.bad_mask.mean(), 0.5)
        constrained = pareto_beam(
            candidates,
            real_size=4,
            bad_size=4,
            bad_kinds=np.array(["S1", "S2", "S1", "S2"]),
            satisfaction_floor=0.5,
            max_rules=2,
            width=4,
            min_gain=0.0,
            real_strata={"tail": np.array([False, False, True, True])},
            stratum_floors={"tail": 0.5},
        )
        self.assertEqual(len(constrained.indices), 1)

    def test_historical_masks_reproduce_offline_missing_value_semantics(self):
        frame = pd.DataFrame(
            {
                "bl_min": [0.9, 0.75, 0.7, np.nan],
                "fi": [0.4, 0.6, 0.4, np.nan],
                "cn_an_mean": [4.0, 4.0, 4.0, np.nan],
                "bl_mean": [1.0, 1.0, 1.0, np.nan],
                "madz_range": [10.0, 10.0, 10.0, np.nan],
                "mad_max": [5.0, 5.0, 5.0, np.nan],
                "frac_like_bonds": [0.0, 0.0, 0.0, np.nan],
            }
        )
        masks = historical_rule_masks(frame)
        np.testing.assert_array_equal(
            masks["L1"],
            np.array([True, True, False, True]),
        )
        np.testing.assert_array_equal(
            masks["L2"],
            np.array([True, False, False, True]),
        )
        self.assertTrue(all(mask[-1] for mask in masks.values()))


if __name__ == "__main__":
    unittest.main()
