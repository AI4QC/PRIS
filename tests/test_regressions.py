import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import apply_rules  # noqa: E402
import paper_data  # noqa: E402


class ApplyRulesRegressionTests(unittest.TestCase):
    def test_missing_features_are_indeterminate_not_pass(self):
        self.assertIsNone(apply_rules.judge({}, apply_rules.R_SINGLE))
        self.assertIsNone(apply_rules.judge({}, apply_rules.R_FIVE))

    def test_explicit_failure_wins_over_other_unknown_rules(self):
        self.assertFalse(apply_rules.judge({"bl_min": 0.7}, apply_rules.R_FIVE))

    def test_complete_rule_set_can_pass(self):
        features = {
            "bl_min": 0.9,
            "cn_an_mean": 4.0,  # D3 guard is false, so bl_mean is not required.
            "madz_range": 10.0,
            "mad_max": 5.0,
            "fi": 0.4,  # D6 guard is false.
        }
        self.assertTrue(apply_rules.judge(features, apply_rules.R_FIVE))


class PaperDataRegressionTests(unittest.TestCase):
    def test_frontier_generator_uses_current_discovery_values(self):
        original_out = paper_data.OUT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                paper_data.OUT = pathlib.Path(tmp)
                frame = paper_data.fig2_frontier().set_index("setting")
                core = frame.loc["trusted core (no CN rules)"]
                five = frame.loc["core + ionicity-guarded charge rule"]
                self.assertAlmostEqual(core.satisfaction, 0.9511)
                self.assertAlmostEqual(core.exclusion, 0.6173)
                self.assertAlmostEqual(five.satisfaction, 0.9071)
                self.assertAlmostEqual(five.exclusion, 0.7052)
        finally:
            paper_data.OUT = original_out

    def test_paper_data_import_has_no_hardcoded_feature_path_side_effect(self):
        if paper_data.FEATURES is None:
            with self.assertRaisesRegex(RuntimeError, "PRIS_FEATURES"):
                paper_data.feature_file("synth_rank.parquet")


class SplitDisciplineRegressionTests(unittest.TestCase):
    def test_bond_report_does_not_promote_unsplit_rows(self):
        source = (ROOT / "src" / "build_bonds.py").read_text()
        self.assertNotIn('isin(["discovery", "unsplit"])', source)
        self.assertIn('split.eq("discovery")', source)


if __name__ == "__main__":
    unittest.main()
