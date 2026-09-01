import pathlib
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from better_search import (  # noqa: E402
    PairwiseDataset,
    Rule,
    assert_no_forbidden_splits,
    deterministic_group_folds,
    evaluate_rule,
    group_equal_accuracy,
    is_frozen_p1_search_feature,
    make_group_pairs,
)
from advanced_local_features import (  # noqa: E402
    aggregate_bond_valence_sites,
    bond_valence_local_features,
    composition_guard_features,
    load_bad_records,
    load_false_positive_records,
    load_real_records,
    resolve_bond_valence_parameter,
    site_bond_valence_statistics,
)


class SplitDisciplineTests(unittest.TestCase):
    def test_forbidden_or_missing_split_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "lockbox"):
            assert_no_forbidden_splits(
                pd.DataFrame({"split": ["discovery", "lockbox"]}),
                allowed={"discovery"},
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            assert_no_forbidden_splits(
                pd.DataFrame({"split": ["discovery", None]}),
                allowed={"discovery"},
            )

    def test_group_folds_are_deterministic_and_keep_groups_together(self):
        groups = np.array(["b", "a", "b", "c", "d", "a"])
        first = deterministic_group_folds(groups, n_splits=3, seed=17)
        second = deterministic_group_folds(groups, n_splits=3, seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[0], first[2])
        self.assertEqual(first[1], first[5])
        self.assertEqual(set(first), {0, 1, 2})


class RuleSemanticsTests(unittest.TestCase):
    def test_frozen_p1_vocabulary_excludes_signed_and_tail_extensions(self):
        self.assertTrue(
            is_frozen_p1_search_feature(
                "bvloc_an_absolute_mismatch_max"
            )
        )
        self.assertTrue(
            is_frozen_p1_search_feature("bvloc_cat_effective_cn_q95")
        )
        self.assertFalse(
            is_frozen_p1_search_feature("bvloc_an_relative_mismatch_min")
        )
        self.assertFalse(
            is_frozen_p1_search_feature("bvloc_cat_vector_asymmetry_q05")
        )
        self.assertFalse(
            is_frozen_p1_search_feature("bvloc_bond_parameter_coverage")
        )

    def test_explicit_failure_wins_over_missing_other_values(self):
        rules = [
            Rule("bl_min", ">=", 0.8),
            Rule("mad_max", "<=", 15.0),
        ]
        self.assertFalse(evaluate_rule({"bl_min": 0.7}, rules))

    def test_missing_required_value_is_indeterminate(self):
        self.assertIsNone(evaluate_rule({}, [Rule("bl_min", ">=", 0.8)]))

    def test_false_guard_makes_rule_not_applicable(self):
        guarded = Rule(
            "bl_mean",
            "<=",
            1.08,
            guard_feature="cn_an_mean",
            guard_op="<=",
            guard_threshold=3.33,
        )
        self.assertTrue(evaluate_rule({"cn_an_mean": 4.0}, [guarded]))
        self.assertIsNone(evaluate_rule({}, [guarded]))


class GroupEqualFormulaTests(unittest.TestCase):
    def test_pair_builder_is_antisymmetric_and_group_balanced(self):
        frame = pd.DataFrame(
            {
                "composition": ["A", "A", "B", "B", "B"],
                "energy": [0.0, 1.0, 0.0, 1.0, 2.0],
                "x": [0.0, 2.0, 0.0, 1.0, 3.0],
            }
        )
        pairs = make_group_pairs(
            frame,
            group_col="composition",
            target_col="energy",
            feature_cols=["x"],
        )
        self.assertIsInstance(pairs, PairwiseDataset)
        self.assertEqual(len(pairs.y), 8)
        np.testing.assert_allclose(pairs.X[0], -pairs.X[1])
        self.assertEqual(pairs.y[0], 1 - pairs.y[1])
        self.assertAlmostEqual(pairs.target_gaps[0], 1.0)
        self.assertAlmostEqual(pairs.target_gaps[0], pairs.target_gaps[1])
        group_weights = pd.Series(pairs.sample_weight).groupby(pairs.groups).sum()
        np.testing.assert_allclose(group_weights.to_numpy(), np.ones(2))
        filtered = make_group_pairs(
            frame,
            group_col="composition",
            target_col="energy",
            feature_cols=["x"],
            min_gap=1.5,
        )
        self.assertEqual(len(filtered.y), 2)
        np.testing.assert_allclose(filtered.target_gaps, np.array([2.0, 2.0]))

    def test_accuracy_is_group_equal_not_pair_row_weighted(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 0, 0, 0])
        groups = np.array(["small", "large", "large", "large"])
        self.assertAlmostEqual(
            group_equal_accuracy(y_true, y_pred, groups),
            0.5,
        )


class BondValenceLocalFeatureTests(unittest.TestCase):
    def test_composition_guard_matches_pauling_ionicity_definition(self):
        from pymatgen.core import Lattice, Structure
        from pymatgen.core.periodic_table import Element

        structure = Structure(
            Lattice.cubic(4.0),
            ["Ca", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        out = composition_guard_features(structure, [2.0, -2.0])
        delta = float(Element("O").X - Element("Ca").X)
        self.assertAlmostEqual(out["dchi"], delta)
        self.assertAlmostEqual(out["fi"], 1 - np.exp(-0.25 * delta**2))

    def test_frozen_nearest_valence_fallback_is_explicit(self):
        parameters = {("Ca", 3, "O", -2): (1.9, 0.37)}
        exact = resolve_bond_valence_parameter(
            ("Ca", 3, "O", -2),
            parameters,
            policy="frozen-fallback",
        )
        nearest = resolve_bond_valence_parameter(
            ("Ca", 2, "O", -1),
            parameters,
            policy="frozen-fallback",
        )
        self.assertEqual(exact, (1.9, 0.37, "exact"))
        self.assertEqual(nearest, (1.9, 0.37, "nearest_valence"))
        self.assertIsNone(
            resolve_bond_valence_parameter(
                ("Ca", 2, "O", -1),
                parameters,
                policy="exact",
            )
        )

    def test_symmetric_bonds_have_zero_vector_asymmetry(self):
        stats = site_bond_valence_statistics(
            strengths=np.array([1.0, 1.0]),
            unit_vectors=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            formal_valence=2.0,
        )
        self.assertAlmostEqual(stats["relative_mismatch"], 0.0)
        self.assertAlmostEqual(stats["effective_cn"], 2.0)
        self.assertAlmostEqual(stats["vector_asymmetry"], 0.0)

    def test_entropy_coordination_is_invariant_to_strength_scale(self):
        first = site_bond_valence_statistics(
            strengths=np.array([1.0, 3.0]),
            unit_vectors=np.eye(2, 3),
            formal_valence=4.0,
        )
        second = site_bond_valence_statistics(
            strengths=np.array([2.0, 6.0]),
            unit_vectors=np.eye(2, 3),
            formal_valence=8.0,
        )
        self.assertAlmostEqual(first["effective_cn"], second["effective_cn"])
        self.assertAlmostEqual(first["relative_mismatch"], 0.0)
        self.assertAlmostEqual(second["relative_mismatch"], 0.0)

    def test_aggregation_separates_charge_sign_and_reports_coverage(self):
        site_stats = [
            {
                "relative_mismatch": 0.1,
                "effective_cn": 4.0,
                "vector_asymmetry": 0.2,
            },
            None,
            {
                "relative_mismatch": 0.3,
                "effective_cn": 2.0,
                "vector_asymmetry": 0.8,
            },
        ]
        out = aggregate_bond_valence_sites(site_stats, [2.0, 1.0, -2.0])
        self.assertAlmostEqual(out["bvloc_site_coverage"], 2 / 3)
        self.assertAlmostEqual(out["bvloc_cat_coverage"], 0.5)
        self.assertAlmostEqual(out["bvloc_an_coverage"], 1.0)
        self.assertAlmostEqual(out["bvloc_cat_relative_mismatch_max"], 0.1)
        self.assertAlmostEqual(out["bvloc_an_vector_asymmetry_q95"], 0.8)

    def test_structure_features_use_frozen_parameters_and_bond_directions(self):
        from pymatgen.core import Lattice, Structure

        structure = Structure(
            Lattice.cubic(10.0),
            ["Ca", "O", "O"],
            [[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.4, 0.5, 0.5]],
        )
        neighbors = [
            [
                {"site_index": 1, "image": (0, 0, 0)},
                {"site_index": 2, "image": (0, 0, 0)},
            ],
            [],
            [],
        ]
        features = bond_valence_local_features(
            structure,
            [2.0, -1.0, -1.0],
            neighbors=neighbors,
            parameters={("Ca", 2, "O", -1): (1.0, 0.37)},
        )
        self.assertAlmostEqual(features["bvloc_bond_parameter_coverage"], 1.0)
        self.assertAlmostEqual(features["bvloc_site_coverage"], 1.0)
        self.assertAlmostEqual(features["bvloc_cat_relative_mismatch_max"], 0.0)
        self.assertAlmostEqual(features["bvloc_cat_effective_cn_mean"], 2.0)
        self.assertAlmostEqual(features["bvloc_cat_vector_asymmetry_max"], 0.0)
        self.assertAlmostEqual(features["bvloc_an_vector_asymmetry_mean"], 1.0)

    def test_structure_features_disclose_fallback_parameter_fraction(self):
        from pymatgen.core import Lattice, Structure

        structure = Structure(
            Lattice.cubic(10.0),
            ["Ca", "O"],
            [[0.5, 0.5, 0.5], [0.6, 0.5, 0.5]],
        )
        features = bond_valence_local_features(
            structure,
            [2.0, -1.0],
            neighbors=[
                [{"site_index": 1, "image": (0, 0, 0)}],
                [],
            ],
            parameters={("Ca", 3, "O", -2): (1.0, 0.37)},
            parameter_policy="frozen-fallback",
        )
        self.assertAlmostEqual(features["bvloc_bond_parameter_coverage"], 1.0)
        self.assertAlmostEqual(features["bvloc_parameter_nearest_valence_fraction"], 1.0)
        self.assertAlmostEqual(features["bvloc_parameter_exact_fraction"], 0.0)

    def test_real_record_loader_never_returns_lockbox_or_unknown_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pd.DataFrame(
                {
                    "source_id": ["d", "c", "l", "u"],
                    "split": ["discovery", "calibration", "lockbox", None],
                }
            ).to_parquet(root / "real_all.parquet", index=False)
            pd.DataFrame(
                {
                    "source_id": ["d", "c", "l", "u"],
                    "blob_offset": [1, 2, 3, 4],
                    "blob_length": [10, 20, 30, 40],
                    "n_elements": [2, 2, 2, 2],
                    "n_sites": [4, 5, 6, 7],
                }
            ).to_parquet(root / "provenance.parquet", index=False)
            records = load_real_records(root, max_sites=80)
        self.assertEqual([row["sid"] for row in records], ["d", "c"])
        self.assertEqual(
            {row["split"] for row in records},
            {"discovery", "calibration"},
        )

    def test_bad_record_loader_follows_existing_ids_and_parent_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pd.DataFrame(
                {
                    "sid": ["d_S2", "d_S1", "l_S1"],
                    "kind": ["S2", "S1", "S1"],
                    "parent": ["d", "d", "l"],
                }
            ).to_parquet(root / "phys_bad.parquet", index=False)
            pd.DataFrame(
                {
                    "source_id": ["d", "l"],
                    "split": ["discovery", "lockbox"],
                }
            ).to_parquet(root / "splits.parquet", index=False)
            pd.DataFrame(
                {
                    "source_id": ["d", "l"],
                    "blob_offset": [1, 2],
                    "blob_length": [10, 20],
                    "n_elements": [2, 2],
                    "n_sites": [4, 5],
                }
            ).to_parquet(root / "provenance.parquet", index=False)
            records = load_bad_records(root, max_sites=80)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sid"], "d")
        self.assertEqual(records[0]["split"], "discovery")
        self.assertEqual(records[0]["kinds"], ("S1", "S2"))

    def test_false_positive_loader_resolves_only_frozen_audit_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pd.DataFrame({"sid": ["b", "a", "missing"]}).to_parquet(
                root / "false_positive.parquet",
                index=False,
            )
            database = root / "materials.sqlite"
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE materials "
                "(material_id TEXT, blob_offset INTEGER, blob_length INTEGER)"
            )
            connection.executemany(
                "INSERT INTO materials VALUES (?, ?, ?)",
                [("a", 1, 10), ("b", 2, 20), ("extra", 3, 30)],
            )
            connection.commit()
            connection.close()
            records = load_false_positive_records(root, database)
        self.assertEqual([record["sid"] for record in records], ["b", "a"])
        self.assertTrue(
            all(record["split"] == "false_positive_audit" for record in records)
        )


if __name__ == "__main__":
    unittest.main()
