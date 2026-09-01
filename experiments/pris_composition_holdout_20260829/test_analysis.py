from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.pris_composition_holdout_20260829.analysis import (
    LAWSET_NAMES,
    attach_identity_and_novelty,
    canonical_composition_key,
    cluster_bootstrap,
    cohort_masks,
    law_masks,
    load_analysis_tables,
    metric_estimates,
)


FEATURE_ROOT = Path(os.environ.get("PRIS_FEATURES", "features/"))
LAW_ROOT = Path(os.environ.get("PRIS_LAW_TABLES", "law_tables/"))


def test_formula_multiples_share_one_exact_composition_key() -> None:
    assert canonical_composition_key("Li2O2") == canonical_composition_key("Li4O4")
    assert canonical_composition_key("Ca3P2O8") == canonical_composition_key("Ca6P4O16")
    assert canonical_composition_key("Li2O2") != canonical_composition_key("Li2O")


def _toy_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real = pd.DataFrame(
        {
            "source_id": ["d1", "d2", "h1", "h2", "h3"],
            "split": ["discovery", "discovery", "calibration", "calibration", "calibration"],
        }
    )
    bad = pd.DataFrame(
        {
            "sid": ["b1", "b2", "b3"],
            "parent": ["h1", "h2", "h3"],
            "kind": ["S1", "S2", "S3"],
            "psplit": ["calibration", "calibration", "calibration"],
        }
    )
    provenance = pd.DataFrame(
        {
            "source_id": ["d1", "d2", "h1", "h2", "h3"],
            "formula": ["Na1Cl1", "Li2O2", "Na2Cl2", "Li2O1", "Mg1O1"],
            "chemical_system": ["Cl-Na", "Li-O", "Cl-Na", "Li-O", "Mg-O"],
        }
    )
    return real, bad, provenance


def test_identity_annotation_uses_discovery_real_only_and_parent_inheritance() -> None:
    real, bad, provenance = _toy_tables()
    real_out, bad_out = attach_identity_and_novelty(real, bad, provenance)

    real_status = real_out.set_index("source_id")["composition_seen_in_discovery"].to_dict()
    bad_status = bad_out.set_index("parent")["composition_seen_in_discovery"].to_dict()
    assert real_status == {"d1": True, "d2": True, "h1": True, "h2": False, "h3": False}
    assert bad_status == {"h1": True, "h2": False, "h3": False}
    assert bad_out.set_index("parent").loc["h3", "chemical_system_seen_in_discovery"] == False


@pytest.mark.parametrize("bad_split", ["lockbox", None, "other"])
def test_identity_annotation_rejects_non_analysis_splits(bad_split: str | None) -> None:
    real, bad, provenance = _toy_tables()
    real.loc[0, "split"] = bad_split
    with pytest.raises(ValueError, match="discovery and calibration"):
        attach_identity_and_novelty(real, bad, provenance)


def test_identity_annotation_rejects_damaged_parent_split_mismatch() -> None:
    real, bad, provenance = _toy_tables()
    bad.loc[0, "psplit"] = "discovery"
    with pytest.raises(ValueError, match="parent split"):
        attach_identity_and_novelty(real, bad, provenance)


def test_cohort_masks_are_nested_and_disjoint_where_required() -> None:
    real, bad, provenance = _toy_tables()
    real, _ = attach_identity_and_novelty(real, bad, provenance)
    masks = cohort_masks(real, split_column="split")
    assert masks["heldout_all"].sum() == 3
    assert masks["composition_shared"].sum() == 1
    assert masks["composition_unseen"].sum() == 2
    assert masks["chemical_system_unseen"].sum() == 1
    assert not np.any(masks["composition_shared"] & masks["composition_unseen"])
    assert np.all(masks["chemical_system_unseen"] <= masks["composition_unseen"])


def test_frozen_law_masks_follow_published_missing_value_convention() -> None:
    passing = {
        "bl_min": 0.90,
        "bl_mean": 1.00,
        "cn_an_mean": 3.0,
        "madz_range": 10.0,
        "mad_max": 5.0,
        "frac_like_bonds": 0.0,
        "fi": 0.8,
        "wyckoff_econ_001": 0.5,
        "bv_rel_mean": 0.5,
    }
    failing_contact = {**passing, "bl_min": 0.70}
    missing = {key: np.nan for key in passing}
    masks = law_masks(pd.DataFrame([passing, failing_contact, missing]))

    assert tuple(masks.columns) == LAWSET_NAMES
    assert masks.iloc[0].all()
    assert not masks.iloc[1].any()
    assert masks.iloc[2].all()


def test_metric_estimates_include_row_and_composition_equal_weighting() -> None:
    frame = pd.DataFrame(
        {
            "composition_key": ["A", "A", "A", "B"],
            "passed": [True, True, False, False],
        }
    )
    result = metric_estimates(frame, frame["passed"].to_numpy())
    assert result["estimate_micro"] == pytest.approx(0.5)
    assert result["estimate_composition_equal"] == pytest.approx((2 / 3 + 0) / 2)


def test_cluster_bootstrap_is_deterministic_and_cluster_based() -> None:
    frame = pd.DataFrame(
        {
            "composition_key": ["A", "A", "B", "C", "C", "C"],
            "passed": [True, False, True, False, False, True],
        }
    )
    first = cluster_bootstrap(frame, frame.passed.to_numpy(), replicates=250, seed=7)
    second = cluster_bootstrap(frame, frame.passed.to_numpy(), replicates=250, seed=7)
    assert first == second
    assert 0 <= first["micro_ci_low"] <= first["micro_ci_high"] <= 1
    assert 0 <= first["composition_equal_ci_low"] <= first["composition_equal_ci_high"] <= 1


@pytest.mark.skipif(not (FEATURE_ROOT / "provenance.parquet").exists(), reason="external PRIS feature store absent")
def test_external_tables_reproduce_published_heldout_metrics() -> None:
    real, bad, provenance = load_analysis_tables(FEATURE_ROOT, LAW_ROOT)
    real, bad = attach_identity_and_novelty(real, bad, provenance)
    real_masks, bad_masks = law_masks(real), law_masks(bad)
    real_held = cohort_masks(real, split_column="split")["heldout_all"]
    bad_held = cohort_masks(bad, split_column="psplit")["heldout_all"]

    expected_satisfaction = {
        "Set 1": 0.9918821974702662,
        "Set 1-prime": 0.9894279781008116,
        "Set 2": 0.9579006985085896,
        "Set 3": 0.917122899754578,
        "Set 4": 0.8180101944496885,
    }
    expected_detection = {
        "Set 1": 0.2890365448504983,
        "Set 1-prime": 0.3837209302325581,
        "Set 2": 0.6121262458471761,
        "Set 3": 0.7004429678848283,
        "Set 4": 0.9111295681063123,
    }
    for name in LAWSET_NAMES:
        assert real_masks.loc[real_held, name].mean() == pytest.approx(expected_satisfaction[name])
        assert 1 - bad_masks.loc[bad_held, name].mean() == pytest.approx(expected_detection[name])
