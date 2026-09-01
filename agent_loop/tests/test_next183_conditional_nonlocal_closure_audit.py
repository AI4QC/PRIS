from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next183_conditional_nonlocal_closure_audit import (
    CLEANLINESS_NAMES,
    CLOSURE_FEATURES,
    CONJUNCTIONS,
    EXPECTED_INPUT_SHA256,
    HYPOTHESES,
    compute_nonlocal_cleanliness,
    conditional_certificate,
    eligibility_from_metrics,
    reconstruct_family_means,
    run_conditional_nonlocal_closure_audit,
    select_conditional_closure_hypothesis,
)


def test_frozen_next182_manifest_identity_is_the_fresh_formal_hash() -> None:
    assert (
        EXPECTED_INPUT_SHA256["next182_manifest"]
        == "31e5db748cc56fa17f9c1ef2aaf38fb78d0653fd35e3ce158f87e7f9a0a0c2e1"
    )


def test_reconstructs_each_capped_family_mean_from_exact_weighted_terms() -> None:
    desired = {
        "cov_a": [0.1, 0.7],
        "sivr_b": [0.3, 0.1],
        "cmvo_a": [0.2, 0.1],
        "hcid_b": [0.4, 0.7],
        "bvtbd_a": [0.1, 0.2],
        "mhcr_a": [0.5, 0.6],
    }
    features = pd.DataFrame(
        {name: np.sinh(values) for name, values in desired.items()}
    )
    terms = [
        {
            "term_id": name,
            "feature": name,
            "direction": 1,
            "transform": "asinh",
            "center": 0.0,
            "scale": 1.0,
        }
        for name in desired
    ]
    base_spec = {
        "base_term_ids": list(desired),
        "base_weights": [1.0] * len(desired),
    }
    families = reconstruct_family_means(
        features=features,
        physical_terms=terms,
        base_spec=base_spec,
        base_support=np.ones(len(features), dtype=bool),
    )
    assert set(families) == {
        "local_geometry",
        "charge_flow_feasibility",
        "contact_robustness",
        "valence_transport",
    }
    np.testing.assert_allclose(families["local_geometry"], [0.2, 0.3])
    np.testing.assert_allclose(families["charge_flow_feasibility"], [0.3, 0.3])
    np.testing.assert_allclose(families["valence_transport"], [0.1, 0.2])
    np.testing.assert_allclose(families["contact_robustness"], [0.5, 0.5])


def test_nonlocal_cleanliness_formulas_are_exact_and_exclude_local_geometry() -> None:
    families = {
        "local_geometry": np.asarray([0.5, 0.5, 0.5, np.nan]),
        "charge_flow_feasibility": np.asarray([0.0, 0.1, 0.5, np.nan]),
        "contact_robustness": np.asarray([0.0, 0.2, 0.25, np.nan]),
        "valence_transport": np.asarray([0.0, 0.3, 0.0, np.nan]),
    }
    cleanliness = compute_nonlocal_cleanliness(
        family_means=families,
        base_support=np.asarray([True, True, True, False]),
    )
    assert tuple(cleanliness) == CLEANLINESS_NAMES == (
        "clean_max",
        "clean_mean",
        "clean_product",
    )
    np.testing.assert_allclose(cleanliness["clean_max"][:3], [1.0, 0.4, 0.0])
    np.testing.assert_allclose(cleanliness["clean_mean"][:3], [1.0, 0.6, 0.5])
    np.testing.assert_allclose(
        cleanliness["clean_product"][:3], [1.0, 0.192, 0.0]
    )
    assert all(np.isnan(values[3]) for values in cleanliness.values())


def test_cleanliness_rejects_out_of_range_or_support_mismatch() -> None:
    families = {
        "local_geometry": np.asarray([0.0]),
        "charge_flow_feasibility": np.asarray([0.0]),
        "contact_robustness": np.asarray([0.0]),
        "valence_transport": np.asarray([0.500001]),
    }
    with pytest.raises(ValueError, match="family mean"):
        compute_nonlocal_cleanliness(
            family_means=families, base_support=np.asarray([True])
        )
    with pytest.raises(ValueError, match="support"):
        compute_nonlocal_cleanliness(
            family_means={key: np.asarray([0.0]) for key in families},
            base_support=np.asarray([True, False]),
        )


def test_conditional_certificate_supports_only_finite_bounded_conjunctions() -> None:
    closure = np.asarray([0.8, 0.5, np.nan, 1.1])
    clean = np.asarray([0.25, 0.75, 0.5, 0.5])
    with pytest.raises(ValueError, match="outside"):
        conditional_certificate(
            closure=closure, cleanliness=clean, conjunction="product"
        )
    closure[-1] = 0.2
    product = conditional_certificate(
        closure=closure, cleanliness=clean, conjunction="product"
    )
    minimum = conditional_certificate(
        closure=closure, cleanliness=clean, conjunction="minimum"
    )
    np.testing.assert_allclose(product[[0, 1, 3]], [0.2, 0.375, 0.1])
    np.testing.assert_allclose(minimum[[0, 1, 3]], [0.25, 0.5, 0.2])
    assert np.isnan(product[2]) and np.isnan(minimum[2])


def test_hypothesis_universe_is_exactly_six_by_three_by_two_high_directions() -> None:
    assert CLOSURE_FEATURES == (
        "psndc_crystalnn_closure_mean",
        "psndc_crystalnn_closure_min",
        "psndc_crystalnn_closure_q10",
        "psndc_crystalnn_volume_mean",
        "psndc_crystalnn_volume_q10",
        "psndc_voronoi_closure_min",
    )
    assert CONJUNCTIONS == ("product", "minimum")
    assert len(HYPOTHESES) == 36
    assert set(HYPOTHESES.values()) == {
        (feature, clean, conjunction, 1)
        for feature in CLOSURE_FEATURES
        for clean in CLEANLINESS_NAMES
        for conjunction in CONJUNCTIONS
    }


def test_eligibility_reuses_frozen_cross_source_gates() -> None:
    kwargs = dict(
        scigen_full_support=0.90,
        wyformer_full_support=0.90,
        scigen_shell_worst_auc=0.55,
        scigen_shell_evaluable_folds=5,
        wyformer_shell_pooled_auc=0.55,
        scigen_full_pooled_auc=0.50,
        wyformer_full_pooled_auc=0.50,
    )
    assert eligibility_from_metrics(**kwargs)
    kwargs["wyformer_shell_pooled_auc"] = 0.549999
    assert not eligibility_from_metrics(**kwargs)


def test_selector_ranks_only_eligible_rows_deterministically() -> None:
    records = pd.DataFrame(
        {
            "hypothesis": ["b", "a", "c"],
            "eligible_for_search": [True, True, False],
            "ranking_min_auc": [0.60, 0.60, 0.99],
            "ranking_mean_auc": [0.70, 0.70, 0.99],
        }
    )
    table, selected = select_conditional_closure_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(
        inspect.signature(run_conditional_nonlocal_closure_audit).parameters
    )
    assert "next182_dir" in parameters
    assert "next179_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_conditional_nonlocal_closure_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT183 input is missing"):
        run_conditional_nonlocal_closure_audit(**kwargs)
