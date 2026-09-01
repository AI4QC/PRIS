from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next186_local_nonlocal_contradiction_relief_audit import (
    CLOSURE_FEATURES,
    CONJUNCTIONS,
    HYPOTHESES,
    SURPLUS_NAMES,
    compute_local_nonlocal_surplus,
    contradiction_relief,
    eligibility_from_metrics,
    run_local_nonlocal_contradiction_relief_audit,
    select_contradiction_relief_hypothesis,
)


def test_local_nonlocal_surplus_formulas_are_exact_in_original_risk_units() -> None:
    families = {
        "local_geometry": np.asarray([0.5, 0.4, 0.3, np.nan]),
        "charge_flow_feasibility": np.asarray([0.1, 0.2, 0.4, np.nan]),
        "contact_robustness": np.asarray([0.2, 0.1, 0.2, np.nan]),
        "valence_transport": np.asarray([0.3, 0.3, 0.1, np.nan]),
    }
    surplus = compute_local_nonlocal_surplus(
        family_means=families,
        base_support=np.asarray([True, True, True, False]),
    )
    assert tuple(surplus) == SURPLUS_NAMES == ("surplus_max", "surplus_mean")
    np.testing.assert_allclose(surplus["surplus_max"][:3], [0.2, 0.1, 0.0])
    np.testing.assert_allclose(
        surplus["surplus_mean"][:3], [0.3, 0.2, 1.0 / 15.0]
    )
    assert all(np.isnan(values[3]) for values in surplus.values())


def test_surplus_rejects_out_of_range_family_or_support_mismatch() -> None:
    families = {
        "local_geometry": np.asarray([0.500001]),
        "charge_flow_feasibility": np.asarray([0.0]),
        "contact_robustness": np.asarray([0.0]),
        "valence_transport": np.asarray([0.0]),
    }
    with pytest.raises(ValueError, match="family mean"):
        compute_local_nonlocal_surplus(
            family_means=families, base_support=np.asarray([True])
        )
    with pytest.raises(ValueError, match="support"):
        compute_local_nonlocal_surplus(
            family_means={key: np.asarray([0.0]) for key in families},
            base_support=np.asarray([True, False]),
        )


def test_relief_conjunctions_are_bounded_and_preserve_missing_values() -> None:
    closure = np.asarray([0.8, 0.5, np.nan, 1.1])
    surplus = np.asarray([0.25, 0.4, 0.1, 0.1])
    with pytest.raises(ValueError, match="outside"):
        contradiction_relief(
            closure=closure, surplus=surplus, conjunction="product"
        )
    closure[-1] = 0.2
    product = contradiction_relief(
        closure=closure, surplus=surplus, conjunction="product"
    )
    minimum = contradiction_relief(
        closure=closure, surplus=surplus, conjunction="minimum"
    )
    np.testing.assert_allclose(product[[0, 1, 3]], [0.2, 0.2, 0.02])
    np.testing.assert_allclose(minimum[[0, 1, 3]], [0.25, 0.25, 0.1])
    assert np.isnan(product[2]) and np.isnan(minimum[2])


def test_hypothesis_universe_is_exactly_six_by_two_by_two_high_directions() -> None:
    assert CLOSURE_FEATURES == (
        "psndc_crystalnn_closure_mean",
        "psndc_crystalnn_closure_min",
        "psndc_crystalnn_closure_q10",
        "psndc_crystalnn_volume_mean",
        "psndc_crystalnn_volume_q10",
        "psndc_voronoi_closure_min",
    )
    assert CONJUNCTIONS == ("product", "minimum")
    assert len(HYPOTHESES) == 24
    assert set(HYPOTHESES.values()) == {
        (feature, surplus_name, conjunction, 1)
        for feature in CLOSURE_FEATURES
        for surplus_name in SURPLUS_NAMES
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
    kwargs["scigen_full_support"] = 0.899999
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
    table, selected = select_contradiction_relief_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(
        inspect.signature(
            run_local_nonlocal_contradiction_relief_audit
        ).parameters
    )
    assert "next185_dir" in parameters and "next179_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_local_nonlocal_contradiction_relief_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT186 input is missing"):
        run_local_nonlocal_contradiction_relief_audit(**kwargs)
