from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next192_signed_safe_margin_audit import (
    HYPOTHESES,
    complementary_safe_family_means,
    eligibility_from_metrics,
    safe_margin_statistics,
    run_signed_safe_margin_audit,
    select_safe_margin_hypothesis,
)


def test_hypothesis_universe_is_exactly_ten_protection_high_summaries() -> None:
    names = (
        "safe_local_geometry",
        "safe_charge_flow_feasibility",
        "safe_valence_transport",
        "safe_contact_robustness",
        "safe_family_mean",
        "safe_family_min",
        "safe_family_second_min",
        "safe_nonlocal_mean",
        "safe_nonlocal_min",
        "safe_local_nonlocal_min",
    )
    assert HYPOTHESES == {name: (name, 1) for name in names}


def test_complementary_hinge_uses_exact_sign_weight_cap_and_support() -> None:
    negative = float(np.sinh(-1.0))
    positive = float(np.sinh(1.0))
    features = pd.DataFrame(
        {
            "a": [negative, positive, np.nan],
            "b": [negative, positive, 0.0],
            "c": [negative, positive, 0.0],
            "d": [negative, positive, 0.0],
        }
    )
    terms = [
        {
            "term_id": term_id,
            "feature": feature,
            "direction": 1,
            "transform": "asinh",
            "center": 0.0,
            "scale": 1.0,
        }
        for term_id, feature in (
            ("cov_test", "a"),
            ("cmvo_test", "b"),
            ("bvtbd_test", "c"),
            ("mhcr_test", "d"),
        )
    ]
    result = complementary_safe_family_means(
        features=features,
        physical_terms=terms,
        base_spec={
            "base_term_ids": [term["term_id"] for term in terms],
            "base_weights": [1.0, 2.0, 1.0, 1.0],
        },
        base_support=[True, True, False],
    )
    assert set(result) == {
        "local_geometry",
        "charge_flow_feasibility",
        "valence_transport",
        "contact_robustness",
    }
    for values in result.values():
        assert values[:2] == pytest.approx([0.5, 0.0])
        assert np.isnan(values[2])


def test_safe_margin_statistics_match_frozen_family_consensus_definitions() -> None:
    statistics = safe_margin_statistics(
        {
            "local_geometry": np.array([0.1, 0.4]),
            "charge_flow_feasibility": np.array([0.2, 0.3]),
            "valence_transport": np.array([0.3, 0.2]),
            "contact_robustness": np.array([0.4, 0.1]),
        }
    )
    expected = {
        "safe_local_geometry": [0.1, 0.4],
        "safe_charge_flow_feasibility": [0.2, 0.3],
        "safe_valence_transport": [0.3, 0.2],
        "safe_contact_robustness": [0.4, 0.1],
        "safe_family_mean": [0.25, 0.25],
        "safe_family_min": [0.1, 0.1],
        "safe_family_second_min": [0.2, 0.2],
        "safe_nonlocal_mean": [0.3, 0.2],
        "safe_nonlocal_min": [0.2, 0.1],
        "safe_local_nonlocal_min": [0.1, 0.2],
    }
    assert set(statistics) == set(expected)
    for name, values in expected.items():
        assert statistics[name] == pytest.approx(values)


def test_eligibility_reuses_every_frozen_cross_source_gate() -> None:
    passing = dict(
        scigen_full_support=0.90,
        wyformer_full_support=0.90,
        scigen_shell_worst_auc=0.55,
        scigen_shell_evaluable_folds=5,
        wyformer_shell_pooled_auc=0.55,
        scigen_full_pooled_auc=0.50,
        wyformer_full_pooled_auc=0.50,
    )
    assert eligibility_from_metrics(**passing)
    for key in passing:
        failing = dict(passing)
        failing[key] = 4 if key.endswith("evaluable_folds") else float(failing[key]) - 1.0e-6
        assert not eligibility_from_metrics(**failing)


def test_selector_ranks_only_eligible_rows_deterministically() -> None:
    records = pd.DataFrame(
        {
            "hypothesis": ["b", "a", "c"],
            "eligible_for_search": [True, True, False],
            "ranking_min_auc": [0.60, 0.60, 0.99],
            "ranking_mean_auc": [0.70, 0.70, 0.99],
        }
    )
    table, selected = select_safe_margin_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(inspect.signature(run_signed_safe_margin_audit).parameters)
    assert "next190_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_signed_safe_margin_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT192 input is missing"):
        run_signed_safe_margin_audit(**kwargs)
