from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.next223_dual_evidence_consensus_search import (
    BETA_FRACTIONS,
    EXPECTED_CONTROL_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    EXPECTED_TOTAL_CANDIDATE_COUNT,
    PROTECTION_BUDGET_FRACTIONS,
    _assert_record_reproduction,
    build_dual_evidence_candidate_specs,
    dual_evidence_consensus_score,
    run_dual_evidence_consensus_search,
)


def _normalizations() -> dict[str, dict[str, object]]:
    return {
        name: {
            "hypothesis": name,
            "feature": name.split("__")[0],
            "direction": name.split("__")[1],
            "q_lo": 0.0,
            "q_hi": 1.0,
        }
        for name in ("a__protected_high", "b__protected_high")
    }


def test_dual_evidence_score_allocates_ordered_protection_and_risk_budgets() -> None:
    score, support, active, delta = dual_evidence_consensus_score(
        base_score=np.full(4, 0.4),
        current_delta=np.zeros(4),
        base_support=np.ones(4, dtype=bool),
        protection_values=np.array([1.0, 0.0, 1.0, 0.0]),
        protection_direction="protected_high",
        protection_q_lo=0.0,
        protection_q_hi=1.0,
        risk_values=np.array([1.0, 0.0, 0.0, 1.0]),
        risk_direction="protected_high",
        risk_q_lo=0.0,
        risk_q_hi=1.0,
        lower=0.2,
        upper=0.6,
        beta_fraction=0.25,
        protection_budget_fraction=0.75,
    )
    np.testing.assert_array_equal(support, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(active, np.ones(4, dtype=bool))
    np.testing.assert_allclose(delta, [-0.15, 0.05, -0.10, 0.0])
    np.testing.assert_allclose(score, [0.25, 0.45, 0.30, 0.40])

    swapped, _, _, _ = dual_evidence_consensus_score(
        base_score=np.full(4, 0.4),
        current_delta=np.zeros(4),
        base_support=np.ones(4, dtype=bool),
        protection_values=np.array([1.0, 0.0, 0.0, 1.0]),
        protection_direction="protected_high",
        protection_q_lo=0.0,
        protection_q_hi=1.0,
        risk_values=np.array([1.0, 0.0, 1.0, 0.0]),
        risk_direction="protected_high",
        risk_q_lo=0.0,
        risk_q_hi=1.0,
        lower=0.2,
        upper=0.6,
        beta_fraction=0.25,
        protection_budget_fraction=0.75,
    )
    assert not np.array_equal(score, swapped)


def test_equal_budget_diagonal_exactly_reduces_to_univariate_signed_term() -> None:
    values = np.array([0.0, 0.5, 1.0])
    score, _, _, delta = dual_evidence_consensus_score(
        base_score=np.full(3, 0.4),
        current_delta=np.zeros(3),
        base_support=np.ones(3, dtype=bool),
        protection_values=values,
        protection_direction="protected_high",
        protection_q_lo=0.0,
        protection_q_hi=1.0,
        risk_values=values,
        risk_direction="protected_high",
        risk_q_lo=0.0,
        risk_q_hi=1.0,
        lower=0.2,
        upper=0.6,
        beta_fraction=0.25,
        protection_budget_fraction=0.5,
    )
    np.testing.assert_allclose(delta, [0.1, 0.0, -0.1])
    np.testing.assert_allclose(score, [0.5, 0.4, 0.3])


def test_dual_evidence_score_uses_original_band_floor_and_pair_missing_off() -> None:
    base = np.array([0.19, 0.2, 0.4, 0.59, 0.6])
    current = np.array([0.0, -0.3, -0.5, 0.05, 0.0])
    score, support, active, delta = dual_evidence_consensus_score(
        base_score=base,
        current_delta=current,
        base_support=np.ones(5, dtype=bool),
        protection_values=np.array([1.0, 1.0, np.nan, 0.0, 0.0]),
        protection_direction="protected_high",
        protection_q_lo=0.0,
        protection_q_hi=1.0,
        risk_values=np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        risk_direction="protected_high",
        risk_q_lo=0.0,
        risk_q_hi=1.0,
        lower=0.2,
        upper=0.6,
        beta_fraction=0.25,
        protection_budget_fraction=0.5,
    )
    np.testing.assert_array_equal(active, [False, True, False, True, False])
    np.testing.assert_array_equal(support, np.ones(5, dtype=bool))
    assert delta[0] == current[0]
    assert delta[2] == current[2]
    assert score[1] == 0.0
    assert score[2] == 0.0
    assert score[3] == pytest.approx(0.74)
    assert score[4] == base[4]


def test_candidate_specs_freeze_exact_ordered_allocated_grammar() -> None:
    specs = build_dual_evidence_candidate_specs(
        current_path_key="next222-final",
        current_terms=[{"hypothesis": "existing"}],
        normalizations=_normalizations(),
        beta_fractions=(1 / 64, 1 / 32),
        protection_budget_fractions=(0.25, 0.5, 0.75),
    )
    assert len(specs) == 25
    assert specs[0]["protection_hypothesis"] is None
    controls = [spec for spec in specs if spec["is_reproduction_control"]]
    eligible = [spec for spec in specs if spec["eligible_new_candidate"]]
    assert len(controls) == 4
    assert len(eligible) == 20
    assert len({str(spec["candidate_key"]) for spec in specs}) == 25
    assert BETA_FRACTIONS == (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
    assert PROTECTION_BUDGET_FRACTIONS == (0.25, 0.5, 0.75)
    assert EXPECTED_TOTAL_CANDIDATE_COUNT == 7261
    assert EXPECTED_CONTROL_COUNT == 110
    assert EXPECTED_ELIGIBLE_COUNT == 7150


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_dual_evidence_consensus_search).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_control_reproduction_equates_json_none_with_parquet_nan() -> None:
    observed = {
        "safe_threshold": None,
        "scigen_pooled_auc": 0.8,
        "scigen_macro_auc": 0.7,
        "scigen_worst_auc": 0.6,
        "wyformer_pooled_auc": 0.75,
        "wyformer_macro_auc": 0.7,
        "wyformer_worst_auc": 0.55,
        "passes_source_auc_gates": True,
        "passes_safe_all_cells": False,
        "passes_broad_all_cells": False,
        "passes_all_discovery_gates": False,
    }
    expected = {**observed, "safe_threshold": np.nan}
    _assert_record_reproduction(observed, expected)


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT223 input is missing"):
        run_dual_evidence_consensus_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 223)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 223)},
            design_path=tmp_path / "design223",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
