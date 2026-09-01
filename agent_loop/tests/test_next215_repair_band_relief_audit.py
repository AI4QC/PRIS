from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next215_repair_band_relief_audit import (
    REPAIR_LOWER_THRESHOLD,
    REPAIR_UPPER_THRESHOLD,
    USED_NEXT214_FEATURES,
    audit_one_source,
    directional_protection,
    ranking_auc_value,
    repair_band_mask,
    run_repair_band_relief_audit,
    select_relief_hypotheses,
)


def test_directional_protection_has_exact_protected_positive_semantics() -> None:
    values = np.array([1.0, 2.0, np.nan])
    np.testing.assert_array_equal(
        directional_protection(values, "protected_high")[:2], [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        directional_protection(values, "protected_low")[:2], [-1.0, -2.0]
    )
    with pytest.raises(ValueError, match="NEXT215 protection direction differs"):
        directional_protection(values, "unknown")


def test_repair_band_is_lower_inclusive_upper_exclusive_and_extreme_only() -> None:
    lower = REPAIR_LOWER_THRESHOLD
    upper = REPAIR_UPPER_THRESHOLD
    score = np.array(
        [lower, np.nextafter(upper, -np.inf), upper, np.nextafter(lower, -np.inf), np.nan]
    )
    support = np.array([True, True, True, True, True])
    endpoint = np.array([0.5, 2.5, 0.5, 2.5, 0.5])
    np.testing.assert_array_equal(
        repair_band_mask(
            score=score,
            support=support,
            endpoint=endpoint,
            lower=lower,
            upper=upper,
        ),
        [True, True, False, False, False],
    )

    intermediate = endpoint.copy()
    intermediate[0] = 1.5
    assert not repair_band_mask(
        score=score,
        support=support,
        endpoint=intermediate,
        lower=lower,
        upper=upper,
    )[0]


def test_source_audit_requires_coverage_counts_and_protection_auc() -> None:
    endpoint = np.array([0.5, 0.5, 2.5, 2.5] * 2)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cohort = np.ones(8, dtype=bool)
    values = np.array([1.0, 0.9, 0.1, 0.0] * 2)
    result = audit_one_source(
        values=values,
        endpoint=endpoint,
        cohort=cohort,
        folds=folds,
        direction="protected_high",
        expected_folds=(0, 1),
        minimum_coverage=1.0,
        minimum_class_count=2,
        minimum_aggregate_auc=0.9,
        minimum_macro_auc=0.9,
        minimum_worst_auc=0.9,
    )
    assert result["passes_source_gates"] is True
    assert result["aggregate_auc"] == pytest.approx(1.0)
    assert result["worst_fold_auc"] == pytest.approx(1.0)

    missing = values.copy()
    missing[0] = np.nan
    failed = audit_one_source(
        values=missing,
        endpoint=endpoint,
        cohort=cohort,
        folds=folds,
        direction="protected_high",
        expected_folds=(0, 1),
        minimum_coverage=1.0,
        minimum_class_count=2,
        minimum_aggregate_auc=0.9,
        minimum_macro_auc=0.9,
        minimum_worst_auc=0.9,
    )
    assert failed["passes_source_gates"] is False


def test_missing_fold_auc_gets_noncompetitive_ranking_value() -> None:
    assert ranking_auc_value(None) == -np.inf
    assert ranking_auc_value(0.625) == pytest.approx(0.625)
    with pytest.raises(ValueError, match="NEXT215 ranking AUC differs"):
        ranking_auc_value(float("nan"))


def test_ranking_vetoes_opposite_directions_and_next214_used_features() -> None:
    used = sorted(USED_NEXT214_FEATURES)[0]
    records = pd.DataFrame(
        {
            "hypothesis": [
                "b__protected_low",
                "a__protected_low",
                "a__protected_high",
                f"{used}__protected_high",
            ],
            "feature": ["b", "a", "a", used],
            "direction": [
                "protected_low",
                "protected_low",
                "protected_high",
                "protected_high",
            ],
            "passes_raw_gates": [True, True, True, True],
            "ranking_min_worst_fold_auc": [0.60, 0.60, 0.60, 0.99],
            "ranking_min_aggregate_auc": [0.70, 0.70, 0.70, 0.99],
            "ranking_mean_aggregate_auc": [0.75, 0.75, 0.75, 0.99],
        }
    )
    table, selected = select_relief_hypotheses(records)
    assert table.loc[table["feature"] == "a", "eligible_for_search"].sum() == 0
    used_row = table.loc[table["feature"] == used].iloc[0]
    assert not bool(used_row["eligible_for_search"])
    assert used_row["ineligibility_reason"] == "already_in_next214_path"
    assert selected is not None
    assert selected["hypothesis"] == "b__protected_low"

    records.loc[2, "passes_raw_gates"] = False
    table, selected = select_relief_hypotheses(records)
    assert table.loc[table["feature"] == "a", "eligible_for_search"].sum() == 1
    assert selected is not None
    assert selected["hypothesis"] == "a__protected_low"


def test_empty_eligibility_is_a_predeclared_stop() -> None:
    records = pd.DataFrame(
        {
            "hypothesis": ["a__protected_low", "a__protected_high"],
            "feature": ["a", "a"],
            "direction": ["protected_low", "protected_high"],
            "passes_raw_gates": [False, False],
            "ranking_min_worst_fold_auc": [0.4, 0.4],
            "ranking_min_aggregate_auc": [0.4, 0.4],
            "ranking_mean_aggregate_auc": [0.4, 0.4],
        }
    )
    table, selected = select_relief_hypotheses(records)
    assert not table["eligible_for_search"].any()
    assert selected is None


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_repair_band_relief_audit).parameters)
    assert "next214_dir" in parameters and "next213_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_repair_band_relief_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT215 input is missing"):
        run_repair_band_relief_audit(**kwargs)
