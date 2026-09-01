from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next207_residual_x0_feature_audit import (
    audit_one_source,
    blocked_feature_name,
    directional_risk,
    run_residual_x0_feature_audit,
    select_auditable_features,
    select_residual_hypothesis,
)


@pytest.mark.parametrize(
    "name",
    [
        "raw_material_id",
        "source_member_bytes",
        "generated_space_group",
        "natoms",
        "geom_species_count",
        "_encoded",
        "pauling_pauling_p2_value",
        "family_supported",
        "family_site_count",
        "family_edge_count",
    ],
)
def test_identifier_support_and_count_columns_are_blocked(name: str) -> None:
    assert blocked_feature_name(name)


def test_schema_selection_keeps_only_sorted_numeric_physical_columns() -> None:
    frame = pd.DataFrame(
        {
            "z_feature": [1.0, 2.0],
            "a_feature": [2, 3],
            "text": ["x", "y"],
            "natoms": [2, 4],
            "foo_supported": [True, False],
        }
    )
    assert select_auditable_features(frame) == ("a_feature", "z_feature")


def test_directional_risk_has_exact_protection_semantics() -> None:
    values = np.array([1.0, 2.0, np.nan])
    np.testing.assert_array_equal(
        directional_risk(values, "protected_low")[:2], [1.0, 2.0]
    )
    np.testing.assert_array_equal(
        directional_risk(values, "protected_high")[:2], [-1.0, -2.0]
    )
    with pytest.raises(ValueError, match="NEXT207 protection direction differs"):
        directional_risk(values, "unknown")


def test_source_audit_requires_coverage_counts_and_consistent_auc() -> None:
    endpoint = np.array([0.5, 0.5, 2.5, 2.5] * 2)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    cohort = np.ones(8, dtype=bool)
    values = np.array([0.0, 0.1, 0.9, 1.0] * 2)
    result = audit_one_source(
        values=values,
        endpoint=endpoint,
        cohort=cohort,
        folds=folds,
        direction="protected_low",
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
        direction="protected_low",
        expected_folds=(0, 1),
        minimum_coverage=1.0,
        minimum_class_count=2,
        minimum_aggregate_auc=0.9,
        minimum_macro_auc=0.9,
        minimum_worst_auc=0.9,
    )
    assert failed["passes_source_gates"] is False


def test_eligible_ranking_is_deterministic_and_rejects_opposite_pair() -> None:
    records = pd.DataFrame(
        {
            "hypothesis": ["b__protected_low", "a__protected_low", "a__protected_high"],
            "feature": ["b", "a", "a"],
            "direction": ["protected_low", "protected_low", "protected_high"],
            "passes_raw_gates": [True, True, True],
            "ranking_min_worst_fold_auc": [0.6, 0.6, 0.6],
            "ranking_min_aggregate_auc": [0.7, 0.7, 0.7],
            "ranking_mean_aggregate_auc": [0.75, 0.75, 0.75],
        }
    )
    table, selected = select_residual_hypothesis(records)
    assert table.loc[table["feature"] == "a", "eligible_for_search"].sum() == 0
    assert table.loc[table["feature"] == "b", "eligible_for_search"].sum() == 1
    assert selected is not None
    assert selected["hypothesis"] == "b__protected_low"

    records.loc[2, "passes_raw_gates"] = False
    table, selected = select_residual_hypothesis(records)
    assert table.loc[table["feature"] == "a", "eligible_for_search"].sum() == 1
    assert selected is not None
    assert selected["hypothesis"] == "a__protected_low"


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_residual_x0_feature_audit).parameters)
    assert "next206_dir" in parameters and "next205_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_residual_x0_feature_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT207 input is missing"):
        run_residual_x0_feature_audit(**kwargs)
