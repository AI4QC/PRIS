from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next233_final_stagewise_margin_local_feature_audit import (
    audit_one_source,
    build_rejected_extreme_cohort,
    run_final_stagewise_margin_local_feature_audit,
    select_auditable_features,
    select_residual_hypothesis,
)


def test_final_rejected_extreme_cohort_is_inclusive_and_support_preserving() -> None:
    cohort = build_rejected_extreme_cohort(
        score=np.array([0.1, 0.2, 0.3, np.nan, 0.4]),
        support=np.array([True, True, True, True, False]),
        endpoint=np.array([0.5, 0.5, 1.5, 2.5, 2.5]),
        threshold=0.2,
    )
    np.testing.assert_array_equal(cohort, [False, True, False, False, False])


def test_final_schema_selection_reuses_exact_raw_numeric_policy() -> None:
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


def test_final_source_audit_keeps_opposite_direction_veto() -> None:
    endpoint = np.array([0.5, 0.5, 2.5, 2.5] * 2)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = audit_one_source(
        values=np.array([0.0, 0.1, 0.9, 1.0] * 2),
        endpoint=endpoint,
        cohort=np.ones(8, dtype=bool),
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
    rows = pd.DataFrame(
        {
            "hypothesis": ["a__protected_low", "a__protected_high"],
            "feature": ["a", "a"],
            "direction": ["protected_low", "protected_high"],
            "passes_raw_gates": [True, True],
            "ranking_min_worst_fold_auc": [0.6, 0.6],
            "ranking_min_aggregate_auc": [0.7, 0.7],
            "ranking_mean_aggregate_auc": [0.75, 0.75],
        }
    )
    table, selected = select_residual_hypothesis(rows)
    assert table["eligible_for_search"].sum() == 0
    assert selected is None


def test_final_audit_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(
        inspect.signature(
            run_final_stagewise_margin_local_feature_audit
        ).parameters
    )
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_final_audit_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT233 input is missing"):
        run_final_stagewise_margin_local_feature_audit(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 233)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 233)},
            design_path=tmp_path / "design233",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
