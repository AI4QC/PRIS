from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next236_chemistry_conditioned_feature_audit import (
    _ranking_auc,
    assign_lower_inclusive_strata,
    chemistry_conditioned_protection,
    fit_conditioned_cutoffs,
    run_chemistry_conditioned_feature_audit,
    select_conditioned_hypotheses,
)


def test_unavailable_auc_is_ineligible_and_sorted_below_evaluable_values() -> None:
    assert _ranking_auc(None) == -1.0
    assert _ranking_auc(0.5) == 0.5


def test_stratum_boundary_values_stay_in_lower_stratum() -> None:
    got = assign_lower_inclusive_strata(
        np.array([0.0, 1.0, 1.5, 2.0, 3.0, np.nan]),
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_array_equal(got, [0, 0, 1, 1, 2, -1])


def test_conditioned_cutoffs_and_protection_are_endpoint_free_and_bounded() -> None:
    signature = inspect.signature(fit_conditioned_cutoffs)
    assert not any("endpoint" in name for name in signature.parameters)
    conditioner = np.repeat(np.arange(4, dtype=float), 16)
    values = np.tile(np.arange(16, dtype=float), 4)
    model = fit_conditioned_cutoffs(
        values=values,
        conditioner=conditioner,
        stratum_edges=np.array([0.0, 1.0, 2.0]),
    )
    protection = chemistry_conditioned_protection(
        values=np.array([0.0, 15.0, 0.0, 15.0]),
        conditioner=np.array([0.0, 0.0, 3.0, 3.0]),
        direction="protected_high",
        model=model,
    )
    np.testing.assert_allclose(protection, [0.0, 1.0, 0.0, 1.0])
    low = chemistry_conditioned_protection(
        values=np.array([0.0, 15.0]),
        conditioner=np.array([0.0, 3.0]),
        direction="protected_low",
        model=model,
    )
    np.testing.assert_allclose(low, [1.0, 0.0])


def test_conditioned_selection_applies_pairwise_opposite_direction_veto() -> None:
    frame = pd.DataFrame(
        {
            "hypothesis": ["a__c__protected_low", "a__c__protected_high"],
            "feature": ["a", "a"],
            "conditioner": ["c", "c"],
            "direction": ["protected_low", "protected_high"],
            "passes_raw_gates": [True, True],
            "ranking_min_worst_fold_auc": [0.6, 0.6],
            "ranking_min_aggregate_auc": [0.7, 0.7],
            "ranking_mean_aggregate_auc": [0.75, 0.75],
        }
    )
    table, selected = select_conditioned_hypotheses(frame)
    assert table["eligible_for_search"].sum() == 0
    assert selected is None


def test_formal_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(
        inspect.signature(run_chemistry_conditioned_feature_audit).parameters
    )
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT236 input is missing"):
        run_chemistry_conditioned_feature_audit(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 236)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 236)},
            design_path=tmp_path / "design236",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
