from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.next222_forward_stagewise_signed_redistribution import (
    build_stage_specs,
    cumulative_signed_score,
    run_forward_stagewise_signed_redistribution,
    strictly_improves,
)


def test_cumulative_score_uses_original_band_and_fixed_nonnegative_floor() -> None:
    base = np.array([0.19, 0.2, 0.4, 0.59, 0.6])
    delta = np.array([0.0, -0.15, -0.5, 0.05, 0.0])
    score, support, active, proposed_delta = cumulative_signed_score(
        base_score=base,
        current_delta=delta,
        base_support=np.ones(5, dtype=bool),
        feature_values=np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        lower=0.2,
        upper=0.6,
        beta_fraction=0.25,
    )
    np.testing.assert_array_equal(active, [False, True, True, True, False])
    np.testing.assert_array_equal(support, np.ones(5, dtype=bool))
    assert proposed_delta[0] == delta[0]
    assert score[1] == 0.0
    assert score[2] == 0.0
    assert score[3] == pytest.approx(0.74)
    assert score[4] == base[4]


def test_stage_specs_include_unchanged_and_all_unused_hypotheses() -> None:
    normalizations = {
        name: {
            "hypothesis": name,
            "feature": name.split("__")[0],
            "direction": name.split("__")[1],
            "q_lo": 0.0,
            "q_hi": 1.0,
        }
        for name in (
            "a__protected_high",
            "b__protected_low",
            "c__protected_high",
        )
    }
    specs = build_stage_specs(
        current_path_key="start",
        current_terms=[{"hypothesis": "a__protected_high"}],
        normalizations=normalizations,
        beta_fractions=(1 / 64, 1 / 32),
    )
    assert len(specs) == 5
    assert specs[0]["proposed_hypothesis"] is None
    assert {spec["proposed_hypothesis"] for spec in specs[1:]} == {
        "b__protected_low",
        "c__protected_high",
    }
    assert len({str(spec["candidate_key"]) for spec in specs}) == 5


def test_strict_improvement_uses_failure_count_then_shortfall() -> None:
    current = {"failed_constraint_count": 6, "normalized_shortfall_sum": 0.25}
    assert strictly_improves(
        {"failed_constraint_count": 5, "normalized_shortfall_sum": 1.0}, current
    )
    assert strictly_improves(
        {"failed_constraint_count": 6, "normalized_shortfall_sum": 0.24}, current
    )
    assert not strictly_improves(
        {"failed_constraint_count": 6, "normalized_shortfall_sum": 0.25}, current
    )


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(
        inspect.signature(run_forward_stagewise_signed_redistribution).parameters
    )
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT222 input is missing"):
        run_forward_stagewise_signed_redistribution(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 222)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 222)},
            design_path=tmp_path / "design222",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
