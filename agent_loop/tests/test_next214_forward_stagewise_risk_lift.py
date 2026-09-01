from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next214_forward_stagewise_risk_lift import (
    MAX_TERMS,
    build_stage_specs,
    run_forward_stagewise_risk_lift,
    strictly_improves,
)


def test_stage_specs_include_unchanged_path_and_exclude_used_hypotheses() -> None:
    normalizations = {
        "a__protected_low": {
            "feature": "a",
            "direction": "protected_low",
            "q_lo": 0.0,
            "q_hi": 1.0,
        },
        "b__protected_high": {
            "feature": "b",
            "direction": "protected_high",
            "q_lo": 1.0,
            "q_hi": 2.0,
        },
        "c__protected_low": {
            "feature": "c",
            "direction": "protected_low",
            "q_lo": 2.0,
            "q_hi": 3.0,
        },
    }
    specs = build_stage_specs(
        current_path_key="path",
        current_terms=(
            {"hypothesis": "a__protected_low", "amplitude_fraction": 1 / 16},
        ),
        normalizations=normalizations,
        amplitude_fractions=(1 / 16, 1 / 8),
        risk_scale=0.4,
        residual_threshold=0.2,
    )
    assert len(specs) == 5
    assert specs[0]["proposed_hypothesis"] is None
    assert {spec["proposed_hypothesis"] for spec in specs[1:]} == {
        "b__protected_high",
        "c__protected_low",
    }
    assert len({str(spec["candidate_key"]) for spec in specs}) == len(specs)
    assert MAX_TERMS == 8


def test_strict_improvement_prefers_fewer_failures_then_shortfall() -> None:
    current = {"failed_constraint_count": 6, "normalized_shortfall_sum": 0.3}
    assert strictly_improves(
        {"failed_constraint_count": 5, "normalized_shortfall_sum": 0.9}, current
    )
    assert strictly_improves(
        {"failed_constraint_count": 6, "normalized_shortfall_sum": 0.29}, current
    )
    assert not strictly_improves(
        {"failed_constraint_count": 6, "normalized_shortfall_sum": 0.3 - 5e-13},
        current,
    )
    assert not strictly_improves(
        {"failed_constraint_count": 7, "normalized_shortfall_sum": 0.0}, current
    )


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_forward_stagewise_risk_lift).parameters)
    assert "next213_dir" in parameters and "next212_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_loop_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_forward_stagewise_risk_lift)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT214 input is missing"):
        run_forward_stagewise_risk_lift(**kwargs)
