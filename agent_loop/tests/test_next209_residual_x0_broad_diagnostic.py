from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next209_residual_x0_broad_diagnostic import (
    candidate_key_sha256,
    run_residual_x0_broad_diagnostic,
    select_closest_residual,
    select_diagnostic_candidates,
    verify_sole_inactive_base,
)


def test_selects_exact_auc_safe_non_broad_population_and_digest() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["z", "a", "m", "b"],
            "passes_source_auc_gates": [True, True, False, True],
            "passes_safe_all_cells": [True, True, True, False],
            "passes_broad_all_cells": [False, True, False, False],
        }
    )
    selected = select_diagnostic_candidates(frame)
    assert selected["candidate_key"].tolist() == ["z"]
    assert candidate_key_sha256(selected) == (
        "594e519ae499312b29433b7dd8a97ff068defcba9755b6d5d00e84c524d67b06"
    )


def test_sole_candidate_must_be_the_inactive_base() -> None:
    valid = pd.DataFrame(
        {
            "candidate_key": ["base"],
            "feature": [None],
            "direction": [None],
            "comparison": [None],
            "cutoff": [None],
            "exception_fraction_numerator": [0],
            "exception_active_rows": [0],
            "exception_active_scigen": [0],
            "exception_active_wyformer": [0],
        }
    )
    row = verify_sole_inactive_base(valid)
    assert row["candidate_key"] == "base"
    invalid = valid.copy()
    invalid.loc[0, "feature"] = "x"
    with pytest.raises(ValueError, match="NEXT209 sole candidate is not the inactive base"):
        verify_sole_inactive_base(invalid)


def test_closest_residual_ranking_and_identity_tie_break_are_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["z", "b", "a"],
            "failed_constraint_count": [1, 1, 1],
            "normalized_shortfall_sum": [0.2, 0.1, 0.1],
        }
    )
    assert select_closest_residual(frame)["candidate_key"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_residual_x0_broad_diagnostic).parameters)
    assert "next208_dir" in parameters and "next207_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_diagnostic_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_residual_x0_broad_diagnostic)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT209 input is missing"):
        run_residual_x0_broad_diagnostic(**kwargs)
