from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next213_two_signal_broad_diagnostic import (
    candidate_key_sha256,
    run_two_signal_broad_diagnostic,
    select_closest_residual,
    select_diagnostic_candidates,
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


def test_closest_residual_ranking_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["z", "b", "a", "few"],
            "failed_constraint_count": [2, 2, 2, 1],
            "normalized_shortfall_sum": [0.1, 0.05, 0.05, 0.9],
        }
    )
    assert select_closest_residual(frame)["candidate_key"] == "few"
    assert select_closest_residual(frame.iloc[:3])["candidate_key"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_two_signal_broad_diagnostic).parameters)
    assert "next212_dir" in parameters and "next211_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_diagnostic_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_two_signal_broad_diagnostic)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT213 input is missing"):
        run_two_signal_broad_diagnostic(**kwargs)
