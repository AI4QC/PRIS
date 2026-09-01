from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next198_discrete_protected_exception_broad_residual import (
    run_discrete_protected_exception_broad_residual,
    select_closest_residual,
    select_diagnostic_candidates,
)


def test_selects_only_auc_and_safe_candidates_that_fail_broad() -> None:
    published = pd.DataFrame(
        {
            "candidate_key": ["d", "c", "b", "a"],
            "passes_source_auc_gates": [True, True, False, True],
            "passes_safe_all_cells": [True, False, True, True],
            "passes_broad_all_cells": [False, False, False, True],
        }
    )
    selected = select_diagnostic_candidates(published)
    assert selected["candidate_key"].tolist() == ["d"]


def test_closest_residual_uses_frozen_lexicographic_tie_breaking() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["b", "a", "c"],
            "failed_constraint_count": [2, 2, 1],
            "normalized_shortfall_sum": [0.1, 0.1, 0.9],
        }
    )
    assert select_closest_residual(frame)["candidate_key"] == "c"
    assert select_closest_residual(frame.iloc[:2])["candidate_key"] == "a"


def test_formal_interface_has_only_discovery_endpoints() -> None:
    parameters = tuple(
        inspect.signature(
            run_discrete_protected_exception_broad_residual
        ).parameters
    )
    assert "next197_dir" in parameters and "next196_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_diagnostic_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(
        run_discrete_protected_exception_broad_residual
    )
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"search_workers", "require_formal_inputs"}
    }
    kwargs["search_workers"] = 1
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT198 input is missing"):
        run_discrete_protected_exception_broad_residual(**kwargs)
