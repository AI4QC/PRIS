from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next178_amplitude_extension_broad_residual_diagnostic import (
    run_amplitude_extension_broad_residual_diagnostic,
    select_diagnostic_candidates,
)


def test_selector_keeps_only_auc_safe_nonbroad_rows_in_canonical_order() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["d", "b", "a", "c"],
            "passes_source_auc_gates": [True, True, False, True],
            "passes_safe_all_cells": [True, True, True, False],
            "passes_broad_all_cells": [False, True, False, False],
        }
    )
    assert select_diagnostic_candidates(frame)["candidate_key"].tolist() == ["d"]


def test_selector_rejects_duplicate_or_incomplete_candidate_records() -> None:
    duplicate = pd.DataFrame(
        {
            "candidate_key": ["a", "a"],
            "passes_source_auc_gates": [True, True],
            "passes_safe_all_cells": [True, True],
            "passes_broad_all_cells": [False, False],
        }
    )
    with pytest.raises(ValueError, match="NEXT178 published candidate schema differs"):
        select_diagnostic_candidates(duplicate)
    with pytest.raises(ValueError, match="NEXT178 published candidate schema differs"):
        select_diagnostic_candidates(duplicate.drop(columns="passes_safe_all_cells"))


def test_formal_interface_has_no_validation_or_replication_endpoint() -> None:
    parameters = tuple(
        inspect.signature(
            run_amplitude_extension_broad_residual_diagnostic
        ).parameters
    )
    assert "next177_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_diagnostic_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_amplitude_extension_broad_residual_diagnostic)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"search_workers", "require_formal_inputs"}
    }
    kwargs["search_workers"] = 1
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT178 input is missing"):
        run_amplitude_extension_broad_residual_diagnostic(**kwargs)
