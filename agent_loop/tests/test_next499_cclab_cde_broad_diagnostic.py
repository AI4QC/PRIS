from __future__ import annotations

import hashlib
import inspect

import pandas as pd
import pytest


def test_selection_is_exact_auc_safe_non_broad_population() -> None:
    from src.next499_cclab_cde_broad_diagnostic import select_diagnostic_candidates

    frame = pd.DataFrame(
        {
            "candidate_key": ["a", "b", "c", "d", "e"],
            "eligible_new_candidate": [True, True, True, False, True],
            "passes_source_auc_gates": [True, True, False, True, True],
            "passes_safe_all_cells": [True, True, True, True, False],
            "passes_broad_all_cells": [False, True, False, False, False],
        }
    )
    selected = select_diagnostic_candidates(frame)
    assert selected["candidate_key"].tolist() == ["a"]


def test_candidate_digest_is_sorted_and_order_independent() -> None:
    from src.next499_cclab_cde_broad_diagnostic import candidate_key_sha256

    expected = hashlib.sha256("a\nz".encode()).hexdigest()
    assert candidate_key_sha256(pd.DataFrame({"candidate_key": ["z", "a"]})) == expected
    assert candidate_key_sha256(pd.DataFrame({"candidate_key": ["a", "z"]})) == expected


def test_closest_residual_uses_failure_count_shortfall_then_key() -> None:
    from src.next499_cclab_cde_broad_diagnostic import select_closest_residual

    frame = pd.DataFrame(
        {
            "candidate_key": ["z", "b", "a"],
            "failed_constraint_count": [4, 3, 3],
            "normalized_shortfall_sum": [0.01, 0.2, 0.2],
        }
    )
    assert select_closest_residual(frame)["candidate_key"] == "a"


def test_interface_excludes_validation_and_replication() -> None:
    from src.next499_cclab_cde_broad_diagnostic import (
        run_cclab_cde_broad_diagnostic,
    )

    parameters = tuple(inspect.signature(run_cclab_cde_broad_diagnostic).parameters)
    assert {"next496_dir", "next497_dir", "next498_dir"} <= set(parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert not any(
        token in name
        for name in parameters
        for token in ("validation", "replication")
    )


def test_diagnostic_fails_closed_on_missing_input(tmp_path) -> None:
    from src.next499_cclab_cde_broad_diagnostic import (
        REQUIRED_DESIGN_STAGES,
        REQUIRED_STAGES,
        run_cclab_cde_broad_diagnostic,
    )

    with pytest.raises(FileNotFoundError, match="NEXT499 input is missing"):
        run_cclab_cde_broad_diagnostic(
            scigen_feature_dir=tmp_path / "sf",
            scigen_discovery_endpoint_dir=tmp_path / "se",
            wyformer_feature_dir=tmp_path / "wf",
            wyformer_discovery_endpoint_dir=tmp_path / "we",
            stage_dirs={stage: tmp_path / f"n{stage}" for stage in REQUIRED_STAGES},
            next135_freeze_path=tmp_path / "freeze",
            design_paths={
                stage: tmp_path / f"d{stage}" for stage in REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design",
            next412_dir=tmp_path / "n412",
            next496_dir=tmp_path / "n496",
            next497_dir=tmp_path / "n497",
            next498_dir=tmp_path / "n498",
            output_dir=tmp_path / "out",
            require_formal_inputs=False,
        )
