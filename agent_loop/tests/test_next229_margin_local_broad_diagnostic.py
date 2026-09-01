from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next229_margin_local_broad_diagnostic import (
    candidate_key_sha256,
    run_margin_local_broad_diagnostic,
    select_closest_residual,
    select_diagnostic_candidates,
)


def test_diagnostic_population_is_exact_eligible_auc_safe_nonbroad_subset() -> None:
    published = pd.DataFrame(
        {
            "candidate_key": ["d", "c", "b", "a"],
            "eligible_new_candidate": [True, True, True, False],
            "passes_source_auc_gates": [True, True, False, True],
            "passes_safe_all_cells": [True, True, True, True],
            "passes_broad_all_cells": [False, True, False, False],
        }
    )
    selected = select_diagnostic_candidates(published)
    assert selected["candidate_key"].tolist() == ["d"]


def test_candidate_digest_is_sorted_and_closest_rank_is_frozen() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["z", "a", "b"],
            "failed_constraint_count": [6, 5, 5],
            "normalized_shortfall_sum": [0.01, 0.2, 0.2],
        }
    )
    assert candidate_key_sha256(frame) == candidate_key_sha256(frame.iloc[::-1])
    closest = select_closest_residual(frame)
    assert closest["candidate_key"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_margin_local_broad_diagnostic).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_diagnostic_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT229 input is missing"):
        run_margin_local_broad_diagnostic(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 229)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 229)},
            design_path=tmp_path / "design229",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
