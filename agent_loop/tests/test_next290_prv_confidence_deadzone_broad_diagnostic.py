from __future__ import annotations

import hashlib
import inspect

import pandas as pd
import pytest

from src.next290_prv_confidence_deadzone_broad_diagnostic import (
    _next289_catalogue_identity_ok,
    candidate_key_sha256,
    run_prv_confidence_deadzone_broad_diagnostic,
    select_closest_residual,
    select_diagnostic_candidates,
)


def test_next289_catalogue_identity_does_not_require_one_sided_monotonicity() -> None:
    assert _next289_catalogue_identity_ok(
        {"candidate_count": 85, "score_never_exceeds_next224": False}
    )
    assert _next289_catalogue_identity_ok({"candidate_count": 85})
    assert not _next289_catalogue_identity_ok({"candidate_count": 84})


def test_diagnostic_selection_is_exact_auc_safe_non_broad_population() -> None:
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
    first = pd.DataFrame({"candidate_key": ["z", "a"]})
    second = pd.DataFrame({"candidate_key": ["a", "z"]})
    expected = hashlib.sha256("a\nz".encode()).hexdigest()
    assert candidate_key_sha256(first) == expected
    assert candidate_key_sha256(second) == expected


def test_closest_residual_uses_failure_count_then_shortfall_then_key() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["z", "b", "a"],
            "failed_constraint_count": [4, 3, 3],
            "normalized_shortfall_sum": [0.01, 0.2, 0.2],
        }
    )
    assert select_closest_residual(frame)["candidate_key"] == "a"


def test_diagnostic_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(
        inspect.signature(
            run_prv_confidence_deadzone_broad_diagnostic
        ).parameters
    )
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_diagnostic_fails_closed_on_missing_input(tmp_path) -> None:
    from src.next290_prv_confidence_deadzone_broad_diagnostic import (
        REQUIRED_DESIGN_STAGES,
        REQUIRED_STAGES,
    )

    with pytest.raises(FileNotFoundError, match="NEXT290 input is missing"):
        run_prv_confidence_deadzone_broad_diagnostic(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in REQUIRED_STAGES},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}" for stage in REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design289",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
