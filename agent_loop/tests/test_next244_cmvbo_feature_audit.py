from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next244_cmvbo_feature_audit import (
    HYPOTHESES,
    _index_cmvbo_by_prefixed_material_id,
    run_cmvbo_feature_audit,
    select_eligible_hypotheses,
)


def test_hypothesis_universe_and_directions_are_frozen() -> None:
    assert HYPOTHESES == (
        ("cmvbo_bar_q4_mean", "protected_high"),
        ("cmvbo_bar_q4_q10", "protected_high"),
        ("cmvbo_bar_q4_std", "protected_low"),
        ("cmvbo_bar_q6_mean", "protected_high"),
        ("cmvbo_bar_q6_q10", "protected_high"),
        ("cmvbo_bar_q6_std", "protected_low"),
        ("cmvbo_coherence_q4_mean", "protected_high"),
        ("cmvbo_coherence_q4_q10", "protected_high"),
        ("cmvbo_coherence_q6_mean", "protected_high"),
        ("cmvbo_coherence_q6_q10", "protected_high"),
        ("cmvbo_neighbor_corr_q4_mean", "protected_high"),
        ("cmvbo_neighbor_corr_q4_q10", "protected_high"),
        ("cmvbo_neighbor_corr_q6_mean", "protected_high"),
        ("cmvbo_neighbor_corr_q6_q10", "protected_high"),
        ("cmvbo_neighbor_corr_joint_q10", "protected_high"),
    )


def test_selection_uses_only_frozen_gates_and_reporting_rank() -> None:
    frame = pd.DataFrame(
        {
            "hypothesis": ["z", "a", "b"],
            "passes_raw_gates": [True, True, False],
            "ranking_min_worst_fold_auc": [0.6, 0.6, 0.9],
            "ranking_min_aggregate_auc": [0.7, 0.7, 0.9],
            "ranking_mean_aggregate_auc": [0.8, 0.8, 0.9],
        }
    )
    selected, leader = select_eligible_hypotheses(frame)
    assert selected.set_index("hypothesis")["eligible_for_search"].to_dict() == {
        "a": True,
        "b": False,
        "z": True,
    }
    assert leader is not None and leader["hypothesis"] == "a"


def test_audit_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(inspect.signature(run_cmvbo_feature_audit).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_cmvbo_identity_alignment_adds_reconstruction_source_prefix() -> None:
    table = pd.DataFrame(
        {"material_id": ["a", "b"], "cmvbo_bar_q4_mean": [0.2, 0.3]}
    )
    indexed = _index_cmvbo_by_prefixed_material_id(
        table=table,
        source="scigen",
        expected_material_ids=pd.Series(["scigen:a", "scigen:b"]),
    )
    assert indexed.index.tolist() == ["scigen:a", "scigen:b"]
    with pytest.raises(ValueError, match="material identity differs"):
        _index_cmvbo_by_prefixed_material_id(
            table=table,
            source="scigen",
            expected_material_ids=pd.Series(["scigen:a", "scigen:c"]),
        )


def test_audit_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT244 input is missing"):
        run_cmvbo_feature_audit(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 244)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}" for stage in range(202, 244)
            },
            design_path=tmp_path / "design244",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
