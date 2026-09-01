from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next256_dvci_feature_audit import (
    HYPOTHESES,
    NEXT255_SOURCE_PATH,
    REQUIRED_DESIGN_STAGES,
    REQUIRED_STAGES,
    _index_dvci_by_prefixed_material_id,
    run_dvci_feature_audit,
    select_eligible_hypotheses,
)


def test_hypothesis_universe_and_directions_are_frozen() -> None:
    assert HYPOTHESES == (
        ("dvci_tightness_mean", "protected_high"),
        ("dvci_tightness_q10", "protected_high"),
        ("dvci_tightness_q25", "protected_high"),
        ("dvci_tightness_lower_quartile_mean", "protected_high"),
        ("dvci_volume_mean", "protected_high"),
        ("dvci_volume_q10", "protected_high"),
        ("dvci_volume_q25", "protected_high"),
        ("dvci_volume_lower_quartile_mean", "protected_high"),
        ("dvci_eigenratio_mean", "protected_high"),
        ("dvci_eigenratio_q10", "protected_high"),
        ("dvci_eigenratio_q25", "protected_high"),
        ("dvci_eigenratio_lower_quartile_mean", "protected_high"),
        ("dvci_closure_mean", "protected_high"),
        ("dvci_closure_q10", "protected_high"),
        ("dvci_closure_q25", "protected_high"),
        ("dvci_closure_lower_quartile_mean", "protected_high"),
    )


def test_next255_provenance_uses_exact_executed_source_path() -> None:
    assert NEXT255_SOURCE_PATH == "src/next255_delaunay_void_cage_isotropy.py"


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
    parameters = tuple(inspect.signature(run_dvci_feature_audit).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_dvci_identity_alignment_adds_reconstruction_source_prefix() -> None:
    table = pd.DataFrame(
        {"material_id": ["a", "b"], "dvci_tightness_mean": [0.2, 0.3]}
    )
    indexed = _index_dvci_by_prefixed_material_id(
        table=table,
        source="scigen",
        expected_material_ids=pd.Series(["scigen:a", "scigen:b"]),
    )
    assert indexed.index.tolist() == ["scigen:a", "scigen:b"]
    with pytest.raises(ValueError, match="material identity differs"):
        _index_dvci_by_prefixed_material_id(
            table=table,
            source="scigen",
            expected_material_ids=pd.Series(["scigen:a", "scigen:c"]),
        )


def test_audit_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT256 input is missing"):
        run_dvci_feature_audit(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in REQUIRED_STAGES},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}" for stage in REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design255",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
