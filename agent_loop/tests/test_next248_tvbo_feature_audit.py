from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next248_tvbo_feature_audit import (
    HYPOTHESES,
    NEXT247_SOURCE_PATH,
    _index_tvbo_by_prefixed_material_id,
    run_tvbo_feature_audit,
    select_eligible_hypotheses,
)


def test_hypothesis_universe_and_directions_are_frozen() -> None:
    assert HYPOTHESES == (
        ("tvbo_w4_abs_mean", "protected_high"),
        ("tvbo_w4_abs_q10", "protected_high"),
        ("tvbo_w4_abs_std", "protected_low"),
        ("tvbo_w6_abs_mean", "protected_high"),
        ("tvbo_w6_abs_q10", "protected_high"),
        ("tvbo_w6_abs_std", "protected_low"),
        ("tvbo_bar_w4_abs_mean", "protected_high"),
        ("tvbo_bar_w4_abs_q10", "protected_high"),
        ("tvbo_bar_w4_abs_std", "protected_low"),
        ("tvbo_bar_w6_abs_mean", "protected_high"),
        ("tvbo_bar_w6_abs_q10", "protected_high"),
        ("tvbo_bar_w6_abs_std", "protected_low"),
        ("tvbo_w4_coarse_delta_mean", "protected_low"),
        ("tvbo_w4_coarse_delta_q90", "protected_low"),
        ("tvbo_w6_coarse_delta_mean", "protected_low"),
        ("tvbo_w6_coarse_delta_q90", "protected_low"),
    )


def test_next247_provenance_uses_exact_executed_source_path() -> None:
    assert NEXT247_SOURCE_PATH == "src/next247_third_order_voronoi_bond_order.py"


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
    parameters = tuple(inspect.signature(run_tvbo_feature_audit).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_tvbo_identity_alignment_adds_reconstruction_source_prefix() -> None:
    table = pd.DataFrame(
        {"material_id": ["a", "b"], "tvbo_w4_abs_mean": [0.2, 0.3]}
    )
    indexed = _index_tvbo_by_prefixed_material_id(
        table=table,
        source="scigen",
        expected_material_ids=pd.Series(["scigen:a", "scigen:b"]),
    )
    assert indexed.index.tolist() == ["scigen:a", "scigen:b"]
    with pytest.raises(ValueError, match="material identity differs"):
        _index_tvbo_by_prefixed_material_id(
            table=table,
            source="scigen",
            expected_material_ids=pd.Series(["scigen:a", "scigen:c"]),
        )


def test_audit_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT248 input is missing"):
        run_tvbo_feature_audit(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 248)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}" for stage in range(202, 248)
            },
            design_path=tmp_path / "design248",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
