from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next275_species_resolved_radical_packing as n275
import src.next276_species_resolved_radical_feature_audit as n


def test_hypothesis_universe_is_exactly_ten_features_times_two() -> None:
    assert n.HYPOTHESES == tuple(
        (feature, direction)
        for feature in n275.FEATURE_NAMES
        for direction in ("protected_low", "protected_high")
    )
    assert len(n.HYPOTHESES) == 20


def test_next275_provenance_uses_exact_executed_source_path() -> None:
    assert n.NEXT275_SOURCE_PATH == "src/next275_species_resolved_radical_packing.py"


def test_bounded_mapping_supports_both_directions_and_abstention() -> None:
    values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, np.nan])
    low = n.bounded_protection(
        values=values, direction="protected_low", q_lo=0.25, q_hi=0.75
    )
    high = n.bounded_protection(
        values=values, direction="protected_high", q_lo=0.25, q_hi=0.75
    )
    np.testing.assert_allclose(low[:5], [1.0, 1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(high[:5], [0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.isnan(low[5]) and np.isnan(high[5])


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
    selected, leader = n.select_eligible_hypotheses(frame)
    assert selected.set_index("hypothesis")["eligible_for_search"].to_dict() == {
        "a": True,
        "b": False,
        "z": True,
    }
    assert leader is not None and leader["hypothesis"] == "a"


def test_audit_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(
        inspect.signature(n.run_species_resolved_radical_feature_audit).parameters
    )
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_species_resolved_identity_alignment_adds_source_prefix() -> None:
    table = pd.DataFrame(
        {"material_id": ["a", "b"], "prvs_volume_within_cv": [0.2, 0.3]}
    )
    indexed = n._index_prvs_by_prefixed_material_id(
        table=table,
        source="scigen",
        expected_material_ids=pd.Series(["scigen:a", "scigen:b"]),
    )
    assert indexed.index.tolist() == ["scigen:a", "scigen:b"]
    with pytest.raises(ValueError, match="material identity differs"):
        n._index_prvs_by_prefixed_material_id(
            table=table,
            source="scigen",
            expected_material_ids=pd.Series(["scigen:a", "scigen:c"]),
        )


def test_audit_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT276 input is missing"):
        n.run_species_resolved_radical_feature_audit(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in n.REQUIRED_STAGES},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}"
                for stage in n.REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design275",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
