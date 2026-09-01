from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next260_pvbp_feature_audit import (
    HYPOTHESES,
    NEXT259_SOURCE_PATH,
    REQUIRED_DESIGN_STAGES,
    REQUIRED_STAGES,
    _index_pvbp_by_prefixed_material_id,
    bounded_protection,
    run_pvbp_feature_audit,
    select_eligible_hypotheses,
)


def test_hypothesis_universe_and_directions_are_frozen() -> None:
    assert HYPOTHESES == tuple(
        (f"pvbp_{metric}_{statistic}", "protected_low")
        for metric in (
            "isolation_any",
            "isolation_3d",
            "prominence_any",
            "radius",
        )
        for statistic in ("mean", "q75", "q90", "upper_quartile_mean")
    )


def test_next259_provenance_uses_exact_executed_source_path() -> None:
    assert NEXT259_SOURCE_PATH == "src/next259_periodic_void_bottleneck_persistence.py"


def test_protected_low_mapping_is_monotone_and_preserves_abstention() -> None:
    values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, np.nan])
    mapped = bounded_protection(
        values=values,
        direction="protected_low",
        q_lo=0.25,
        q_hi=0.75,
    )
    np.testing.assert_allclose(mapped[:5], [1.0, 1.0, 0.5, 0.0, 0.0])
    assert np.isnan(mapped[5])


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
    parameters = tuple(inspect.signature(run_pvbp_feature_audit).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_pvbp_identity_alignment_adds_reconstruction_source_prefix() -> None:
    table = pd.DataFrame(
        {"material_id": ["a", "b"], "pvbp_isolation_any_mean": [0.2, 0.3]}
    )
    indexed = _index_pvbp_by_prefixed_material_id(
        table=table,
        source="scigen",
        expected_material_ids=pd.Series(["scigen:a", "scigen:b"]),
    )
    assert indexed.index.tolist() == ["scigen:a", "scigen:b"]
    with pytest.raises(ValueError, match="material identity differs"):
        _index_pvbp_by_prefixed_material_id(
            table=table,
            source="scigen",
            expected_material_ids=pd.Series(["scigen:a", "scigen:c"]),
        )


def test_audit_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT260 input is missing"):
        run_pvbp_feature_audit(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in REQUIRED_STAGES},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}" for stage in REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design259",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
