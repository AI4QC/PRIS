from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next237_chemistry_conditioned_margin_local_search as n237
from src.next237_chemistry_conditioned_margin_local_search import (
    AMPLITUDE_FRACTIONS,
    LOCAL_WIDTH_FRACTIONS,
    build_conditioned_candidate_specs,
    conditioned_margin_local_score,
    materialize_conditioned_candidates,
    run_chemistry_conditioned_margin_local_search,
    select_best_new_record,
)


def test_conditioned_score_uses_bounded_certificate_and_triangular_margin() -> None:
    score, support, active, weight = conditioned_margin_local_score(
        base_score=np.array([0.5, 0.5, 0.9, 0.1]),
        base_support=np.ones(4, dtype=bool),
        protection=np.array([1.0, 0.0, 1.0, 0.0]),
        threshold=0.5,
        repair_width=0.8,
        local_width_fraction=1.0,
        amplitude_fraction=0.5,
    )
    np.testing.assert_array_equal(support, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(active, np.ones(4, dtype=bool))
    np.testing.assert_allclose(weight, [1.0, 1.0, 0.5, 0.5])
    np.testing.assert_allclose(score, [0.1, 0.9, 0.7, 0.3])


def test_specs_cover_complete_conditioned_grid() -> None:
    eligible = pd.DataFrame(
        {
            "hypothesis": ["a__c__protected_high"],
            "feature": ["a"],
            "conditioner": ["c"],
            "direction": ["protected_high"],
            "stratum_edges_json": ["[1,2,3]"],
            "q_lo_by_stratum_json": ["[0,0,0,0]"],
            "q_hi_by_stratum_json": ["[1,1,1,1]"],
        }
    )
    specs = build_conditioned_candidate_specs(
        base_candidate_key="base",
        eligible_table=eligible,
        local_width_fractions=(0.5, 1.0),
        amplitude_fractions=(0.25, 0.5, 1.0),
    )
    assert len(specs) == 7
    assert specs[0]["is_reproduction_control"] is True
    assert all(spec["eligible_new_candidate"] for spec in specs[1:])
    assert all(
        spec["normalization_population"]
        == "COMPOSITION_CONDITIONED_COMBINED_DISCOVERY"
        for spec in specs
    )
    assert LOCAL_WIDTH_FRACTIONS == (
        1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0
    )
    assert AMPLITUDE_FRACTIONS == (1 / 4, 1 / 2, 1.0)


def test_materializer_preserves_base_support_and_activity() -> None:
    features = pd.DataFrame(
        {"source_dataset": ["scigen", "wyformer"], "a": [1.0, 0.0], "c": [0.0, 0.0]}
    )
    eligible = pd.DataFrame(
        {
            "hypothesis": ["a__c__protected_high"],
            "feature": ["a"],
            "conditioner": ["c"],
            "direction": ["protected_high"],
            "stratum_edges_json": ["[1,2,3]"],
            "q_lo_by_stratum_json": ["[0,0,0,0]"],
            "q_hi_by_stratum_json": ["[1,1,1,1]"],
        }
    )
    specs = build_conditioned_candidate_specs(
        base_candidate_key="base",
        eligible_table=eligible,
        threshold=0.5,
        repair_width=0.8,
        local_width_fractions=(0.25,),
        amplitude_fractions=(0.5,),
    )
    virtual, terms, runtime, activity = materialize_conditioned_candidates(
        features=features,
        base_score=np.array([0.5, 0.5]),
        base_support=np.ones(2, dtype=bool),
        specs=specs,
    )
    corrected = np.arcsinh(virtual[terms[1]["feature"]].to_numpy()) / terms[1]["scale"]
    np.testing.assert_allclose(corrected, [0.4, 0.6])
    assert len(runtime) == 2
    assert activity[str(specs[1]["candidate_key"])] == {
        "rows": 2,
        "scigen": 1,
        "wyformer": 1,
    }


def test_formal_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(
        inspect.signature(run_chemistry_conditioned_margin_local_search).parameters
    )
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_reporting_selection_excludes_reproduction_control(monkeypatch) -> None:
    records = pd.DataFrame(
        {
            "candidate_key": ["control", "new"],
            "eligible_new_candidate": [False, True],
            "passes_source_auc_gates": [True, True],
            "passes_safe_all_cells": [True, True],
        }
    )
    seen: list[pd.DataFrame] = []

    def fake_select(frame: pd.DataFrame) -> pd.Series:
        seen.append(frame.copy())
        return frame.iloc[0]

    monkeypatch.setattr(n237.n223, "select_best_eligible_record", fake_select)
    selected = select_best_new_record(records)
    assert selected is not None and selected["candidate_key"] == "new"
    assert len(seen) == 1
    assert seen[0]["candidate_key"].tolist() == ["new"]


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT237 input is missing"):
        run_chemistry_conditioned_margin_local_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 237)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 237)},
            design_path=tmp_path / "design237",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
