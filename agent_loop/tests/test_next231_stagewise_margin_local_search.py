from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next231_stagewise_margin_local_search import (
    AMPLITUDE_FRACTIONS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    LOCAL_WIDTH_FRACTIONS,
    build_stagewise_candidate_specs,
    materialize_stagewise_candidates,
    run_stagewise_margin_local_search,
    stagewise_margin_local_score,
)


def test_stagewise_score_is_signed_triangular_with_wide_fraction_support() -> None:
    score, support, active, weight, protection = stagewise_margin_local_score(
        base_score=np.array([0.5, 0.5, 0.9, 0.1]),
        base_support=np.ones(4, dtype=bool),
        feature_values=np.array([1.0, 0.0, 1.0, 0.0]),
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        threshold=0.5,
        repair_width=0.8,
        local_width_fraction=1.0,
        amplitude_fraction=0.5,
    )
    np.testing.assert_array_equal(support, np.ones(4, dtype=bool))
    np.testing.assert_array_equal(active, np.ones(4, dtype=bool))
    np.testing.assert_allclose(weight, [1.0, 1.0, 0.5, 0.5])
    np.testing.assert_allclose(protection, [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(score, [0.1, 0.9, 0.7, 0.3])


def test_stagewise_score_missing_off_floor_and_edge_zero() -> None:
    score, _, active, weight, _ = stagewise_margin_local_score(
        base_score=np.array([0.1, 0.1, 0.3]),
        base_support=np.ones(3, dtype=bool),
        feature_values=np.array([1.0, np.nan, 1.0]),
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        threshold=0.1,
        repair_width=0.8,
        local_width_fraction=0.25,
        amplitude_fraction=1.0,
    )
    np.testing.assert_array_equal(active, [True, False, False])
    np.testing.assert_allclose(score, [0.0, 0.1, 0.3])
    np.testing.assert_allclose(weight, [1.0, 0.0, 0.0])


def test_specs_cover_complete_second_stage_grid_with_global_normalization() -> None:
    features = pd.DataFrame(
        {
            "a": np.arange(32, dtype=float),
            "b": np.arange(32, dtype=float)[::-1],
        }
    )
    specs = build_stagewise_candidate_specs(
        base_candidate_key="next229-frontier",
        eligible_hypotheses=("a__protected_low", "b__protected_high"),
        features=features,
        base_score=np.linspace(0.0, 1.0, 32),
        base_support=np.ones(32, dtype=bool),
        threshold=0.5,
        repair_width=0.8,
        local_width_fractions=(0.5, 1.0),
        amplitude_fractions=(0.25, 0.5, 1.0),
    )
    assert len(specs) == 13
    assert specs[0]["eligible_new_candidate"] is False
    assert all(spec["eligible_new_candidate"] for spec in specs[1:])
    assert len({str(spec["candidate_key"]) for spec in specs}) == 13
    assert specs[1]["q_lo"] == 1.0 and specs[1]["q_hi"] == 29.0
    assert all(
        spec["normalization_population"] == "ALL_FINITE_COMBINED_DISCOVERY"
        for spec in specs
    )
    assert LOCAL_WIDTH_FRACTIONS == (
        1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0
    )
    assert AMPLITUDE_FRACTIONS == (1 / 4, 1 / 2, 1.0)
    assert EXPECTED_CANDIDATE_COUNT == 631
    assert EXPECTED_ELIGIBLE_COUNT == 630


def test_materializer_encodes_exact_score_and_activity() -> None:
    features = pd.DataFrame(
        {"source_dataset": ["scigen", "wyformer"], "a": [1.0, 0.0]}
    )
    specs = build_stagewise_candidate_specs(
        base_candidate_key="base",
        eligible_hypotheses=("a__protected_high",),
        features=features,
        base_score=np.array([0.5, 0.5]),
        base_support=np.ones(2, dtype=bool),
        threshold=0.5,
        repair_width=0.8,
        local_width_fractions=(0.25,),
        amplitude_fractions=(0.5,),
    )
    virtual, terms, runtime, activity = materialize_stagewise_candidates(
        features=features,
        base_score=np.array([0.5, 0.5]),
        base_support=np.ones(2, dtype=bool),
        specs=specs,
    )
    assert len(terms) == len(runtime) == 2
    corrected = np.arcsinh(virtual[terms[1]["feature"]].to_numpy()) / terms[1]["scale"]
    np.testing.assert_allclose(corrected, [0.4, 0.6])
    assert activity[str(specs[1]["candidate_key"])] == {
        "rows": 2,
        "scigen": 1,
        "wyformer": 1,
    }


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_stagewise_margin_local_search).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT231 input is missing"):
        run_stagewise_margin_local_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 231)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 231)},
            design_path=tmp_path / "design231",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
