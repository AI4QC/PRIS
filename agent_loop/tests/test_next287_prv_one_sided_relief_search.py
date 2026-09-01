from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next287_prv_one_sided_relief_search import (
    AMPLITUDE_FRACTIONS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    LOCAL_WIDTH_FRACTIONS,
    SCORE_COMPOSITION,
    build_prv_one_sided_candidate_specs,
    materialize_prv_one_sided_candidates,
    prv_one_sided_relief_score,
    run_prv_one_sided_relief_search,
    select_best_new_record,
)


def _eligible_table() -> pd.DataFrame:
    features = ["prv_chebyshev_ratio_cv", "prv_volume_ratio_cv"]
    return pd.DataFrame(
        {
            "hypothesis": [f"{name}__protected_low" for name in features],
            "feature": features,
            "direction": ["protected_low"] * 2,
            "q_lo": [0.011985598809152042, 0.01182477813930403],
            "q_hi": [0.28180046821941024, 0.5950219074124739],
        }
    )


def test_one_sided_relief_is_exact_local_nonincreasing_and_support_preserving() -> None:
    base = np.array([0.15, 0.15, 0.15, 0.17, 0.19, 0.11, 0.15, 0.15])
    support = np.array([True, True, True, True, True, True, True, False])
    protection = np.array([1.0, 0.5, 0.0, 1.0, 1.0, 1.0, np.nan, 1.0])
    score, got_support, active, weight = prv_one_sided_relief_score(
        base_score=base,
        base_support=support,
        protection=protection,
        threshold=0.15,
        repair_width=0.08,
        local_width_fraction=1 / 2,
        amplitude_fraction=1 / 2,
    )
    assert np.array_equal(got_support, support)
    assert active.tolist() == [True, True, True, True, False, False, False, False]
    assert np.allclose(weight, [1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(score[:7], [0.13, 0.14, 0.15, 0.16, 0.19, 0.11, 0.15])
    assert np.isnan(score[7])
    assert np.all(score[support] >= 0.0)
    assert np.all(score[support] <= base[support])


def test_one_sided_relief_rejects_out_of_grid_or_malformed_inputs() -> None:
    kwargs = {
        "base_score": [0.15],
        "base_support": [True],
        "protection": [0.5],
        "threshold": 0.15,
        "repair_width": 0.08,
        "local_width_fraction": 1 / 2,
        "amplitude_fraction": 1 / 2,
    }
    with pytest.raises(ValueError, match="NEXT287 one-sided relief inputs differ"):
        prv_one_sided_relief_score(**{**kwargs, "protection": [1.01]})
    with pytest.raises(ValueError, match="NEXT287 one-sided relief inputs differ"):
        prv_one_sided_relief_score(**{**kwargs, "local_width_fraction": 0.3})
    with pytest.raises(ValueError, match="NEXT287 one-sided relief inputs differ"):
        prv_one_sided_relief_score(**{**kwargs, "amplitude_fraction": 0.3})


def test_frozen_grid_builds_one_control_and_forty_two_new_candidates() -> None:
    specs = build_prv_one_sided_candidate_specs(
        base_candidate_key="base", eligible_table=_eligible_table()
    )
    assert LOCAL_WIDTH_FRACTIONS == (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
    assert AMPLITUDE_FRACTIONS == (1 / 4, 1 / 2, 1.0)
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 43
    assert EXPECTED_ELIGIBLE_COUNT == 42
    assert sum(bool(spec["eligible_new_candidate"]) for spec in specs) == 42
    assert specs[0]["is_reproduction_control"] is True
    assert len({str(spec["candidate_key"]) for spec in specs}) == 43
    assert {str(spec["score_composition"]) for spec in specs} == {SCORE_COMPOSITION}


def test_materializer_encodes_the_exact_physical_one_sided_score() -> None:
    features = pd.DataFrame(
        {
            "source_dataset": ["scigen", "wyformer"],
            "prv_chebyshev_ratio_cv": [0.0, 1.0],
            "prv_volume_ratio_cv": [0.0, 1.0],
        }
    )
    specs = build_prv_one_sided_candidate_specs(
        base_candidate_key="base",
        eligible_table=_eligible_table(),
        local_width_fractions=(1 / 2,),
        amplitude_fractions=(1 / 2,),
    )
    virtual, terms, runtime, activity = materialize_prv_one_sided_candidates(
        features=features,
        base_score=np.array([0.15, 0.15]),
        base_support=np.array([True, True]),
        specs=specs,
    )
    assert len(terms) == len(runtime) == len(activity) == 3
    for term in terms:
        decoded = np.arcsinh(virtual[str(term["feature"])].to_numpy(float))
        assert np.all(decoded >= 0.0)
        assert np.all(decoded <= 0.15 + 1.0e-15)


def test_reporting_selection_excludes_the_reproduction_control() -> None:
    records = pd.DataFrame(
        {
            "candidate_key": ["control", "new"],
            "eligible_new_candidate": [False, True],
            "passes_source_auc_gates": [True, True],
            "passes_safe_all_cells": [True, True],
            "passes_all_discovery_gates": [True, False],
            "passes_broad_all_cells": [True, False],
            "safe_passing_cells": [12, 11],
            "safe_worst_cell_severe_recall": [1.0, 0.9],
            "safe_worst_cell_precision_lower": [1.0, 0.9],
            "scigen_pooled_auc": [1.0, 0.6],
            "wyformer_pooled_auc": [1.0, 0.6],
            "term_count": [1, 1],
        }
    )
    selected = select_best_new_record(records)
    assert selected is not None and selected["candidate_key"] == "new"


def test_search_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(inspect.signature(run_prv_one_sided_relief_search).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_search_fails_closed_on_missing_input(tmp_path) -> None:
    from src.next287_prv_one_sided_relief_search import (
        REQUIRED_DESIGN_STAGES,
        REQUIRED_STAGES,
    )

    with pytest.raises(FileNotFoundError, match="NEXT287 input is missing"):
        run_prv_one_sided_relief_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in REQUIRED_STAGES},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}" for stage in REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design287",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
