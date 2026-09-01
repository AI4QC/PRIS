from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next285_power_cell_shape_volume_margin_local_search import (
    AMPLITUDE_FRACTIONS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    LOCAL_WIDTH_FRACTIONS,
    build_psvc_candidate_specs,
    psvc_margin_local_score,
    run_power_cell_shape_volume_margin_local_search,
    select_best_new_record,
)


def _eligible_table() -> pd.DataFrame:
    features = [
        "psvc_inflated_asphericity_mean",
        "psvc_inflated_asphericity_q90",
        "psvc_small_inflated_asphericity_mean",
    ]
    return pd.DataFrame(
        {
            "hypothesis": [f"{name}__protected_low" for name in features],
            "feature": features,
            "direction": ["protected_low"] * 3,
            "q_lo": [0.00018635855, 0.00042500845, 0.0],
            "q_hi": [0.06414149265, 0.1862978152, 0.12410135085],
        }
    )


def test_frozen_grid_builds_one_control_and_sixty_three_new_candidates() -> None:
    specs = build_psvc_candidate_specs(
        base_candidate_key="base", eligible_table=_eligible_table()
    )
    assert LOCAL_WIDTH_FRACTIONS == (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
    assert AMPLITUDE_FRACTIONS == (1 / 4, 1 / 2, 1.0)
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 64
    assert EXPECTED_ELIGIBLE_COUNT == 63
    assert sum(bool(spec["eligible_new_candidate"]) for spec in specs) == 63
    assert specs[0]["is_reproduction_control"] is True


def test_margin_local_score_is_signed_local_and_support_preserving() -> None:
    base = np.array([0.15, 0.15, 0.5])
    support = np.array([True, True, True])
    protection = np.array([1.0, 0.0, 0.5])
    score, got_support, active, weight = psvc_margin_local_score(
        base_score=base,
        base_support=support,
        protection=protection,
        threshold=0.15,
        repair_width=0.08,
        local_width_fraction=1 / 2,
        amplitude_fraction=1 / 2,
    )
    assert np.array_equal(got_support, support)
    assert active.tolist() == [True, True, False]
    assert weight.tolist() == [1.0, 1.0, 0.0]
    assert score[0] < base[0] < score[1]
    assert score[2] == base[2]


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
    parameters = tuple(
        inspect.signature(run_power_cell_shape_volume_margin_local_search).parameters
    )
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert "amendment_path" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_search_fails_closed_on_missing_input(tmp_path) -> None:
    from src.next285_power_cell_shape_volume_margin_local_search import (
        REQUIRED_DESIGN_STAGES,
        REQUIRED_STAGES,
    )

    with pytest.raises(FileNotFoundError, match="NEXT285 input is missing"):
        run_power_cell_shape_volume_margin_local_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in REQUIRED_STAGES},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}" for stage in REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design283",
            amendment_path=tmp_path / "amendment283",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
