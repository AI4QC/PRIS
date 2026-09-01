from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


def _eligible_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hypothesis": [
                "cclab_cde_conservative_domain_extension__protected_high"
            ],
            "feature": ["cclab_cde_conservative_domain_extension"],
            "direction": ["protected_high"],
            "q_lo": [0.0731047591],
            "q_hi": [0.8827890278],
        }
    )


def test_frozen_grid_has_one_control_and_twenty_one_new_candidates() -> None:
    from src.next498_cclab_cde_margin_local_search import (
        AMPLITUDE_FRACTIONS,
        EXPECTED_CANDIDATE_COUNT,
        EXPECTED_ELIGIBLE_COUNT,
        LOCAL_WIDTH_FRACTIONS,
        build_cclab_cde_candidate_specs,
    )

    specs = build_cclab_cde_candidate_specs(
        base_candidate_key="base", eligible_table=_eligible_table()
    )
    assert LOCAL_WIDTH_FRACTIONS == (
        1 / 64,
        1 / 32,
        1 / 16,
        1 / 8,
        1 / 4,
        1 / 2,
        1.0,
    )
    assert AMPLITUDE_FRACTIONS == (1 / 4, 1 / 2, 1.0)
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 22
    assert EXPECTED_ELIGIBLE_COUNT == 21
    assert sum(bool(spec["eligible_new_candidate"]) for spec in specs) == 21
    assert specs[0]["is_reproduction_control"] is True


def test_score_is_signed_local_nonnegative_and_support_preserving() -> None:
    from src.next498_cclab_cde_margin_local_search import (
        cclab_cde_margin_local_score,
    )

    base = np.array([0.15, 0.15, 0.5, 0.01])
    support = np.array([True, True, True, False])
    protection = np.array([1.0, 0.0, 0.5, 0.0])
    score, got_support, active, weight = cclab_cde_margin_local_score(
        base_score=base,
        base_support=support,
        protection=protection,
        threshold=0.15,
        repair_width=0.08,
        local_width_fraction=1 / 2,
        amplitude_fraction=1 / 2,
    )
    assert np.array_equal(got_support, support)
    assert active.tolist() == [True, True, False, False]
    assert weight.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert 0.0 <= score[0] < base[0] < score[1]
    assert score[2] == base[2]


def test_reporting_selection_excludes_reproduction_control() -> None:
    from src.next498_cclab_cde_margin_local_search import select_best_new_record

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


def test_interface_and_formula_boundary_exclude_validation_replication_and_dft() -> None:
    from src.next498_cclab_cde_margin_local_search import (
        _formula_from_spec,
        run_cclab_cde_margin_local_search,
    )

    parameters = tuple(inspect.signature(run_cclab_cde_margin_local_search).parameters)
    assert "next497_dir" in parameters and "next496_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name
        for name in parameters
        for token in ("validation", "replication")
    )
    formula = _formula_from_spec(None)
    assert formula["selected"] is False
    assert formula["dft_values_used_by_executable_formula"] is False
    assert formula["learned_energy_force_stress_proxy_used"] is False
    assert formula["model_or_proxy_potential_used"] is False
    assert formula["physical_relaxation_executed"] is False


def test_search_fails_closed_on_missing_input(tmp_path) -> None:
    from src.next498_cclab_cde_margin_local_search import (
        REQUIRED_DESIGN_STAGES,
        REQUIRED_STAGES,
        run_cclab_cde_margin_local_search,
    )

    with pytest.raises(FileNotFoundError, match="NEXT498 input is missing"):
        run_cclab_cde_margin_local_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in REQUIRED_STAGES},
            next135_freeze_path=tmp_path / "next135",
            design_paths={
                stage: tmp_path / f"design{stage}"
                for stage in REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design",
            next412_dir=tmp_path / "next412",
            next496_dir=tmp_path / "next496",
            next497_dir=tmp_path / "next497",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
