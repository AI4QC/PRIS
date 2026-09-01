from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from experiments.next363_fcbe_label_blind_probe import (
    BASE_FEATURE_FILES,
    DESIGN_SHA256,
    PRIOR_FILE_NAMES,
    PRIOR_MODULES,
    evaluate_probe_gates,
    maximum_label_free_spearman,
    probe_source_hashes,
    run_label_blind_probe,
    select_probe_ids,
)


def test_novelty_population_includes_base_charge_spectrum_and_formal_rfnc() -> None:
    assert BASE_FEATURE_FILES == {
        "scigen": "features_discovery.parquet",
        "wyformer": "wyformer_x0_features_discovery.parquet",
    }
    assert tuple(PRIOR_MODULES)[-4:] == ("next323", "next347", "next355", "next359")
    assert len(PRIOR_MODULES) == 28
    assert set(PRIOR_FILE_NAMES) == set(PRIOR_MODULES)


def test_probe_selection_is_deterministic_and_unique() -> None:
    frame = pd.DataFrame(
        {
            "material_id": [f"m-{index:03d}" for index in range(24)],
            "natoms": [2 + index % 5 for index in range(24)],
            "chemical_system": ["A-B", "A-C", "B-C"] * 8,
        }
    ).sample(frac=1.0, random_state=7)
    selected = select_probe_ids(frame, count=8)
    assert selected == select_probe_ids(frame.iloc[::-1], count=8)
    assert len(selected) == len(set(selected)) == 8


def test_maximum_spearman_can_select_base_charge_spectrum_control() -> None:
    fcbe = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior = pd.DataFrame(
        {
            "base:csf_gaussian_t060": [5, 4, 3, 2, 1, 0],
            "next359:rfnc_directional_covering_floor_q10": [0, 1, 0, 1, 0, 1],
        }
    )
    result = maximum_label_free_spearman(fcbe, prior)
    assert result["feature"] == "base:csf_gaussian_t060"
    assert result["absolute_correlation"] == 1.0


def test_probe_gates_use_positive_domain_and_strict_novelty_boundary() -> None:
    sources = {
        source: {
            "supported": 80,
            "unique_rounded_10": 80,
            "minimum": 0.01,
            "maximum": 17.9,
            "maximum_invariance_error": 0.0,
            "maximum_label_free_spearman": {"absolute_correlation": 0.5},
        }
        for source in ("scigen", "wyformer")
    }
    assert all(evaluate_probe_gates(sources, count=80).values())
    sources["scigen"]["minimum"] = 0.0
    assert evaluate_probe_gates(sources, count=80)["closed_domain"] is False
    sources["scigen"]["minimum"] = 0.01
    sources["scigen"]["maximum_label_free_spearman"]["absolute_correlation"] = 0.90
    assert evaluate_probe_gates(sources, count=80)["novel"] is False


def test_probe_runner_exposes_only_geometry_and_label_free_inputs() -> None:
    assert tuple(inspect.signature(run_label_blind_probe).parameters) == (
        "scigen_cohort_dir", "wyformer_cohort_dir", "scigen_base_feature_dir",
        "wyformer_base_feature_dir", "prior_feature_dirs", "count",
    )
    assert not any(
        token in name
        for name in inspect.signature(run_label_blind_probe).parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_probe_source_hashes_cover_complete_frozen_execution() -> None:
    hashes = probe_source_hashes()
    assert tuple(hashes) == (
        "design",
        "src/next19_valence_transport.py",
        "src/next36_charge_spectrum_features.py",
        "src/next267_periodic_radical_voronoi_packing.py",
        "src/next295_positive_contact_force_closure.py",
        "src/next363_first_charge_bragg_extinction.py",
        "experiments/next363_fcbe_label_blind_probe.py",
        "tests/test_next363_first_charge_bragg_extinction.py",
        "tests/test_next363_fcbe_label_blind_probe.py",
    )
    assert hashes["design"] == DESIGN_SHA256 == (
        "8184c6866d9f1f62aa61342b7d3ce39c87051e7b34393884c490fce6fa0568e9"
    )
    assert all(len(value) == 64 for value in hashes.values())
