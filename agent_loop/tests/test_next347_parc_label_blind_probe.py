from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from experiments.next347_parc_label_blind_probe import (
    DESIGN_SHA256,
    PRIOR_MODULES,
    evaluate_probe_gates,
    maximum_label_free_spearman,
    probe_source_hashes,
    run_label_blind_probe,
    select_probe_ids,
)


def test_prior_population_is_exactly_frozen_with_allocation_and_green_controls() -> None:
    assert tuple(PRIOR_MODULES) == (
        "next166", "next168", "next173", "next179", "next239", "next243",
        "next247", "next251", "next255", "next259", "next263", "next267",
        "next271", "next275", "next279", "next283", "next291", "next295",
        "next299", "next303", "next307", "next311", "next315", "next319",
        "next323",
    )
    assert "prv_allocation_total_variation" in PRIOR_MODULES["next267"].FEATURE_NAMES
    assert "prpa_volume_geary" in PRIOR_MODULES["next279"].FEATURE_NAMES
    assert "pcgr_charge_resistance" in PRIOR_MODULES["next315"].FEATURE_NAMES


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


def test_maximum_spearman_uses_joint_finite_nondegenerate_columns() -> None:
    parc = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior = pd.DataFrame(
        {
            "increasing": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "partial": [np.nan, 0.0, 1.0, 0.0, 1.0, np.nan],
            "constant": [1.0] * 6,
        }
    )
    result = maximum_label_free_spearman(parc, prior)
    assert result["feature"] == "increasing"
    assert result["correlation"] == pytest.approx(1.0)
    assert result["joint_finite"] == 6


def test_probe_gates_include_strict_domain_support_invariance_and_novelty() -> None:
    sources = {
        source: {
            "supported": 80,
            "unique_rounded_10": 80,
            "minimum": 0.1,
            "maximum": 0.9,
            "maximum_invariance_error": 0.0,
            "maximum_label_free_spearman": {"absolute_correlation": 0.5},
        }
        for source in ("scigen", "wyformer")
    }
    assert all(evaluate_probe_gates(sources, count=80).values())
    sources["wyformer"]["maximum_invariance_error"] = 2.0e-8
    assert evaluate_probe_gates(sources, count=80)["invariant"] is False


def test_probe_runner_exposes_no_outcome_interface() -> None:
    assert tuple(inspect.signature(run_label_blind_probe).parameters) == (
        "scigen_cohort_dir", "wyformer_cohort_dir", "prior_feature_dirs", "count",
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
        "src/next267_periodic_radical_voronoi_packing.py",
        "src/next331_radical_facet_minimum_participation.py",
        "src/next339_periodic_geometric_homogenized_transmissivity.py",
        "src/next347_periodic_allocation_redistribution_capacity.py",
        "experiments/next347_parc_label_blind_probe.py",
        "tests/test_next347_periodic_allocation_redistribution_capacity.py",
        "tests/test_next347_parc_label_blind_probe.py",
    )
    assert hashes["design"] == DESIGN_SHA256 == (
        "b6815af26a012ac27b04341dab03b26c0b0c78f5bb0a3b10feec7d2fc8d1e5e2"
    )
    assert all(len(value) == 64 for value in hashes.values())
