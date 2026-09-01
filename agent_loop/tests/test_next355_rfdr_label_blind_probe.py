from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from experiments.next355_rfdr_label_blind_probe import (
    DESIGN_SHA256,
    PRIOR_MODULES,
    evaluate_probe_gates,
    maximum_label_free_spearman,
    probe_source_hashes,
    run_label_blind_probe,
    select_probe_ids,
)


def test_prior_population_includes_formal_through_parc_and_inline_pdsr() -> None:
    assert tuple(PRIOR_MODULES)[-3:] == ("next319", "next323", "next347")
    assert len(PRIOR_MODULES) == 26
    assert "next351:pdsr_deviatoric_retention_floor" not in PRIOR_MODULES


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


def test_maximum_spearman_can_select_inline_pdsr_control() -> None:
    rfdr = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior = pd.DataFrame(
        {
            "next351:pdsr_deviatoric_retention_floor": [0, 1, 2, 3, 4, 5],
            "partial": [np.nan, 0.0, 1.0, 0.0, 1.0, np.nan],
        }
    )
    result = maximum_label_free_spearman(rfdr, prior)
    assert result["feature"] == "next351:pdsr_deviatoric_retention_floor"
    assert result["absolute_correlation"] == 1.0


def test_probe_gates_retain_strict_novelty_boundary() -> None:
    sources = {
        source: {
            "supported": 80,
            "unique_rounded_10": 80,
            "minimum": 0.0,
            "maximum": 1.0,
            "maximum_invariance_error": 0.0,
            "maximum_label_free_spearman": {"absolute_correlation": 0.5},
        }
        for source in ("scigen", "wyformer")
    }
    assert all(evaluate_probe_gates(sources, count=80).values())
    sources["scigen"]["maximum_label_free_spearman"]["absolute_correlation"] = 0.90
    assert evaluate_probe_gates(sources, count=80)["novel"] is False


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
        "src/next295_positive_contact_force_closure.py",
        "src/next339_periodic_geometric_homogenized_transmissivity.py",
        "src/next351_periodic_deviatoric_strain_rigidity.py",
        "src/next355_radical_facet_deviatoric_rigidity.py",
        "experiments/next355_rfdr_label_blind_probe.py",
        "tests/test_next355_radical_facet_deviatoric_rigidity.py",
        "tests/test_next355_rfdr_label_blind_probe.py",
    )
    assert hashes["design"] == DESIGN_SHA256 == (
        "cd86db09780a28eb4ddbc993837a46ab9f6852c9bcf35c6bdd719752c6d59059"
    )
    assert all(len(value) == 64 for value in hashes.values())

