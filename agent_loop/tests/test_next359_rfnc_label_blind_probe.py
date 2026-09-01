from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from experiments.next359_rfnc_label_blind_probe import (
    DESIGN_SHA256,
    PRIOR_FILE_NAMES,
    PRIOR_MODULES,
    evaluate_probe_gates,
    maximum_label_free_spearman,
    probe_source_hashes,
    run_label_blind_probe,
    select_probe_ids,
)


def test_prior_population_includes_every_formal_stage_through_rfdr() -> None:
    assert tuple(PRIOR_MODULES)[-4:] == ("next319", "next323", "next347", "next355")
    assert tuple(PRIOR_MODULES)[-1] == "next355"
    assert len(PRIOR_MODULES) == 27
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


def test_maximum_spearman_can_select_formal_rfdr_control() -> None:
    rfnc = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior = pd.DataFrame(
        {
            "next355:rfdr_deviatoric_retention_floor": [0, 1, 2, 3, 4, 5],
            "partial": [np.nan, 0.0, 1.0, 0.0, 1.0, np.nan],
        }
    )
    result = maximum_label_free_spearman(rfnc, prior)
    assert result["feature"] == "next355:rfdr_deviatoric_retention_floor"
    assert result["absolute_correlation"] == 1.0


def test_probe_gates_retain_strict_novelty_boundary() -> None:
    sources = {
        source: {
            "supported": 80,
            "unique_rounded_10": 80,
            "minimum": 0.01,
            "maximum": 0.99,
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
        "src/next327_radical_facet_positive_enclosure.py",
        "src/next339_periodic_geometric_homogenized_transmissivity.py",
        "src/next359_radical_facet_normal_covering.py",
        "experiments/next359_rfnc_label_blind_probe.py",
        "tests/test_next359_radical_facet_normal_covering.py",
        "tests/test_next359_rfnc_label_blind_probe.py",
    )
    assert hashes["design"] == DESIGN_SHA256 == (
        "b7a278ccaacc81800938edb21a10096aaa70aac3af3aeb3c336f53e940e52dac"
    )
    assert all(len(value) == 64 for value in hashes.values())
