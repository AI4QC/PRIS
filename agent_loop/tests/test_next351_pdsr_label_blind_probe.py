from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from experiments.next351_pdsr_label_blind_probe import (
    DESIGN_SHA256,
    PRIOR_MODULES,
    evaluate_probe_gates,
    maximum_label_free_spearman,
    probe_source_hashes,
    run_label_blind_probe,
    select_probe_ids,
)


def test_prior_population_extends_frozen_parc_population_without_audit() -> None:
    assert tuple(PRIOR_MODULES)[-3:] == ("next319", "next323", "next347")
    assert len(PRIOR_MODULES) == 26
    assert "parc_allocation_redistribution_protection" in PRIOR_MODULES["next347"].FEATURE_NAMES
    assert not any(stage in PRIOR_MODULES for stage in ("next348", "next349", "next350"))


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
    pdsr = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior = pd.DataFrame(
        {
            "increasing": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "partial": [np.nan, 0.0, 1.0, 0.0, 1.0, np.nan],
            "constant": [1.0] * 6,
        }
    )
    result = maximum_label_free_spearman(pdsr, prior)
    assert result["feature"] == "increasing"
    assert result["correlation"] == 1.0
    assert result["joint_finite"] == 6


def test_probe_gates_include_closed_domain_support_invariance_and_novelty() -> None:
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
    sources["wyformer"]["maximum_label_free_spearman"]["absolute_correlation"] = 0.90
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
        "src/next19_valence_transport.py",
        "src/next267_periodic_radical_voronoi_packing.py",
        "src/next295_positive_contact_force_closure.py",
        "src/next351_periodic_deviatoric_strain_rigidity.py",
        "experiments/next351_pdsr_label_blind_probe.py",
        "tests/test_next351_periodic_deviatoric_strain_rigidity.py",
        "tests/test_next351_pdsr_label_blind_probe.py",
    )
    assert hashes["design"] == DESIGN_SHA256 == (
        "53f2aa97144e35ed4b62dbf506dbb8610cd98fb304b8c27683e549241fe6a408"
    )
    assert all(len(value) == 64 for value in hashes.values())

