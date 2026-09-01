from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from experiments.next331_rfmp_label_blind_probe import (
    DESIGN_SHA256,
    PRIOR_MODULES,
    evaluate_probe_gates,
    maximum_label_free_spearman,
    probe_source_hashes,
    run_label_blind_probe,
    select_probe_ids,
)


def test_prior_population_is_exactly_frozen_and_includes_closest_area_controls() -> None:
    assert tuple(PRIOR_MODULES) == (
        "next239",
        "next243",
        "next247",
        "next251",
        "next255",
        "next259",
        "next263",
        "next267",
        "next271",
        "next275",
        "next279",
        "next283",
        "next291",
        "next295",
        "next299",
        "next303",
        "next307",
        "next311",
        "next315",
        "next319",
        "next323",
    )
    assert "mvbo_facet_evenness_q10" in PRIOR_MODULES["next239"].FEATURE_NAMES
    assert "psvc_sphericity_q10" in PRIOR_MODULES["next283"].FEATURE_NAMES
    assert "pgce_all_facet_participation_floor" in PRIOR_MODULES["next323"].FEATURE_NAMES


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


def test_probe_selection_refuses_duplicate_or_insufficient_identity() -> None:
    frame = pd.DataFrame(
        {"material_id": ["a", "a"], "natoms": [2, 3], "chemical_system": ["A-B", "A-C"]}
    )
    with pytest.raises(ValueError, match="identity"):
        select_probe_ids(frame, count=1)
    with pytest.raises(ValueError, match="count"):
        select_probe_ids(frame.drop_duplicates("material_id"), count=2)


def test_maximum_spearman_uses_joint_finite_nondegenerate_columns() -> None:
    rfmp = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior = pd.DataFrame(
        {
            "increasing": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "partial": [np.nan, 0.0, 1.0, 0.0, 1.0, np.nan],
            "constant": [1.0] * 6,
            "text": ["x"] * 6,
        }
    )
    result = maximum_label_free_spearman(rfmp, prior)
    assert result["feature"] == "increasing"
    assert result["correlation"] == pytest.approx(1.0)
    assert result["joint_finite"] == 6


def test_probe_runner_exposes_no_outcome_interface() -> None:
    assert tuple(inspect.signature(run_label_blind_probe).parameters) == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "prior_feature_dirs",
        "count",
    )


def test_probe_gates_fail_closed_on_a_quantized_zero_value() -> None:
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
    sources["wyformer"]["minimum"] = 0.0
    gates = evaluate_probe_gates(sources, count=80)
    assert gates["strict_domain"] is False
    assert gates["support"] is True
    assert not any(
        token in name
        for name in inspect.signature(run_label_blind_probe).parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_probe_source_hashes_cover_the_frozen_executed_artifacts() -> None:
    hashes = probe_source_hashes()
    assert tuple(hashes) == (
        "design",
        "src/next331_radical_facet_minimum_participation.py",
        "experiments/next331_rfmp_label_blind_probe.py",
        "tests/test_next331_radical_facet_minimum_participation.py",
        "tests/test_next331_rfmp_label_blind_probe.py",
    )
    assert hashes["design"] == DESIGN_SHA256 == (
        "1fce65f3cd7d0394329abadd748d49c2d9bfcd657b29ceb9e5b4d912b2322b54"
    )
    assert all(len(value) == 64 for value in hashes.values())
