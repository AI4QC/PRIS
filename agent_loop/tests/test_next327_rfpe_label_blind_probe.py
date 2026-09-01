from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from experiments.next327_rfpe_label_blind_probe import (
    maximum_label_free_spearman,
    probe_source_hashes,
    run_label_blind_probe,
    select_probe_ids,
)


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
    rfpe = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior = pd.DataFrame(
        {
            "increasing": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "partial": [np.nan, 0.0, 1.0, 0.0, 1.0, np.nan],
            "constant": [1.0] * 6,
            "text": ["x"] * 6,
        }
    )
    result = maximum_label_free_spearman(rfpe, prior)
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
    assert not any(
        token in name
        for name in inspect.signature(run_label_blind_probe).parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_probe_source_hashes_cover_the_frozen_executed_artifacts() -> None:
    hashes = probe_source_hashes()
    assert tuple(hashes) == (
        "design",
        "src/next327_radical_facet_positive_enclosure.py",
        "experiments/next327_rfpe_label_blind_probe.py",
        "tests/test_next327_radical_facet_positive_enclosure.py",
        "tests/test_next327_rfpe_label_blind_probe.py",
    )
    assert hashes["design"] == (
        "5211e4f853c74497bc5a31a7a86ce3adc01ea008b7833b7557dcbbf7885cbfa9"
    )
    assert all(len(value) == 64 for value in hashes.values())
