from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from experiments.next323_pgce_label_blind_probe import (
    maximum_label_free_spearman,
    run_label_blind_probe,
    select_probe_ids,
)


def test_probe_selection_is_deterministic_stratified_and_unique() -> None:
    frame = pd.DataFrame(
        {
            "material_id": [f"m-{index:03d}" for index in range(24)],
            "natoms": [2 + index % 5 for index in range(24)],
            "chemical_system": ["A-B", "A-C", "B-C"] * 8,
        }
    ).sample(frac=1.0, random_state=7)
    selected = select_probe_ids(frame, count=8)
    repeated = select_probe_ids(frame.iloc[::-1], count=8)
    assert selected == repeated
    assert len(selected) == len(set(selected)) == 8
    assert set(selected).issubset(set(frame["material_id"]))


def test_probe_selection_refuses_duplicate_or_insufficient_identity() -> None:
    frame = pd.DataFrame(
        {
            "material_id": ["a", "a"],
            "natoms": [2, 3],
            "chemical_system": ["A-B", "A-C"],
        }
    )
    with pytest.raises(ValueError, match="identity"):
        select_probe_ids(frame, count=1)
    with pytest.raises(ValueError, match="count"):
        select_probe_ids(frame.drop_duplicates("material_id"), count=2)


def test_maximum_spearman_uses_only_joint_finite_nondegenerate_columns() -> None:
    pgce = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prior = pd.DataFrame(
        {
            "increasing": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "partial": [np.nan, 0.0, 1.0, 0.0, 1.0, np.nan],
            "constant": [1.0] * 6,
            "text": ["x"] * 6,
        }
    )
    result = maximum_label_free_spearman(pgce, prior)
    assert result["feature"] == "increasing"
    assert result["correlation"] == pytest.approx(1.0)
    assert result["joint_finite"] == 6


def test_probe_helpers_expose_no_outcome_interface() -> None:
    names = tuple(inspect.signature(select_probe_ids).parameters)
    assert names == ("frame", "count")
    assert not any(
        token in name
        for name in names
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )
    runner = tuple(inspect.signature(run_label_blind_probe).parameters)
    assert runner == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "prior_feature_dirs",
        "count",
    )
    assert not any(
        token in name
        for name in runner
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )
