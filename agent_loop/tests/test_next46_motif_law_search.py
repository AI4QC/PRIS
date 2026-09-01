"""Contracts for additive NEXT46 motif formula search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_combine_three_feature_tables_is_exact_and_nonoverlapping() -> None:
    from src.next46_motif_law_search import combine_three_feature_tables

    base = pd.DataFrame({"material_id": ["a", "b"], "x": [1.0, 2.0]})
    rich = pd.DataFrame({"material_id": ["a", "b"], "y": [3.0, 4.0]})
    motif = pd.DataFrame({"material_id": ["a", "b"], "z": [5.0, 6.0]})
    combined = combine_three_feature_tables(
        base,
        rich,
        motif,
        base_features=("x",),
        rich_features=("y",),
        motif_features=("z",),
    )
    assert combined.columns.tolist() == ["material_id", "x", "y", "z"]
    with pytest.raises(ValueError, match="overlap"):
        combine_three_feature_tables(
            base,
            rich,
            motif.rename(columns={"z": "x"}),
            base_features=("x",),
            rich_features=("y",),
            motif_features=("x",),
        )


def test_motif_search_selection_keeps_validation_labels_sealed() -> None:
    from src.next46_motif_law_search import search_motif_candidate

    rng = np.random.default_rng(46)
    n = 600
    x = rng.random(n)
    motif = rng.random(n)
    features = pd.DataFrame({"x": x, "motif": motif})
    split = np.asarray(["discovery"] * 360 + ["validation"] * 240)
    first_endpoint = np.where(x + motif > 1.25, 0.3, 0.0)
    second_endpoint = first_endpoint.copy()
    second_endpoint[360:] = np.where(second_endpoint[360:] > 0.1, 0.0, 0.3)
    kwargs = {
        "features": features,
        "material_ids": [f"id-{index}" for index in range(n)],
        "split": split,
        "candidate_features": ("x", "motif"),
    }
    first = search_motif_candidate(endpoint=first_endpoint, **kwargs)
    second = search_motif_candidate(endpoint=second_endpoint, **kwargs)
    assert first["selected_formula"] == second["selected_formula"]
    assert first["discovery_metrics"] == second["discovery_metrics"]
    assert first["validation_metrics"] != second["validation_metrics"]
