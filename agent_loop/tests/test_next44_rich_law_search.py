"""Contracts for joining NEXT43/NEXT44 and reusing the fixed finite search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_combine_feature_tables_requires_exact_one_to_one_identity() -> None:
    from src.next44_rich_law_search import combine_feature_tables

    base = pd.DataFrame({"material_id": ["a", "b"], "base_x": [1.0, 2.0]})
    rich = pd.DataFrame({"material_id": ["a", "b"], "rich_x": [3.0, 4.0]})
    combined = combine_feature_tables(
        base,
        rich,
        base_features=("base_x",),
        rich_features=("rich_x",),
    )
    assert combined.columns.tolist() == ["material_id", "base_x", "rich_x"]
    assert combined.rich_x.tolist() == [3.0, 4.0]
    with pytest.raises(ValueError, match="identity"):
        combine_feature_tables(
            base,
            rich.iloc[:1],
            base_features=("base_x",),
            rich_features=("rich_x",),
        )


def test_combine_feature_tables_rejects_duplicate_or_overlapping_features() -> None:
    from src.next44_rich_law_search import combine_feature_tables

    duplicate = pd.DataFrame({"material_id": ["a", "a"], "x": [1.0, 2.0]})
    clean = pd.DataFrame({"material_id": ["a", "b"], "y": [3.0, 4.0]})
    with pytest.raises(ValueError, match="identity"):
        combine_feature_tables(
            duplicate,
            clean,
            base_features=("x",),
            rich_features=("y",),
        )
    left = pd.DataFrame({"material_id": ["a", "b"], "x": [1.0, 2.0]})
    right = pd.DataFrame({"material_id": ["a", "b"], "x": [3.0, 4.0]})
    with pytest.raises(ValueError, match="overlap"):
        combine_feature_tables(
            left,
            right,
            base_features=("x",),
            rich_features=("x",),
        )


def test_rich_search_selection_still_ignores_validation_labels() -> None:
    from src.next44_rich_law_search import search_rich_candidate

    ids = np.asarray([f"id-{index}" for index in range(500)])
    split = np.asarray(["discovery"] * 300 + ["validation"] * 200)
    x = np.tile(np.linspace(0.0, 1.0, 100), 5)
    y = np.cos(np.arange(500))
    features = pd.DataFrame({"base_x": x, "rich_y": y})
    first = np.where(x > 0.72, 0.3, 0.0)
    second = first.copy()
    second[300:] = np.where(second[300:] > 0.1, 0.0, 0.3)
    result_a = search_rich_candidate(
        features=features,
        material_ids=ids,
        endpoint=first,
        split=split,
        candidate_features=("base_x", "rich_y"),
    )
    result_b = search_rich_candidate(
        features=features,
        material_ids=ids,
        endpoint=second,
        split=split,
        candidate_features=("base_x", "rich_y"),
    )
    assert result_a["selected_formula"] == result_b["selected_formula"]
    assert result_a["discovery_metrics"] == result_b["discovery_metrics"]
    assert result_a["validation_metrics"] != result_b["validation_metrics"]
