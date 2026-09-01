"""Contracts for exhaustive two-condition analytic guard search."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_bitset_metrics_match_reference_array_metrics() -> None:
    from src.next23_evaluate import _decision_metrics
    from src.next45_exhaustive_rectangles import bitset_decision_metrics

    endpoint = np.asarray([0.0, 0.05, 0.2, 0.6, 0.3, 0.0])
    supported = np.asarray([True, True, True, True, False, True])
    reject = np.asarray([False, True, True, True, True, False]) & supported
    expected = _decision_metrics(supported=supported, reject=reject, endpoint=endpoint)
    actual = bitset_decision_metrics(
        supported=supported, reject=reject, endpoint=endpoint
    )
    assert actual == expected


def test_exhaustive_rectangles_can_recover_a_conditional_tail() -> None:
    from src.next45_exhaustive_rectangles import search_exhaustive_rectangles

    rng = np.random.default_rng(45)
    n = 800
    x = rng.random(n)
    y = rng.random(n)
    endpoint = np.where((x >= 0.6) & (y <= 0.4), 0.3, 0.0)
    split = np.asarray(["discovery"] * 500 + ["validation"] * 300)
    result = search_exhaustive_rectangles(
        features=pd.DataFrame({"x": x, "y": y}),
        material_ids=[f"id-{index}" for index in range(n)],
        endpoint=endpoint,
        split=split,
        candidate_features=("x", "y"),
        tail_fractions=(0.4, 0.5, 0.6),
    )
    formula = result["selected_formula"]
    assert formula["kind"] == "conjunctive"
    assert {term["feature"] for term in formula["terms"]} == {"x", "y"}
    assert result["discovery_metrics"]["rejection_precision"] > 0.85
    assert result["validation_metrics"]["rejection_precision"] > 0.85


def test_rectangle_selection_does_not_read_validation_labels() -> None:
    from src.next45_exhaustive_rectangles import search_exhaustive_rectangles

    rng = np.random.default_rng(450)
    n = 600
    features = pd.DataFrame({"x": rng.random(n), "y": rng.random(n), "z": rng.random(n)})
    split = np.asarray(["discovery"] * 360 + ["validation"] * 240)
    endpoint_a = np.where((features.x > 0.55) & (features.y < 0.45), 0.3, 0.0)
    endpoint_b = endpoint_a.copy()
    endpoint_b[360:] = np.where(endpoint_b[360:] > 0.1, 0.0, 0.3)
    kwargs = {
        "features": features,
        "material_ids": [f"id-{index}" for index in range(n)],
        "split": split,
        "candidate_features": ("x", "y", "z"),
        "tail_fractions": (0.4, 0.5, 0.6),
    }
    first = search_exhaustive_rectangles(endpoint=endpoint_a, **kwargs)
    second = search_exhaustive_rectangles(endpoint=endpoint_b, **kwargs)
    assert first["selected_formula"] == second["selected_formula"]
    assert first["discovery_metrics"] == second["discovery_metrics"]
    assert first["validation_metrics"] != second["validation_metrics"]
