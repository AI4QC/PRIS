from __future__ import annotations

import numpy as np
import pandas as pd

from src.next57_odac23_discovery_search import (
    apply_odac23_formula,
    search_discovery_rule,
)


def test_finite_search_recovers_protected_vs_severe_signal() -> None:
    n = 240
    severe = np.arange(n) < 72
    protected = (np.arange(n) >= 72) & (np.arange(n) < 200)
    endpoint = np.full(n, 0.10)
    endpoint[severe] = 0.30
    endpoint[protected] = 0.01
    signal = np.where(severe, 4.0, np.where(protected, -3.0, 0.0))
    features = pd.DataFrame(
        {
            "combined_supported": True,
            "periodic_dimension_max": 3.0,
            "periodic_framework_fraction": 1.0,
            "defective": np.arange(n) % 2 == 0,
            "open_metal_site": (np.arange(n) // 2) % 2 == 0,
            "signal": signal,
            "noise": np.sin(np.arange(n)),
        }
    )

    result = search_discovery_rule(
        features=features,
        endpoint=endpoint,
        candidate_features=("signal", "noise"),
    )

    assert result["passes_discovery_gates"]
    assert result["selected_formula"]["terms"][0]["feature"] == "signal"
    score, supported, reject = apply_odac23_formula(
        features, result["selected_formula"]
    )
    assert supported.all()
    assert np.isfinite(score).all()
    assert (reject & severe).sum() > 0
    assert not (reject & protected).any()


def test_missing_term_fails_open_to_keep() -> None:
    features = pd.DataFrame(
        {
            "combined_supported": [True, True],
            "periodic_dimension_max": [3.0, 3.0],
            "periodic_framework_fraction": [1.0, 1.0],
            "x": [1.0, np.nan],
        }
    )
    formula = {
        "kind": "additive",
        "terms": [{"feature": "x", "direction": 1, "center": 0.0, "scale": 1.0, "weight": 1.0}],
        "threshold": 0.5,
        "missing_policy": "KEEP",
        "domain_gate": {
            "periodic_dimension_max_min": 1.0,
            "periodic_framework_fraction_min": 0.5,
        },
    }

    _score, supported, reject = apply_odac23_formula(features, formula)

    assert supported.tolist() == [True, False]
    assert reject.tolist() == [True, False]
