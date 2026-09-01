from __future__ import annotations

import numpy as np
import pandas as pd

from src.next62_odac23_conjunctive_search import (
    apply_conjunctive_formula,
    search_conjunctive_rule,
)


def test_conjunction_recovers_joint_tail_without_protected_rejections() -> None:
    rng = np.random.default_rng(7)
    n = 600
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    severe = (x > 0.65) & (y > 0.65)
    protected = (x < 0.2) | (y < 0.2)
    endpoint = np.full(n, 0.10)
    endpoint[protected] = 0.01
    endpoint[severe] = 0.30
    features = pd.DataFrame(
        {
            "combined_supported": True,
            "periodic_dimension_max": 3.0,
            "periodic_framework_fraction": 1.0,
            "defective": np.arange(n) % 2 == 0,
            "open_metal_site": (np.arange(n) // 2) % 2 == 0,
            "x": x,
            "y": y,
            "noise": rng.normal(size=n),
        }
    )

    result = search_conjunctive_rule(
        features=features,
        endpoint=endpoint,
        candidate_features=("x", "y", "noise"),
    )

    assert result["passes_discovery_gates"]
    assert {term["feature"] for term in result["selected_formula"]["terms"]} == {"x", "y"}
    _score, supported, reject = apply_conjunctive_formula(
        features, result["selected_formula"]
    )
    assert supported.all()
    assert not (reject & protected).any()
    assert (reject & severe).sum() > 0


def test_conjunction_missing_value_fails_open() -> None:
    features = pd.DataFrame(
        {
            "combined_supported": [True, True],
            "periodic_dimension_max": [3.0, 3.0],
            "periodic_framework_fraction": [1.0, 1.0],
            "x": [2.0, np.nan],
        }
    )
    formula = {
        "kind": "conjunction",
        "terms": [
            {
                "feature": "x",
                "direction": 1,
                "center": 0.0,
                "scale": 1.0,
                "cutoff": 1.0,
                "quantile": 0.9,
            }
        ],
        "missing_policy": "KEEP",
        "domain_gate": {
            "periodic_dimension_max_min": 1.0,
            "periodic_framework_fraction_min": 0.5,
        },
    }

    score, supported, reject = apply_conjunctive_formula(features, formula)

    assert score[0] == 1.0
    assert supported.tolist() == [True, False]
    assert reject.tolist() == [True, False]
