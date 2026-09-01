from __future__ import annotations

import numpy as np
import pandas as pd

from src.next57_odac23_discovery_search import DOMAIN_GATE
from src.next74_odac23_one_shot_internal_validation import evaluate_frozen_formula


def test_frozen_formula_evaluation_does_not_recalibrate() -> None:
    rng = np.random.default_rng(74)
    n = 800
    x = rng.normal(size=n)
    endpoint = np.full(n, 0.10)
    endpoint[x >= 0.7] = 0.30
    endpoint[x <= -0.7] = 0.01
    features = pd.DataFrame(
        {
            "combined_supported": True,
            "periodic_dimension_max": 3.0,
            "periodic_framework_fraction": 1.0,
            "defective": np.arange(n) % 2 == 0,
            "open_metal_site": (np.arange(n) // 2) % 2 == 0,
            "x": x,
        }
    )
    formula = {
        "kind": "additive",
        "terms": [
            {"feature": "x", "direction": 1, "center": 0.0, "scale": 1.0, "weight": 1.0}
        ],
        "threshold": 0.7,
        "missing_policy": "KEEP",
        "domain_gate": dict(DOMAIN_GATE),
    }

    result = evaluate_frozen_formula(features=features, endpoint=endpoint, formula=formula)

    assert result["passes_gates"]
    assert result["evaluated_threshold"] == formula["threshold"]
    assert result["formula_recalibrated"] is False
