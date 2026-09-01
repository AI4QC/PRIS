from __future__ import annotations

import numpy as np
import pandas as pd

from src.next57_odac23_discovery_search import DOMAIN_GATE
from src.next73_odac23_exact_threshold_calibration import calibrate_exact_threshold


def test_exact_threshold_calibration_seals_fixed_terms() -> None:
    rng = np.random.default_rng(73)
    n = 700
    x = rng.normal(size=n)
    endpoint = np.full(n, 0.10)
    endpoint[x >= 0.65] = 0.30
    endpoint[x <= -0.65] = 0.01
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
    frozen = {
        "kind": "additive",
        "terms": [
            {"feature": "x", "direction": 1, "center": 0.0, "scale": 1.0, "weight": 1.0}
        ],
        "threshold": 999.0,
        "missing_policy": "KEEP",
        "domain_gate": dict(DOMAIN_GATE),
    }

    result = calibrate_exact_threshold(features=features, endpoint=endpoint, frozen_formula=frozen)

    assert result["passes_discovery_gates"]
    assert result["selected_formula"]["terms"] == frozen["terms"]
    assert result["selected_formula"]["threshold"] != frozen["threshold"]
