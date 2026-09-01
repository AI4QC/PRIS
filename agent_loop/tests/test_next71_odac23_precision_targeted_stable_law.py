from __future__ import annotations

import numpy as np
import pandas as pd

from src.next71_odac23_precision_targeted_stable_law import (
    search_precision_targeted_stable_law,
)


def test_precision_targeted_stable_law_recovers_all_row_signal() -> None:
    rng = np.random.default_rng(71)
    n = 700
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    latent = 2.2 * x - 1.4 * y
    endpoint = np.full(n, 0.10)
    endpoint[latent >= 0.9] = 0.30
    endpoint[latent <= -0.7] = 0.01
    features = pd.DataFrame(
        {
            "material_id": [f"m-{i}" for i in range(n)],
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

    result = search_precision_targeted_stable_law(
        features=features,
        endpoint=endpoint,
        candidate_features=("x", "y", "noise"),
        c_values=(0.1, 1.0),
        protected_multipliers=(1.0, 2.0),
        term_counts=(3,),
    )

    assert result["passes_discovery_gates"]
    assert {term["feature"] for term in result["selected_formula"]["terms"]} >= {"x", "y"}
    assert not result["selected_fit_reached_iteration_limit"]
