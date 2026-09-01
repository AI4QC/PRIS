from __future__ import annotations

import numpy as np
import pandas as pd

from src.next68_odac23_sparse_stable_law import search_sparse_stable_law


def test_sparse_stable_law_recovers_explicit_signal() -> None:
    rng = np.random.default_rng(19)
    n = 500
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    latent = 2.0 * x - 1.5 * y
    severe = latent > 0.8
    protected = latent < -0.5
    endpoint = np.full(n, 0.10)
    endpoint[severe] = 0.30
    endpoint[protected] = 0.01
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

    result = search_sparse_stable_law(
        features=features,
        endpoint=endpoint,
        candidate_features=("x", "y", "noise"),
        c_values=(0.1, 0.3, 1.0),
        term_counts=(3,),
    )

    assert result["passes_discovery_gates"]
    assert {term["feature"] for term in result["selected_formula"]["terms"]} >= {"x", "y"}
    assert all(term["weight"] > 0 for term in result["selected_formula"]["terms"])
