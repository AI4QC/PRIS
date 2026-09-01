from __future__ import annotations

import numpy as np
import pandas as pd

from src.next57_odac23_discovery_search import DOMAIN_GATE
from src.next72_odac23_anchored_tail_correction_search import (
    search_anchored_tail_correction,
)


def test_anchored_tail_correction_adds_missing_signal() -> None:
    rng = np.random.default_rng(72)
    n = 800
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    latent = x + 2.5 * y
    endpoint = np.full(n, 0.10)
    endpoint[latent >= 0.8] = 0.30
    endpoint[latent <= -0.8] = 0.01
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
    anchor = {
        "kind": "additive",
        "terms": [
            {"feature": "x", "direction": 1, "center": 0.0, "scale": 1.0, "weight": 1.0}
        ],
        "threshold": 0.0,
        "missing_policy": "KEEP",
        "domain_gate": dict(DOMAIN_GATE),
    }

    result = search_anchored_tail_correction(
        features=features,
        endpoint=endpoint,
        anchor_formula=anchor,
        candidate_features=("y", "noise"),
        single_weights=(0.5, 1.0, 2.0, 4.0),
        pair_weights=(0.5, 1.0),
        pair_shortlist=2,
    )

    assert result["passes_discovery_gates"]
    assert "y" in {term["feature"] for term in result["selected_formula"]["terms"]}
