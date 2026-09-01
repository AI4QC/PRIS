from __future__ import annotations

import numpy as np
import pandas as pd

from src.next67_odac23_monotone_expanded_search import search_expanded_additive_rule


def test_expanded_additive_search_recovers_simple_signal() -> None:
    n = 240
    endpoint = np.full(n, 0.10)
    endpoint[:72] = 0.30
    endpoint[72:200] = 0.01
    features = pd.DataFrame(
        {
            "combined_supported": True,
            "periodic_dimension_max": 3.0,
            "periodic_framework_fraction": 1.0,
            "defective": np.arange(n) % 2 == 0,
            "open_metal_site": (np.arange(n) // 2) % 2 == 0,
            "signal": np.r_[np.full(72, 4.0), np.full(128, -3.0), np.zeros(40)],
            "noise": np.sin(np.arange(n)),
        }
    )

    result = search_expanded_additive_rule(
        features=features,
        endpoint=endpoint,
        candidate_features=("signal", "noise"),
        anchor_features=("signal",),
    )

    assert result["passes_discovery_gates"]
    assert result["selected_formula"]["terms"][0]["feature"] == "signal"
    assert result["candidate_count"] > 0
