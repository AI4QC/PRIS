from __future__ import annotations

import numpy as np
import pandas as pd

from src.next50_framework_rule_search import (
    apply_framework_formula,
    search_source_balanced_framework_rule,
)


def test_source_balanced_search_prefers_within_source_signal_and_stays_finite() -> None:
    rows = []
    endpoint = []
    sources = []
    material_ids = []
    for source_index, source in enumerate(("alpha", "beta", "gamma")):
        for index in range(80):
            severe = index % 4 == 0
            rows.append(
                {
                    "signal": float(severe),
                    "source_confounded": float(source_index),
                    "noise": float((index * 17 + source_index) % 23),
                    "periodic_dimension_max": 3.0,
                    "periodic_framework_fraction": 1.0,
                }
            )
            endpoint.append(0.65 if severe else 0.05)
            sources.append(source)
            material_ids.append(f"{source}-{index}")
    features = pd.DataFrame(rows)

    result = search_source_balanced_framework_rule(
        features=features,
        material_ids=material_ids,
        source_families=sources,
        endpoint=endpoint,
        candidate_features=("signal", "source_confounded", "noise"),
    )

    formula = result["selected_formula"]
    assert 1 <= len(formula["terms"]) <= 3
    assert formula["terms"][0]["feature"] == "signal"
    assert result["full_development_metrics"]["passes_primary_gates"]
    assert result["source_balanced_diagnostics"]["worst_source_auc"] == 1.0
    assert result["source_balanced_diagnostics"]["macro_source_auc"] == 1.0

    contaminated = features.copy()
    contaminated.loc[0, "signal"] = np.nan
    _score, supported, rejected = apply_framework_formula(contaminated, formula)
    assert not supported[0]
    assert not rejected[0]
