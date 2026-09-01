from __future__ import annotations

import numpy as np
import pandas as pd

from src.next550_omc25_two_sided_contact_evaluate import evaluate_scores


def test_evaluator_uses_frozen_energy_response_thresholds() -> None:
    table = pd.DataFrame(
        {
            "material_id": [f"REF{i // 2}-{i}" for i in range(20)],
            "source_shard": ["s0"] * 10 + ["s1"] * 10,
            "tcse_risk": np.linspace(0.01, 0.99, 20),
            "risk_low_q10": np.linspace(0.02, 0.98, 20),
            "risk_high_q50": np.linspace(0.03, 0.97, 20),
            "next31_risk_score": np.linspace(0.04, 0.96, 20),
            "energy_drop_pa": np.linspace(0.0, 0.08, 20),
        }
    )

    result = evaluate_scores(table, bootstrap_draws=20, seed=7)

    assert result["counts"]["large_response"] == 10
    assert result["counts"]["protected"] == 3
    assert result["scores"]["tcse_risk"]["roc_auc"] == 1.0
    assert result["scores"]["tcse_risk"]["top_15_percent"]["rows"] == 3
    assert result["retrospective_only"] is True
    assert result["scientific_success_claim"] is False


def test_evaluator_reports_component_comparators() -> None:
    table = pd.DataFrame(
        {
            "material_id": [f"R{i}-{i}" for i in range(8)],
            "source_shard": ["s"] * 8,
            "tcse_risk": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
            "risk_low_q10": [0.2, 0.1, 0.4, 0.3, 0.5, 0.8, 0.7, 0.9],
            "risk_high_q50": [0.1, 0.3, 0.2, 0.4, 0.7, 0.5, 0.9, 0.8],
            "next31_risk_score": [0.1, 0.2, 0.4, 0.3, 0.6, 0.8, 0.7, 0.9],
            "energy_drop_pa": [0, 0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07],
        }
    )

    result = evaluate_scores(table, bootstrap_draws=10, seed=11)

    assert set(result["scores"]) == {
        "tcse_risk",
        "risk_low_q10",
        "risk_high_q50",
        "next31_risk_score",
    }
    assert result["scores"]["tcse_risk"]["roc_auc"] == 1.0
