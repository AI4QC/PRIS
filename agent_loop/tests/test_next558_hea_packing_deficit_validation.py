from __future__ import annotations

import numpy as np
import pandas as pd

from src.next558_hea_packing_deficit_validation import evaluate_validation


def test_validation_evaluator_is_deterministic_and_uses_frozen_score() -> None:
    n = 40
    table = pd.DataFrame(
        {
            "fid": [f"id{i}" for i in range(n)],
            "chemical_system": [f"S{i // 4}" for i in range(n)],
            "size_family": (
                ["ordered"] * 10 + ["sqs"] * 10
                + ["ordered"] * 10 + ["sqs"] * 10
            ),
            "primitive_covalent_packing_fraction__risk_low": np.linspace(0.01, 0.99, n),
            "dft_waste": [False] * 20 + [True] * 20,
            "waste_severity": np.linspace(0.1, 2.0, n),
            "protected": [True] * 5 + [False] * 35,
            "pauling_risk": [np.nan] * n,
        }
    )

    first = evaluate_validation(table, bootstrap_draws=20, seed=9)
    second = evaluate_validation(table, bootstrap_draws=20, seed=9)

    assert first == second
    assert first["candidate"]["overall"]["roc_auc"] == 1.0
    assert first["candidate"]["ordered"]["roc_auc"] == 1.0
    assert first["candidate"]["sqs"]["roc_auc"] == 1.0
    assert first["pauling_control"]["coverage"] == 0.0
