from __future__ import annotations

import numpy as np
import pandas as pd

import src.next559_hea_entropy_packing_discovery as n559
from src.next561_hea_entropy_packing_confirmation import evaluate_confirmation


def _strong_table() -> pd.DataFrame:
    rows = 400
    base = np.tile(np.linspace(0.0, 1.0, 100, endpoint=True), 4)
    labels = base >= 0.5
    return pd.DataFrame(
        {
            "fid": [f"fid-{index:04d}" for index in range(rows)],
            "chemical_system": [f"sys-{index % 40:02d}" for index in range(rows)],
            "size_family": np.where(np.arange(rows) % 2 == 0, "ordered", "sqs"),
            "replication_stratum": np.where(
                np.arange(rows) < 100, "unseen_chemical_system", "new_identity_known_system"
            ),
            n559.SCORE: base,
            n559.ENTROPY_RISK: np.tile([0.25, 0.75], rows // 2),
            n559.PACKING_RISK: np.tile([0.75, 0.25], rows // 2),
            "dft_waste": labels,
            "waste_severity": 1.0 + 3.0 * base,
            "protected": base < 0.20,
            "pauling_risk": np.nan,
        }
    )


def test_confirmation_passes_only_when_all_frozen_clauses_pass() -> None:
    result = evaluate_confirmation(_strong_table(), bootstrap_draws=100, seed=7)

    assert result["confirmation_pass"] is True
    assert all(result["confirmation_clauses"].values())
    assert result["counts"]["pauling_supported"] == 0


def test_confirmation_rejects_joint_score_without_component_margin() -> None:
    table = _strong_table()
    table[n559.ENTROPY_RISK] = table[n559.SCORE]
    result = evaluate_confirmation(table, bootstrap_draws=100, seed=7)

    assert result["confirmation_pass"] is False
    assert result["confirmation_clauses"]["auc_margin_over_entropy_at_least_0p03"] is False
