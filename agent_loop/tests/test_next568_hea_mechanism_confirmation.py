from __future__ import annotations

import numpy as np
import pandas as pd

from src.next568_hea_mechanism_confirmation import evaluate_confirmation


def test_final_confirmation_applies_the_frozen_gate_panel() -> None:
    rows = 500
    score = np.tile(np.linspace(0.0, 1.0, 100), 5)
    table = pd.DataFrame(
        {
            "fid": [f"fid-{index:04d}" for index in range(rows)],
            "chemical_system": [f"sys-{index % 50:02d}" for index in range(rows)],
            "size_family": np.where(np.arange(rows) % 2, "sqs", "ordered"),
            "dft_waste": score >= 0.5,
            "waste_severity": 1.0 + score,
            "protected": score < 0.2,
            "pauling_risk": np.nan,
            "u_H": np.tile([0.25, 0.75], rows // 2),
            "u_M": np.tile([0.75, 0.25], rows // 2),
            "MEMAX": score,
        }
    )

    result = evaluate_confirmation(table, "MEMAX", bootstrap_draws=100, seed=13)

    assert result["confirmation_pass"] is True
    assert all(result["confirmation_clauses"].values())
    assert result["scientific_success_claim"] is True
