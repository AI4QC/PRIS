from __future__ import annotations

import numpy as np
import pandas as pd

from src.next566b_hea_mechanism_selection import evaluate_selection


def _selection_table() -> pd.DataFrame:
    rows = 500
    score = np.tile(np.linspace(0.0, 1.0, 100), 5)
    return pd.DataFrame(
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
            "u_Z": np.tile([0.2, 0.8], rows // 2),
            "MEMAX": score,
            "MEPU24": score,
            "ZEPU24": score,
        }
    )


def test_selection_uses_all_frozen_gates_and_deterministic_tie_break() -> None:
    result = evaluate_selection(_selection_table(), bootstrap_draws=100, seed=11)

    assert result["selection_pass"] is True
    assert result["selected_candidate"] == "MEMAX"
    assert all(result["candidates"]["MEMAX"]["selection_clauses"].values())
