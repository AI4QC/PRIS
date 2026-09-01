"""Contracts for NEXT29 fixed-contact transport to generated inorganic CSP."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_transport_metrics_compare_identical_cohort_pauling_controls() -> None:
    from src.next29_omatg_contact_transport import evaluate_frames

    rows = 1000
    ids = [f"generated-{index:04d}" for index in range(rows)]
    reference_match = np.zeros(rows, dtype=bool)
    reference_match[500:] = True
    reject = np.zeros(rows, dtype=bool)
    reject[:118] = True
    reject[500:502] = True
    predictions = pd.DataFrame(
        {
            "material_id": ids,
            "analytic_supported": True,
            "next28_risk_score": np.where(reject, 7.0, 2.0),
            "reject": reject,
        }
    )
    joined = pd.DataFrame(
        {
            "material_id": ids,
            "analytic_supported": False,
            "next23_risk_score": -99.0,
            "reject": False,
            "reference_match": reference_match,
            "corrected_rmsd": np.where(reference_match, 0.02, 0.5),
        }
    )
    for name in ("p2", "p3", "p4", "p5", "p2_p5"):
        joined[f"pauling_{name}_decision"] = "ABSTAIN"
    result = evaluate_frames(predictions=predictions, endpoint_and_pauling=joined)
    assert result["fixed_contact_rule"]["passes_primary_gates"] is True
    assert result["fixed_contact_rule"]["nonmatches_rejected"] == 118
    assert result["best_safe_pauling_savings_lower"] == 0.0
    assert result["beyond_pauling_on_this_endpoint"] is True


def test_transport_metrics_fail_open_for_unsupported_rows() -> None:
    from src.next29_omatg_contact_transport import evaluate_frames

    predictions = pd.DataFrame(
        {
            "material_id": ["a", "b"],
            "analytic_supported": [False, True],
            "next28_risk_score": [np.nan, 7.0],
            "reject": [True, True],
        }
    )
    joined = pd.DataFrame(
        {
            "material_id": ["a", "b"],
            "reference_match": [True, False],
            "corrected_rmsd": [0.01, 0.5],
            **{
                f"pauling_{name}_decision": ["ABSTAIN", "REJECT"]
                for name in ("p2", "p3", "p4", "p5", "p2_p5")
            },
        }
    )
    result = evaluate_frames(predictions=predictions, endpoint_and_pauling=joined)
    assert result["fixed_contact_rule"]["rejected"] == 1
    assert result["fixed_contact_rule"]["supported"] == 1
