from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from src.next96_wyformer_dual_operating_candidate import (
    BROAD_THRESHOLD,
    SAFE_THRESHOLD,
    _prediction_frame,
    apply_dual_formula,
    freeze_wyformer_dual_operating_candidate,
    pauling_dominance,
)


def test_freeze_runner_has_no_validation_or_replication_endpoint_argument() -> None:
    parameters = inspect.signature(freeze_wyformer_dual_operating_candidate).parameters
    assert "discovery_endpoint_dir" in parameters
    assert not any("validation" in name or "replication" in name for name in parameters)


def test_dual_formula_uses_same_score_and_inclusive_fixed_thresholds() -> None:
    frame = pd.DataFrame({"risk": [0.0, BROAD_THRESHOLD, SAFE_THRESHOLD, np.nan]})
    formula = {
        "kind": "nonnegative_sum_of_at_most_three_one_sided_robust_hinges",
        "missing_policy": "ABSTAIN",
        "terms": [
            {
                "term_id": "risk__high",
                "feature": "risk",
                "direction": 1,
                "transform": "asinh",
                "group": "test",
                "center": 0.0,
                "scale": 1.0,
                "weight": 1.0,
            }
        ],
    }
    # asinh changes raw values, so use thresholds equal to the transformed
    # values for this unit test rather than the formal constants.
    broad = float(np.arcsinh(BROAD_THRESHOLD))
    safe = float(np.arcsinh(SAFE_THRESHOLD))
    score, supported, safe_reject, broad_reject = apply_dual_formula(
        frame, formula, safe_threshold=safe, broad_threshold=broad
    )
    assert supported.tolist() == [True, True, True, False]
    assert np.isnan(score[-1])
    assert broad_reject.tolist() == [False, True, True, False]
    assert safe_reject.tolist() == [False, False, True, False]


def test_pauling_dominance_requires_all_five_comparisons() -> None:
    pauling = {
        "coverage_lower": 0.4,
        "protected_kept": 90,
        "severe_rejected": 30,
        "severe_rejection_precision_lower": 0.3,
        "savings_lower": 0.35,
    }
    law = {
        "coverage_lower": 0.9,
        "protected_kept": 90,
        "severe_rejected": 31,
        "severe_rejection_precision_lower": 0.5,
        "savings_lower": 0.4,
    }
    result = pauling_dominance(law, pauling)
    assert result["passes_all"] is True
    worse = dict(law, severe_rejected=30)
    assert pauling_dominance(worse, pauling)["passes_all"] is False


def test_prediction_frame_freezes_identity_group_and_pauling_columns() -> None:
    frame = pd.DataFrame(
        {
            "material_id": ["m1", "m2"],
            "reduced_formula": ["AB", "A2B"],
            "crystal_system": ["cubic", "orthorhombic"],
            "pauling_p2_p5_decision": ["KEEP", "REJECT"],
        }
    )
    predictions = _prediction_frame(
        frame=frame,
        score=np.array([0.1, 1.2]),
        supported=np.array([True, False]),
        safe_reject=np.array([False, False]),
        broad_reject=np.array([False, False]),
    )
    assert predictions.columns.tolist() == [
        "material_id",
        "reduced_formula",
        "crystal_system",
        "pauling_p2_p5_decision",
        "score",
        "supported",
        "safe_decision",
        "broad_decision",
    ]
    assert predictions.loc[1, "safe_decision"] == "ABSTAIN"
