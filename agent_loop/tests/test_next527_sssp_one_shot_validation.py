import numpy as np
import pandas as pd
import pytest

import src.next527_sssp_one_shot_validation as n


def _synthetic_source(*, reverse=False):
    rows = 600
    severe = np.arange(rows) >= rows // 2
    sssp = np.where(severe, 0.20, 0.85).astype(float)
    if reverse:
        sssp = 1.0 - sssp
    return pd.DataFrame(
        {
            "material_id": [f"m{i}" for i in range(rows)],
            "reduced_formula": [f"F{i}" for i in range(rows)],
            "sssp_same_sign_shell_purity_q10": sssp,
            "sssp_supported": True,
            "pauling_p2_p5_decision": np.where(
                np.arange(rows) % 3 == 0, "REJECT", "ABSTAIN"
            ),
            "endpoint": np.where(severe, 2.0, 0.0),
        }
    )


def test_evaluate_source_passes_strong_zero_dft_signal_and_dominates_pauling():
    result = n.evaluate_sssp_source(
        frame=_synthetic_source(), threshold=0.5231805323, bootstrap_draws=100
    )
    assert result["passes_source_gates"] is True
    assert result["pooled_auc"] == pytest.approx(1.0)
    assert result["worst_fold_auc"] == pytest.approx(1.0)
    assert result["cluster_bootstrap_auc_95"][0] == pytest.approx(1.0)
    assert result["pauling_dominance"]["passes_all"] is True


def test_evaluate_source_fails_reversed_signal_without_changing_direction():
    result = n.evaluate_sssp_source(
        frame=_synthetic_source(reverse=True),
        threshold=0.5231805323,
        bootstrap_draws=50,
    )
    assert result["passes_source_gates"] is False
    assert result["pooled_auc"] == pytest.approx(0.0)
    assert result["formula_or_threshold_modified"] is False


def test_evaluate_source_rejects_nonfinite_endpoint_and_bad_decision_schema():
    bad = _synthetic_source()
    bad.loc[0, "endpoint"] = np.nan
    with pytest.raises(ValueError, match="evaluation frame"):
        n.evaluate_sssp_source(frame=bad, threshold=0.5231805323, bootstrap_draws=10)
    bad = _synthetic_source()
    bad.loc[0, "pauling_p2_p5_decision"] = "MAYBE"
    with pytest.raises(ValueError, match="evaluation frame"):
        n.evaluate_sssp_source(frame=bad, threshold=0.5231805323, bootstrap_draws=10)


def test_validation_boundary_is_sequential_and_replication_closed():
    assert n.ROLE == "internal_validation"
    assert n.BOOTSTRAP_SEED == 20260813
    assert n.GATES["protected_recall_lower"] == 0.95
    assert n.GATES["severe_rejection_precision_lower"] == 0.60
