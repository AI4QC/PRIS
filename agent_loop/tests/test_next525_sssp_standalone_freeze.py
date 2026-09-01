import numpy as np
import pytest

import src.next525_sssp_standalone_freeze as n


def _source(values, endpoint):
    values = np.asarray(values, dtype=float)
    return {
        "sssp": values,
        "supported": np.isfinite(values),
        "endpoint": np.asarray(endpoint, dtype=float),
    }


def test_select_shared_threshold_uses_one_threshold_and_lexicographic_rank():
    sources = {
        "left": _source(
            [0.10, 0.20, 0.30, 0.80, 0.90, 0.95],
            [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        ),
        "right": _source(
            [0.12, 0.22, 0.32, 0.82, 0.92, 0.96],
            [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        ),
    }
    result = n.select_shared_threshold(
        sources=sources,
        minimum_protected_recall_lower=0.0,
        minimum_savings_lower=0.0,
    )
    assert result["threshold"] == pytest.approx(0.32)
    assert set(result["source_metrics"]) == {"left", "right"}
    assert result["source_metrics"]["left"]["severe_rejected"] == 3
    assert result["source_metrics"]["right"]["severe_rejected"] == 3


def test_select_shared_threshold_rejects_bad_arrays_and_no_feasible_candidate():
    with pytest.raises(ValueError, match="source arrays"):
        n.select_shared_threshold(
            sources={"x": _source([0.1, 0.2], [2.0])},
            minimum_protected_recall_lower=0.0,
            minimum_savings_lower=0.0,
        )
    with pytest.raises(RuntimeError, match="no shared threshold"):
        n.select_shared_threshold(
            sources={"x": _source([0.1, 0.9], [2.0, 0.0])},
            minimum_protected_recall_lower=1.0,
            minimum_savings_lower=1.0,
        )


def test_formula_boundary_is_strictly_zero_dft_and_keeps_unsupported():
    formula = n.formula_payload(threshold=0.5231805323)
    assert formula["feature"] == "sssp_same_sign_shell_purity_q10"
    assert formula["reject_when"] == "supported and SSSP <= threshold"
    assert formula["missing_policy"] == "ABSTAIN"
    assert formula["dft_inputs"] == []
    assert formula["learned_model_inputs"] == []
    assert formula["relaxation_inputs"] == []
