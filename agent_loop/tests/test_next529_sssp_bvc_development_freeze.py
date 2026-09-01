import numpy as np
import pytest

import src.next529_sssp_bvc_development_freeze as n


def _cell(sssp, mismatch, endpoint):
    return {
        "sssp": np.asarray(sssp, dtype=float),
        "sssp_supported": np.ones(len(sssp), dtype=bool),
        "scbv": np.asarray(mismatch, dtype=float),
        "endpoint": np.asarray(endpoint, dtype=float),
    }


def test_apply_formula_is_conjunctive_and_scbv_missing_keeps():
    result = n.apply_sssp_bvc(
        sssp=[0.2, 0.2, 0.8, np.nan],
        sssp_supported=[True, True, True, False],
        scbv=[0.5, np.nan, 0.5, 0.5],
        scbv_supported=[True, False, True, True],
        sssp_threshold=0.5231805323,
        scbv_threshold=0.33695346214642063,
    )
    assert result["supported"].tolist() == [True, True, True, False]
    assert result["reject"].tolist() == [True, False, False, False]
    assert result["risk"][0] > 0.0
    assert result["risk"][1] == 0.0
    assert np.isnan(result["risk"][3])


def test_select_scbv_threshold_uses_all_cells_and_frozen_rank():
    cells = {
        "a": _cell(
            [0.2, 0.2, 0.2, 0.8, 0.8],
            [0.2, 0.4, 0.8, 0.9, 0.1],
            [0.0, 2.0, 2.0, 0.0, 0.0],
        ),
        "b": _cell(
            [0.2, 0.2, 0.2, 0.8, 0.8],
            [0.1, 0.5, 0.9, 0.8, 0.2],
            [0.0, 2.0, 2.0, 0.0, 0.0],
        ),
    }
    selected = n.select_scbv_threshold(
        cells=cells,
        sssp_threshold=0.5231805323,
        minimum_protected_recall_lower=0.0,
        minimum_savings_lower=0.0,
    )
    assert selected["threshold"] == pytest.approx(0.4)
    assert selected["cell_metrics"]["a"]["severe_rejected"] == 2
    assert selected["cell_metrics"]["b"]["severe_rejected"] == 2


def test_formula_payload_has_no_dft_model_or_relaxation_inputs():
    formula = n.formula_payload(
        sssp_threshold=0.5231805323,
        scbv_threshold=0.33695346214642063,
    )
    assert formula["dft_inputs"] == []
    assert formula["learned_model_inputs"] == []
    assert formula["relaxation_inputs"] == []
    assert formula["scbv_missing_policy"] == "KEEP"
