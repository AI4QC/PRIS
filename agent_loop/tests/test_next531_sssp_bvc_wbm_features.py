import numpy as np
import pandas as pd

import src.next531_sssp_bvc_wbm_features as n


def test_apply_frozen_formula_preserves_scbv_missing_keep_policy():
    table = pd.DataFrame(
        {
            "sssp_same_sign_shell_purity_q10": [0.2, 0.2, 0.8, np.nan],
            "sssp_supported": [True, True, True, False],
            "scbv_mismatch_rms": [0.5, np.nan, 0.5, 0.5],
            "scbv_supported": [True, False, True, True],
        }
    )
    result = n.apply_frozen_formula(table)
    assert result["formula_supported"].tolist() == [True, True, True, False]
    assert result["reject"].tolist() == [True, False, False, False]
    assert result["risk_score"].iloc[0] > 0.0
    assert result["risk_score"].iloc[1] == 0.0


def test_label_free_gates_require_coverage_and_nondegeneracy():
    table = pd.DataFrame(
        {
            "sssp_same_sign_shell_purity_q10": np.linspace(0.1, 0.9, 100),
            "sssp_supported": True,
            "scbv_mismatch_rms": np.linspace(0.2, 0.8, 100),
            "scbv_supported": True,
        }
    )
    result = n.label_free_gate_statistics(table)
    assert result["passes"] is True
    table.loc[:20, "sssp_supported"] = False
    table.loc[:20, "sssp_same_sign_shell_purity_q10"] = np.nan
    assert n.label_free_gate_statistics(table)["passes"] is False


def test_feature_boundary_flags_keep_external_endpoint_closed():
    assert n.BOUNDARY_FLAGS["wbm_summary_opened"] is False
    assert n.BOUNDARY_FLAGS["external_endpoint_opened"] is False
    assert n.BOUNDARY_FLAGS["dft_values_used_by_features"] is False
