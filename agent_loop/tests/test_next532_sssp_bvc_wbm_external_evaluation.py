from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.next529_sssp_bvc_development_freeze as n529
import src.next532_sssp_bvc_wbm_external_evaluation as n532
from src.next87_scigen_sparse_law_search import assign_group_folds


def _group_for_fold(fold: int) -> str:
    for index in range(10_000):
        value = f"synthetic_formula_{fold}_{index}"
        if int(assign_group_folds(np.asarray([value], dtype=object))[0]) == fold:
            return value
    raise AssertionError("could not construct a deterministic fold group")


def _frame(*, reversed_signal: bool = False) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        group = _group_for_fold(fold)
        for label, endpoint in (("protected", 0.05), ("severe", 0.80)):
            for index in range(30):
                should_reject = label == "severe"
                if reversed_signal:
                    should_reject = not should_reject
                sssp = 0.20 if should_reject else 0.80
                scbv = 0.90 if should_reject else 0.10
                pauling = (
                    "REJECT" if label == "severe" and index < 6
                    else "REJECT" if label == "protected" and index < 3
                    else "KEEP" if index < 15
                    else "ABSTAIN"
                )
                rows.append(
                    {
                        "material_id": f"m-{fold}-{label}-{index}",
                        "rk": group,
                        "sssp_same_sign_shell_purity_q10": sssp,
                        "sssp_supported": True,
                        "scbv_mismatch_rms": scbv,
                        "scbv_supported": True,
                        "pauling_p2_p5_decision": pauling,
                        "endpoint": endpoint,
                    }
                )
    frame = pd.DataFrame(rows)
    applied = n529.apply_sssp_bvc(
        sssp=frame["sssp_same_sign_shell_purity_q10"].to_numpy(float),
        sssp_supported=frame["sssp_supported"].to_numpy(bool),
        scbv=frame["scbv_mismatch_rms"].to_numpy(float),
        scbv_supported=frame["scbv_supported"].to_numpy(bool),
        sssp_threshold=n529.SSSP_THRESHOLD,
        scbv_threshold=n529.EXPECTED_SCBV_THRESHOLD,
    )
    frame["risk_score"] = applied["risk"]
    frame["formula_supported"] = applied["supported"]
    frame["reject"] = applied["reject"]
    frame["sssp_bvc_decision"] = np.where(
        ~applied["supported"], "ABSTAIN",
        np.where(applied["reject"], "REJECT", "KEEP"),
    )
    return frame


def test_strong_frozen_formula_passes_and_dominates_pauling() -> None:
    result = n532.evaluate_wbm_external(frame=_frame())
    assert result["passes_all_external_gates"] is True
    assert result["metrics"]["severe_rejected"] == 150
    assert result["gate_checks"]["five_formula_folds"] is True
    assert result["pauling_dominance"]["passes_all"] is True
    assert result["binary_reject_auc_all_extremes"] == pytest.approx(1.0)


def test_reversed_frozen_formula_fails_without_direction_repair() -> None:
    result = n532.evaluate_wbm_external(frame=_frame(reversed_signal=True))
    assert result["passes_all_external_gates"] is False
    assert result["formula_or_threshold_modified"] is False
    assert result["gate_checks"]["protected_recall_lower"] is False
    assert result["binary_reject_auc_all_extremes"] == pytest.approx(0.0)


def test_nonfinite_endpoint_and_modified_frozen_decision_fail_closed() -> None:
    frame = _frame()
    frame.loc[0, "endpoint"] = np.nan
    with pytest.raises(ValueError, match="evaluation frame differs"):
        n532.evaluate_wbm_external(frame=frame)

    frame = _frame()
    frame.loc[0, "reject"] = not bool(frame.loc[0, "reject"])
    with pytest.raises(ValueError, match="frozen predictions differ"):
        n532.evaluate_wbm_external(frame=frame)


def test_endpoint_is_offline_only_and_never_enters_formula_boundary() -> None:
    assert n532.ENDPOINT_COLUMN == "site_stats_fingerprint_init_final_norm_diff"
    assert n532.ENDPOINT_COLUMNS_OPENED == ("material_id", n532.ENDPOINT_COLUMN)
    assert n532.EXECUTABLE_INPUT_BOUNDARY == (
        "composition", "one raw initial fully periodic geometry"
    )
    assert n532.PROHIBITED_EXECUTABLE_INPUTS == (
        "DFT values", "relaxed or later geometry", "trajectory",
        "learned energy force stress", "MLIP or proxy potential", "relaxation",
    )
