"""Tests for the sealed label-free CHSC incremental-overlap analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_m5_baseline_uses_complete_composition_groups() -> None:
    from src.next12_chsc_label_free_overlap import _m5_baseline

    features = pd.DataFrame(
        {
            "sid": ["a", "b", "c", "d"],
            "rk": ["x", "x", "y", "y"],
            "m5_prediction_ok": [True, True, True, False],
            "m5_energy_ev_per_atom": [-1.0, -0.8, -2.0, np.nan],
        }
    )
    result = _m5_baseline(features, threshold=0.12119269371032715)
    assert result["m5_decision"].tolist() == [
        "KEEP",
        "REJECT",
        "ABSTAIN",
        "ABSTAIN",
    ]


def test_incremental_composition_counts_only_new_chsc_rejects() -> None:
    from src.next12_chsc_label_free_overlap import _transition_table

    baseline = pd.DataFrame(
        {"sid": ["a", "b", "c"], "rk": ["x", "x", "y"], "m5_decision": ["KEEP", "REJECT", "ABSTAIN"]}
    )
    phsc = pd.DataFrame(
        {
            "sid": ["a", "b", "c"],
            "rk": ["x", "x", "y"],
            "phsc_status": ["resolved_nonnegative", "resolved_negative", "abstain_unsupported_geometry"],
        }
    )
    chsc = pd.DataFrame(
        {
            "sid": ["a", "b", "c"],
            "rk": ["x", "x", "y"],
            "chsc_status": ["resolved_negative", "resolved_nonnegative", "abstain_unsupported_geometry"],
        }
    )
    table, summary = _transition_table(baseline, phsc, chsc)
    assert table["m5_phsc_decision"].tolist() == ["KEEP", "REJECT", "ABSTAIN"]
    assert table["m5_phsc_chsc_decision"].tolist() == ["REJECT", "REJECT", "ABSTAIN"]
    assert summary["chsc_net_reject_delta_over_m5_phsc"] == 1
    assert summary["new_chsc_reject_sids"] == ["a"]


def test_cli_accepts_no_label_or_endpoint_argument() -> None:
    from src.next12_chsc_label_free_overlap import main

    for forbidden in ("--labels", "--endpoint", "--dft", "--threshold"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
