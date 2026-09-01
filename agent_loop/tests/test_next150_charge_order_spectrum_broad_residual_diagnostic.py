import pandas as pd
import pytest

import src.next150_charge_order_spectrum_broad_residual_diagnostic as n150


def test_select_safe_candidates_preserves_auc_status_weight_and_threshold() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["b", "a"],
            "safe_threshold": [2.0, 1.0],
            "passes_safe_all_cells": [True, False],
            "passes_source_auc_gates": [False, True],
            "charge_order_spectrum_protection_weight": [1.0, 0.0],
        }
    )
    selected = n150.select_safe_candidates(frame)
    assert selected[
        ["candidate_key", "charge_order_spectrum_protection_weight", "passes_source_auc_gates"]
    ].values.tolist() == [["b", 1.0, False]]


def test_select_safe_candidates_rejects_duplicate_identity() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["x", "x"],
            "safe_threshold": [1.0, 1.0],
            "passes_safe_all_cells": [True, True],
            "passes_source_auc_gates": [False, False],
            "charge_order_spectrum_protection_weight": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="schema"):
        n150.select_safe_candidates(frame)
