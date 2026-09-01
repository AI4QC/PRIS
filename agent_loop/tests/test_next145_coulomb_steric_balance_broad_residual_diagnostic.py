import pandas as pd
import pytest

import src.next145_coulomb_steric_balance_broad_residual_diagnostic as n145


def test_select_safe_candidates_preserves_weight_and_threshold() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["b", "a"],
            "safe_threshold": [2.0, 1.0],
            "passes_safe_all_cells": [True, False],
            "coulomb_steric_balance_protection_weight": [0.1, 0.0],
        }
    )
    selected = n145.select_safe_candidates(frame)
    assert selected[
        ["candidate_key", "coulomb_steric_balance_protection_weight"]
    ].values.tolist() == [["b", 0.1]]


def test_select_safe_candidates_rejects_duplicate_identity() -> None:
    frame = pd.DataFrame(
        {
            "candidate_key": ["x", "x"],
            "safe_threshold": [1.0, 1.0],
            "passes_safe_all_cells": [True, True],
            "coulomb_steric_balance_protection_weight": [0.1, 0.1],
        }
    )
    with pytest.raises(ValueError, match="schema"):
        n145.select_safe_candidates(frame)
