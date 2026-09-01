from __future__ import annotations

import pandas as pd
import pytest

from src.next131_protected_broad_residual_diagnostic import select_safe_candidates


def test_select_safe_candidates_keeps_only_published_safe12_rows_in_key_order() -> None:
    records = pd.DataFrame(
        {
            "candidate_key": ["z", "b", "a"],
            "safe_threshold": [3.0, 2.0, 1.0],
            "passes_safe_all_cells": [False, True, True],
            "protection_term_id": [None, "p", None],
            "protection_weight": [0.0, 0.25, 0.0],
        }
    )
    selected = select_safe_candidates(records)
    assert selected["candidate_key"].tolist() == ["a", "b"]
    assert selected["safe_threshold"].tolist() == [1.0, 2.0]


def test_select_safe_candidates_fails_closed_on_missing_or_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="schema"):
        select_safe_candidates(pd.DataFrame({"candidate_key": ["a"]}))
    records = pd.DataFrame(
        {
            "candidate_key": ["a"],
            "safe_threshold": [float("nan")],
            "passes_safe_all_cells": [True],
            "protection_term_id": [None],
            "protection_weight": [0.0],
        }
    )
    with pytest.raises(ValueError, match="threshold"):
        select_safe_candidates(records)
