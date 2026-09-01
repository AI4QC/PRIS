from __future__ import annotations

from src.next567_hea_mechanism_confirmation_cohort import selection_hash


def test_confirmation_selection_hash_is_distinct_and_deterministic() -> None:
    assert selection_hash("abc") == selection_hash("abc")
    assert selection_hash("abc") != selection_hash("abd")
