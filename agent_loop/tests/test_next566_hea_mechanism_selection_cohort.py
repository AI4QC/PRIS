from __future__ import annotations

from src.next566_hea_mechanism_selection_cohort import selection_hash


def test_selection_hash_is_fid_deterministic() -> None:
    assert selection_hash("abc") == selection_hash("abc")
    assert selection_hash("abc") != selection_hash("abd")
