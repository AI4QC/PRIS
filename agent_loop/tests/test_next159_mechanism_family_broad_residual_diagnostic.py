from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.next159_mechanism_family_broad_residual_diagnostic import (
    EXPECTED_CANDIDATE_KEY_SHA256,
    EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT,
    _closest,
    select_auc_safe_candidates,
)


def test_frozen_next158_auc_safe_population_identity() -> None:
    path = Path(
        "$PRIS_ARCHIVE/"
        "next158_mechanism_family_consensus_search_v1/"
        "next158_mechanism_family_consensus_candidate_search.parquet"
    )
    selected = select_auc_safe_candidates(pd.read_parquet(path))
    digest = hashlib.sha256(
        "\n".join(selected["candidate_key"].astype(str)).encode()
    ).hexdigest()
    assert len(selected) == EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT == 3
    assert digest == EXPECTED_CANDIDATE_KEY_SHA256
    assert set(selected["coordination_protection_weight"]) == {0.0}
    assert set(selected["packing_protection_weight"]) == {0.5}


def test_closest_prioritizes_failed_count_then_shortfall() -> None:
    records = pd.DataFrame(
        [
            {
                "candidate_key": "a",
                "failed_constraint_count": 6,
                "normalized_shortfall_sum": 0.8,
                "best_threshold": 1.0,
            },
            {
                "candidate_key": "b",
                "failed_constraint_count": 7,
                "normalized_shortfall_sum": 0.1,
                "best_threshold": 0.5,
            },
            {
                "candidate_key": "c",
                "failed_constraint_count": 6,
                "normalized_shortfall_sum": 0.7,
                "best_threshold": 1.2,
            },
        ]
    )
    assert _closest(records)["candidate_key"] == "c"
