from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.next154_top2_attenuation_broad_residual_diagnostic import (
    EXPECTED_CANDIDATE_KEY_SHA256,
    EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT,
    select_auc_safe_candidates,
    summarize_by_gamma,
)


def test_frozen_auc_safe_population_identity() -> None:
    path = Path(
        "$PRIS_ARCHIVE/"
        "next153_top2_attenuation_homotopy_search_v1/"
        "next153_top2_attenuation_homotopy_candidate_search.parquet"
    )
    selected = select_auc_safe_candidates(pd.read_parquet(path))
    digest = hashlib.sha256(
        "\n".join(selected["candidate_key"].astype(str)).encode()
    ).hexdigest()
    assert len(selected) == EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT == 22
    assert digest == EXPECTED_CANDIDATE_KEY_SHA256
    assert selected["top2_attenuation"].value_counts().sort_index().to_dict() == {
        0.0: 11,
        0.1: 11,
    }


def test_gamma_summary_uses_failed_count_before_shortfall() -> None:
    records = pd.DataFrame(
        [
            {"top2_attenuation": 0.0, "candidate_key": "a", "failed_constraint_count": 6, "normalized_shortfall_sum": 0.8, "best_threshold": 1.0},
            {"top2_attenuation": 0.0, "candidate_key": "b", "failed_constraint_count": 7, "normalized_shortfall_sum": 0.1, "best_threshold": 0.5},
            {"top2_attenuation": 0.1, "candidate_key": "c", "failed_constraint_count": 6, "normalized_shortfall_sum": 0.7, "best_threshold": 1.2},
        ]
    )
    summary = summarize_by_gamma(records)
    assert summary["gamma=0"]["closest_candidate_key"] == "a"
    assert summary["gamma=0.1"]["closest_candidate_key"] == "c"
    assert summary["gamma=0.1"]["minimum_normalized_shortfall_sum_at_best_count"] == 0.7

