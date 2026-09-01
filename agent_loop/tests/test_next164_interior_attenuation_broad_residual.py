from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.next164_interior_attenuation_broad_residual import (
    EXPECTED_CANDIDATE_KEY_SHA256,
    EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT,
    select_auc_safe_candidates,
)


def test_frozen_next163_auc_safe_population_identity() -> None:
    path = Path(
        "$PRIS_ARCHIVE/"
        "next163_interior_family_attenuation_search_v1/"
        "next163_interior_family_attenuation_candidate_search.parquet"
    )
    selected = select_auc_safe_candidates(pd.read_parquet(path))
    digest = hashlib.sha256(
        "\n".join(selected["candidate_key"].astype(str)).encode()
    ).hexdigest()
    assert len(selected) == EXPECTED_DIAGNOSTIC_CANDIDATE_COUNT == 12
    assert digest == EXPECTED_CANDIDATE_KEY_SHA256
    assert (
        selected["dominant_family_attenuation"]
        .value_counts()
        .sort_index()
        .to_dict()
        == {0.01: 3, 0.025: 3, 0.05: 3, 0.075: 3}
    )
