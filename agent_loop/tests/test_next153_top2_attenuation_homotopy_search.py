from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next132_extended_coordination_protection_search as n132
from src.next153_top2_attenuation_homotopy_search import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CANDIDATE_KEY_SHA256,
    build_candidate_specs,
    compose_homotopy_score,
    top2_attenuated_sum,
)


def test_top2_attenuation_has_exact_frozen_endpoints() -> None:
    values = np.array([[1.0, 4.0, 2.0, 3.0], [5.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(top2_attenuated_sum(values, 0.0), [10.0, 5.0])
    np.testing.assert_allclose(top2_attenuated_sum(values, 0.5), [6.5, 2.5])
    np.testing.assert_allclose(top2_attenuated_sum(values, 1.0), [3.0, 0.0])


def test_homotopy_score_preserves_base_support() -> None:
    contributions = np.array(
        [[5.0, 4.0, 3.0, 2.0], [4.0, 3.0, np.nan, 1.0], [5.0, 2.0, 1.0, 0.0]]
    )
    term_support = np.array(
        [[True, True, True, True], [True, True, False, True], [True, True, True, True]]
    )
    score, supported = compose_homotopy_score(
        contributions=contributions,
        term_support=term_support,
        gamma=0.5,
        coordination_protection=np.array([1.0, np.nan, 2.0]),
        coordination_active=np.array([True, False, False]),
        packing_protection=np.array([0.5, 1.0, 1.0]),
        packing_active=np.array([False, True, True]),
    )
    assert supported.tolist() == [True, False, True]
    np.testing.assert_allclose(score[[0, 2]], [8.5, 4.0])
    assert np.isnan(score[1])


def test_frozen_homotopy_candidate_universe_identity() -> None:
    root = Path("$PRIS_ARCHIVE/")
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(root / "next125_mhcr_frontier_rescue_v1" / n130.n125.SEARCH_NAME)
    )
    bases = n132.select_extended_bases(
        pd.read_parquet(root / "next130_coordination_protection_search_v1" / n130.SEARCH_NAME),
        all_bases,
    )
    physical_ids = {
        term_id
        for value in bases["term_ids_json"]
        for term_id in json.loads(str(value))
    }
    specs = build_candidate_specs(bases=bases, physical_term_ids=physical_ids)
    digest = hashlib.sha256(
        "\n".join(str(spec["candidate_key"]) for spec in specs).encode()
    ).hexdigest()
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 66
    assert digest == EXPECTED_CANDIDATE_KEY_SHA256
    assert sorted({spec["top2_attenuation"] for spec in specs}) == [
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
