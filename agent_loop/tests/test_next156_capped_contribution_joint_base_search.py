from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next132_extended_coordination_protection_search as n132
from src.next156_capped_contribution_joint_base_search import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CANDIDATE_KEY_SHA256,
    build_candidate_specs,
    capped_contribution_sum,
    compose_capped_joint_score,
)


def test_capped_contribution_sum_caps_each_term_independently() -> None:
    values = np.array([[0.0, 0.25, 1.0, 4.0], [0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_allclose(capped_contribution_sum(values), [1.25, 1.0])


def test_capped_joint_score_preserves_support_and_active_policy() -> None:
    values = np.array([[0.0, 0.25, 1.0, 4.0], [0.1, np.nan, 0.3, 0.4]])
    support = np.array([[True, True, True, True], [True, False, True, True]])
    score, supported = compose_capped_joint_score(
        contributions=values,
        term_support=support,
        coordination_protection=np.array([0.5, np.nan]),
        coordination_active=np.array([True, False]),
        coordination_weight=1.0,
        packing_protection=np.array([1.0, 1.0]),
        packing_active=np.array([True, True]),
        packing_weight=0.25,
    )
    assert supported.tolist() == [True, False]
    np.testing.assert_allclose(score[0], 0.5)
    assert np.isnan(score[1])


def test_frozen_capped_candidate_universe_identity() -> None:
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
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 176
    assert digest == EXPECTED_CANDIDATE_KEY_SHA256
    assert {spec["aggregation"] for spec in specs} == {"sum_clip_each_0p5"}
    assert {spec["contribution_cap"] for spec in specs} == {0.5}

