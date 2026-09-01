from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.next130_coordination_protection_search as n130
import src.next132_extended_coordination_protection_search as n132
from src.next163_interior_family_attenuation_search import (
    DOMINANT_FAMILY_ATTENUATIONS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CANDIDATE_KEY_SHA256,
    _verify_endpoint_reproduction,
    build_candidate_specs,
    compose_family_dominance_attenuation_score,
    family_dominance_attenuated_sum,
)


TERM_IDS = [
    "cov_demo",
    "scbv_demo",
    "cmvo_demo",
    "bvtbd_demo",
    "mhcr_demo",
]


def test_interior_attenuation_values_share_the_frozen_formula() -> None:
    values = np.array([[0.0, 1.0, 0.25, 0.4, 0.5]])
    expected = {0.01: 1.395, 0.025: 1.3875, 0.05: 1.375, 0.075: 1.3625}
    assert set(DOMINANT_FAMILY_ATTENUATIONS) == set(expected)
    for gamma, value in expected.items():
        np.testing.assert_allclose(
            family_dominance_attenuated_sum(values, TERM_IDS, gamma), [value]
        )


def test_interior_score_preserves_support_and_active_policy() -> None:
    values = np.array(
        [[0.0, 1.0, 0.25, 0.4, 0.5], [0.1, np.nan, 0.3, 0.4, 0.5]]
    )
    support = np.array(
        [[True, True, True, True, True], [True, False, True, True, True]]
    )
    score, supported = compose_family_dominance_attenuation_score(
        contributions=values,
        term_support=support,
        term_ids=TERM_IDS,
        attenuation=0.05,
        coordination_protection=np.array([0.5, np.nan]),
        coordination_active=np.array([True, False]),
        coordination_weight=1.0,
        packing_protection=np.array([1.0, 1.0]),
        packing_active=np.array([True, True]),
        packing_weight=0.25,
    )
    assert supported.tolist() == [True, False]
    np.testing.assert_allclose(score[0], 0.625)
    assert np.isnan(score[1])


def test_frozen_interior_candidate_universe_identity() -> None:
    root = Path("$PRIS_ARCHIVE/")
    all_bases = n130.n127.select_next125_bases(
        pd.read_parquet(
            root / "next125_mhcr_frontier_rescue_v1" / n130.n125.SEARCH_NAME
        )
    )
    bases = n132.select_extended_bases(
        pd.read_parquet(
            root
            / "next130_coordination_protection_search_v1"
            / n130.SEARCH_NAME
        ),
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
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 704
    assert digest == EXPECTED_CANDIDATE_KEY_SHA256
    assert {spec["dominant_family_attenuation"] for spec in specs} == set(
        DOMINANT_FAMILY_ATTENUATIONS
    )


def test_endpoint_reproduction_treats_none_and_nan_as_undefined() -> None:
    record = {
        "candidate_key": "x",
        "scigen_pooled_auc": 0.8,
        "scigen_macro_auc": 0.7,
        "scigen_worst_auc": 0.6,
        "wyformer_pooled_auc": 0.8,
        "wyformer_macro_auc": 0.7,
        "wyformer_worst_auc": 0.6,
        "safe_threshold": None,
        "passes_source_auc_gates": True,
        "passes_safe_all_cells": False,
        "passes_broad_all_cells": False,
        "passes_all_discovery_gates": False,
    }
    published = pd.DataFrame([{**record, "safe_threshold": np.nan}] * 176)
    published["candidate_key"] = ["x"] + [f"unused-{i}" for i in range(175)]
    rerun = [record] + [
        {**record, "candidate_key": f"unused-{i}"} for i in range(175)
    ]
    _verify_endpoint_reproduction(
        rerun=rerun, published=published, endpoint="synthetic"
    )
