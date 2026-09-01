from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next87_scigen_sparse_law_search import _term_risk

from src.next170_local_directional_rigidity_attenuation_search import (
    ATTENUATIONS,
    ELIGIBLE_FEATURES,
    EXPECTED_CANDIDATE_COUNT,
    attenuate_score,
    build_candidate_specs,
    materialize_attenuation_candidates,
    run_local_directional_rigidity_attenuation_search,
)


def test_candidate_universe_is_exactly_one_base_plus_five_by_eight() -> None:
    assert ELIGIBLE_FEATURES == (
        "pldr_crystalnn_tightness_min",
        "pldr_crystalnn_tightness_q10",
        "pldr_crystalnn_tightness_mean",
        "pldr_crystalnn_volume_q10",
        "pldr_crystalnn_volume_mean",
    )
    assert ATTENUATIONS == (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)
    specs = build_candidate_specs(base_candidate_key="frozen-base")
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 41
    assert len({spec["candidate_key"] for spec in specs}) == 41
    base = [spec for spec in specs if spec["directional_rigidity_feature"] is None]
    assert len(base) == 1
    assert base[0]["attenuation"] == 0.0
    assert {
        (spec["directional_rigidity_feature"], spec["attenuation"])
        for spec in specs
        if spec["directional_rigidity_feature"] is not None
    } == {(feature, alpha) for feature in ELIGIBLE_FEATURES for alpha in ATTENUATIONS}


def test_candidate_keys_are_canonical_and_deterministic() -> None:
    left = build_candidate_specs(base_candidate_key="frozen-base")
    right = build_candidate_specs(base_candidate_key="frozen-base")
    assert left == right
    for spec in left:
        assert json.dumps(json.loads(spec["candidate_key"]), sort_keys=True, separators=(",", ":")) == spec["candidate_key"]


def test_attenuation_is_monotone_bounded_and_keeps_missing_feature_at_base() -> None:
    base = np.asarray([0.0, 0.2, 0.5, 1.0, 0.7])
    support = np.asarray([True, True, True, True, False])
    feature = np.asarray([0.0, 0.25, 0.5, 1.0, np.nan])
    unchanged, unchanged_support = attenuate_score(
        base_score=base, base_support=support, feature=feature, attenuation=0.0
    )
    score, corrected_support = attenuate_score(
        base_score=base, base_support=support, feature=feature, attenuation=0.4
    )
    np.testing.assert_array_equal(unchanged, base)
    np.testing.assert_array_equal(unchanged_support, support)
    np.testing.assert_array_equal(corrected_support, support)
    np.testing.assert_allclose(score[:4], base[:4] * (1.0 - 0.4 * feature[:4]))
    assert score[4] == base[4]
    assert np.all(score >= 0.0)
    assert np.all(score <= base)


def test_materialized_virtual_terms_recover_every_exact_corrected_score() -> None:
    features = pd.DataFrame(
        {
            name: [0.0, 0.5, 1.0, np.nan]
            for name in ELIGIBLE_FEATURES
        }
    )
    base_score = np.asarray([0.1, 0.2, 0.5, 0.8])
    base_support = np.asarray([True, True, True, False])
    specs = build_candidate_specs(base_candidate_key="frozen-base")
    frame, terms, runtime = materialize_attenuation_candidates(
        features=features,
        base_score=base_score,
        base_support=base_support,
        specs=specs,
    )
    assert len(terms) == len(runtime) == EXPECTED_CANDIDATE_COUNT
    for spec, term in zip(specs, terms, strict=True):
        recovered, recovered_support = _term_risk(frame, term)
        raw_feature = (
            np.full(len(frame), np.nan)
            if spec["directional_rigidity_feature"] is None
            else frame[str(spec["directional_rigidity_feature"])].to_numpy(float)
        )
        expected, expected_support = attenuate_score(
            base_score=base_score,
            base_support=base_support,
            feature=raw_feature,
            attenuation=float(spec["attenuation"]),
        )
        np.testing.assert_allclose(recovered[base_support], expected[base_support])
        np.testing.assert_array_equal(recovered_support, expected_support)


def test_formal_search_interface_has_no_validation_or_replication_endpoint() -> None:
    parameters = tuple(
        inspect.signature(run_local_directional_rigidity_attenuation_search).parameters
    )
    assert "next168_dir" in parameters
    assert "next169_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_before_opening_any_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_local_directional_rigidity_attenuation_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"search_workers", "require_formal_inputs"}
    }
    kwargs["search_workers"] = 1
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT170 input is missing"):
        run_local_directional_rigidity_attenuation_search(**kwargs)
