from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next87_scigen_sparse_law_search import _term_risk
from src.next175_weighted_rigidity_repair_width_search import (
    ATTENUATIONS,
    BROAD_THRESHOLD,
    ELIGIBLE_FEATURES,
    EXPECTED_CANDIDATE_COUNT,
    REPAIR_WIDTH,
    SAFE_THRESHOLD,
    build_candidate_specs,
    interval_width_protect_score,
    materialize_interval_width_candidates,
    run_weighted_rigidity_repair_width_search,
)


def test_interval_width_operator_has_exact_frozen_bounds_and_amplitude() -> None:
    base = np.asarray(
        [
            BROAD_THRESHOLD - 1.0e-12,
            BROAD_THRESHOLD,
            0.35,
            SAFE_THRESHOLD - 1.0e-12,
            SAFE_THRESHOLD,
        ]
    )
    support = np.ones(len(base), dtype=bool)
    feature = np.ones(len(base))
    score, corrected_support, active = interval_width_protect_score(
        base_score=base,
        base_support=support,
        feature=feature,
        attenuation=0.40,
    )
    np.testing.assert_array_equal(active, [False, True, True, True, False])
    np.testing.assert_allclose(
        score[active], np.maximum(0.0, base[active] - 0.40 * REPAIR_WIDTH)
    )
    np.testing.assert_array_equal(score[~active], base[~active])
    np.testing.assert_array_equal(corrected_support, support)


def test_operator_is_bounded_and_missing_or_unsupported_keeps_base() -> None:
    base = np.asarray([0.10, 0.25, 0.40, 0.50, 0.60, 0.30])
    support = np.asarray([True, True, True, True, True, False])
    feature = np.asarray([1.0, 0.0, 0.5, 1.0, 1.0, np.nan])
    corrected, corrected_support, active = interval_width_protect_score(
        base_score=base,
        base_support=support,
        feature=feature,
        attenuation=1.0,
    )
    np.testing.assert_array_equal(active, [False, True, True, True, False, False])
    assert np.all(corrected >= 0.0)
    assert np.all(corrected <= base)
    assert corrected[0] == base[0]
    assert corrected[4] == base[4]
    assert corrected[5] == base[5]
    np.testing.assert_array_equal(corrected_support, support)


def test_candidate_universe_is_exactly_one_base_plus_five_by_eight() -> None:
    assert ATTENUATIONS == (0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)
    assert ELIGIBLE_FEATURES == (
        "pwldr_crystalnn_tightness_min",
        "pwldr_crystalnn_tightness_q10",
        "pwldr_crystalnn_tightness_mean",
        "pwldr_crystalnn_volume_q10",
        "pwldr_crystalnn_volume_mean",
    )
    specs = build_candidate_specs(base_candidate_key="frozen-base")
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 41
    assert len({spec["candidate_key"] for spec in specs}) == 41
    for spec in specs:
        assert json.dumps(
            json.loads(spec["candidate_key"]), sort_keys=True, separators=(",", ":")
        ) == spec["candidate_key"]


def test_materialized_terms_recover_every_exact_corrected_score() -> None:
    features = pd.DataFrame(
        {name: [0.0, 0.5, 1.0, np.nan] for name in ELIGIBLE_FEATURES}
    )
    base_score = np.asarray([0.10, 0.30, 0.50, 0.80])
    base_support = np.asarray([True, True, True, False])
    specs = build_candidate_specs(base_candidate_key="frozen-base")
    frame, terms, runtime = materialize_interval_width_candidates(
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
            if spec["weighted_rigidity_feature"] is None
            else frame[str(spec["weighted_rigidity_feature"])].to_numpy(float)
        )
        expected, expected_support, _ = interval_width_protect_score(
            base_score=base_score,
            base_support=base_support,
            feature=raw_feature,
            attenuation=float(spec["attenuation"]),
        )
        np.testing.assert_allclose(recovered[base_support], expected[base_support])
        np.testing.assert_array_equal(recovered_support, expected_support)


def test_formal_interface_has_only_discovery_endpoints() -> None:
    parameters = tuple(
        inspect.signature(run_weighted_rigidity_repair_width_search).parameters
    )
    assert "next173_dir" in parameters and "next174_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_weighted_rigidity_repair_width_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"search_workers", "require_formal_inputs"}
    }
    kwargs["search_workers"] = 1
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT175 input is missing"):
        run_weighted_rigidity_repair_width_search(**kwargs)
