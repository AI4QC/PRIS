from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next212_two_signal_risk_lift_search import (
    EXPECTED_CANDIDATE_COUNT,
    anchored_two_signal_score,
    build_candidate_specs,
    materialize_two_signal_candidates,
    run_two_signal_risk_lift_search,
)
from src.next87_scigen_sparse_law_search import _term_risk


def _anchor() -> dict[str, object]:
    return {
        "candidate_key": "anchor-key",
        "feature": "primary",
        "hypothesis": "primary__protected_low",
        "direction": "protected_low",
        "q_lo": 0.0,
        "q_hi": 2.0,
        "amplitude_fraction": 1 / 16,
        "amplitude_numerator": 1,
        "amplitude_denominator": 16,
        "risk_scale": 0.4,
        "residual_threshold": 0.2,
    }


def _secondary_specs() -> list[dict[str, object]]:
    return [
        {
            "candidate_key": "ignored-a",
            "feature": "primary",
            "hypothesis": "primary__protected_low",
            "direction": "protected_low",
            "q_lo": 0.0,
            "q_hi": 2.0,
        },
        {
            "candidate_key": "ignored-b",
            "feature": "secondary",
            "hypothesis": "secondary__protected_high",
            "direction": "protected_high",
            "q_lo": 1.0,
            "q_hi": 3.0,
        },
    ]


def test_secondary_lift_is_nonnegative_thresholded_missing_safe_and_support_preserving() -> None:
    anchor_score = np.array([0.1, 0.3, 0.4, 0.5])
    activation_score = np.array([0.1, 0.3, 0.4, 0.5])
    support = np.array([True, True, True, False])
    values = np.array([1.0, 1.0, np.nan, 3.0])
    score, got_support, active = anchored_two_signal_score(
        anchor_score=anchor_score,
        activation_score=activation_score,
        base_support=support,
        feature_values=values,
        direction="protected_high",
        q_lo=1.0,
        q_hi=3.0,
        residual_threshold=0.2,
        amplitude_fraction=0.5,
        risk_scale=0.4,
    )
    np.testing.assert_allclose(score, [0.1, 0.5, 0.4, 0.5])
    np.testing.assert_array_equal(got_support, support)
    np.testing.assert_array_equal(active, [False, True, False, False])


def test_candidate_grid_excludes_anchor_hypothesis_and_is_deterministic() -> None:
    specs = build_candidate_specs(
        anchor_spec=_anchor(),
        next210_specs=_secondary_specs(),
        amplitude_fractions=(1 / 16, 1 / 8),
    )
    assert len(specs) == 3
    assert specs[0]["secondary_hypothesis"] is None
    assert [spec["secondary_hypothesis"] for spec in specs[1:]] == [
        "secondary__protected_high",
        "secondary__protected_high",
    ]
    assert [spec["secondary_amplitude_numerator"] for spec in specs[1:]] == [1, 2]
    assert len({str(spec["candidate_key"]) for spec in specs}) == len(specs)
    assert EXPECTED_CANDIDATE_COUNT == 216


def test_virtual_term_round_trip_reproduces_materialized_scores() -> None:
    features = pd.DataFrame(
        {
            "primary": [0.0, 1.0, 2.0],
            "secondary": [1.0, 2.0, 3.0],
        }
    )
    anchor_score = np.array([0.1, 0.3, 0.4])
    activation_score = anchor_score.copy()
    support = np.ones(3, dtype=bool)
    specs = build_candidate_specs(
        anchor_spec=_anchor(),
        next210_specs=_secondary_specs(),
        amplitude_fractions=(1 / 16,),
    )
    virtual, terms, runtime = materialize_two_signal_candidates(
        features=features,
        anchor_score=anchor_score,
        activation_score=activation_score,
        base_support=support,
        specs=specs,
    )
    assert len(terms) == len(runtime) == 2
    assert [json.loads(str(spec["candidate_key"]))["secondary_hypothesis"] for spec in specs] == [
        None,
        "secondary__protected_high",
    ]
    for spec, term in zip(specs, terms, strict=True):
        values = (
            np.full(3, np.nan)
            if spec["secondary_feature"] is None
            else features[str(spec["secondary_feature"])].to_numpy(float)
        )
        expected, expected_support, _ = anchored_two_signal_score(
            anchor_score=anchor_score,
            activation_score=activation_score,
            base_support=support,
            feature_values=values,
            direction=spec["secondary_direction"],
            q_lo=spec["secondary_q_lo"],
            q_hi=spec["secondary_q_hi"],
            residual_threshold=float(spec["residual_threshold"]),
            amplitude_fraction=float(spec["secondary_amplitude_fraction"]),
            risk_scale=float(spec["risk_scale"]),
        )
        score, got_support = _term_risk(virtual, term)
        np.testing.assert_allclose(score[got_support], expected[expected_support])
        np.testing.assert_array_equal(got_support, expected_support)


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_two_signal_risk_lift_search).parameters)
    assert "next211_dir" in parameters and "next210_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_two_signal_risk_lift_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT212 input is missing"):
        run_two_signal_risk_lift_search(**kwargs)
