from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next210_residual_risk_lift_search import (
    AMPLITUDE_FRACTIONS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    RISK_SCALE,
    bounded_directional_risk,
    build_candidate_specs,
    materialize_residual_risk_lift_candidates,
    residual_risk_lift_score,
    robust_risk_cutoffs,
    run_residual_risk_lift_search,
)


def test_robust_cutoffs_are_endpoint_free_and_use_inverted_cdf() -> None:
    assert tuple(inspect.signature(robust_risk_cutoffs).parameters) == ("values",)
    assert robust_risk_cutoffs([0.0, 1.0, 2.0, 3.0, np.nan]) == (0.0, 3.0)
    with pytest.raises(ValueError, match="NEXT210 robust risk cutoffs are degenerate"):
        robust_risk_cutoffs([1.0, 1.0, np.nan])


def test_directional_risk_is_bounded_and_has_exact_semantics() -> None:
    values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, np.nan])
    low = bounded_directional_risk(values, "protected_low", 0.0, 2.0)
    high = bounded_directional_risk(values, "protected_high", 0.0, 2.0)
    np.testing.assert_allclose(low[:5], [0.0, 0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(high[:5], [1.0, 1.0, 0.5, 0.0, 0.0])
    assert np.isnan(low[-1]) and np.isnan(high[-1])
    with pytest.raises(ValueError, match="NEXT210 protection direction differs"):
        bounded_directional_risk(values, "unknown", 0.0, 2.0)


def test_risk_lift_is_nonnegative_thresholded_and_support_preserving() -> None:
    score, support, active = residual_risk_lift_score(
        base_score=[0.1, 0.2, 0.3, 0.4, 0.5],
        base_support=[True, True, True, True, False],
        feature_values=[0.0, 1.0, 2.0, np.nan, 3.0],
        direction="protected_low",
        q_lo=0.0,
        q_hi=2.0,
        residual_threshold=0.2,
        amplitude_fraction=0.5,
        risk_scale=0.4,
    )
    assert support.tolist() == [True, True, True, True, False]
    assert active.tolist() == [False, True, True, False, False]
    np.testing.assert_allclose(score, [0.1, 0.3, 0.5, 0.4, 0.5])
    assert np.all(score >= np.array([0.1, 0.2, 0.3, 0.4, 0.5]))


def test_candidate_grid_is_deterministic_and_records_normalization() -> None:
    assert AMPLITUDE_FRACTIONS == (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
    features = pd.DataFrame(
        {"low": np.arange(32.0), "high": np.arange(100.0, 132.0)}
    )
    kwargs = {
        "base_candidate_key": "base",
        "eligible_hypotheses": (
            "low__protected_low",
            "high__protected_high",
        ),
        "features": features,
        "base_score": np.arange(32.0),
        "base_support": np.ones(32, dtype=bool),
        "residual_threshold": 1.0,
        "amplitude_fractions": (0.25, 0.5),
        "risk_scale": 0.4,
    }
    first = build_candidate_specs(**kwargs)
    assert first == build_candidate_specs(**kwargs)
    assert len(first) == 1 + 2 * 2
    assert first[0]["feature"] is None
    payloads = [json.loads(spec["candidate_key"]) for spec in first[1:]]
    assert all(payload["q_hi"] > payload["q_lo"] for payload in payloads)
    assert all(payload["risk_scale"] == 0.4 for payload in payloads)
    assert {payload["amplitude_numerator"] for payload in payloads} == {4, 8}


def test_materializer_round_trips_every_lifted_score() -> None:
    features = pd.DataFrame(
        {"material_id": ["a", "b", "c"], "x": [0.0, 1.0, 2.0]}
    )
    specs = build_candidate_specs(
        base_candidate_key="base",
        eligible_hypotheses=("x__protected_low",),
        features=features,
        base_score=[0.2, 0.4, 0.6],
        base_support=[True, True, True],
        residual_threshold=0.2,
        amplitude_fractions=(0.5,),
        risk_scale=0.4,
    )
    extended, terms, runtime = materialize_residual_risk_lift_candidates(
        features=features,
        base_score=[0.2, 0.4, 0.6],
        base_support=[True, True, True],
        specs=specs,
    )
    assert len(terms) == len(runtime) == len(specs) == 2
    for spec, term in zip(specs, terms, strict=True):
        recovered = (
            np.arcsinh(extended[term["feature"]].to_numpy(float))
            / float(term["scale"])
        )
        expected, expected_support, _ = residual_risk_lift_score(
            base_score=[0.2, 0.4, 0.6],
            base_support=[True, True, True],
            feature_values=(
                [np.nan, np.nan, np.nan]
                if spec["feature"] is None
                else features[str(spec["feature"])]
            ),
            direction=spec["direction"],
            q_lo=spec["q_lo"],
            q_hi=spec["q_hi"],
            residual_threshold=spec["residual_threshold"],
            amplitude_fraction=spec["amplitude_fraction"],
            risk_scale=spec["risk_scale"],
        )
        np.testing.assert_allclose(recovered[expected_support], expected[expected_support])


def test_formal_constants_and_interface_are_exact_and_sealed() -> None:
    assert EXPECTED_ELIGIBLE_COUNT == 44
    assert EXPECTED_CANDIDATE_COUNT == 1 + 44 * 5 == 221
    assert RISK_SCALE == pytest.approx(0.5415470292150686 - 0.21976295573076796)
    parameters = tuple(inspect.signature(run_residual_risk_lift_search).parameters)
    assert "next209_dir" in parameters and "next208_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_residual_risk_lift_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT210 input is missing"):
        run_residual_risk_lift_search(**kwargs)
