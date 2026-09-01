from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next208_residual_x0_exception_search import (
    BOUNDARY_FLAGS,
    EXCEPTION_FRACTIONS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    build_candidate_specs,
    empirical_exception_cutoff,
    materialize_residual_x0_exception_candidates,
    residual_x0_exception_score,
    run_residual_x0_exception_search,
)


def test_empirical_cutoff_is_endpoint_free_and_uses_inverted_cdf() -> None:
    parameters = tuple(inspect.signature(empirical_exception_cutoff).parameters)
    assert parameters == ("values", "direction", "exception_fraction")
    values = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    assert empirical_exception_cutoff(values, "protected_low", 0.5) == 1.0
    assert empirical_exception_cutoff(values, "protected_high", 0.25) == 2.0
    with pytest.raises(ValueError, match="NEXT208 protection direction differs"):
        empirical_exception_cutoff(values, "unknown", 0.5)


def test_exception_score_is_fixed_zero_fail_open_and_support_preserving() -> None:
    score, support, active = residual_x0_exception_score(
        base_score=[0.10, 0.20, 0.30, 0.40, 0.50],
        base_support=[True, True, True, True, False],
        feature_values=[0.0, 0.2, np.nan, 0.1, 0.0],
        direction="protected_low",
        cutoff=0.2,
        residual_threshold=0.2,
    )
    assert support.tolist() == [True, True, True, True, False]
    assert active.tolist() == [False, True, False, True, False]
    np.testing.assert_allclose(score, [0.10, 0.0, 0.30, 0.0, 0.50])

    high, _, high_active = residual_x0_exception_score(
        base_score=[0.20, 0.30, 0.40],
        base_support=[True, True, True],
        feature_values=[1.0, 2.0, 3.0],
        direction="protected_high",
        cutoff=2.0,
        residual_threshold=0.2,
    )
    assert high_active.tolist() == [False, True, True]
    np.testing.assert_allclose(high, [0.20, 0.0, 0.0])


def test_candidate_grid_is_exact_deterministic_and_records_realized_cutoffs() -> None:
    assert EXCEPTION_FRACTIONS == tuple(k / 16 for k in range(1, 16))
    features = pd.DataFrame(
        {
            "low": [0.0, 1.0, 2.0, 3.0],
            "high": [10.0, 20.0, 30.0, 40.0],
        }
    )
    kwargs = {
        "base_candidate_key": "base",
        "eligible_hypotheses": (
            "low__protected_low",
            "high__protected_high",
        ),
        "features": features,
        "base_score": [0.0, 1.0, 2.0, 3.0],
        "base_support": [True, True, True, True],
        "residual_threshold": 1.0,
        "exception_fractions": (0.25, 0.5),
    }
    first = build_candidate_specs(**kwargs)
    second = build_candidate_specs(**kwargs)
    assert first == second
    assert len(first) == 1 + 2 * 2
    assert first[0]["feature"] is None
    assert len({spec["candidate_key"] for spec in first}) == len(first)
    payloads = [json.loads(spec["candidate_key"]) for spec in first[1:]]
    assert all(payload["residual_threshold"] == 1.0 for payload in payloads)
    assert all(payload["score_composition"].endswith("else_keep_base") for payload in payloads)
    low_half = next(
        payload
        for payload in payloads
        if payload["feature"] == "low" and payload["exception_fraction_numerator"] == 8
    )
    assert low_half["cutoff"] == 2.0


def test_materializer_round_trips_every_corrected_score() -> None:
    features = pd.DataFrame(
        {"material_id": ["a", "b", "c"], "x": [0.0, 1.0, np.nan]}
    )
    specs = build_candidate_specs(
        base_candidate_key="base",
        eligible_hypotheses=("x__protected_low",),
        features=features,
        base_score=[0.2, 0.4, 0.6],
        base_support=[True, True, True],
        residual_threshold=0.2,
        exception_fractions=(0.5,),
    )
    extended, terms, runtime = materialize_residual_x0_exception_candidates(
        features=features,
        base_score=[0.2, 0.4, 0.6],
        base_support=[True, True, True],
        specs=specs,
    )
    assert len(terms) == len(runtime) == len(specs) == 2
    for spec, term in zip(specs, terms, strict=True):
        encoded = extended[term["feature"]].to_numpy(float)
        recovered = np.arcsinh(encoded) / float(term["scale"])
        expected, expected_support, _ = residual_x0_exception_score(
            base_score=[0.2, 0.4, 0.6],
            base_support=[True, True, True],
            feature_values=(
                [np.nan, np.nan, np.nan]
                if spec["feature"] is None
                else features[str(spec["feature"])]
            ),
            direction=spec["direction"],
            cutoff=spec["cutoff"],
            residual_threshold=spec["residual_threshold"],
        )
        np.testing.assert_allclose(recovered[expected_support], expected[expected_support])


def test_formal_constants_and_boundary_flags_are_exact() -> None:
    assert EXPECTED_ELIGIBLE_COUNT == 44
    assert EXPECTED_CANDIDATE_COUNT == 1 + 44 * 15 == 661
    assert BOUNDARY_FLAGS == {
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "opened_validation_outputs_used": False,
        "scigen_replication_endpoint_opened": False,
        "wyformer_replication_endpoint_opened": False,
    }


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_residual_x0_exception_search).parameters)
    assert "next206_dir" in parameters and "next207_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_residual_x0_exception_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT208 input is missing"):
        run_residual_x0_exception_search(**kwargs)
