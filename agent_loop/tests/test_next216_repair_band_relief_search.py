from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next216_repair_band_relief_search import (
    AMPLITUDE_FRACTIONS,
    EXPECTED_CANDIDATE_COUNT,
    bounded_directional_protection,
    build_candidate_specs,
    repair_band_relief_score,
    robust_protection_cutoffs,
    run_repair_band_relief_search,
)


def test_robust_cutoffs_use_endpoint_blind_inverted_cdf_quantiles() -> None:
    values = np.arange(16, dtype=float)
    values[3] = np.nan
    finite = values[np.isfinite(values)]
    assert robust_protection_cutoffs(values) == (
        float(np.quantile(finite, 1 / 16, method="inverted_cdf")),
        float(np.quantile(finite, 15 / 16, method="inverted_cdf")),
    )
    with pytest.raises(ValueError, match="NEXT216 robust protection cutoffs"):
        robust_protection_cutoffs([1.0, 1.0, np.nan])


def test_bounded_certificate_has_exact_protection_directions() -> None:
    values = np.array([0.0, 5.0, 10.0, np.nan])
    np.testing.assert_allclose(
        bounded_directional_protection(values, "protected_high", 0.0, 10.0)[:3],
        [0.0, 0.5, 1.0],
    )
    np.testing.assert_allclose(
        bounded_directional_protection(values, "protected_low", 0.0, 10.0)[:3],
        [1.0, 0.5, 0.0],
    )
    assert np.isnan(
        bounded_directional_protection(values, "protected_low", 0.0, 10.0)[3]
    )


def test_relief_is_exactly_lower_inclusive_upper_exclusive() -> None:
    lower = 0.2
    upper = 0.6
    score = np.array(
        [np.nextafter(lower, -np.inf), lower, 0.4, np.nextafter(upper, -np.inf), upper]
    )
    feature = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    corrected, support, active = repair_band_relief_score(
        base_score=score,
        base_support=np.ones(5, dtype=bool),
        feature_values=feature,
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        lower=lower,
        upper=upper,
        amplitude_fraction=0.5,
    )
    np.testing.assert_array_equal(support, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(active, [False, True, False, True, False])
    assert corrected[0] == score[0]
    assert corrected[1] == pytest.approx(score[1] * 0.5)
    assert corrected[2] == score[2]
    assert corrected[3] == pytest.approx(score[3] * 0.5)
    assert corrected[4] == score[4]


def test_candidate_specs_include_one_base_and_exact_feature_amplitudes() -> None:
    features = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0)[::-1]})
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.6, 0.7])
    specs = build_candidate_specs(
        base_candidate_key="next214-final",
        eligible_hypotheses=("a__protected_high", "b__protected_low"),
        features=features,
        base_score=score,
        base_support=np.ones(8, dtype=bool),
        lower=0.2,
        upper=0.6,
        amplitude_fractions=(1 / 16, 1 / 8),
    )
    assert len(specs) == 5
    assert specs[0]["hypothesis"] is None
    assert {spec["hypothesis"] for spec in specs[1:]} == {
        "a__protected_high",
        "b__protected_low",
    }
    assert len({str(spec["candidate_key"]) for spec in specs}) == len(specs)
    assert AMPLITUDE_FRACTIONS == (1 / 16, 1 / 8, 1 / 4, 1 / 2)
    assert EXPECTED_CANDIDATE_COUNT == 89


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_repair_band_relief_search).parameters)
    assert "next215_dir" in parameters and "next214_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_repair_band_relief_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT216 input is missing"):
        run_repair_band_relief_search(**kwargs)
