from __future__ import annotations

import math

import pandas as pd
import pytest

from src.next126_hall_profile_persistence import (
    FEATURE_NAME,
    compute_hall_profile_persistence,
    materialize_hall_profile_persistence,
)


def test_profile_persistence_has_exact_step_semantics_and_bounds() -> None:
    assert compute_hall_profile_persistence(0.0, 0.0, 0.0, 0.0) == 0.0
    assert compute_hall_profile_persistence(0.0, 0.0, 0.0, 1.0) == 0.0
    assert compute_hall_profile_persistence(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)
    observed = compute_hall_profile_persistence(0.0, 0.25, 0.5, 1.0)
    assert observed == pytest.approx((0.15 * 0.25 + 0.25 * 0.5) / 0.45)
    assert 0.0 <= observed <= 1.0


@pytest.mark.parametrize(
    "profile",
    [
        (0.2, 0.1, 0.3, 0.4),
        (0.0, 0.1, 0.6, 0.5),
        (-0.1, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.1),
        (0.0, 0.0, math.nan, 1.0),
    ],
)
def test_invalid_profiles_fail_closed(profile) -> None:
    with pytest.raises(ValueError, match="Hall profile"):
        compute_hall_profile_persistence(*profile)


def test_materialization_preserves_support_and_is_fail_open() -> None:
    table = pd.DataFrame(
        {
            "material_id": ["a", "b", "c"],
            "mhcr_expanded_supported": [True, True, False],
            "mhcr_expanded_failure": [None, None, "unsupported"],
            "mhcr_expanded_negative_deficit_gain_tau05": [0.0, 0.2, float("nan")],
            "mhcr_expanded_negative_deficit_gain_tau10": [0.0, 0.2, float("nan")],
            "mhcr_expanded_negative_deficit_gain_tau25": [0.0, 0.2, float("nan")],
            "mhcr_expanded_negative_deficit_gain_tau50": [0.0, 0.4, float("nan")],
        }
    )
    result = materialize_hall_profile_persistence(table)
    assert result["material_id"].tolist() == ["a", "b", "c"]
    assert result["mhpp_supported"].tolist() == [True, True, False]
    assert result.loc[0, FEATURE_NAME] == 0.0
    assert result.loc[1, FEATURE_NAME] == pytest.approx(0.5)
    assert math.isnan(float(result.loc[2, FEATURE_NAME]))
