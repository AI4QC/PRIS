from __future__ import annotations

import math

import pandas as pd
import pytest

from src.next129_coordination_protection import (
    CLIP_NORMALIZED,
    FEATURE_NAME,
    compute_coordination_protection,
    materialize_coordination_protection,
)


def test_protection_is_zero_below_center_and_bounded_above() -> None:
    assert compute_coordination_protection(0.0) == 0.0
    assert compute_coordination_protection(math.expm1(2.1671471220989416)) == pytest.approx(0.0)
    assert 0.0 < compute_coordination_protection(12.0) <= CLIP_NORMALIZED
    assert compute_coordination_protection(1.0e9) == pytest.approx(CLIP_NORMALIZED)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_invalid_coordination_is_unsupported(value: float) -> None:
    assert compute_coordination_protection(value) is None


def test_materialization_is_fail_open_for_invalid_rows() -> None:
    table = pd.DataFrame(
        {"material_id": ["a", "b", "c"], "cov_coord110_mean": [12.0, 0.0, math.nan]}
    )
    result = materialize_coordination_protection(table)
    assert result["coordination_protection_supported"].tolist() == [True, True, False]
    assert result.loc[0, FEATURE_NAME] > 0.0
    assert result.loc[1, FEATURE_NAME] == 0.0
    assert math.isnan(float(result.loc[2, FEATURE_NAME]))
