from __future__ import annotations

import numpy as np

from src.next155_robust_contribution_transform_audit import (
    FIXED_DIRECTIONS,
    robust_transform_statistics,
)


def test_robust_transform_statistics_match_frozen_definitions() -> None:
    values = np.array([[0.0, 0.25, 1.0, 4.0]])
    stats = robust_transform_statistics(values)
    assert set(stats) == set(FIXED_DIRECTIONS)
    np.testing.assert_allclose(stats["sum_all"], [5.25])
    np.testing.assert_allclose(stats["sum_sqrt"], [3.5])
    np.testing.assert_allclose(stats["sum_log1p"], [np.log1p(values).sum()])
    np.testing.assert_allclose(stats["sum_tanh"], [np.tanh(values).sum()])
    np.testing.assert_allclose(stats["sum_rational"], [(values / (1.0 + values)).sum()])
    np.testing.assert_allclose(stats["sum_clip_0p25"], [0.75])
    np.testing.assert_allclose(stats["sum_clip_0p5"], [1.25])
    np.testing.assert_allclose(stats["sum_clip_1"], [2.25])
    np.testing.assert_allclose(stats["sum_clip_2"], [3.25])
    assert set(FIXED_DIRECTIONS.values()) == {-1}

