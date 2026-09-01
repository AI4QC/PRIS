import numpy as np

import src.next151_violation_multiplicity_audit as n151


def test_multiplicity_statistics_match_fixed_definitions() -> None:
    contributions = np.array(
        [
            [4.0, 1.0, 0.5, 0.0],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    stats = n151.multiplicity_statistics(contributions)

    assert np.allclose(stats["sum_all"], [5.5, 8.0])
    assert np.allclose(stats["max_one"], [4.0, 2.0])
    assert np.allclose(stats["second_largest"], [1.0, 2.0])
    assert np.allclose(stats["third_largest"], [0.5, 2.0])
    assert np.allclose(stats["top2_sum"], [5.0, 4.0])
    assert np.allclose(stats["top3_sum"], [5.5, 6.0])
    assert np.allclose(stats["sum_minus_max"], [1.5, 6.0])
    assert np.allclose(stats["sum_minus_top2"], [0.5, 4.0])
    assert np.allclose(stats["max_fraction"], [4.0 / 5.5, 0.25])
    assert np.allclose(stats["effective_violation_count"], [30.25 / 17.25, 4.0])
    assert stats["count_gt_0p1"].tolist() == [3.0, 4.0]
    assert stats["count_gt_1"].tolist() == [1.0, 4.0]


def test_fixed_directions_encode_single_dominant_vs_distributed_hypothesis() -> None:
    assert n151.FIXED_DIRECTIONS["max_fraction"] == 1
    assert all(
        direction == -1
        for name, direction in n151.FIXED_DIRECTIONS.items()
        if name != "max_fraction"
    )
