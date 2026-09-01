from __future__ import annotations

import numpy as np

from src.next157_mechanism_family_consensus_audit import (
    FAMILY_PREFIXES,
    FIXED_DIRECTIONS,
    mechanism_family_statistics,
)


def test_mechanism_family_statistics_equalize_family_multiplicity() -> None:
    term_ids = [
        "cov_example__high",
        "scbv_example__high",
        "cmvo_example__high",
        "bvtbd_example__high",
        "mhcr_example__high",
    ]
    values = np.array([[1.0, 3.0, 2.0, 4.0, 5.0]])
    stats = mechanism_family_statistics(values, term_ids)
    assert set(stats) == set(FIXED_DIRECTIONS)
    assert set(FAMILY_PREFIXES) == {
        "local_geometry",
        "charge_flow_feasibility",
        "valence_transport",
        "contact_robustness",
    }
    np.testing.assert_allclose(stats["family_mean_sum"], [13.0])
    np.testing.assert_allclose(stats["family_max_sum"], [14.0])
    np.testing.assert_allclose(stats["family_max_second"], [4.0])
    np.testing.assert_allclose(stats["family_max_third"], [3.0])
    np.testing.assert_allclose(stats["family_max_sum_minus_largest"], [9.0])
    np.testing.assert_allclose(
        stats["family_max_geomean1p"],
        [np.exp(np.mean(np.log1p([3.0, 2.0, 4.0, 5.0]))) - 1.0],
    )
    np.testing.assert_allclose(stats["family_active_count_0p25"], [4.0])
    np.testing.assert_allclose(stats["family_active_count_0p5"], [4.0])
    np.testing.assert_allclose(stats["family_capped_mean_sum"], [2.0])
    np.testing.assert_allclose(
        stats["family_rational_mean_sum"],
        [0.625 + 2.0 / 3.0 + 0.8 + 5.0 / 6.0],
    )
    assert set(FIXED_DIRECTIONS.values()) == {-1}

