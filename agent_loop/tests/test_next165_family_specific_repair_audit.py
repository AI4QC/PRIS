from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.next165_family_specific_repair_audit import (
    FAMILY_PREFIXES,
    FIXED_DIRECTIONS,
    family_repair_statistics,
    select_family_repair_statistic,
)


TERM_IDS = [
    "cov_demo",
    "scbv_demo",
    "cmvo_demo",
    "bvtbd_demo",
    "mhcr_demo",
]


def test_family_repair_statistics_match_frozen_concentration_catalogue() -> None:
    values = np.array(
        [
            [0.2, 0.8, 0.4, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    stats = family_repair_statistics(values, TERM_IDS)

    assert set(stats) == set(FIXED_DIRECTIONS)
    assert len(stats) == 15
    assert set(FAMILY_PREFIXES) == {
        "local_geometry",
        "charge_flow_feasibility",
        "valence_transport",
        "contact_robustness",
    }
    assert {
        name: FIXED_DIRECTIONS[name]
        for name in FIXED_DIRECTIONS
        if any(name.startswith(f"{family}_") for family in FAMILY_PREFIXES)
    } == {
        name: 1
        for family in FAMILY_PREFIXES
        for name in (
            f"{family}_share",
            f"{family}_margin",
            f"{family}_is_dominant",
        )
    }
    assert FIXED_DIRECTIONS["largest_family_share"] == 1
    assert FIXED_DIRECTIONS["effective_family_count"] == -1
    assert FIXED_DIRECTIONS["normalized_family_entropy"] == -1

    family_values = np.array([0.35, 0.4, 0.1, 0.0])
    shares = family_values / family_values.sum()
    expected_entropy = -np.sum(
        shares[shares > 0] * np.log(shares[shares > 0])
    ) / np.log(4.0)
    for family, expected in zip(FAMILY_PREFIXES, shares, strict=True):
        np.testing.assert_allclose(stats[f"{family}_share"], [expected, 0.0])
    np.testing.assert_allclose(
        stats["charge_flow_feasibility_margin"], [0.05, 0.0]
    )
    np.testing.assert_allclose(
        stats["charge_flow_feasibility_is_dominant"], [1.0, 0.0]
    )
    for family in ("local_geometry", "valence_transport", "contact_robustness"):
        np.testing.assert_allclose(stats[f"{family}_margin"], [0.0, 0.0])
        np.testing.assert_allclose(
            stats[f"{family}_is_dominant"], [0.0, 0.0]
        )
    np.testing.assert_allclose(stats["largest_family_share"], [shares.max(), 0.0])
    np.testing.assert_allclose(
        stats["effective_family_count"],
        [family_values.sum() ** 2 / np.square(family_values).sum(), 0.0],
    )
    np.testing.assert_allclose(
        stats["normalized_family_entropy"], [expected_entropy, 0.0]
    )


def test_family_repair_statistics_reject_missing_family() -> None:
    with pytest.raises(ValueError, match="family coverage"):
        family_repair_statistics(np.ones((1, 4)), TERM_IDS[:-1])


def test_select_family_repair_statistic_uses_frozen_lexicographic_ranking() -> None:
    table, selected = select_family_repair_statistic(
        pd.DataFrame(
            [
                {
                    "statistic": "eligible-lower-min",
                    "eligible_for_search": True,
                    "ranking_min_auc": 0.56,
                    "ranking_mean_auc": 0.70,
                },
                {
                    "statistic": "eligible-top-z",
                    "eligible_for_search": True,
                    "ranking_min_auc": 0.57,
                    "ranking_mean_auc": 0.71,
                },
                {
                    "statistic": "eligible-top-a",
                    "eligible_for_search": True,
                    "ranking_min_auc": 0.57,
                    "ranking_mean_auc": 0.71,
                },
                {
                    "statistic": "ineligible",
                    "eligible_for_search": False,
                    "ranking_min_auc": 0.99,
                    "ranking_mean_auc": 0.99,
                },
            ]
        )
    )
    assert table["statistic"].tolist() == [
        "eligible-top-a",
        "eligible-top-z",
        "eligible-lower-min",
        "ineligible",
    ]
    assert selected is not None
    assert selected["statistic"] == "eligible-top-a"
