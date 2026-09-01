from __future__ import annotations

import pandas as pd

from src.next167_periodic_contact_topology_audit import (
    HYPOTHESES,
    select_topology_hypothesis,
)


def test_topology_hypotheses_match_the_frozen_two_graph_catalogue() -> None:
    assert len(HYPOTHESES) == 14
    for mode in ("voronoi", "crystalnn"):
        assert HYPOTHESES[f"{mode}_rank_max__high"] == (
            f"pct_{mode}_rank_max",
            1,
        )
        assert HYPOTHESES[f"{mode}_rank_mean__high"] == (
            f"pct_{mode}_rank_mean",
            1,
        )
        for rank in range(4):
            assert HYPOTHESES[f"{mode}_rank{rank}_fraction__high"] == (
                f"pct_{mode}_rank{rank}_fraction",
                1,
            )
        assert HYPOTHESES[f"{mode}_rank0_fraction__low"] == (
            f"pct_{mode}_rank0_fraction",
            -1,
        )


def test_select_topology_hypothesis_applies_frozen_eligibility_ranking() -> None:
    table, selected = select_topology_hypothesis(
        pd.DataFrame(
            [
                {
                    "hypothesis": "eligible-b",
                    "eligible_for_search": True,
                    "ranking_min_auc": 0.56,
                    "ranking_mean_auc": 0.70,
                },
                {
                    "hypothesis": "eligible-a",
                    "eligible_for_search": True,
                    "ranking_min_auc": 0.56,
                    "ranking_mean_auc": 0.70,
                },
                {
                    "hypothesis": "ineligible",
                    "eligible_for_search": False,
                    "ranking_min_auc": 0.99,
                    "ranking_mean_auc": 0.99,
                },
            ]
        )
    )
    assert table["hypothesis"].tolist() == [
        "eligible-a",
        "eligible-b",
        "ineligible",
    ]
    assert selected is not None
    assert selected["hypothesis"] == "eligible-a"
