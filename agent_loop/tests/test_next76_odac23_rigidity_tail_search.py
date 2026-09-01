from __future__ import annotations

from src.next76_odac23_rigidity_tail_search import replication_ready


def test_replication_ready_requires_precision_safety_margin() -> None:
    base = {
        "coverage_lower": 0.99,
        "protected_recall_lower": 0.98,
        "reject_precision_lower": 0.749,
        "savings_lower": 0.03,
        "pooled_extreme_auc": 0.80,
        "macro_stratum_auc": 0.75,
        "worst_stratum_auc": 0.70,
    }

    assert not replication_ready(base)
    assert replication_ready({**base, "reject_precision_lower": 0.75})
