from __future__ import annotations

from src.next77_odac23_analytic_electrostatic_features import (
    ANALYTIC_ELECTROSTATIC_FEATURE_NAMES,
)
from src.next78_odac23_electrostatic_tail_search import (
    CANDIDATE_FEATURE_NAMES,
    replication_ready,
)


def test_candidate_catalogue_contains_each_electrostatic_feature_once() -> None:
    assert len(CANDIDATE_FEATURE_NAMES) == len(set(CANDIDATE_FEATURE_NAMES))
    assert set(ANALYTIC_ELECTROSTATIC_FEATURE_NAMES).issubset(CANDIDATE_FEATURE_NAMES)


def test_replication_ready_preserves_precision_safety_margin() -> None:
    metrics = {
        "coverage_lower": 0.99,
        "protected_recall_lower": 0.98,
        "reject_precision_lower": 0.749,
        "savings_lower": 0.03,
        "pooled_extreme_auc": 0.80,
        "macro_stratum_auc": 0.75,
        "worst_stratum_auc": 0.70,
    }

    assert not replication_ready(metrics)
    assert replication_ready({**metrics, "reject_precision_lower": 0.75})
