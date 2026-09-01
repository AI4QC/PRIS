from __future__ import annotations

from src.next79_odac23_electrostatic_residual_guard import (
    CANDIDATE_FEATURE_NAMES,
    replication_ready,
)


def test_final_catalogue_is_one_preselected_electrostatic_residual() -> None:
    assert CANDIDATE_FEATURE_NAMES == ("aefi_residual_q95",)


def test_adaptive_search_requires_point_eight_precision_lower_bound() -> None:
    metrics = {
        "coverage_lower": 0.99,
        "protected_recall_lower": 0.98,
        "reject_precision_lower": 0.799,
        "savings_lower": 0.03,
        "pooled_extreme_auc": 0.80,
        "macro_stratum_auc": 0.75,
        "worst_stratum_auc": 0.70,
    }

    assert not replication_ready(metrics)
    assert replication_ready({**metrics, "reject_precision_lower": 0.80})
