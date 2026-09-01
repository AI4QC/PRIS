from __future__ import annotations

import numpy as np

from src.next128_broad_residual_diagnostic import diagnose_broad_threshold_tables


def test_diagnostic_finds_exact_broad_threshold() -> None:
    tables = {
        "thresholds": np.asarray([3.0, 2.0, 1.0]),
        "coverage_lower": np.asarray([0.8]),
        "protected_kept": np.asarray([[9, 8, 7]]),
        "rejected_severe": np.asarray([[2, 4, 5]]),
        "precision_lower": np.asarray([[0.7, 0.65, 0.55]]),
        "savings_lower": np.asarray([[0.1, 0.25, 0.4]]),
    }
    cells = [{"cell_id": "source:all", "kind": "source_aggregate"}]
    pauling = {
        "source:all": {
            "coverage_lower": 0.5,
            "protected_kept": 8,
            "severe_rejected": 3,
            "severe_rejection_precision_lower": 0.6,
            "savings_lower": 0.2,
        }
    }
    result = diagnose_broad_threshold_tables(
        tables=tables,
        cells=cells,
        pauling_by_cell=pauling,
        safe_threshold=3.5,
    )
    assert result["passes_broad"] is True
    assert result["best_threshold"] == 2.0
    assert result["failed_constraint_count"] == 0


def test_diagnostic_reports_strict_equality_and_aggregate_precision_failures() -> None:
    tables = {
        "thresholds": np.asarray([2.0]),
        "coverage_lower": np.asarray([0.8]),
        "protected_kept": np.asarray([[8]]),
        "rejected_severe": np.asarray([[3]]),
        "precision_lower": np.asarray([[0.44]]),
        "savings_lower": np.asarray([[0.2]]),
    }
    cells = [{"cell_id": "source:all", "kind": "source_aggregate"}]
    pauling = {
        "source:all": {
            "coverage_lower": 0.5,
            "protected_kept": 8,
            "severe_rejected": 3,
            "severe_rejection_precision_lower": 0.44,
            "savings_lower": 0.2,
        }
    }
    result = diagnose_broad_threshold_tables(
        tables=tables,
        cells=cells,
        pauling_by_cell=pauling,
        safe_threshold=3.0,
    )
    assert result["passes_broad"] is False
    assert result["failed_constraint_count"] == 4
    assert {(item["component"], item["comparator"]) for item in result["failures"]} == {
        ("severe_rejected", ">"),
        ("severe_precision_lower", ">"),
        ("savings_lower", ">"),
        ("aggregate_precision_lower", ">="),
    }
