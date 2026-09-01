from __future__ import annotations

import pandas as pd
import pytest

from experiments.property_design_20260821.forward_screen import (
    join_cohort_tables,
    summarize_queue,
)


def test_join_cohort_tables_requires_exact_one_to_one_structure_match():
    labels = pd.DataFrame(
        {"material_id": ["mp-1", "mp-2"], "made": [True, False]}
    )
    structures = pd.DataFrame(
        {"material_id": ["mp-1", "mp-2"], "structure_json": ["{}", "{}"]}
    )

    joined = join_cohort_tables(labels, structures)

    assert joined.material_id.tolist() == ["mp-1", "mp-2"]
    assert len(joined) == 2


def test_join_cohort_tables_fails_on_missing_structure():
    labels = pd.DataFrame(
        {"material_id": ["mp-1", "mp-2"], "made": [True, False]}
    )
    structures = pd.DataFrame(
        {"material_id": ["mp-1"], "structure_json": ["{}"]}
    )

    with pytest.raises(ValueError, match="missing structures for 1 material IDs"):
        join_cohort_tables(labels, structures)


def test_join_cohort_tables_fails_on_duplicate_structure_id():
    labels = pd.DataFrame({"material_id": ["mp-1"], "made": [True]})
    structures = pd.DataFrame(
        {"material_id": ["mp-1", "mp-1"], "structure_json": ["{}", "{}"]}
    )

    with pytest.raises(ValueError, match="structure table contains duplicate material IDs"):
        join_cohort_tables(labels, structures)


def test_summarize_queue_removes_only_reject_and_keeps_no_verdict():
    frame = pd.DataFrame(
        {
            "material_id": ["a", "b", "c", "d", "e"],
            "fit_valid": [True, True, True, True, True],
            "bulk_modulus_gpa": [250.0, 240.0, 230.0, 100.0, 260.0],
            "made": [True, True, False, True, True],
            "rung_L4_verdict": ["pass", "reject", "no verdict", "reject", "no verdict"],
        }
    )

    result = summarize_queue(
        frame, verdict_column="rung_L4_verdict", bulk_threshold_gpa=200.0
    )

    assert result["high_property_count"] == 4
    assert result["queue_removed_count"] == 1
    assert result["queue_retained_count"] == 3
    assert result["queue_reduction"] == pytest.approx(0.25)
    assert result["verdict_counts"] == {"pass": 1, "reject": 1, "no verdict": 2}
    assert result["good_material_count"] == 3
    assert result["good_material_removed_count"] == 1
    assert result["good_material_removal_rate"] == pytest.approx(1 / 3)


def test_summarize_queue_excludes_invalid_property_fit_from_high_property_set():
    frame = pd.DataFrame(
        {
            "material_id": ["a", "b"],
            "fit_valid": [False, True],
            "bulk_modulus_gpa": [500.0, 250.0],
            "made": [True, True],
            "rung_L4_verdict": ["reject", "pass"],
        }
    )

    result = summarize_queue(
        frame, verdict_column="rung_L4_verdict", bulk_threshold_gpa=200.0
    )

    assert result["high_property_count"] == 1
    assert result["queue_removed_count"] == 0
    assert result["good_material_count"] == 1
