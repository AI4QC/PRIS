import json

import pandas as pd
import pytest

import src.next136_conjunctive_broad_residual_diagnostic as n136


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_key": "b",
                "safe_threshold": 2.0,
                "passes_safe_all_cells": True,
                "conjunctive_term_ids_json": json.dumps(["packing"]),
                "conjunctive_weights_json": json.dumps([0.5]),
                "conjunctive_term_count": 1,
            },
            {
                "candidate_key": "a",
                "safe_threshold": 1.0,
                "passes_safe_all_cells": False,
                "conjunctive_term_ids_json": "[]",
                "conjunctive_weights_json": "[]",
                "conjunctive_term_count": 0,
            },
            {
                "candidate_key": "c",
                "safe_threshold": 3.0,
                "passes_safe_all_cells": True,
                "conjunctive_term_ids_json": json.dumps(["packing", "volume"]),
                "conjunctive_weights_json": json.dumps([0.25, 0.1]),
                "conjunctive_term_count": 2,
            },
        ]
    )


def test_select_safe_candidates_preserves_only_published_safe_rows() -> None:
    selected = n136.select_safe_candidates(_records())

    assert selected["candidate_key"].tolist() == ["b", "c"]
    assert selected["safe_threshold"].tolist() == [2.0, 3.0]


def test_select_safe_candidates_rejects_duplicate_or_nonfinite_identity() -> None:
    duplicate = pd.concat([_records(), _records().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="schema"):
        n136.select_safe_candidates(duplicate)

    nonfinite = _records()
    nonfinite.loc[0, "safe_threshold"] = float("nan")
    with pytest.raises(ValueError, match="threshold"):
        n136.select_safe_candidates(nonfinite)


def test_summarize_groups_reports_closest_configuration() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_key": "x",
                "conjunctive_term_count": 1,
                "conjunctive_term_ids_json": '["packing"]',
                "conjunctive_weights_json": "[0.5]",
                "failed_constraint_count": 6,
                "normalized_shortfall_sum": 0.7,
                "best_threshold": 2.0,
            },
            {
                "candidate_key": "y",
                "conjunctive_term_count": 1,
                "conjunctive_term_ids_json": '["packing"]',
                "conjunctive_weights_json": "[0.5]",
                "failed_constraint_count": 5,
                "normalized_shortfall_sum": 0.9,
                "best_threshold": 3.0,
            },
        ]
    )

    summary = n136.summarize_groups(frame)
    assert summary["by_conjunctive_term_count"]["1"]["minimum_failed_constraint_count"] == 5
    assert summary["by_configuration"]['["packing"]@[0.5]']["closest_candidate_key"] == "y"
