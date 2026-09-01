import json

import pandas as pd
import pytest

import src.next138_bottleneck_broad_residual_diagnostic as n138


def test_select_safe_candidates_preserves_bottleneck_identity() -> None:
    records = pd.DataFrame(
        [
            {
                "candidate_key": "b",
                "safe_threshold": 2.5,
                "passes_safe_all_cells": True,
                "bottleneck_term_ids_json": json.dumps(["packing_min"]),
                "bottleneck_weights_json": json.dumps([0.25]),
                "bottleneck_term_count": 1,
            },
            {
                "candidate_key": "a",
                "safe_threshold": 1.5,
                "passes_safe_all_cells": False,
                "bottleneck_term_ids_json": "[]",
                "bottleneck_weights_json": "[]",
                "bottleneck_term_count": 0,
            },
        ]
    )

    selected = n138.select_safe_candidates(records)
    assert selected["candidate_key"].tolist() == ["b"]
    assert selected["safe_threshold"].tolist() == [2.5]


def test_select_safe_candidates_rejects_mismatched_configuration() -> None:
    records = pd.DataFrame(
        [
            {
                "candidate_key": "x",
                "safe_threshold": 1.0,
                "passes_safe_all_cells": True,
                "bottleneck_term_ids_json": '["packing_min"]',
                "bottleneck_weights_json": "[]",
                "bottleneck_term_count": 1,
            }
        ]
    )
    with pytest.raises(ValueError, match="identity"):
        n138.select_safe_candidates(records)
