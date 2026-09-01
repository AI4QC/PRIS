from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from src.next95_wyformer_sparse_law_search import (
    DEFAULT_GATES,
    _endpoint_numeric,
    evaluate_fixed_threshold_folds,
    recalibrate_terms,
    run_wyformer_sparse_search,
)


def test_runner_accepts_only_discovery_endpoint() -> None:
    parameters = inspect.signature(run_wyformer_sparse_search).parameters
    assert "discovery_endpoint_dir" in parameters
    assert not any("validation" in name or "replication" in name for name in parameters)


def test_endpoint_numeric_excludes_middle_only_from_binary_extremes() -> None:
    values = _endpoint_numeric(pd.Series(["protected", "middle", "severe"]))
    assert values.tolist() == [1.0, 1.5, 2.0]


def test_term_recalibration_uses_discovery_robust_statistics() -> None:
    features = pd.DataFrame({"risk": [*map(float, range(8)), np.nan]})
    template = [
        {
            "term_id": "risk__high",
            "feature": "risk",
            "direction": 1,
            "transform": "log1p_nonnegative",
            "group": "test",
        }
    ]
    terms, excluded = recalibrate_terms(features, template, min_coverage=0.75)
    assert not excluded
    assert len(terms) == 1
    term = terms[0]
    transformed = np.log1p(np.arange(8, dtype=float))
    assert np.isclose(term["center"], np.median(transformed))
    assert np.isclose(
        term["scale"],
        (np.quantile(transformed, 0.9) - np.quantile(transformed, 0.1)) / 2,
    )
    assert term["direction"] == 1


def test_fixed_formula_and_threshold_must_pass_each_composition_fold() -> None:
    groups = np.array([f"g{i}" for i in range(50) for _ in range(4)], dtype=object)
    endpoint = np.tile(np.array([1.0, 1.0, 2.0, 2.0]), 50)
    score = np.tile(np.array([0.0, 0.1, 2.0, 3.0]), 50)
    supported = np.ones(len(score), dtype=bool)
    gates = dict(DEFAULT_GATES)
    gates.update(
        {
            "coverage_lower": 0.80,
            "protected_recall_lower": 0.80,
            "severe_rejection_precision_lower": 0.80,
            "savings_lower": 0.10,
        }
    )
    result = evaluate_fixed_threshold_folds(
        score=score,
        supported=supported,
        endpoint=endpoint,
        reduced_formula=groups,
        threshold=1.0,
        gates=gates,
    )
    assert result["passes_all_folds"] is True
    assert result["passing_folds"] == 5
    assert len(result["folds"]) == 5
