from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next188_contradiction_severity_audit import (
    HYPOTHESES,
    eligibility_from_metrics,
    reverse_auc_evaluation,
    run_contradiction_severity_audit,
    select_contradiction_severity_hypothesis,
)


def test_hypothesis_universe_is_exactly_twenty_four_severe_high_directions() -> None:
    assert len(HYPOTHESES) == 24
    assert {value[3] for value in HYPOTHESES.values()} == {-1}
    assert all(name.endswith("__severe_high") for name in HYPOTHESES)


def test_auc_reversal_inverts_pooled_and_each_evaluable_fold_exactly() -> None:
    reversed_metrics = reverse_auc_evaluation(
        {
            "pooled_auc": 0.7,
            "macro_auc": 0.5,
            "worst_auc": 0.2,
            "evaluable_folds": 2,
            "fold_aucs_json": "[0.2,0.8,null]",
            "protected": 10,
            "severe": 20,
        }
    )
    assert reversed_metrics["pooled_auc"] == pytest.approx(0.3)
    assert reversed_metrics["macro_auc"] == pytest.approx(0.5)
    assert reversed_metrics["worst_auc"] == pytest.approx(0.2)
    assert reversed_metrics["fold_aucs"] == pytest.approx([0.8, 0.2])
    assert reversed_metrics["evaluable_folds"] == 2
    assert reversed_metrics["protected"] == 10
    assert reversed_metrics["severe"] == 20


def test_eligibility_reuses_frozen_cross_source_gates() -> None:
    kwargs = dict(
        scigen_full_support=0.90,
        wyformer_full_support=0.90,
        scigen_shell_worst_auc=0.55,
        scigen_shell_evaluable_folds=5,
        wyformer_shell_pooled_auc=0.55,
        scigen_full_pooled_auc=0.50,
        wyformer_full_pooled_auc=0.50,
    )
    assert eligibility_from_metrics(**kwargs)
    kwargs["scigen_shell_evaluable_folds"] = 4
    assert not eligibility_from_metrics(**kwargs)


def test_selector_ranks_only_eligible_rows_deterministically() -> None:
    records = pd.DataFrame(
        {
            "hypothesis": ["b", "a", "c"],
            "eligible_for_search": [True, True, False],
            "ranking_min_auc": [0.60, 0.60, 0.99],
            "ranking_mean_auc": [0.70, 0.70, 0.99],
        }
    )
    table, selected = select_contradiction_severity_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(inspect.signature(run_contradiction_severity_audit).parameters)
    assert "next186_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_contradiction_severity_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT188 input is missing"):
        run_contradiction_severity_audit(**kwargs)
