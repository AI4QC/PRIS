from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next180_strong_neighborhood_directional_closure_audit import (
    HYPOTHESES,
    eligibility_from_metrics,
    run_strong_neighborhood_directional_closure_audit,
    select_strong_closure_hypothesis,
)


def test_hypothesis_universe_is_exactly_ten_frozen_high_directions() -> None:
    assert len(HYPOTHESES) == 10
    assert set(HYPOTHESES.values()) == {
        (f"psndc_{mode}_{suffix}", 1)
        for mode in ("voronoi", "crystalnn")
        for suffix in (
            "closure_min",
            "closure_q10",
            "closure_mean",
            "volume_q10",
            "volume_mean",
        )
    }


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
    kwargs["scigen_shell_worst_auc"] = 0.549999
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
    table, selected = select_strong_closure_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(
        inspect.signature(
            run_strong_neighborhood_directional_closure_audit
        ).parameters
    )
    assert "next179_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_strong_neighborhood_directional_closure_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT180 input is missing"):
        run_strong_neighborhood_directional_closure_audit(**kwargs)
