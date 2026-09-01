from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next174_weighted_local_directional_rigidity_audit import (
    HYPOTHESES,
    eligibility_from_metrics,
    run_weighted_local_directional_rigidity_audit,
    select_weighted_rigidity_hypothesis,
)


def test_hypothesis_universe_is_exactly_ten_frozen_high_directions() -> None:
    assert len(HYPOTHESES) == 10
    assert set(HYPOTHESES.values()) == {
        (f"pwldr_{mode}_{suffix}", 1)
        for mode in ("voronoi", "crystalnn")
        for suffix in (
            "tightness_min",
            "tightness_q10",
            "tightness_mean",
            "volume_q10",
            "volume_mean",
        )
    }


def test_eligibility_requires_every_frozen_gate() -> None:
    kwargs = {
        "scigen_full_support": 0.90,
        "wyformer_full_support": 0.90,
        "scigen_shell_worst_auc": 0.55,
        "scigen_shell_evaluable_folds": 5,
        "wyformer_shell_pooled_auc": 0.55,
        "scigen_full_pooled_auc": 0.50,
        "wyformer_full_pooled_auc": 0.50,
    }
    assert eligibility_from_metrics(**kwargs)
    for key in kwargs:
        changed = dict(kwargs)
        changed[key] = 4 if key == "scigen_shell_evaluable_folds" else float(changed[key]) - 1.0e-12
        assert not eligibility_from_metrics(**changed)


def test_selection_is_deterministic_and_eligible_first() -> None:
    records = pd.DataFrame(
        {
            "hypothesis": ["b", "a", "c"],
            "eligible_for_search": [True, True, False],
            "ranking_min_auc": [0.60, 0.60, 0.99],
            "ranking_mean_auc": [0.70, 0.70, 0.99],
        }
    )
    ranked, selected = select_weighted_rigidity_hypothesis(records)
    assert ranked["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_no_validation_or_replication_endpoint(tmp_path) -> None:
    signature = inspect.signature(run_weighted_local_directional_rigidity_audit)
    parameters = tuple(signature.parameters)
    assert "next173_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )
    kwargs = {
        name: tmp_path / name
        for name in parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT174 input is missing"):
        run_weighted_local_directional_rigidity_audit(**kwargs)
