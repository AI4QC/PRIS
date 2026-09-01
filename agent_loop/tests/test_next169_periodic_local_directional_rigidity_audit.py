from __future__ import annotations

import inspect
import pandas as pd

from src.next169_periodic_local_directional_rigidity_audit import (
    HYPOTHESES,
    eligibility_from_metrics,
    run_periodic_local_directional_rigidity_audit,
    select_directional_rigidity_hypothesis,
)


def test_hypothesis_schema_is_exactly_ten_high_direction_features() -> None:
    assert HYPOTHESES == {
        f"{mode}_{suffix}__high": (f"pldr_{mode}_{suffix}", 1)
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
    passing = {
        "scigen_full_support": 0.90,
        "wyformer_full_support": 0.90,
        "scigen_shell_worst_auc": 0.55,
        "scigen_shell_evaluable_folds": 5,
        "wyformer_shell_pooled_auc": 0.55,
        "scigen_full_pooled_auc": 0.50,
        "wyformer_full_pooled_auc": 0.50,
    }
    assert eligibility_from_metrics(**passing) is True
    for key in passing:
        failing = dict(passing)
        if key.endswith("evaluable_folds"):
            failing[key] = 4
        else:
            failing[key] = float(failing[key]) - 1.0e-6
        assert eligibility_from_metrics(**failing) is False


def test_selection_is_deterministic_and_never_selects_an_ineligible_record() -> None:
    records = pd.DataFrame(
        [
            {
                "hypothesis": "z_ineligible",
                "eligible_for_search": False,
                "ranking_min_auc": 0.99,
                "ranking_mean_auc": 0.99,
            },
            {
                "hypothesis": "b_eligible",
                "eligible_for_search": True,
                "ranking_min_auc": 0.60,
                "ranking_mean_auc": 0.70,
            },
            {
                "hypothesis": "a_eligible",
                "eligible_for_search": True,
                "ranking_min_auc": 0.60,
                "ranking_mean_auc": 0.70,
            },
        ]
    )
    table, selected = select_directional_rigidity_hypothesis(records)
    assert table["hypothesis"].tolist() == [
        "a_eligible",
        "b_eligible",
        "z_ineligible",
    ]
    assert selected is not None
    assert selected["hypothesis"] == "a_eligible"


def test_formal_audit_interface_cannot_accept_validation_or_replication_endpoints() -> None:
    parameters = tuple(
        inspect.signature(run_periodic_local_directional_rigidity_audit).parameters
    )
    assert "next168_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )
