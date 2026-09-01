"""Contracts for the fixed-threshold NEXT28 contact-coordination law."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _development(rows: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"dev-{index:04d}" for index in range(rows)]
    shard = np.asarray([f"s{index % 6}" for index in range(rows)])
    score = np.full(rows, 2.0)
    score[:72] = 7.0
    positive = np.zeros(rows, dtype=bool)
    positive[:71] = True
    features = pd.DataFrame(
        {
            "material_id": ids,
            "development_shard": shard,
            "periodic_contact_coord105": score,
            "analytic_supported": True,
        }
    )
    endpoints = pd.DataFrame(
        {
            "material_id": ids,
            "force0_max": np.where(positive, 1.1, 0.1),
            "force0_rms": 0.1,
            "energy_drop_pa": 0.01,
            "stress0_norm": 0.01,
        }
    )
    return features, endpoints


def test_freeze_and_apply_fixed_contact_coordination_law(tmp_path: Path) -> None:
    from src.next28_contact_coordination import (
        FROZEN_RULE_NAME,
        PREDICTIONS_NAME,
        apply_frozen_rule,
        freeze_rule,
    )

    features, endpoints = _development()
    freeze_dir = tmp_path / "freeze"
    freeze = freeze_rule(features=features, endpoints=endpoints, output_dir=freeze_dir)
    assert freeze["eligible"] is True
    rule = freeze["rule"]
    assert rule["formula"] == "reject iff periodic_contact_coord105 >= 6.3"
    assert rule["threshold"] == 6.3
    assert rule["maximum_terms"] == 1

    holdout = features.drop(columns=["development_shard"]).iloc[:3].copy()
    holdout["material_id"] = ["future-a", "future-b", "future-c"]
    holdout["periodic_contact_coord105"] = [6.29, 6.3, np.nan]
    prediction_dir = tmp_path / "predictions"
    manifest = apply_frozen_rule(
        frozen_rule_path=freeze_dir / FROZEN_RULE_NAME,
        rule_manifest_path=freeze_dir / "MANIFEST.json",
        features=holdout,
        output_dir=prediction_dir,
    )
    predictions = pd.read_parquet(prediction_dir / PREDICTIONS_NAME)
    assert predictions.reject.tolist() == [False, True, False]
    assert predictions.analytic_supported.tolist() == [True, True, False]
    assert manifest["labels_opened"] is False
    assert manifest["threshold_refit"] is False
