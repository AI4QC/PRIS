"""Shard-stable freeze contracts for the NEXT27 periodic pressure law."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _tables(broken_second_shard: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_parts = []
    endpoint_parts = []
    columns = [
        "periodic_nonbond_vdw_min",
        "periodic_nonbond_vdw_q01",
        "periodic_nonbond_vdw_q05",
        "periodic_overlap2_pa",
        "periodic_overlap3_pa",
        "periodic_repulsion12_pa",
        "periodic_contact_coord100",
        "periodic_contact_coord105",
        "periodic_contact_coord110",
        "periodic_nearest_mean",
        "periodic_nearest_q10",
        "periodic_pairs_pa",
    ]
    for shard, offset in (("a", 0.0), ("b", -0.4)):
        n = 200
        risk = np.linspace(-2.0, 2.0, n) + offset
        ids = [f"{shard}-{index:04d}" for index in range(n)]
        features = pd.DataFrame({"material_id": ids, "development_shard": shard})
        for column in columns:
            if "vdw" in column or "nearest" in column:
                features[column] = 1.0 - 0.1 * risk
            else:
                features[column] = risk + 3.0
        positive = risk >= 1.0
        if shard == "b" and broken_second_shard:
            positive[:] = False
        endpoints = pd.DataFrame(
            {
                "material_id": ids,
                "force0_max": np.where(positive, 1.1, 0.2),
                "force0_rms": 0.1,
                "energy_drop_pa": 0.01,
                "stress0_norm": 0.005,
            }
        )
        feature_parts.append(features)
        endpoint_parts.append(endpoints)
    return pd.concat(feature_parts, ignore_index=True), pd.concat(endpoint_parts, ignore_index=True)


def test_search_freezes_only_rule_that_is_stable_in_every_development_shard(
    tmp_path: Path,
) -> None:
    from src.next27_development import FROZEN_RULE_NAME, search_and_freeze

    features, endpoints = _tables()
    output = tmp_path / "freeze"
    manifest = search_and_freeze(features=features, endpoints=endpoints, output_dir=output)
    rule = pd.read_json(output / FROZEN_RULE_NAME, typ="series")

    assert manifest["eligible"] is True
    assert bool(rule["eligible"])
    assert 1 <= len(rule["terms"]) <= 2
    assert set(rule["development_shard_metrics"]) == {"a", "b"}
    for metrics in rule["development_shard_metrics"].values():
        assert metrics["endpoint_positive_precision"] >= 0.80
        assert metrics["endpoint_negative_protection"] >= 0.95
        assert metrics["savings"] >= 0.05


def test_search_refuses_formula_when_one_shard_has_no_transportable_signal(
    tmp_path: Path,
) -> None:
    from src.next27_development import search_and_freeze

    features, endpoints = _tables(broken_second_shard=True)
    manifest = search_and_freeze(
        features=features, endpoints=endpoints, output_dir=tmp_path / "failed"
    )
    assert manifest["eligible"] is False


def test_frozen_next27_application_is_label_free_and_fail_open(tmp_path: Path) -> None:
    from src.next27_apply_rule import apply_frozen_rule
    from src.next27_development import search_and_freeze

    features, endpoints = _tables()
    freeze = tmp_path / "freeze"
    search_and_freeze(features=features, endpoints=endpoints, output_dir=freeze)
    prospective = features.drop(columns="development_shard").copy()
    prospective.loc[0, [column for column in prospective if column != "material_id"]] = np.nan
    output = tmp_path / "predictions"
    manifest = apply_frozen_rule(
        frozen_rule_path=freeze / "FROZEN_RULE.json",
        rule_manifest_path=freeze / "MANIFEST.json",
        features=prospective,
        output_dir=output,
    )
    predictions = pd.read_parquet(output / "next27_predictions.parquet")

    assert manifest["labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert not bool(predictions.loc[0, "reject"])
