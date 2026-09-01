from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next31_omc25_energy_ranking import (
    apply_frozen_rule,
    assemble_label_free_features,
    compute_energy_risk,
    fit_frozen_rule,
    freeze_development_rule,
)


def _feature_manifest(path: Path, feature_path: Path) -> None:
    import hashlib

    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "labels_opened": False,
                "endpoint_fields_read": False,
                "relaxed_structures_opened": False,
                "model_or_proxy_potential_used": False,
                "outputs_sha256": {feature_path.name: digest},
            }
        ),
        encoding="utf-8",
    )


def test_assemble_label_free_features_validates_manifests_and_shards(
    tmp_path: Path,
) -> None:
    feature_paths: list[Path] = []
    manifest_paths: list[Path] = []
    for shard, material_id in (("data0001", "a"), ("data0002", "b")):
        folder = tmp_path / shard
        folder.mkdir()
        feature_path = folder / "next27_periodic_packing_features.parquet"
        pd.DataFrame(
            {
                "material_id": [material_id],
                "periodic_nonbond_vdw_q05": [0.9],
                "periodic_contact_coord105": [3.0],
                "analytic_supported": [True],
            }
        ).to_parquet(feature_path, index=False)
        manifest_path = folder / "MANIFEST.json"
        _feature_manifest(manifest_path, feature_path)
        feature_paths.append(feature_path)
        manifest_paths.append(manifest_path)

    output_dir = tmp_path / "combined"
    manifest = assemble_label_free_features(
        feature_paths=feature_paths,
        feature_manifest_paths=manifest_paths,
        source_shards=("data0001", "data0002"),
        output_dir=output_dir,
    )

    combined = pd.read_parquet(output_dir / "next31_label_free_features.parquet")
    assert combined["source_shard"].tolist() == ["data0001", "data0002"]
    assert combined["material_id"].tolist() == ["a", "b"]
    assert manifest["labels_opened"] is False
    assert manifest["counts"] == {"rows": 2, "shards": 2}

    bad_manifest = json.loads(manifest_paths[0].read_text())
    bad_manifest["labels_opened"] = True
    manifest_paths[0].write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="label-free boundary"):
        assemble_label_free_features(
            feature_paths=feature_paths,
            feature_manifest_paths=manifest_paths,
            source_shards=("data0001", "data0002"),
            output_dir=tmp_path / "bad-combined",
        )


def _rule() -> dict[str, object]:
    return {
        "eligible": True,
        "q05_median": 1.0,
        "q05_iqr": 0.1,
        "coord105_median": 2.0,
        "coord105_iqr": 2.0,
        "threshold": 2.0,
    }


def test_energy_risk_is_two_term_dimensionless_formula_and_fails_open() -> None:
    features = pd.DataFrame(
        {
            "periodic_nonbond_vdw_q05": [0.89, 1.0, np.nan],
            "periodic_contact_coord105": [4.0, 2.0, 8.0],
        }
    )

    score, supported, reject = compute_energy_risk(features, _rule())

    assert score[:2].tolist() == pytest.approx([2.1, 0.0])
    assert np.isnan(score[2])
    assert supported.tolist() == [True, True, False]
    assert reject.tolist() == [True, False, False]


def test_fit_rule_uses_only_named_development_shards() -> None:
    features = pd.DataFrame(
        {
            "material_id": [f"id-{index}" for index in range(40)],
            "source_shard": ["dev-a"] * 20 + ["confirm-z"] * 20,
            "periodic_nonbond_vdw_q05": np.linspace(0.8, 1.2, 40),
            "periodic_contact_coord105": np.linspace(0.0, 8.0, 40),
        }
    )
    endpoints = features[["material_id", "source_shard"]].copy()
    endpoints["energy_drop_pa"] = [0.05] * 20 + [0.0] * 20
    changed = endpoints.copy()
    changed.loc[changed["source_shard"].eq("confirm-z"), "energy_drop_pa"] = 99.0

    first = fit_frozen_rule(
        features=features,
        endpoints=endpoints,
        development_shards=("dev-a",),
        rejection_fraction=0.10,
    )
    second = fit_frozen_rule(
        features=features,
        endpoints=changed,
        development_shards=("dev-a",),
        rejection_fraction=0.10,
    )

    assert first == second
    assert first["confirmation_rows_used"] == 0
    assert first["development_shards"] == ["dev-a"]


def test_apply_rule_seals_label_free_predictions_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    rule_path = tmp_path / "rule.json"
    rule_path.write_text(json.dumps(_rule()), encoding="utf-8")
    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "material_id": ["a", "b"],
            "source_shard": ["fresh", "fresh"],
            "periodic_nonbond_vdw_q05": [0.89, 1.1],
            "periodic_contact_coord105": [4.0, 1.0],
            "analytic_supported": [True, True],
        }
    ).to_parquet(features_path, index=False)
    output_dir = tmp_path / "predictions"

    manifest = apply_frozen_rule(
        frozen_rule_path=rule_path,
        feature_paths=(features_path,),
        output_dir=output_dir,
    )

    predictions = pd.read_parquet(output_dir / "next31_predictions.parquet")
    assert predictions["reject"].tolist() == [True, False]
    assert not any(
        token in column.lower()
        for column in predictions.columns
        for token in ("energy", "force", "stress", "endpoint", "label", "dft")
    )
    assert manifest["labels_opened"] is False
    with pytest.raises(FileExistsError):
        apply_frozen_rule(
            frozen_rule_path=rule_path,
            feature_paths=(features_path,),
            output_dir=output_dir,
        )


def test_freeze_development_rule_publishes_exact_exposed_shards(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.parquet"
    endpoint_path = tmp_path / "endpoints.parquet"
    features = pd.DataFrame(
        {
            "material_id": [f"id-{index}" for index in range(40)],
            "source_shard": ["dev-a"] * 20 + ["other"] * 20,
            "periodic_nonbond_vdw_q05": np.linspace(0.8, 1.2, 40),
            "periodic_contact_coord105": np.linspace(0.0, 8.0, 40),
            "analytic_supported": [True] * 40,
        }
    )
    endpoints = features[["material_id", "source_shard"]].copy()
    endpoints["energy_drop_pa"] = np.linspace(0.0, 0.1, 40)
    features.to_parquet(feature_path, index=False)
    endpoints.to_parquet(endpoint_path, index=False)
    output_dir = tmp_path / "freeze"

    manifest = freeze_development_rule(
        feature_paths=(feature_path,),
        endpoints_path=endpoint_path,
        development_shards=("dev-a",),
        output_dir=output_dir,
        rejection_fraction=0.10,
    )

    rule = json.loads((output_dir / "NEXT31_FROZEN_ENERGY_RULE.json").read_text())
    assert rule["development_shards"] == ["dev-a"]
    assert rule["confirmation_rows_used"] == 0
    assert manifest["labels_opened_for_development"] is True
    assert manifest["confirmation_labels_used_for_selection"] is False
