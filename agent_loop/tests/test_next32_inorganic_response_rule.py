from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next32_inorganic_response_rule import (
    FROZEN_RULE_NAME,
    MANIFEST_NAME,
    PREDICTIONS_NAME,
    PROMOTION_GATE_NAMES,
    REJECTION_FRACTIONS,
    SCAN_NAME,
    apply_frozen_rule,
    classify_dft_response,
    compute_inorganic_risk,
    freeze_development_rule,
    promotion_gates,
    scan_development_candidates,
    wilson_one_sided,
)
from src.next32_inorganic_response_features import (
    FEATURE_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next32_omat24_cohort import ENDPOINT_NAME, ENDPOINT_PROTOCOL


def test_frozen_severe_and_protected_endpoint_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "force_max": [1.0, 0.49, 0.49, 0.49, 0.51],
            "force_rms": [0.1, 0.40, 0.19, 0.19, 0.19],
            "stress_norm": [0.001, 0.001, 0.030, 0.014, 0.014],
        }
    )

    severe, protected = classify_dft_response(frame)

    assert severe.tolist() == [True, True, True, False, False]
    assert protected.tolist() == [False, False, False, True, False]


def test_one_sided_wilson_bounds_are_directional_and_finite() -> None:
    lower = wilson_one_sided(90, 100, bound="lower")
    upper = wilson_one_sided(90, 100, bound="upper")

    assert 0.82 < lower < 0.90
    assert 0.90 < upper < 0.96
    assert wilson_one_sided(0, 0, bound="lower") == 0.0
    assert wilson_one_sided(0, 0, bound="upper") == 1.0
    with pytest.raises(ValueError):
        wilson_one_sided(2, 1, bound="lower")


def test_promotion_requires_all_exact_preregistered_gates() -> None:
    metrics = {
        "coverage_lb": 0.95,
        "protected_recall_lb": 0.98,
        "severe_precision_lb": 0.90,
        "savings_lb": 0.05,
        "auc": 0.85,
        "precision_lift_over_prevalence": 0.20,
    }

    gates = promotion_gates(metrics)

    assert tuple(gates) == PROMOTION_GATE_NAMES
    assert all(gates.values())
    for name in metrics:
        failed = dict(metrics)
        failed[name] = np.nextafter(metrics[name], -np.inf)
        assert not all(promotion_gates(failed).values()), name


def _development(n: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    material_id = [f"dev-{index:04d}" for index in range(n)]
    features = pd.DataFrame(
        {
            "material_id": material_id,
            "source_name": "rattled-relax",
            "cov_overlap2_pa": np.linspace(2.0, 0.0, n),
        }
    )
    severe = np.arange(n) < 100
    endpoints = pd.DataFrame(
        {
            "material_id": material_id,
            "force_max": np.where(severe, 1.5, 0.1),
            "force_rms": np.where(severe, 0.5, 0.1),
            "stress_norm": np.where(severe, 0.04, 0.005),
        }
    )
    return features, endpoints


def test_bounded_scan_promotes_one_frozen_development_only_formula() -> None:
    features, endpoints = _development()

    scan, rule = scan_development_candidates(features, endpoints)

    assert rule is not None
    assert rule["eligible"] is True
    assert rule["confirmation_rows_used"] == 0
    assert rule["confirmation_labels_used_for_selection"] is False
    assert rule["rejection_fraction"] in REJECTION_FRACTIONS
    assert rule["terms"] == ["cov_overlap2_high"]
    assert rule["development_gates_passed"] is True
    assert scan.promotion.any()


def test_applying_confirmation_features_cannot_change_development_constants() -> None:
    features, endpoints = _development()
    _scan, rule = scan_development_candidates(features, endpoints)
    assert rule is not None
    frozen = copy.deepcopy(rule)
    confirmation = pd.DataFrame(
        {
            "material_id": ["confirm-a", "confirm-b"],
            "source_name": ["rattled-300", "rattled-1000"],
            "cov_overlap2_pa": [1.0e9, -1.0e9],
        }
    )

    score, supported, rejected = compute_inorganic_risk(confirmation, rule)

    assert np.isfinite(score[supported]).all()
    assert len(rejected) == 2
    assert rule == frozen
    assert json.dumps(rule, sort_keys=True, allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed_development_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    features, endpoints = _development()
    feature_path = tmp_path / FEATURE_NAME
    endpoint_path = tmp_path / ENDPOINT_NAME
    features.to_parquet(feature_path, index=False)
    endpoints.to_parquet(endpoint_path, index=False)
    feature_manifest_path = tmp_path / "feature-manifest.json"
    endpoint_manifest_path = tmp_path / "endpoint-manifest.json"
    feature_manifest_path.write_text(
        json.dumps(
            {
                "protocol": FEATURE_PROTOCOL,
                "labels_opened": False,
                "endpoint_fields_read": False,
                "model_or_proxy_potential_used": False,
                "outputs_sha256": {FEATURE_NAME: _sha256(feature_path)},
            }
        )
        + "\n"
    )
    endpoint_manifest_path.write_text(
        json.dumps(
            {
                "protocol": ENDPOINT_PROTOCOL,
                "labels_opened": True,
                "outputs_sha256": {ENDPOINT_NAME: _sha256(endpoint_path)},
            }
        )
        + "\n"
    )
    return feature_path, feature_manifest_path, endpoint_path, endpoint_manifest_path


def test_freeze_then_apply_publishes_hash_locked_label_free_predictions(
    tmp_path: Path,
) -> None:
    feature_path, feature_manifest_path, endpoint_path, endpoint_manifest_path = (
        _sealed_development_inputs(tmp_path)
    )
    freeze_dir = tmp_path / "freeze"

    freeze_manifest = freeze_development_rule(
        feature_path=feature_path,
        feature_manifest_path=feature_manifest_path,
        endpoints_path=endpoint_path,
        endpoints_manifest_path=endpoint_manifest_path,
        output_dir=freeze_dir,
    )

    assert freeze_manifest["promoted"] is True
    assert (freeze_dir / SCAN_NAME).is_file()
    assert (freeze_dir / FROZEN_RULE_NAME).is_file()
    assert freeze_manifest["outputs_sha256"][FROZEN_RULE_NAME] == _sha256(
        freeze_dir / FROZEN_RULE_NAME
    )
    prediction_dir = tmp_path / "predictions"
    prediction_manifest = apply_frozen_rule(
        frozen_rule_path=freeze_dir / FROZEN_RULE_NAME,
        frozen_rule_manifest_path=freeze_dir / MANIFEST_NAME,
        feature_paths=[feature_path],
        feature_manifest_paths=[feature_manifest_path],
        output_dir=prediction_dir,
    )
    predictions = pd.read_parquet(prediction_dir / PREDICTIONS_NAME)
    assert len(predictions) == 1000
    assert predictions.reject.sum() == 100
    assert prediction_manifest["labels_opened"] is False
    assert prediction_manifest["endpoint_fields_read"] is False
    assert not any(
        token in column.lower()
        for column in predictions
        for token in ("energy", "force", "stress", "dft", "endpoint", "label")
    )
    with pytest.raises(FileExistsError):
        apply_frozen_rule(
            frozen_rule_path=freeze_dir / FROZEN_RULE_NAME,
            frozen_rule_manifest_path=freeze_dir / MANIFEST_NAME,
            feature_paths=[feature_path],
            feature_manifest_paths=[feature_manifest_path],
            output_dir=prediction_dir,
        )


def test_application_rejects_feature_manifest_after_labels_open(tmp_path: Path) -> None:
    feature_path, feature_manifest_path, endpoint_path, endpoint_manifest_path = (
        _sealed_development_inputs(tmp_path)
    )
    freeze_dir = tmp_path / "freeze"
    freeze_development_rule(
        feature_path=feature_path,
        feature_manifest_path=feature_manifest_path,
        endpoints_path=endpoint_path,
        endpoints_manifest_path=endpoint_manifest_path,
        output_dir=freeze_dir,
    )
    manifest = json.loads(feature_manifest_path.read_text())
    manifest["labels_opened"] = True
    feature_manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="label-free"):
        apply_frozen_rule(
            frozen_rule_path=freeze_dir / FROZEN_RULE_NAME,
            frozen_rule_manifest_path=freeze_dir / MANIFEST_NAME,
            feature_paths=[feature_path],
            feature_manifest_paths=[feature_manifest_path],
            output_dir=tmp_path / "predictions",
        )
