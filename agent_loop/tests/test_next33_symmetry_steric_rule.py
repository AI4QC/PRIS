from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next33_symmetry_steric_rule import (
    CANDIDATE_TERM_SETS,
    DIAGNOSTIC_ONLY_TERM_SETS,
    FROZEN_RULE_NAME,
    MANIFEST_NAME,
    PREDICTIONS_NAME,
    REJECTION_FRACTIONS,
    SCAN_NAME,
    SYMMETRY_TERMS,
    TERM_CATALOGUE,
    apply_frozen_rule,
    compute_symmetry_steric_risk,
    freeze_development_rule,
    scan_development_candidates,
)
from src.next33_symmetry_steric_features import (
    FEATURE_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next32_omat24_cohort import ENDPOINT_NAME, ENDPOINT_PROTOCOL


def test_frozen_catalogue_has_only_geometry_terms_and_symmetry_singles_are_diagnostic() -> None:
    assert TERM_CATALOGUE["cov_q05_low"] == {
        "feature": "cov_q05",
        "direction": -1.0,
        "transform": "identity",
    }
    assert TERM_CATALOGUE["steric_rep12_vector_rms_high"]["direction"] == 1.0
    assert set(DIAGNOSTIC_ONLY_TERM_SETS) == {(name,) for name in SYMMETRY_TERMS}
    assert set(DIAGNOSTIC_ONLY_TERM_SETS).issubset(CANDIDATE_TERM_SETS)
    for terms in CANDIDATE_TERM_SETS:
        if any(term in SYMMETRY_TERMS for term in terms) and terms not in DIAGNOSTIC_ONLY_TERM_SETS:
            assert any(term not in SYMMETRY_TERMS for term in terms)
    assert not any(
        token in str(specification["feature"]).lower()
        for specification in TERM_CATALOGUE.values()
        for token in ("energy", "force", "stress", "dft", "relax", "label", "target")
    )


def _endpoints(n: int = 1000) -> pd.DataFrame:
    severe = np.arange(n) < 100
    return pd.DataFrame(
        {
            "material_id": [f"dev-{index:04d}" for index in range(n)],
            "force_max": np.where(severe, 1.5, 0.1),
            "force_rms": np.where(severe, 0.5, 0.1),
            "stress_norm": np.where(severe, 0.04, 0.005),
        }
    )


def test_bounded_scan_promotes_physical_steric_term_with_fixed_fraction() -> None:
    endpoints = _endpoints()
    features = pd.DataFrame(
        {
            "material_id": endpoints.material_id,
            "steric_rep12_vector_rms": np.linspace(2.0, 0.0, len(endpoints)),
        }
    )

    scan, rule = scan_development_candidates(features, endpoints)

    assert rule is not None
    assert rule["terms"] == ["steric_rep12_vector_rms_high"]
    assert rule["rejection_fraction"] in REJECTION_FRACTIONS
    assert rule["development_gates_passed"] is True
    assert rule["symmetry_only"] is False
    assert scan.promotion.sum() == 1


def test_symmetry_only_candidate_cannot_promote_even_when_all_statistical_gates_pass() -> None:
    endpoints = _endpoints()
    features = pd.DataFrame(
        {
            "material_id": endpoints.material_id,
            "sym_recovery_residual_rms_rel": np.linspace(2.0, 0.0, len(endpoints)),
        }
    )

    scan, rule = scan_development_candidates(features, endpoints)

    row = scan.loc[
        scan.candidate_id.eq("sym_residual_rms_high")
        & scan.rejection_fraction.eq(0.10)
    ].iloc[0]
    assert row.gates_passed
    assert not row.promotion_eligible
    assert not row.promotion
    assert rule is None


def test_zero_iqr_disables_only_formulas_that_require_that_term() -> None:
    endpoints = _endpoints()
    features = pd.DataFrame(
        {
            "material_id": endpoints.material_id,
            "steric_rep12_pa": 1.0,
            "steric_rep12_vector_rms": np.linspace(2.0, 0.0, len(endpoints)),
        }
    )

    scan, rule = scan_development_candidates(features, endpoints)

    constant = scan.loc[scan.candidate_id.eq("steric_rep12_pa_high")]
    varying = scan.loc[scan.candidate_id.eq("steric_rep12_vector_rms_high")]
    assert not constant.enabled.any()
    assert varying.enabled.all()
    assert rule is not None


def test_applying_extreme_confirmation_features_does_not_change_frozen_constants() -> None:
    endpoints = _endpoints()
    features = pd.DataFrame(
        {
            "material_id": endpoints.material_id,
            "steric_rep12_vector_rms": np.linspace(2.0, 0.0, len(endpoints)),
        }
    )
    _scan, rule = scan_development_candidates(features, endpoints)
    assert rule is not None
    frozen = copy.deepcopy(rule)
    confirmation = pd.DataFrame(
        {
            "material_id": ["confirm-a", "confirm-b"],
            "source_name": ["rattled-300", "rattled-1000"],
            "steric_rep12_vector_rms": [1.0e12, -1.0e12],
        }
    )

    score, supported, rejected = compute_symmetry_steric_risk(confirmation, rule)

    assert supported.all()
    assert np.isfinite(score).all()
    assert len(rejected) == 2
    assert rule == frozen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    endpoints = _endpoints()
    features = pd.DataFrame(
        {
            "material_id": endpoints.material_id,
            "source_name": "rattled-relax",
            "parent_id": [f"parent-{index:04d}" for index in range(len(endpoints))],
            "steric_rep12_vector_rms": np.linspace(2.0, 0.0, len(endpoints)),
        }
    )
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


def test_freeze_and_label_free_application_are_hash_locked_and_no_replace(
    tmp_path: Path,
) -> None:
    feature_path, feature_manifest_path, endpoint_path, endpoint_manifest_path = (
        _sealed_inputs(tmp_path)
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
    predictions_dir = tmp_path / "predictions"
    prediction_manifest = apply_frozen_rule(
        frozen_rule_path=freeze_dir / FROZEN_RULE_NAME,
        frozen_rule_manifest_path=freeze_dir / MANIFEST_NAME,
        feature_paths=[feature_path],
        feature_manifest_paths=[feature_manifest_path],
        output_dir=predictions_dir,
    )
    predictions = pd.read_parquet(predictions_dir / PREDICTIONS_NAME)
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
            output_dir=predictions_dir,
        )


def test_application_rejects_feature_manifest_with_opened_labels(tmp_path: Path) -> None:
    feature_path, feature_manifest_path, endpoint_path, endpoint_manifest_path = (
        _sealed_inputs(tmp_path)
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
