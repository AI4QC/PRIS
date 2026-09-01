from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next32_omat24_cohort import ENDPOINT_NAME, ENDPOINT_PROTOCOL
from src.next36_charge_spectrum_features import (
    FEATURE_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next36_charge_spectrum_rule import (
    CANDIDATE_TERM_SETS,
    FROZEN_RULE_NAME,
    MANIFEST_NAME,
    MECHANISM_PAIRS,
    PREDICTIONS_NAME,
    REJECTION_FRACTIONS,
    SCAN_NAME,
    TERM_CATALOGUE,
    apply_frozen_rule,
    compute_charge_spectrum_risk,
    freeze_development_rule,
    scan_development_candidates,
)


def test_catalogue_is_exact_bounded_and_all_spectrum_terms_are_high_risk() -> None:
    spectrum = {
        name: specification
        for name, specification in TERM_CATALOGUE.items()
        if name.startswith("csf_")
    }
    assert len(spectrum) == 6
    assert all(specification["direction"] == 1.0 for specification in spectrum.values())
    assert len(TERM_CATALOGUE) == 10
    assert len(MECHANISM_PAIRS) == 7
    assert len(CANDIDATE_TERM_SETS) == 17
    assert len(set(CANDIDATE_TERM_SETS)) == 17
    assert tuple(REJECTION_FRACTIONS) == (0.025, 0.05, 0.075, 0.10, 0.15)
    assert not any(
        token in str(specification["feature"]).lower()
        for specification in TERM_CATALOGUE.values()
        for token in (
            "energy",
            "force",
            "stress",
            "dft",
            "relax",
            "label",
            "target",
            "mlip",
        )
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


def test_scan_has_85_rows_and_promotes_perfect_spectrum_term() -> None:
    endpoints = _endpoints()
    features = pd.DataFrame(
        {
            "material_id": endpoints.material_id,
            "csf_gaussian_t040": np.linspace(2.0, 0.0, len(endpoints)),
        }
    )

    scan, rule = scan_development_candidates(features, endpoints)

    assert len(scan) == 85
    assert scan.candidate_id.nunique() == 17
    assert rule is not None
    assert rule["terms"] == ["csf_gaussian_t040_high"]
    assert rule["development_gates_passed"] is True
    assert scan.promotion.sum() == 1


def test_missing_and_zero_iqr_disable_only_dependent_formulas() -> None:
    endpoints = _endpoints()
    features = pd.DataFrame(
        {
            "material_id": endpoints.material_id,
            "csf_gaussian_t040": np.linspace(2.0, 0.0, len(endpoints)),
            "csf_gaussian_t025": 1.0,
        }
    )
    scan, rule = scan_development_candidates(features, endpoints)

    assert not scan.loc[
        scan.candidate_id.eq("csf_gaussian_t025_high"), "enabled"
    ].any()
    assert scan.loc[
        scan.candidate_id.eq("csf_gaussian_t040_high"), "enabled"
    ].all()
    assert rule is not None


def test_replay_does_not_refit_frozen_constants() -> None:
    endpoints = _endpoints()
    features = pd.DataFrame(
        {
            "material_id": endpoints.material_id,
            "csf_gaussian_t040": np.linspace(2.0, 0.0, len(endpoints)),
        }
    )
    _scan, rule = scan_development_candidates(features, endpoints)
    assert rule is not None
    frozen = copy.deepcopy(rule)
    replay = pd.DataFrame(
        {
            "material_id": ["confirm-a", "confirm-b"],
            "source_name": ["rattled-300", "rattled-1000"],
            "csf_gaussian_t040": [1.0e12, -1.0e12],
        }
    )

    score, supported, rejected = compute_charge_spectrum_risk(replay, rule)

    assert supported.all() and np.isfinite(score).all() and len(rejected) == 2
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
            "csf_gaussian_t040": np.linspace(2.0, 0.0, len(endpoints)),
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
                "dft_values_used": False,
                "weighted_charge_spectrum_used": True,
                "thermodynamic_limit_hyperuniformity_claimed": False,
                "electronic_structure_calculation_used": False,
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
    return (
        feature_path,
        feature_manifest_path,
        endpoint_path,
        endpoint_manifest_path,
    )


def test_freeze_and_label_free_replay_are_hash_locked_and_no_replace(tmp_path: Path) -> None:
    feature_path, feature_manifest_path, endpoint_path, endpoint_manifest_path = (
        _sealed_inputs(tmp_path)
    )
    freeze_dir = tmp_path / "freeze"
    manifest = freeze_development_rule(
        feature_path=feature_path,
        feature_manifest_path=feature_manifest_path,
        endpoints_path=endpoint_path,
        endpoints_manifest_path=endpoint_manifest_path,
        output_dir=freeze_dir,
    )

    assert manifest["promoted"] is True
    assert (freeze_dir / SCAN_NAME).is_file()
    assert (freeze_dir / FROZEN_RULE_NAME).is_file()
    prediction_dir = tmp_path / "predictions"
    replay_manifest = apply_frozen_rule(
        frozen_rule_path=freeze_dir / FROZEN_RULE_NAME,
        frozen_rule_manifest_path=freeze_dir / MANIFEST_NAME,
        feature_paths=[feature_path],
        feature_manifest_paths=[feature_manifest_path],
        output_dir=prediction_dir,
    )
    predictions = pd.read_parquet(prediction_dir / PREDICTIONS_NAME)
    assert predictions.reject.sum() == 100
    assert replay_manifest["labels_opened"] is False
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


def test_application_rejects_opened_feature_manifest(tmp_path: Path) -> None:
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
    value = json.loads(feature_manifest_path.read_text())
    value["labels_opened"] = True
    feature_manifest_path.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="label-free"):
        apply_frozen_rule(
            frozen_rule_path=freeze_dir / FROZEN_RULE_NAME,
            frozen_rule_manifest_path=freeze_dir / MANIFEST_NAME,
            feature_paths=[feature_path],
            feature_manifest_paths=[feature_manifest_path],
            output_dir=tmp_path / "predictions",
        )
