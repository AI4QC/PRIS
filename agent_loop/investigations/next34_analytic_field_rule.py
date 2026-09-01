"""Bounded NEXT34 development scan and immutable label-free rule replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import (
    _publish_directory_no_replace,
    _sha256,
    _strict_json,
)
from src.next32_inorganic_response_rule import (
    REJECTION_FRACTIONS,
    _candidate_metrics,
    _joined_development,
    _term_score,
    classify_dft_response,
    promotion_gates,
)
from src.next32_omat24_cohort import ENDPOINT_NAME, ENDPOINT_PROTOCOL
from src.next34_analytic_field_features import (
    FEATURE_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next34-analytic-electrostatic-field-rule-v1"
SCAN_NAME = "next34_development_scan.parquet"
FROZEN_RULE_NAME = "NEXT34_FROZEN_ANALYTIC_FIELD_RULE.json"
PREDICTIONS_NAME = "next34_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
_FORBIDDEN_FEATURE_TOKENS = (
    "energy",
    "force",
    "stress",
    "dft",
    "endpoint",
    "label",
    "target",
    "relax",
    "mattersim",
    "mlip",
)


def _term(feature: str, direction: float) -> dict[str, object]:
    return {"feature": feature, "direction": direction, "transform": "identity"}


TERM_CATALOGUE: Mapping[str, Mapping[str, object]] = {
    "aefi_field_rms_high": _term("aefi_field_rms", 1.0),
    "aefi_field_q95_high": _term("aefi_field_q95", 1.0),
    "aefi_field_max_high": _term("aefi_field_max", 1.0),
    "aefi_residual_rms_high": _term("aefi_residual_rms", 1.0),
    "aefi_residual_q95_high": _term("aefi_residual_q95", 1.0),
    "aefi_residual_max_high": _term("aefi_residual_max", 1.0),
    "aefi_field_tensor_deviator_high": _term("aefi_field_tensor_deviator", 1.0),
    "steric_rep12_vector_rms_high": _term("steric_rep12_vector_rms", 1.0),
    "steric_rep12_vector_q95_high": _term("steric_rep12_vector_q95", 1.0),
    "steric_rep12_vector_max_high": _term("steric_rep12_vector_max", 1.0),
    "steric_overlap2_vector_rms_high": _term("steric_overlap2_vector_rms", 1.0),
    "steric_rep12_tensor_deviator_high": _term(
        "steric_rep12_tensor_deviator", 1.0
    ),
    "sivr_site_imbalance_high": _term("sivr_site_imbalance_rms", 1.0),
    "sivr_edge_mismatch_high": _term("sivr_edge_mismatch_q95", 1.0),
    "cov_q05_low": _term("cov_q05", -1.0),
}

MECHANISM_PAIRS = (
    ("aefi_field_rms_high", "steric_rep12_vector_rms_high"),
    ("aefi_field_q95_high", "steric_rep12_vector_q95_high"),
    ("aefi_field_max_high", "steric_rep12_vector_max_high"),
    ("aefi_residual_rms_high", "steric_overlap2_vector_rms_high"),
    ("aefi_residual_q95_high", "sivr_site_imbalance_high"),
    ("aefi_residual_max_high", "steric_rep12_vector_max_high"),
    ("aefi_field_tensor_deviator_high", "steric_rep12_tensor_deviator_high"),
    ("aefi_field_rms_high", "cov_q05_low"),
    ("aefi_residual_rms_high", "sivr_site_imbalance_high"),
    ("steric_rep12_vector_rms_high", "sivr_site_imbalance_high"),
    ("cov_q05_low", "sivr_edge_mismatch_high"),
)
CANDIDATE_TERM_SETS = tuple((name,) for name in TERM_CATALOGUE) + MECHANISM_PAIRS


def _fit_identity_term(
    features: pd.DataFrame, name: str
) -> tuple[dict[str, object] | None, str | None]:
    specification = TERM_CATALOGUE[name]
    feature = str(specification["feature"])
    if feature not in features:
        return None, f"{feature} is absent"
    raw = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
    finite = np.isfinite(raw)
    if not finite.any():
        return None, f"{feature} has no finite development values"
    q25, median, q75 = np.quantile(raw[finite], [0.25, 0.50, 0.75])
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= 0.0:
        return None, f"{name} has zero development IQR"
    return {
        "feature": feature,
        "direction": float(specification["direction"]),
        "transform": "identity",
        "median": float(median),
        "iqr": iqr,
    }, None


def _candidate_score(
    features: pd.DataFrame,
    terms: Sequence[str],
    parameters: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    support = np.ones(len(features), dtype=bool)
    score = np.zeros(len(features), dtype=float)
    for term in terms:
        component, component_support = _term_score(features, parameters[term])
        support &= component_support
        score += np.nan_to_num(component, nan=0.0)
    score[~support] = np.nan
    return score, support


def scan_development_candidates(
    features: pd.DataFrame, endpoints: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Evaluate the exact frozen NEXT34 catalogue on exposed development rows."""

    joined = _joined_development(features, endpoints)
    severe, protected = classify_dft_response(joined)
    parameters: dict[str, dict[str, object]] = {}
    disabled: dict[str, str] = {}
    for name in TERM_CATALOGUE:
        parameter, reason = _fit_identity_term(joined, name)
        if parameter is None:
            disabled[name] = reason or "term is disabled"
        else:
            parameters[name] = parameter

    rows: list[dict[str, object]] = []
    passing: list[tuple[dict[str, object], dict[str, dict[str, object]]]] = []
    for terms in CANDIDATE_TERM_SETS:
        candidate_id = "+".join(terms)
        unavailable = [term for term in terms if term not in parameters]
        if unavailable:
            reason = "; ".join(disabled[term] for term in unavailable)
            for fraction in REJECTION_FRACTIONS:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "terms": list(terms),
                        "n_terms": len(terms),
                        "rejection_fraction": fraction,
                        "enabled": False,
                        "disabled_reason": reason,
                        "promotion_eligible": True,
                        "gates_passed": False,
                        "promotion": False,
                    }
                )
            continue
        score, support = _candidate_score(joined, terms, parameters)
        for fraction in REJECTION_FRACTIONS:
            threshold = float(
                np.quantile(score[support], 1.0 - fraction, method="higher")
            )
            reject = support & (score >= threshold)
            metrics = _candidate_metrics(
                score=score,
                support=support,
                reject=reject,
                severe=severe,
                protected=protected,
            )
            gate_inputs = {
                key: float(metrics[key])
                for key in (
                    "coverage_lb",
                    "protected_recall_lb",
                    "severe_precision_lb",
                    "savings_lb",
                    "auc",
                    "precision_lift_over_prevalence",
                )
            }
            gates = promotion_gates(gate_inputs)
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "terms": list(terms),
                "n_terms": len(terms),
                "rejection_fraction": fraction,
                "threshold": threshold,
                "enabled": True,
                "disabled_reason": None,
                "promotion_eligible": True,
                **metrics,
                **{f"gate_{name}": value for name, value in gates.items()},
                "gates_passed": all(gates.values()),
                "promotion": False,
            }
            rows.append(row)
            if all(gates.values()):
                passing.append((row, {term: parameters[term] for term in terms}))
    scan = pd.DataFrame(rows)
    if not passing:
        return scan, None
    winner, winner_parameters = min(
        passing,
        key=lambda item: (
            -float(item[0]["severe_precision_lb"]),
            -float(item[0]["savings_lb"]),
            -float(item[0]["auc"]),
            int(item[0]["n_terms"]),
            float(item[0]["rejection_fraction"]),
            str(item[0]["candidate_id"]),
        ),
    )
    winning = scan.candidate_id.eq(str(winner["candidate_id"])) & scan.rejection_fraction.eq(
        float(winner["rejection_fraction"])
    )
    scan.loc[winning, "promotion"] = True
    terms = list(winner["terms"])
    metric_names = (
        "rows",
        "supported",
        "rejected",
        "severe",
        "protected",
        "rejected_severe",
        "rejected_protected",
        "coverage_lb",
        "protected_recall_lb",
        "severe_precision_lb",
        "savings_lb",
        "auc",
        "severe_prevalence_ub",
        "precision_lift_over_prevalence",
    )
    metrics = {
        name: (
            int(winner[name])
            if isinstance(winner[name], (int, np.integer))
            else float(winner[name])
        )
        for name in metric_names
    }
    gates = {
        name.removeprefix("gate_"): bool(winner[name])
        for name in winner
        if name.startswith("gate_")
    }
    rule: dict[str, object] = {
        "protocol": PROTOCOL,
        "eligible": True,
        "formula": " + ".join(terms),
        "terms": terms,
        "term_parameters": winner_parameters,
        "threshold": float(winner["threshold"]),
        "rejection_fraction": float(winner["rejection_fraction"]),
        "development_rows": len(joined),
        "development_metrics": metrics,
        "development_gates": gates,
        "development_gates_passed": True,
        "confirmation_rows_used": 0,
        "confirmation_labels_used_for_selection": False,
        "dft_or_learned_proxy_used_at_execution": False,
        "classical_analytic_electrostatics_used": any(
            term.startswith("aefi_") for term in terms
        ),
        "same_composition_candidates_used": False,
    }
    return scan, rule


def compute_analytic_field_risk(
    features: pd.DataFrame, rule: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one frozen NEXT34 rule without endpoint access or refitting."""

    if rule.get("protocol") != PROTOCOL or rule.get("eligible") is not True:
        raise ValueError("frozen NEXT34 rule is not eligible")
    raw_terms = rule.get("terms")
    raw_parameters = rule.get("term_parameters")
    if not isinstance(raw_terms, list) or not raw_terms or len(raw_terms) > 2:
        raise ValueError("frozen NEXT34 terms are invalid")
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("frozen NEXT34 term parameters are invalid")
    terms = tuple(str(term) for term in raw_terms)
    if terms not in CANDIDATE_TERM_SETS or set(raw_parameters) != set(terms):
        raise ValueError("frozen NEXT34 formula is not in the catalogue")
    parameters: dict[str, Mapping[str, object]] = {}
    for term in terms:
        value = raw_parameters[term]
        if not isinstance(value, Mapping):
            raise ValueError("frozen NEXT34 term parameters are invalid")
        specification = TERM_CATALOGUE[term]
        if any(
            value.get(key) != specification[key]
            for key in ("feature", "direction", "transform")
        ):
            raise ValueError("frozen NEXT34 term specification changed")
        parameters[term] = value
    score, support = _candidate_score(features, terms, parameters)
    threshold = float(rule.get("threshold", math.nan))
    if not np.isfinite(threshold):
        raise ValueError("frozen NEXT34 threshold is invalid")
    reject = support & (score >= threshold)
    return score, support, reject


def _validate_feature_artifact(
    feature_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    if feature_path.name != FEATURE_NAME or not feature_path.is_file():
        raise ValueError("NEXT34 feature path/name is invalid")
    manifest = _strict_json(manifest_path, role="NEXT34 feature manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != FEATURE_PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("dft_values_used") is not False
        or manifest.get("classical_analytic_electrostatics_used") is not True
        or manifest.get("electronic_structure_calculation_used") is not False
        or manifest.get("model_or_proxy_potential_used") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(FEATURE_NAME) != _sha256(feature_path)
    ):
        raise ValueError("NEXT34 feature artifact crossed the label-free boundary")
    frame = pd.read_parquet(feature_path)
    forbidden = [
        str(column)
        for column in frame
        if any(token in str(column).lower() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            f"NEXT34 feature columns crossed the label-free boundary: {forbidden}"
        )
    if (
        "material_id" not in frame
        or frame.material_id.isna().any()
        or frame.material_id.astype(str).duplicated().any()
    ):
        raise ValueError("NEXT34 feature material IDs are invalid")
    frame = frame.copy()
    frame["material_id"] = frame.material_id.astype(str)
    return frame, manifest


def freeze_development_rule(
    *,
    feature_path: Path,
    feature_manifest_path: Path,
    endpoints_path: Path,
    endpoints_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Open exposed development endpoints and publish one immutable scan."""

    paths = {
        "features": Path(feature_path).resolve(),
        "feature_manifest": Path(feature_manifest_path).resolve(),
        "endpoints": Path(endpoints_path).resolve(),
        "endpoints_manifest": Path(endpoints_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    features, _feature_manifest = _validate_feature_artifact(
        paths["features"], paths["feature_manifest"]
    )
    if paths["endpoints"].name != ENDPOINT_NAME or not paths["endpoints"].is_file():
        raise ValueError("NEXT34 development endpoint path/name is invalid")
    endpoint_manifest = _strict_json(
        paths["endpoints_manifest"], role="NEXT34 development endpoint manifest"
    )
    endpoint_outputs = endpoint_manifest.get("outputs_sha256")
    if (
        endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_manifest.get("labels_opened") is not True
        or not isinstance(endpoint_outputs, Mapping)
        or endpoint_outputs.get(ENDPOINT_NAME) != _sha256(paths["endpoints"])
    ):
        raise ValueError("NEXT34 development endpoint artifact is invalid")
    endpoints = pd.read_parquet(
        paths["endpoints"],
        columns=["material_id", "force_max", "force_rms", "stress_norm"],
    )
    scan, rule = scan_development_candidates(features, endpoints)
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "exposed_rattled_relax_next34_development_scan",
        "labels_opened_for_development": True,
        "confirmation_rows_used": 0,
        "confirmation_labels_used_for_selection": False,
        "promoted": rule is not None,
        "candidate_term_sets": [list(terms) for terms in CANDIDATE_TERM_SETS],
        "rejection_fractions": list(REJECTION_FRACTIONS),
        "inputs_sha256": input_hashes,
        "executed_source_sha256": {
            "src/next34_analytic_field_rule.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        scan_path = staging / SCAN_NAME
        scan.to_parquet(scan_path, index=False)
        outputs = {SCAN_NAME: _sha256(scan_path)}
        if rule is not None:
            rule_path = staging / FROZEN_RULE_NAME
            rule_path.write_text(
                json.dumps(rule, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            outputs[FROZEN_RULE_NAME] = _sha256(rule_path)
        manifest["outputs_sha256"] = outputs
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if any(_sha256(path) != input_hashes[role] for role, path in paths.items()):
            raise RuntimeError("NEXT34 development input changed before publication")
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT34 rule source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def apply_frozen_rule(
    *,
    frozen_rule_path: Path,
    frozen_rule_manifest_path: Path,
    feature_paths: Sequence[Path],
    feature_manifest_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, object]:
    """Publish label-free NEXT34 predictions for one or more sealed cohorts."""

    rule_path = Path(frozen_rule_path).resolve()
    rule_manifest_path = Path(frozen_rule_manifest_path).resolve()
    features = [Path(path).resolve() for path in feature_paths]
    feature_manifests = [Path(path).resolve() for path in feature_manifest_paths]
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if (
        rule_path.name != FROZEN_RULE_NAME
        or not rule_path.is_file()
        or not rule_manifest_path.is_file()
        or not features
        or len(features) != len(feature_manifests)
    ):
        raise ValueError("NEXT34 rule application inputs are invalid")
    rule_manifest = _strict_json(rule_manifest_path, role="NEXT34 frozen-rule manifest")
    outputs = rule_manifest.get("outputs_sha256")
    if (
        rule_manifest.get("protocol") != PROTOCOL
        or rule_manifest.get("promoted") is not True
        or not isinstance(outputs, Mapping)
        or outputs.get(FROZEN_RULE_NAME) != _sha256(rule_path)
    ):
        raise ValueError("NEXT34 frozen rule is not promoted and hash-locked")
    try:
        rule = json.loads(rule_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("NEXT34 frozen rule JSON is invalid") from exc
    if not isinstance(rule, dict):
        raise ValueError("NEXT34 frozen rule JSON is invalid")

    parts: list[pd.DataFrame] = []
    feature_inputs: list[dict[str, str]] = []
    for feature, feature_manifest in zip(features, feature_manifests, strict=True):
        frame, _upstream = _validate_feature_artifact(feature, feature_manifest)
        score, supported, reject = compute_analytic_field_risk(frame, rule)
        source = (
            frame.source_name.astype(str)
            if "source_name" in frame
            else pd.Series([feature.parent.name] * len(frame))
        )
        parent = (
            frame.parent_id.astype(str)
            if "parent_id" in frame
            else pd.Series([""] * len(frame))
        )
        parts.append(
            pd.DataFrame(
                {
                    "material_id": frame.material_id.astype(str),
                    "source_name": source.to_numpy(),
                    "parent_id": parent.to_numpy(),
                    "analytic_supported": supported,
                    "next34_risk_score": score,
                    "reject": reject,
                    "input_role": "unrelaxed_x0_geometry_only",
                }
            )
        )
        feature_inputs.append(
            {
                "feature_path": str(feature),
                "feature_sha256": _sha256(feature),
                "manifest_path": str(feature_manifest),
                "manifest_sha256": _sha256(feature_manifest),
            }
        )
    predictions = pd.concat(parts, ignore_index=True).sort_values(
        ["source_name", "material_id"], kind="stable", ignore_index=True
    )
    if predictions.material_id.duplicated().any():
        raise ValueError("NEXT34 prediction material IDs are duplicated")
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    input_hashes: dict[str, object] = {
        "frozen_rule": _sha256(rule_path),
        "frozen_rule_manifest": _sha256(rule_manifest_path),
        "features": feature_inputs,
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "prospective_label_free_analytic_field_prediction",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "dft_values_used": False,
        "classical_analytic_electrostatics_used": True,
        "electronic_structure_calculation_used": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "counts": {
            "rows": len(predictions),
            "supported": int(predictions.analytic_supported.sum()),
            "rejected": int(predictions.reject.sum()),
            "sources": int(predictions.source_name.nunique()),
        },
        "inputs_sha256": input_hashes,
        "executed_source_sha256": {
            "src/next34_analytic_field_rule.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        prediction_path = staging / PREDICTIONS_NAME
        predictions.to_parquet(prediction_path, index=False)
        manifest["outputs_sha256"] = {PREDICTIONS_NAME: _sha256(prediction_path)}
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if _sha256(rule_path) != input_hashes["frozen_rule"] or _sha256(
            rule_manifest_path
        ) != input_hashes["frozen_rule_manifest"]:
            raise RuntimeError("NEXT34 frozen rule changed before publication")
        for item in feature_inputs:
            if _sha256(Path(item["feature_path"])) != item["feature_sha256"] or _sha256(
                Path(item["manifest_path"])
            ) != item["manifest_sha256"]:
                raise RuntimeError("NEXT34 feature changed before publication")
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT34 rule source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "CANDIDATE_TERM_SETS",
    "FROZEN_RULE_NAME",
    "MANIFEST_NAME",
    "MECHANISM_PAIRS",
    "PREDICTIONS_NAME",
    "PROTOCOL",
    "REJECTION_FRACTIONS",
    "SCAN_NAME",
    "TERM_CATALOGUE",
    "apply_frozen_rule",
    "compute_analytic_field_risk",
    "freeze_development_rule",
    "scan_development_candidates",
]
