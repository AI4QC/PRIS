"""Bounded development and frozen application of the NEXT32 inorganic law."""

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
from scipy.stats import rankdata

from src.next19_feature_build import (
    _publish_directory_no_replace,
    _sha256,
    _strict_json,
)
from src.next32_inorganic_response_features import (
    FEATURE_NAME,
    PROTOCOL as FEATURE_PROTOCOL,
)
from src.next32_omat24_cohort import ENDPOINT_NAME, ENDPOINT_PROTOCOL


PROTOCOL = "2026-08-03-next32-omat24-inorganic-response-rule-v1"
SCAN_NAME = "next32_development_scan.parquet"
FROZEN_RULE_NAME = "NEXT32_FROZEN_INORGANIC_RESPONSE_RULE.json"
PREDICTIONS_NAME = "next32_predictions.parquet"
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
REJECTION_FRACTIONS = (0.025, 0.05, 0.075, 0.10, 0.15)
PROMOTION_GATE_NAMES = (
    "coverage",
    "protected_recall",
    "severe_precision",
    "savings",
    "auc",
    "precision_lift_over_prevalence",
)
GATE_THRESHOLDS: Mapping[str, float] = {
    "coverage_lb": 0.95,
    "protected_recall_lb": 0.98,
    "severe_precision_lb": 0.90,
    "savings_lb": 0.05,
    "auc": 0.85,
    "precision_lift_over_prevalence": 0.20,
}

# The catalogue is fixed before development labels are joined.  ``direction``
# multiplies the robust z score; +1 means that a larger raw term is riskier.
TERM_CATALOGUE: Mapping[str, Mapping[str, object]] = {
    "cov_q01_low": {"feature": "cov_q01", "direction": -1.0, "transform": "identity"},
    "cov_q05_low": {"feature": "cov_q05", "direction": -1.0, "transform": "identity"},
    "cov_contact085_high": {"feature": "cov_contact085_pa", "direction": 1.0, "transform": "identity"},
    "cov_overlap2_high": {"feature": "cov_overlap2_pa", "direction": 1.0, "transform": "identity"},
    "cov_site_q95_high": {"feature": "cov_site_overlap_q95", "direction": 1.0, "transform": "identity"},
    "cov_site_max_high": {"feature": "cov_site_overlap_max", "direction": 1.0, "transform": "identity"},
    "sivr_edge_mismatch_high": {"feature": "sivr_edge_mismatch_q95", "direction": 1.0, "transform": "identity"},
    "sivr_site_imbalance_high": {"feature": "sivr_site_imbalance_rms", "direction": 1.0, "transform": "identity"},
    "sivr_anisotropy_high": {"feature": "sivr_cell_anisotropy", "direction": 1.0, "transform": "identity"},
    "madelung_weak_binding": {"feature": "nm_total_reduced", "direction": 1.0, "transform": "identity"},
    "madelung_site_spread_high": {"feature": "nm_site_spread", "direction": 1.0, "transform": "identity"},
    "scbve_mismatch_high": {"feature": "scbv_mismatch_q95", "direction": 1.0, "transform": "identity"},
    "scbve_vector_asymmetry_high": {"feature": "scbv_vector_asymmetry_rms", "direction": 1.0, "transform": "identity"},
    "scbve_scale_mismatch": {"feature": "scbv_global_scale", "direction": 1.0, "transform": "log_deviation"},
}

_MECHANISM_PAIRS = (
    ("cov_q05_low", "cov_contact085_high"),
    ("cov_q05_low", "cov_overlap2_high"),
    ("cov_q01_low", "cov_site_max_high"),
    ("cov_overlap2_high", "cov_site_q95_high"),
    ("sivr_edge_mismatch_high", "sivr_site_imbalance_high"),
    ("sivr_edge_mismatch_high", "scbve_mismatch_high"),
    ("sivr_site_imbalance_high", "scbve_vector_asymmetry_high"),
    ("madelung_weak_binding", "madelung_site_spread_high"),
    ("scbve_mismatch_high", "scbve_scale_mismatch"),
    ("cov_q05_low", "sivr_edge_mismatch_high"),
    ("cov_overlap2_high", "scbve_mismatch_high"),
    ("cov_site_q95_high", "scbve_vector_asymmetry_high"),
    ("cov_q05_low", "madelung_weak_binding"),
    ("cov_site_q95_high", "madelung_site_spread_high"),
)
CANDIDATE_TERM_SETS = tuple((name,) for name in TERM_CATALOGUE) + _MECHANISM_PAIRS


def classify_dft_response(endpoints: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen severe-response and protected-low-response labels."""

    required = {"force_max", "force_rms", "stress_norm"}
    if not required.issubset(endpoints.columns):
        raise ValueError(f"DFT endpoint lacks columns: {sorted(required - set(endpoints))}")
    values = {
        name: pd.to_numeric(endpoints[name], errors="coerce").to_numpy(float)
        for name in sorted(required)
    }
    if any(not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("DFT response endpoints must be finite")
    severe = (
        (values["force_max"] >= 1.0)
        | (values["force_rms"] >= 0.40)
        | (values["stress_norm"] >= 0.030)
    )
    protected = (
        (values["force_max"] <= 0.50)
        & (values["force_rms"] <= 0.20)
        & (values["stress_norm"] <= 0.015)
    )
    return severe, protected


def wilson_one_sided(successes: int, total: int, *, bound: str) -> float:
    """One-sided 95% Wilson score bound using z=1.6448536269514722."""

    if (
        type(successes) is not int
        or type(total) is not int
        or successes < 0
        or total < 0
        or successes > total
        or bound not in {"lower", "upper"}
    ):
        raise ValueError("invalid Wilson count or bound")
    if total == 0:
        return 0.0 if bound == "lower" else 1.0
    z = 1.6448536269514722
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return float(max(0.0, centre - radius) if bound == "lower" else min(1.0, centre + radius))


def promotion_gates(metrics: Mapping[str, float]) -> dict[str, bool]:
    """Evaluate the six preregistered development gates without rounding."""

    try:
        values = {name: float(metrics[name]) for name in GATE_THRESHOLDS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("development metrics are incomplete") from exc
    return {
        "coverage": bool(np.isfinite(values["coverage_lb"]) and values["coverage_lb"] >= 0.95),
        "protected_recall": bool(np.isfinite(values["protected_recall_lb"]) and values["protected_recall_lb"] >= 0.98),
        "severe_precision": bool(np.isfinite(values["severe_precision_lb"]) and values["severe_precision_lb"] >= 0.90),
        "savings": bool(np.isfinite(values["savings_lb"]) and values["savings_lb"] >= 0.05),
        "auc": bool(np.isfinite(values["auc"]) and values["auc"] >= 0.85),
        "precision_lift_over_prevalence": bool(
            np.isfinite(values["precision_lift_over_prevalence"])
            and values["precision_lift_over_prevalence"] >= 0.20
        ),
    }


def _numeric_feature(frame: pd.DataFrame, feature: str) -> np.ndarray:
    if feature not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)


def _fit_term_parameters(
    features: pd.DataFrame, name: str
) -> tuple[dict[str, object] | None, str | None]:
    specification = TERM_CATALOGUE[name]
    feature = str(specification["feature"])
    raw = _numeric_feature(features, feature)
    transform = str(specification["transform"])
    reference: float | None = None
    if transform == "identity":
        transformed = raw.copy()
    elif transform == "log_deviation":
        eligible = np.isfinite(raw) & (raw > 0.0)
        if not eligible.any():
            return None, f"{feature} has no finite positive development values"
        reference = float(np.median(raw[eligible]))
        transformed = np.full(len(raw), np.nan, dtype=float)
        transformed[eligible] = np.abs(np.log(raw[eligible] / reference))
    else:
        raise RuntimeError(f"unknown frozen transform: {transform}")
    finite = np.isfinite(transformed)
    if not finite.any():
        return None, f"{feature} has no finite development values"
    q25, median, q75 = np.quantile(transformed[finite], [0.25, 0.50, 0.75])
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= 0.0:
        return None, f"{name} has zero development IQR"
    parameter = {
        "feature": feature,
        "direction": float(specification["direction"]),
        "transform": transform,
        "median": float(median),
        "iqr": iqr,
    }
    if reference is not None:
        parameter["reference"] = reference
    return parameter, None


def _term_score(
    features: pd.DataFrame, parameter: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    feature = str(parameter["feature"])
    raw = _numeric_feature(features, feature)
    transform = str(parameter["transform"])
    if transform == "identity":
        transformed = raw
    elif transform == "log_deviation":
        reference = float(parameter["reference"])
        transformed = np.full(len(raw), np.nan, dtype=float)
        eligible = np.isfinite(raw) & (raw > 0.0) & np.isfinite(reference) & (reference > 0.0)
        transformed[eligible] = np.abs(np.log(raw[eligible] / reference))
    else:
        raise ValueError("frozen rule contains an unknown transform")
    median = float(parameter["median"])
    iqr = float(parameter["iqr"])
    direction = float(parameter["direction"])
    if not all(np.isfinite(value) for value in (median, iqr, direction)) or iqr <= 0.0:
        raise ValueError("frozen rule contains invalid robust constants")
    support = np.isfinite(transformed)
    score = np.full(len(features), np.nan, dtype=float)
    score[support] = direction * (transformed[support] - median) / iqr
    return score, support


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


def _auc(labels: np.ndarray, scores: np.ndarray, support: np.ndarray) -> float:
    selected_labels = np.asarray(labels[support], dtype=bool)
    selected_scores = np.asarray(scores[support], dtype=float)
    positives = int(selected_labels.sum())
    negatives = int(len(selected_labels) - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(selected_scores, method="average")
    return float(
        (ranks[selected_labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _candidate_metrics(
    *,
    score: np.ndarray,
    support: np.ndarray,
    reject: np.ndarray,
    severe: np.ndarray,
    protected: np.ndarray,
) -> dict[str, float | int]:
    total = len(score)
    supported_count = int(support.sum())
    rejected_count = int(reject.sum())
    protected_count = int(protected.sum())
    protected_kept = int((protected & ~reject).sum())
    severe_rejected = int((severe & reject).sum())
    severe_count = int(severe.sum())
    precision_lb = wilson_one_sided(severe_rejected, rejected_count, bound="lower")
    prevalence_ub = wilson_one_sided(severe_count, total, bound="upper")
    return {
        "rows": total,
        "supported": supported_count,
        "rejected": rejected_count,
        "severe": severe_count,
        "protected": protected_count,
        "rejected_severe": severe_rejected,
        "rejected_protected": int((protected & reject).sum()),
        "coverage_lb": wilson_one_sided(supported_count, total, bound="lower"),
        "protected_recall_lb": wilson_one_sided(protected_kept, protected_count, bound="lower"),
        "severe_precision_lb": precision_lb,
        "savings_lb": wilson_one_sided(rejected_count, total, bound="lower"),
        "auc": _auc(severe, score, support),
        "severe_prevalence_ub": prevalence_ub,
        "precision_lift_over_prevalence": precision_lb - prevalence_ub,
    }


def _joined_development(
    features: pd.DataFrame, endpoints: pd.DataFrame
) -> pd.DataFrame:
    if "material_id" not in features or "material_id" not in endpoints:
        raise ValueError("development inputs lack material_id")
    left = features.copy()
    right = endpoints.copy()
    for frame, role in ((left, "features"), (right, "endpoints")):
        frame["material_id"] = frame["material_id"].astype(str)
        if frame.material_id.isna().any() or frame.material_id.duplicated().any():
            raise ValueError(f"development {role} material IDs are invalid")
    if set(left.material_id) != set(right.material_id):
        raise ValueError("development endpoint IDs do not exactly match features")
    endpoint_columns = ["material_id", "force_max", "force_rms", "stress_norm"]
    missing = set(endpoint_columns) - set(right)
    if missing:
        raise ValueError(f"development endpoints lack columns: {sorted(missing)}")
    joined = left.merge(right.loc[:, endpoint_columns], on="material_id", validate="one_to_one")
    return joined.sort_values("material_id", kind="stable", ignore_index=True)


def scan_development_candidates(
    features: pd.DataFrame, endpoints: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object] | None]:
    """Evaluate only the frozen candidate catalogue on exposed development rows."""

    joined = _joined_development(features, endpoints)
    severe, protected = classify_dft_response(joined)
    parameters: dict[str, dict[str, object]] = {}
    disabled: dict[str, str] = {}
    for name in TERM_CATALOGUE:
        parameter, reason = _fit_term_parameters(joined, name)
        if parameter is None:
            disabled[name] = reason or "term is disabled"
        else:
            parameters[name] = parameter

    rows: list[dict[str, object]] = []
    candidates: list[tuple[dict[str, object], dict[str, object]]] = []
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
                        "gates_passed": False,
                        "promotion": False,
                    }
                )
            continue
        score, support = _candidate_score(joined, terms, parameters)
        for fraction in REJECTION_FRACTIONS:
            threshold = float(np.quantile(score[support], 1.0 - fraction, method="higher"))
            reject = support & (score >= threshold)
            metrics = _candidate_metrics(
                score=score,
                support=support,
                reject=reject,
                severe=severe,
                protected=protected,
            )
            numeric_metrics = {
                key: float(metrics[key]) for key in GATE_THRESHOLDS
            }
            gates = promotion_gates(numeric_metrics)
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "terms": list(terms),
                "n_terms": len(terms),
                "rejection_fraction": fraction,
                "threshold": threshold,
                "enabled": True,
                "disabled_reason": None,
                **metrics,
                **{f"gate_{name}": value for name, value in gates.items()},
                "gates_passed": all(gates.values()),
                "promotion": False,
            }
            rows.append(row)
            if all(gates.values()):
                candidates.append((row, {term: parameters[term] for term in terms}))

    scan = pd.DataFrame(rows)
    if not candidates:
        return scan, None
    winner, winner_parameters = min(
        candidates,
        key=lambda item: (
            -float(item[0]["severe_precision_lb"]),
            -float(item[0]["savings_lb"]),
            -float(item[0]["auc"]),
            int(item[0]["n_terms"]),
            float(item[0]["rejection_fraction"]),
            str(item[0]["candidate_id"]),
        ),
    )
    winning_mask = (
        scan.candidate_id.eq(winner["candidate_id"])
        & scan.rejection_fraction.eq(winner["rejection_fraction"])
    )
    scan.loc[winning_mask, "promotion"] = True
    gates = {name: bool(winner[f"gate_{name}"]) for name in PROMOTION_GATE_NAMES}
    metrics = {
        key: (int(value) if isinstance(value, (int, np.integer)) else float(value))
        for key, value in winner.items()
        if key
        in {
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
        }
    }
    terms = list(winner["terms"])
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
        "same_composition_candidates_used": False,
    }
    return scan, rule


def compute_inorganic_risk(
    features: pd.DataFrame, rule: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a frozen NEXT32 rule without reading any response endpoint."""

    if rule.get("protocol") != PROTOCOL or rule.get("eligible") is not True:
        raise ValueError("frozen NEXT32 rule is not eligible")
    raw_terms = rule.get("terms")
    raw_parameters = rule.get("term_parameters")
    if (
        not isinstance(raw_terms, list)
        or not raw_terms
        or len(raw_terms) > 2
        or not isinstance(raw_parameters, Mapping)
    ):
        raise ValueError("frozen NEXT32 term schema is invalid")
    terms = tuple(str(term) for term in raw_terms)
    if terms not in CANDIDATE_TERM_SETS or set(raw_parameters) != set(terms):
        raise ValueError("frozen NEXT32 terms differ from the catalogue")
    parameters: dict[str, Mapping[str, object]] = {}
    for term in terms:
        value = raw_parameters[term]
        if not isinstance(value, Mapping):
            raise ValueError("frozen NEXT32 term parameters are invalid")
        specification = TERM_CATALOGUE[term]
        if any(value.get(key) != specification[key] for key in ("feature", "direction", "transform")):
            raise ValueError("frozen NEXT32 term specification changed")
        parameters[term] = value
    score, support = _candidate_score(features, terms, parameters)
    threshold = float(rule.get("threshold", math.nan))
    if not np.isfinite(threshold):
        raise ValueError("frozen NEXT32 threshold is invalid")
    reject = support & (score >= threshold)
    return score, support, reject


def _validate_feature_artifact(
    feature_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate one immutable, label-free NEXT32 feature artifact."""

    if feature_path.name != FEATURE_NAME or not feature_path.is_file():
        raise ValueError("NEXT32 feature path/name is invalid")
    manifest = _strict_json(manifest_path, role="NEXT32 feature manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != FEATURE_PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("model_or_proxy_potential_used") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(FEATURE_NAME) != _sha256(feature_path)
    ):
        raise ValueError("NEXT32 feature artifact crossed the label-free boundary")
    frame = pd.read_parquet(feature_path)
    forbidden = [
        str(column)
        for column in frame
        if any(token in str(column).lower() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"NEXT32 feature columns crossed the label-free boundary: {forbidden}")
    if "material_id" not in frame or frame.material_id.isna().any() or frame.material_id.astype(str).duplicated().any():
        raise ValueError("NEXT32 feature material IDs are invalid")
    frame = frame.copy()
    frame["material_id"] = frame["material_id"].astype(str)
    return frame, manifest


def freeze_development_rule(
    *,
    feature_path: Path,
    feature_manifest_path: Path,
    endpoints_path: Path,
    endpoints_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Open exposed development labels, scan once, and publish without overwrite."""

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
        raise ValueError("NEXT32 development endpoint path/name is invalid")
    endpoint_manifest = _strict_json(
        paths["endpoints_manifest"], role="NEXT32 development endpoint manifest"
    )
    endpoint_outputs = endpoint_manifest.get("outputs_sha256")
    if (
        endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_manifest.get("labels_opened") is not True
        or not isinstance(endpoint_outputs, Mapping)
        or endpoint_outputs.get(ENDPOINT_NAME) != _sha256(paths["endpoints"])
    ):
        raise ValueError("NEXT32 development endpoint artifact is invalid")
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
        "mode": "exposed_rattled_relax_development_scan",
        "labels_opened_for_development": True,
        "confirmation_rows_used": 0,
        "confirmation_labels_used_for_selection": False,
        "promoted": rule is not None,
        "candidate_term_sets": [list(terms) for terms in CANDIDATE_TERM_SETS],
        "rejection_fractions": list(REJECTION_FRACTIONS),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "inputs_sha256": input_hashes,
        "executed_source_sha256": {"src/next32_inorganic_response_rule.py": source_hash},
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
            raise RuntimeError("NEXT32 development input changed before publication")
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT32 development source changed before publication")
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
    """Publish label-free NEXT32 predictions for one or more new cohorts."""

    rule_path = Path(frozen_rule_path).resolve()
    rule_manifest_path = Path(frozen_rule_manifest_path).resolve()
    feature_candidates = [Path(path).resolve() for path in feature_paths]
    feature_manifest_candidates = [
        Path(path).resolve() for path in feature_manifest_paths
    ]
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if (
        rule_path.name != FROZEN_RULE_NAME
        or not rule_path.is_file()
        or not rule_manifest_path.is_file()
        or not feature_candidates
        or len(feature_candidates) != len(feature_manifest_candidates)
    ):
        raise ValueError("NEXT32 rule application inputs are invalid")
    rule_manifest = _strict_json(rule_manifest_path, role="NEXT32 frozen-rule manifest")
    rule_outputs = rule_manifest.get("outputs_sha256")
    if (
        rule_manifest.get("protocol") != PROTOCOL
        or rule_manifest.get("promoted") is not True
        or rule_manifest.get("confirmation_labels_used_for_selection") is not False
        or not isinstance(rule_outputs, Mapping)
        or rule_outputs.get(FROZEN_RULE_NAME) != _sha256(rule_path)
    ):
        raise ValueError("NEXT32 frozen rule is not hash-locked and promoted")
    try:
        rule = json.loads(rule_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("NEXT32 frozen rule JSON is invalid") from exc
    if not isinstance(rule, dict):
        raise ValueError("NEXT32 frozen rule JSON is invalid")

    parts: list[pd.DataFrame] = []
    feature_inputs: list[dict[str, str]] = []
    for feature_candidate, manifest_candidate in zip(
        feature_candidates, feature_manifest_candidates, strict=True
    ):
        frame, _manifest = _validate_feature_artifact(feature_candidate, manifest_candidate)
        score, supported, reject = compute_inorganic_risk(frame, rule)
        source_name = (
            frame["source_name"].astype(str)
            if "source_name" in frame
            else pd.Series([feature_candidate.parent.name] * len(frame))
        )
        parent_id = (
            frame["parent_id"].astype(str)
            if "parent_id" in frame
            else pd.Series([""] * len(frame))
        )
        parts.append(
            pd.DataFrame(
                {
                    "material_id": frame.material_id.astype(str),
                    "source_name": source_name.to_numpy(),
                    "parent_id": parent_id.to_numpy(),
                    "analytic_supported": supported,
                    "next32_risk_score": score,
                    "reject": reject,
                    "input_role": "unrelaxed_x0_geometry_only",
                }
            )
        )
        feature_inputs.append(
            {
                "features_path": str(feature_candidate),
                "features_sha256": _sha256(feature_candidate),
                "manifest_path": str(manifest_candidate),
                "manifest_sha256": _sha256(manifest_candidate),
            }
        )
    predictions = pd.concat(parts, ignore_index=True).sort_values(
        ["source_name", "material_id"], kind="stable", ignore_index=True
    )
    if predictions.material_id.duplicated().any():
        raise ValueError("NEXT32 prediction material IDs are duplicated")
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    input_hashes: dict[str, object] = {
        "frozen_rule": _sha256(rule_path),
        "frozen_rule_manifest": _sha256(rule_manifest_path),
        "feature_artifacts": feature_inputs,
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "prospective_label_free_inorganic_response_prediction",
        "labels_opened": False,
        "endpoint_fields_read": False,
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
        "executed_source_sha256": {"src/next32_inorganic_response_rule.py": source_hash},
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        predictions_path = staging / PREDICTIONS_NAME
        predictions.to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {PREDICTIONS_NAME: _sha256(predictions_path)}
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if _sha256(rule_path) != input_hashes["frozen_rule"] or _sha256(
            rule_manifest_path
        ) != input_hashes["frozen_rule_manifest"]:
            raise RuntimeError("NEXT32 frozen rule input changed before publication")
        for item in feature_inputs:
            if _sha256(Path(item["features_path"])) != item["features_sha256"] or _sha256(
                Path(item["manifest_path"])
            ) != item["manifest_sha256"]:
                raise RuntimeError("NEXT32 feature input changed before publication")
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT32 rule source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "CANDIDATE_TERM_SETS",
    "GATE_THRESHOLDS",
    "PROMOTION_GATE_NAMES",
    "PROTOCOL",
    "REJECTION_FRACTIONS",
    "FROZEN_RULE_NAME",
    "MANIFEST_NAME",
    "PREDICTIONS_NAME",
    "SCAN_NAME",
    "TERM_CATALOGUE",
    "apply_frozen_rule",
    "classify_dft_response",
    "compute_inorganic_risk",
    "freeze_development_rule",
    "promotion_gates",
    "scan_development_candidates",
    "wilson_one_sided",
]
