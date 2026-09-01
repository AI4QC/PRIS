#!/usr/bin/env python3
"""Source-balanced development of an explicit x0-only MOF framework rule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next43_finite_law_search import apply_formula
from src.next48_qmof_external_validation import (
    JOINED_NAME as NEXT48_JOINED_NAME,
    PROTOCOL as NEXT48_PROTOCOL,
)
from src.next50_framework_motif_features import (
    COMBINED_FEATURE_NAMES,
    FEATURES_NAME as NEXT50_FEATURES_NAME,
    PROTOCOL as NEXT50_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next50-source-balanced-framework-rule-development-v1"
FORMULA_NAME = "NEXT50_FRAMEWORK_DEVELOPMENT_CANDIDATE.json"
SEARCH_NAME = "NEXT50_FRAMEWORK_SOURCE_BALANCED_SEARCH.json"
PREDICTIONS_NAME = "next50_qmof_framework_development_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
PROTECTED_MAX = 0.10
SEVERE_MIN = 0.50
ONE_SIDED_95_Z = 1.6448536269514722
DOMAIN_GATE = {
    "periodic_dimension_max_min": 1.0,
    "periodic_framework_fraction_min": 0.50,
}
GATES = {
    "coverage_lower_at_least": 0.95,
    "protected_recall_lower_at_least": 0.95,
    "reject_precision_lower_at_least": 0.70,
    "savings_lower_at_least": 0.02,
    "pooled_severe_auc_at_least": 0.75,
    "macro_source_auc_at_least": 0.65,
    "worst_source_auc_at_least": 0.55,
}
REJECTION_FRACTIONS = tuple(float(value) for value in np.arange(0.02, 0.301, 0.01))
PAIR_WEIGHTS = (0.5, 1.0, 2.0)
PAIR_SHORTLIST = 16
TRIPLE_SHORTLIST = 9


def _wilson_lower(successes: int, total: int) -> float | None:
    if total <= 0:
        return None
    p = successes / total
    z2 = ONE_SIDED_95_Z**2
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    margin = ONE_SIDED_95_Z * math.sqrt(
        (p * (1.0 - p) + z2 / (4.0 * total)) / total
    )
    return float((center - margin) / denominator)


def _safe_auc(score: np.ndarray, truth: np.ndarray) -> float | None:
    finite = np.isfinite(score)
    selected_truth = np.asarray(truth, dtype=bool)[finite]
    if finite.sum() < 2 or len(np.unique(selected_truth)) < 2:
        return None
    return float(roc_auc_score(selected_truth.astype(int), score[finite]))


def apply_framework_formula(
    features: pd.DataFrame, formula: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply an additive formula only inside its explicit periodic-framework gate."""

    if formula.get("domain_gate") != DOMAIN_GATE:
        raise ValueError("NEXT50 framework domain gate differs")
    score, supported, reject = apply_formula(features, formula)
    dimension = pd.to_numeric(
        features.get("periodic_dimension_max"), errors="coerce"
    ).to_numpy(float)
    fraction = pd.to_numeric(
        features.get("periodic_framework_fraction"), errors="coerce"
    ).to_numpy(float)
    domain_finite = np.isfinite(dimension) & np.isfinite(fraction)
    in_domain = (
        domain_finite
        & (dimension >= DOMAIN_GATE["periodic_dimension_max_min"])
        & (fraction >= DOMAIN_GATE["periodic_framework_fraction_min"])
    )
    supported &= in_domain
    reject &= supported
    score = np.asarray(score, dtype=float)
    score[~supported] = np.nan
    return score, supported, reject


def _decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, endpoint: np.ndarray
) -> dict[str, object]:
    supported = np.asarray(supported, dtype=bool)
    reject = np.asarray(reject, dtype=bool) & supported
    endpoint = np.asarray(endpoint, dtype=float)
    if supported.shape != reject.shape or endpoint.shape != supported.shape:
        raise ValueError("NEXT50 decision arrays differ")
    protected = endpoint <= PROTECTED_MAX
    severe = endpoint >= SEVERE_MIN
    n = len(endpoint)
    n_supported = int(supported.sum())
    n_reject = int(reject.sum())
    n_protected = int(protected.sum())
    protected_kept = int((protected & ~reject).sum())
    severe_rejected = int((severe & reject).sum())
    precision = severe_rejected / n_reject if n_reject else None
    protected_recall = protected_kept / n_protected if n_protected else None
    metrics: dict[str, object] = {
        "rows": n,
        "supported": n_supported,
        "rejected": n_reject,
        "protected": n_protected,
        "protected_kept": protected_kept,
        "severe": int(severe.sum()),
        "severe_rejected": severe_rejected,
        "coverage": n_supported / n if n else None,
        "coverage_lower": _wilson_lower(n_supported, n),
        "protected_recall": protected_recall,
        "protected_recall_lower": _wilson_lower(protected_kept, n_protected),
        "rejection_precision": precision,
        "rejection_precision_lower": _wilson_lower(severe_rejected, n_reject),
        "savings": n_reject / n if n else None,
        "savings_lower": _wilson_lower(n_reject, n),
    }
    metrics["passes_primary_gates"] = bool(
        metrics["coverage_lower"] is not None
        and float(metrics["coverage_lower"]) >= GATES["coverage_lower_at_least"]
        and metrics["protected_recall_lower"] is not None
        and float(metrics["protected_recall_lower"])
        >= GATES["protected_recall_lower_at_least"]
        and metrics["rejection_precision_lower"] is not None
        and float(metrics["rejection_precision_lower"])
        >= GATES["reject_precision_lower_at_least"]
        and metrics["savings_lower"] is not None
        and float(metrics["savings_lower"]) >= GATES["savings_lower_at_least"]
    )
    return metrics


def _source_diagnostics(
    *,
    score: np.ndarray,
    supported: np.ndarray,
    reject: np.ndarray,
    endpoint: np.ndarray,
    sources: np.ndarray,
) -> dict[str, object]:
    severe = endpoint >= SEVERE_MIN
    per_source: dict[str, object] = {}
    aucs: list[float] = []
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        auc = _safe_auc(score[mask & supported], severe[mask & supported])
        metrics = _decision_metrics(
            supported=supported[mask], reject=reject[mask], endpoint=endpoint[mask]
        )
        per_source[str(source)] = {"severe_auc": auc, "metrics": metrics}
        if auc is not None:
            aucs.append(float(auc))
    if len(aucs) < 2:
        raise ValueError("NEXT50 requires at least two source families with both classes")
    return {
        "evaluable_source_count": len(aucs),
        "macro_source_auc": float(np.mean(aucs)),
        "worst_source_auc": float(np.min(aucs)),
        "per_source": per_source,
    }


def _candidate_rank(
    metrics: Mapping[str, object], pooled_auc: float, source: Mapping[str, object], terms: int
) -> tuple[float, ...]:
    values = {
        "coverage_lower_at_least": metrics.get("coverage_lower"),
        "protected_recall_lower_at_least": metrics.get("protected_recall_lower"),
        "reject_precision_lower_at_least": metrics.get("rejection_precision_lower"),
        "savings_lower_at_least": metrics.get("savings_lower"),
        "pooled_severe_auc_at_least": pooled_auc,
        "macro_source_auc_at_least": source.get("macro_source_auc"),
        "worst_source_auc_at_least": source.get("worst_source_auc"),
    }
    ratios = [
        float(values[name]) / cutoff if values[name] is not None else -1.0
        for name, cutoff in GATES.items()
    ]
    passes = bool(metrics["passes_primary_gates"] and all(ratio >= 1.0 for ratio in ratios[4:]))
    return (
        1.0 if passes else 0.0,
        min(ratios),
        float(metrics.get("protected_recall_lower") or -1.0),
        float(metrics.get("rejection_precision_lower") or -1.0),
        float(metrics.get("savings_lower") or -1.0),
        float(source["worst_source_auc"]),
        float(source["macro_source_auc"]),
        pooled_auc,
        -float(terms),
    )


def _term(info: Mapping[str, object], *, weight: float) -> dict[str, object]:
    return {
        "feature": str(info["feature"]),
        "direction": int(info["direction"]),
        "center": float(info["center"]),
        "scale": float(info["scale"]),
        "weight": float(weight),
    }


def search_source_balanced_framework_rule(
    *,
    features: pd.DataFrame,
    material_ids: Sequence[str],
    source_families: Sequence[str],
    endpoint: Sequence[float],
    candidate_features: Sequence[str] = COMBINED_FEATURE_NAMES,
) -> dict[str, object]:
    """Search a finite 1--3 term catalogue and rank by worst source behavior."""

    material_ids = np.asarray(material_ids, dtype=str)
    sources = np.asarray(source_families, dtype=str)
    endpoint = np.asarray(endpoint, dtype=float)
    if (
        len(features) != len(material_ids)
        or sources.shape != material_ids.shape
        or endpoint.shape != material_ids.shape
        or not np.isfinite(endpoint).all()
        or len(set(material_ids.tolist())) != len(material_ids)
        or len(set(sources.tolist())) < 2
    ):
        raise ValueError("NEXT50 development arrays differ")
    severe = endpoint >= SEVERE_MIN
    infos: list[dict[str, object]] = []
    for feature in candidate_features:
        if feature not in features:
            continue
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if finite.sum() < 10:
            continue
        center = float(np.median(values[finite]))
        q25, q75 = np.quantile(values[finite], (0.25, 0.75))
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 0.0:
            continue
        raw_source_aucs: list[float] = []
        for source in sorted(set(sources.tolist())):
            mask = finite & (sources == source)
            auc = _safe_auc(values[mask], severe[mask])
            if auc is not None:
                raw_source_aucs.append(auc)
        if len(raw_source_aucs) < 2:
            continue
        direction = 1 if float(np.mean(raw_source_aucs)) >= 0.5 else -1
        directional = [auc if direction == 1 else 1.0 - auc for auc in raw_source_aucs]
        pooled_auc = _safe_auc(direction * values[finite], severe[finite])
        if pooled_auc is None:
            continue
        infos.append(
            {
                "feature": str(feature),
                "direction": direction,
                "center": center,
                "scale": scale,
                "risk": direction * (values - center) / scale,
                "pooled_directional_auc": float(pooled_auc),
                "macro_source_directional_auc": float(np.mean(directional)),
                "worst_source_directional_auc": float(np.min(directional)),
            }
        )
    if not infos:
        raise ValueError("NEXT50 has no source-evaluable finite feature")
    infos.sort(
        key=lambda item: (
            -float(item["worst_source_directional_auc"]),
            -float(item["macro_source_directional_auc"]),
            -float(item["pooled_directional_auc"]),
            str(item["feature"]),
        )
    )

    best: tuple[tuple[float, ...], str, dict[str, object], dict[str, object], float, dict[str, object]] | None = None
    considered = 0

    def consider(selected: Sequence[Mapping[str, object]], weights: Sequence[float]) -> None:
        nonlocal best, considered
        raw_score = np.zeros(len(features), dtype=float)
        finite = np.ones(len(features), dtype=bool)
        for info, weight in zip(selected, weights, strict=True):
            risk = np.asarray(info["risk"], dtype=float)
            finite &= np.isfinite(risk)
            raw_score += float(weight) * risk
        if not finite.any():
            return
        thresholds = {
            float(np.quantile(raw_score[finite], 1.0 - fraction, method="inverted_cdf"))
            for fraction in REJECTION_FRACTIONS
        }
        for threshold in sorted(thresholds):
            formula = {
                "kind": "additive",
                "terms": [
                    _term(info, weight=float(weight))
                    for info, weight in zip(selected, weights, strict=True)
                ],
                "threshold": threshold,
                "missing_policy": "KEEP",
                "domain_gate": dict(DOMAIN_GATE),
            }
            score, supported, reject = apply_framework_formula(features, formula)
            metrics = _decision_metrics(
                supported=supported, reject=reject, endpoint=endpoint
            )
            pooled_auc = _safe_auc(score[supported], severe[supported])
            if pooled_auc is None:
                continue
            source = _source_diagnostics(
                score=score,
                supported=supported,
                reject=reject,
                endpoint=endpoint,
                sources=sources,
            )
            rank = _candidate_rank(metrics, pooled_auc, source, len(selected))
            key = json.dumps(formula, sort_keys=True, separators=(",", ":"))
            considered += 1
            record = (rank, key, formula, metrics, pooled_auc, source)
            if best is None or rank > best[0] or (rank == best[0] and key < best[1]):
                best = record

    for info in infos:
        consider((info,), (1.0,))
    shortlist = infos[: min(PAIR_SHORTLIST, len(infos))]
    for left, right in combinations(shortlist, 2):
        for weight in PAIR_WEIGHTS:
            consider((left, right), (1.0, weight))
    triples = infos[: min(TRIPLE_SHORTLIST, len(infos))]
    for selected in combinations(triples, 3):
        consider(selected, (1.0, 1.0, 1.0))
    if best is None:
        raise RuntimeError("NEXT50 finite source-balanced catalogue is empty")
    rank, _key, formula, metrics, pooled_auc, source = best
    score, supported, reject = apply_framework_formula(features, formula)
    serializable_infos = [
        {key: value for key, value in info.items() if key != "risk"} for info in infos
    ]
    passes = bool(rank[0] == 1.0)
    metrics = dict(metrics)
    metrics["pooled_severe_auc"] = pooled_auc
    metrics["passes_primary_gates"] = passes
    return {
        "selected_formula": formula,
        "full_development_metrics": metrics,
        "source_balanced_diagnostics": source,
        "candidate_count": considered,
        "source_evaluable_feature_count": len(infos),
        "feature_diagnostics": serializable_infos,
        "passes_source_balanced_development_gates": passes,
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _strict_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    def safe(item: object) -> object:
        if isinstance(item, dict):
            return {str(key): safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        return item

    return (json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_qmof_framework_rule_development(
    *,
    feature_path: Path,
    feature_manifest_path: Path,
    evaluation_path: Path,
    evaluation_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Join exposed QMOF labels only after the combined x0 feature table is sealed."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "features": Path(feature_path).resolve(),
        "feature_manifest": Path(feature_manifest_path).resolve(),
        "evaluation": Path(evaluation_path).resolve(),
        "evaluation_manifest": Path(evaluation_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT50 development input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    feature_manifest = _strict_json(paths["feature_manifest"], role="NEXT50 feature manifest")
    evaluation_manifest = _strict_json(
        paths["evaluation_manifest"], role="NEXT48 evaluation manifest"
    )
    feature_outputs = feature_manifest.get("outputs_sha256")
    evaluation_outputs = evaluation_manifest.get("outputs_sha256")
    if (
        paths["features"].name != NEXT50_FEATURES_NAME
        or paths["evaluation"].name != NEXT48_JOINED_NAME
        or feature_manifest.get("protocol") != NEXT50_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or feature_manifest.get("relaxed_coordinate_payloads_opened") is not False
        or feature_manifest.get("model_or_proxy_potential_used") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(paths["features"].name) != hashes["features"]
        or evaluation_manifest.get("protocol") != NEXT48_PROTOCOL
        or evaluation_manifest.get("labels_opened") is not True
        or evaluation_manifest.get("evaluation_only_dft_final_geometry_read") is not True
        or evaluation_manifest.get("law_execution_dft_values_read") is not False
        or not isinstance(evaluation_outputs, Mapping)
        or evaluation_outputs.get(paths["evaluation"].name) != hashes["evaluation"]
    ):
        raise ValueError("NEXT50 feature/label separation contract differs")

    features = pd.read_parquet(paths["features"])
    evaluation = pd.read_parquet(
        paths["evaluation"],
        columns=[
            "material_id",
            "source_family",
            "endpoint_supported",
            "fingerprint_change",
            "next31_risk_score",
            "reject",
            "pauling_p2_p5_decision",
        ],
    )
    joined = features.merge(
        evaluation,
        on="material_id",
        how="inner",
        suffixes=("", "_endpoint"),
        validate="one_to_one",
    )
    if (
        len(joined) != len(features)
        or not joined["source_family"].eq(joined["source_family_endpoint"]).all()
    ):
        raise ValueError("NEXT50 feature and endpoint identities differ")
    eligible = joined["endpoint_supported"].astype(bool) & joined[
        "combined_supported"
    ].astype(bool)
    development = joined.loc[eligible].reset_index(drop=True)
    result = search_source_balanced_framework_rule(
        features=development,
        material_ids=development["material_id"].astype(str),
        source_families=development["source_family"].astype(str),
        endpoint=development["fingerprint_change"].astype(float),
        candidate_features=COMBINED_FEATURE_NAMES,
    )
    formula = result["selected_formula"]
    all_score, all_supported, all_reject = apply_framework_formula(joined, formula)
    predictions = joined.loc[
        :,
        [
            "material_id",
            "source_family",
            "endpoint_supported",
            "fingerprint_change",
            "next31_risk_score",
            "reject",
            "pauling_p2_p5_decision",
        ],
    ].rename(columns={"reject": "next31_reject"})
    predictions["framework_score"] = all_score
    predictions["framework_supported"] = all_supported
    predictions["framework_reject"] = all_reject

    formula_document = {
        "protocol": PROTOCOL,
        "role": "exposed-QMOF development candidate; frozen before ODAC opening",
        "formula": formula,
        "gates": dict(GATES),
        "protected_max": PROTECTED_MAX,
        "severe_min": SEVERE_MIN,
        "execution_input": "one_raw_unrelaxed_periodic_x0_only",
        "execution_uses_dft": False,
        "execution_uses_relaxed_geometry": False,
        "execution_uses_mlip_or_learned_energy_force_stress": False,
        "execution_runs_physical_relaxation": False,
        "requires_independent_odac_confirmation": True,
    }
    search_document = {
        key: value
        for key, value in result.items()
        if key not in {"score", "supported", "reject", "selected_formula"}
    }
    search_document.update(
        {
            "protocol": PROTOCOL,
            "selected_formula": formula,
            "data_role": "QMOF exposed development only",
            "endpoint": "initial-to-PBE-D3(BJ)-relaxed CrystalNN fingerprint L2",
            "candidate_feature_count": len(COMBINED_FEATURE_NAMES),
            "scientific_confirmation": False,
        }
    )
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    output_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "exposed_qmof_source_balanced_framework_rule_development",
        "development_labels_opened": True,
        "offline_dft_final_geometry_label_used": True,
        "offline_dft_energy_used": False,
        "law_execution_dft_values_read": False,
        "law_execution_endpoint_or_later_geometry_read": False,
        "law_execution_mlip_or_model_potential_used": False,
        "law_execution_learned_energy_force_stress_proxy_used": False,
        "law_execution_physical_relaxation_executed": False,
        "thresholds_fit_on_exposed_qmof": True,
        "scientific_confirmation": False,
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next50_framework_rule_search.py": source_hash
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        prediction_path = staging / PREDICTIONS_NAME
        formula_path.write_bytes(_json_bytes(formula_document))
        search_path.write_bytes(_json_bytes(search_document))
        predictions.to_parquet(prediction_path, index=False)
        output_manifest["outputs_sha256"] = {
            path.name: _sha256(path)
            for path in (formula_path, search_path, prediction_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(output_manifest))
        for name, path in paths.items():
            if _sha256(path) != hashes[name]:
                raise RuntimeError(f"NEXT50 input {name} changed before publication")
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT50 search source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_manifest


__all__ = [
    "DOMAIN_GATE",
    "FORMULA_NAME",
    "GATES",
    "MANIFEST_NAME",
    "PREDICTIONS_NAME",
    "PROTOCOL",
    "SEARCH_NAME",
    "apply_framework_formula",
    "run_qmof_framework_rule_development",
    "search_source_balanced_framework_rule",
]
