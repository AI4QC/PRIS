#!/usr/bin/env python3
"""Discovery-only finite search for an explicit ODAC23 x0 rejection rule."""

from __future__ import annotations

import argparse
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
from src.next55_odac23_analytic_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    NEXT55_FEATURE_NAMES,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)
from src.next56_odac23_label_firewall import (
    PROTOCOL as FIREWALL_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
    TOP_MANIFEST_NAME,
)


PROTOCOL = "2026-08-03-next57-odac23-discovery-finite-search-v1"
DESIGN_SHA256 = "15bcf018fc3f7310495efe2af866ce7c31b9b9d8d5adfc7251d69a8cbdb33bac"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "bc4a05b70c0b4d84723c83e4a79e2f20ffce7f55f9ddf3a32cb8b5f3bcef6f1e"
)
EXPECTED_FIREWALL_MANIFEST_SHA256 = (
    "163888a169f649f9863276bba62047050251787b424568ddf85eb9dc3fd2ee8c"
)
EXPECTED_DISCOVERY_MANIFEST_SHA256 = (
    "6a9a9a524ef5535dbcef990c71cbce2abc6e7ab9e7e61e5233f2df7760c1d858"
)
EXPECTED_DISCOVERY_LABEL_SHA256 = (
    "7dc2fd6c582204533d18aedf7905e5471ba25737529a088a993d005566c17ea9"
)
FORMULA_NAME = "NEXT57_ODAC23_DISCOVERY_CANDIDATE.json"
SEARCH_NAME = "NEXT57_ODAC23_DISCOVERY_SEARCH.json"
PREDICTIONS_NAME = "next57_odac23_discovery_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
ENDPOINT_COLUMN = "framework_displacement_p95_median"
PROTECTED_MAX = 0.05
SEVERE_MIN = 0.20
ONE_SIDED_95_Z = 1.6448536269514722
DOMAIN_GATE = {
    "periodic_dimension_max_min": 1.0,
    "periodic_framework_fraction_min": 0.5,
}
GATES = {
    "coverage_lower_at_least": 0.95,
    "protected_recall_lower_at_least": 0.95,
    "reject_precision_lower_at_least": 0.70,
    "savings_lower_at_least": 0.02,
    "pooled_extreme_auc_at_least": 0.75,
    "macro_stratum_auc_at_least": 0.65,
    "worst_stratum_auc_at_least": 0.55,
}
REJECTION_FRACTIONS = tuple(float(value) for value in np.arange(0.02, 0.301, 0.01))
PAIR_WEIGHTS = (0.5, 1.0, 2.0)
PAIR_SHORTLIST = 20
TRIPLE_SHORTLIST = 12


def _wilson_lower(successes: int, total: int) -> float:
    if total <= 0:
        return 0.0
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
    return float(roc_auc_score(selected_truth.astype(int), np.asarray(score)[finite]))


def apply_odac23_formula(
    features: pd.DataFrame, formula: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the additive formula with strict fail-open x0/domain behavior."""

    if formula.get("kind") != "additive" or formula.get("missing_policy") != "KEEP":
        raise ValueError("NEXT57 formula kind/missing policy differs")
    if formula.get("domain_gate") != DOMAIN_GATE:
        raise ValueError("NEXT57 formula domain gate differs")
    terms = formula.get("terms")
    if not isinstance(terms, list) or not 1 <= len(terms) <= 3:
        raise ValueError("NEXT57 formula term count differs")
    n = len(features)
    score = np.zeros(n, dtype=float)
    supported = np.asarray(features.get("combined_supported", False), dtype=bool).copy()
    dimension = pd.to_numeric(features.get("periodic_dimension_max"), errors="coerce").to_numpy(float)
    fraction = pd.to_numeric(features.get("periodic_framework_fraction"), errors="coerce").to_numpy(float)
    supported &= (
        np.isfinite(dimension)
        & np.isfinite(fraction)
        & (dimension >= DOMAIN_GATE["periodic_dimension_max_min"])
        & (fraction >= DOMAIN_GATE["periodic_framework_fraction_min"])
    )
    for term in terms:
        if not isinstance(term, Mapping):
            raise ValueError("NEXT57 formula term differs")
        feature = str(term.get("feature"))
        if feature not in features:
            raise ValueError(f"NEXT57 formula feature is missing: {feature}")
        direction = int(term.get("direction"))
        center = float(term.get("center"))
        scale = float(term.get("scale"))
        weight = float(term.get("weight"))
        if direction not in (-1, 1) or scale <= 0.0 or weight <= 0.0:
            raise ValueError("NEXT57 formula coefficient differs")
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        supported &= np.isfinite(values)
        score += weight * direction * (values - center) / scale
    threshold = float(formula.get("threshold"))
    if not math.isfinite(threshold):
        raise ValueError("NEXT57 formula threshold differs")
    score[~supported] = np.nan
    reject = supported & (score >= threshold)
    return score, supported, reject


def _decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, endpoint: np.ndarray
) -> dict[str, object]:
    protected = endpoint <= PROTECTED_MAX
    severe = endpoint >= SEVERE_MIN
    n = len(endpoint)
    n_supported = int(supported.sum())
    n_rejected = int(reject.sum())
    n_protected = int(protected.sum())
    protected_kept = int((protected & ~reject).sum())
    severe_rejected = int((severe & reject).sum())
    return {
        "rows": n,
        "supported": n_supported,
        "rejected": n_rejected,
        "protected": n_protected,
        "protected_kept": protected_kept,
        "severe": int(severe.sum()),
        "severe_rejected": severe_rejected,
        "coverage": n_supported / n if n else 0.0,
        "coverage_lower": _wilson_lower(n_supported, n),
        "protected_recall": protected_kept / n_protected if n_protected else None,
        "protected_recall_lower": _wilson_lower(protected_kept, n_protected),
        "reject_precision": severe_rejected / n_rejected if n_rejected else None,
        "reject_precision_lower": _wilson_lower(severe_rejected, n_rejected),
        "savings": n_rejected / n if n else 0.0,
        "savings_lower": _wilson_lower(n_rejected, n),
    }


def _stratum_labels(features: pd.DataFrame) -> np.ndarray:
    defective = np.asarray(features["defective"], dtype=bool)
    oms = np.asarray(features["open_metal_site"], dtype=bool)
    return np.asarray(
        [f"defective={int(left)}|oms={int(right)}" for left, right in zip(defective, oms, strict=True)],
        dtype=str,
    )


def _auc_diagnostics(
    *, score: np.ndarray, supported: np.ndarray, endpoint: np.ndarray, strata: np.ndarray
) -> dict[str, object]:
    extreme = (endpoint <= PROTECTED_MAX) | (endpoint >= SEVERE_MIN)
    truth = endpoint >= SEVERE_MIN
    pooled = _safe_auc(score[extreme & supported], truth[extreme & supported])
    values = {}
    aucs = []
    for stratum in sorted(set(strata.tolist())):
        mask = extreme & supported & (strata == stratum)
        auc = _safe_auc(score[mask], truth[mask])
        values[stratum] = {"rows": int(mask.sum()), "auc": auc}
        if auc is not None:
            aucs.append(auc)
    return {
        "pooled_extreme_auc": pooled,
        "macro_stratum_auc": float(np.mean(aucs)) if aucs else None,
        "worst_stratum_auc": float(np.min(aucs)) if aucs else None,
        "evaluable_strata": len(aucs),
        "strata": values,
    }


def _gate_rank(
    metrics: Mapping[str, object], aucs: Mapping[str, object], term_count: int
) -> tuple[float, ...]:
    values = (
        float(metrics["coverage_lower"]),
        float(metrics["protected_recall_lower"]),
        float(metrics["reject_precision_lower"]),
        float(metrics["savings_lower"]),
        float(aucs["pooled_extreme_auc"] or 0.0),
        float(aucs["macro_stratum_auc"] or 0.0),
        float(aucs["worst_stratum_auc"] or 0.0),
    )
    thresholds = tuple(float(value) for value in GATES.values())
    ratios = tuple(value / threshold for value, threshold in zip(values, thresholds, strict=True))
    passed = float(all(value >= threshold for value, threshold in zip(values, thresholds, strict=True)))
    return (
        passed,
        min(ratios),
        values[6],
        values[5],
        values[2],
        values[1],
        values[3],
        values[4],
        -float(term_count),
    )


def _term(info: Mapping[str, object], weight: float) -> dict[str, object]:
    return {
        "feature": str(info["feature"]),
        "direction": int(info["direction"]),
        "center": float(info["center"]),
        "scale": float(info["scale"]),
        "weight": float(weight),
    }


def search_discovery_rule(
    *,
    features: pd.DataFrame,
    endpoint: Sequence[float],
    candidate_features: Sequence[str] = NEXT55_FEATURE_NAMES,
) -> dict[str, object]:
    """Search the frozen finite catalogue using discovery labels only."""

    endpoint = np.asarray(endpoint, dtype=float)
    required = {
        "combined_supported",
        "periodic_dimension_max",
        "periodic_framework_fraction",
        "defective",
        "open_metal_site",
    }
    if (
        len(features) != len(endpoint)
        or not required.issubset(features.columns)
        or not np.isfinite(endpoint).all()
        or not ((endpoint <= PROTECTED_MAX).any() and (endpoint >= SEVERE_MIN).any())
    ):
        raise ValueError("NEXT57 discovery arrays differ")
    strata = _stratum_labels(features)
    base_supported = (
        np.asarray(features["combined_supported"], dtype=bool)
        & (pd.to_numeric(features["periodic_dimension_max"], errors="coerce").to_numpy(float) >= 1.0)
        & (pd.to_numeric(features["periodic_framework_fraction"], errors="coerce").to_numpy(float) >= 0.5)
    )
    extreme = (endpoint <= PROTECTED_MAX) | (endpoint >= SEVERE_MIN)
    truth = endpoint >= SEVERE_MIN
    infos = []
    for feature in candidate_features:
        if feature not in features:
            continue
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        finite = base_supported & np.isfinite(values)
        if int(finite.sum()) < 20:
            continue
        center = float(np.median(values[finite]))
        q25, q75 = np.quantile(values[finite], (0.25, 0.75))
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1.0e-12:
            continue
        raw_aucs = []
        for stratum in sorted(set(strata.tolist())):
            mask = finite & extreme & (strata == stratum)
            auc = _safe_auc(values[mask], truth[mask])
            if auc is not None:
                raw_aucs.append(auc)
        if not raw_aucs:
            continue
        direction = 1 if float(np.mean(raw_aucs)) >= 0.5 else -1
        directional = [auc if direction == 1 else 1.0 - auc for auc in raw_aucs]
        risk = direction * (values - center) / scale
        pooled = _safe_auc(risk[finite & extreme], truth[finite & extreme])
        if pooled is None:
            continue
        infos.append(
            {
                "feature": feature,
                "direction": direction,
                "center": center,
                "scale": scale,
                "risk": risk,
                "pooled_directional_auc": pooled,
                "macro_stratum_directional_auc": float(np.mean(directional)),
                "worst_stratum_directional_auc": float(np.min(directional)),
            }
        )
    if not infos:
        raise ValueError("NEXT57 has no evaluable feature")
    infos.sort(
        key=lambda item: (
            -float(item["worst_stratum_directional_auc"]),
            -float(item["macro_stratum_directional_auc"]),
            -float(item["pooled_directional_auc"]),
            str(item["feature"]),
        )
    )

    best = None
    considered = 0

    def consider(selected: Sequence[Mapping[str, object]], weights: Sequence[float]) -> None:
        nonlocal best, considered
        raw_score = np.zeros(len(features), dtype=float)
        finite = base_supported.copy()
        for info, weight in zip(selected, weights, strict=True):
            risk = np.asarray(info["risk"], dtype=float)
            finite &= np.isfinite(risk)
            raw_score += float(weight) * risk
        if not finite.any():
            return
        score_for_auc = raw_score.copy()
        score_for_auc[~finite] = np.nan
        aucs = _auc_diagnostics(
            score=score_for_auc, supported=finite, endpoint=endpoint, strata=strata
        )
        if aucs["pooled_extreme_auc"] is None:
            return
        thresholds = {
            float(np.quantile(raw_score[finite], 1.0 - fraction, method="inverted_cdf"))
            for fraction in REJECTION_FRACTIONS
        }
        for threshold in sorted(thresholds):
            formula = {
                "kind": "additive",
                "terms": [
                    _term(info, weight)
                    for info, weight in zip(selected, weights, strict=True)
                ],
                "threshold": threshold,
                "missing_policy": "KEEP",
                "domain_gate": dict(DOMAIN_GATE),
            }
            score, supported, reject = apply_odac23_formula(features, formula)
            metrics = _decision_metrics(supported=supported, reject=reject, endpoint=endpoint)
            rank = _gate_rank(metrics, aucs, len(selected))
            key = json.dumps(formula, sort_keys=True, separators=(",", ":"))
            considered += 1
            record = (rank, key, formula, metrics, aucs, score, supported, reject)
            if best is None or rank > best[0] or (rank == best[0] and key < best[1]):
                best = record

    for info in infos:
        consider((info,), (1.0,))
    pair_infos = infos[: min(PAIR_SHORTLIST, len(infos))]
    for left, right in combinations(pair_infos, 2):
        for weight in PAIR_WEIGHTS:
            consider((left, right), (1.0, weight))
    triple_infos = infos[: min(TRIPLE_SHORTLIST, len(infos))]
    for selected in combinations(triple_infos, 3):
        consider(selected, (1.0, 1.0, 1.0))
    if best is None:
        raise RuntimeError("NEXT57 finite catalogue is empty")
    rank, _key, formula, metrics, aucs, score, supported, reject = best
    return {
        "selected_formula": formula,
        "discovery_metrics": {**metrics, **{key: aucs[key] for key in (
            "pooled_extreme_auc", "macro_stratum_auc", "worst_stratum_auc", "evaluable_strata"
        )}},
        "stratum_diagnostics": aucs["strata"],
        "passes_discovery_gates": bool(rank[0] == 1.0),
        "candidate_count": considered,
        "evaluable_feature_count": len(infos),
        "feature_diagnostics": [
            {key: value for key, value in info.items() if key != "risk"} for info in infos
        ],
        "rank": list(rank),
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _strict_json(path: Path, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_discovery_search(
    *,
    feature_dir: Path,
    firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run discovery-only search; no lockbox path exists in this interface."""

    feature_dir = Path(feature_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / MANIFEST_NAME,
        "firewall_manifest": Path(firewall_manifest_path).resolve(),
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT57 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "firewall_manifest": EXPECTED_FIREWALL_MANIFEST_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT57 frozen input hash differs")
    feature_manifest = _strict_json(paths["feature_manifest"], "NEXT55 manifest")
    firewall_manifest = _strict_json(paths["firewall_manifest"], "NEXT56 firewall manifest")
    discovery_manifest = _strict_json(paths["discovery_manifest"], "NEXT56 discovery manifest")
    feature_outputs = feature_manifest.get("outputs_sha256")
    discovery_outputs = discovery_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != SOURCE_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(paths["features"].name) != hashes["features"]
        or firewall_manifest.get("protocol") != FIREWALL_PROTOCOL
        or firewall_manifest.get("endpoint_values_summarized_or_inspected") is not False
        or discovery_manifest.get("protocol") != FIREWALL_PROTOCOL
        or discovery_manifest.get("partition_role") != "discovery"
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get(ROLE_LABELS_NAME) != hashes["discovery_labels"]
    ):
        raise ValueError("NEXT57 provenance boundary differs")

    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT57 received a non-discovery label")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(features) or len(joined) != len(labels):
        raise ValueError("NEXT57 discovery identity coverage differs")
    result = search_discovery_rule(features=joined, endpoint=joined[ENDPOINT_COLUMN].to_numpy(float))
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / frozen discovery role only",
        "endpoint_definition": {
            "column": ENDPOINT_COLUMN,
            "protected_max_angstrom": PROTECTED_MAX,
            "severe_min_angstrom": SEVERE_MIN,
        },
        "gates": GATES,
        "feature_artifact_sha256": hashes["features"],
        "scientific_status": "advance_to_internal_validation"
        if result["passes_discovery_gates"]
        else "discovery_failure_diagnostic_only",
    }
    search_record = {
        key: value
        for key, value in result.items()
        if key not in {"score", "supported", "reject", "selected_formula"}
    }
    predictions = pd.DataFrame(
        {
            "material_id": joined["material_id"].astype(str),
            "partition_role": "discovery",
            ENDPOINT_COLUMN: joined[ENDPOINT_COLUMN].to_numpy(float),
            "protected": joined[ENDPOINT_COLUMN].to_numpy(float) <= PROTECTED_MAX,
            "severe": joined[ENDPOINT_COLUMN].to_numpy(float) >= SEVERE_MIN,
            "risk_score": result["score"],
            "supported": result["supported"],
            "reject": result["reject"],
        }
    )

    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "discovery_only_finite_formula_search",
        "discovery_labels_opened": True,
        "internal_validation_labels_opened": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "passes_discovery_gates": result["passes_discovery_gates"],
        "counts": {
            "rows": len(joined),
            "protected": int(predictions["protected"].sum()),
            "severe": int(predictions["severe"].sum()),
            "supported": int(predictions["supported"].sum()),
            "rejected": int(predictions["reject"].sum()),
            "candidate_formulas": int(result["candidate_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {"src/next57_odac23_discovery_search.py": source_hash},
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        predictions_path = staging / PREDICTIONS_NAME
        formula_path.write_bytes(_json_bytes(formula))
        search_path.write_bytes(_json_bytes(search_record))
        predictions.to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {
            path.name: _sha256(path)
            for path in (formula_path, search_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT57 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT57 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--firewall-manifest", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_discovery_search(
        feature_dir=args.feature_dir,
        firewall_manifest_path=args.firewall_manifest,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps({"passes": manifest["passes_discovery_gates"], **manifest["counts"]}, indent=2, sort_keys=True))


__all__ = [
    "DOMAIN_GATE",
    "GATES",
    "PROTOCOL",
    "apply_odac23_formula",
    "run_discovery_search",
    "search_discovery_rule",
]


if __name__ == "__main__":
    main()
