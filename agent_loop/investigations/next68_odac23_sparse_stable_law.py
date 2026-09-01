#!/usr/bin/env python3
"""Discover and publish a sparse stable explicit ODAC23 x0 law."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next57_odac23_discovery_search import (
    DOMAIN_GATE,
    GATES,
    PROTECTED_MAX,
    REJECTION_FRACTIONS,
    SEVERE_MIN,
    _auc_diagnostics,
    _decision_metrics,
    _gate_rank,
)
from src.next60_odac23_robust_scaffold_endpoint import (
    ENDPOINT_COLUMN,
    PROTOCOL as ENDPOINT_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
)
from src.next65_odac23_physics_couplings import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    NEXT65_FEATURE_NAMES,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next68-odac23-sparse-stable-explicit-law-v1"
DESIGN_SHA256 = "7b55a116c20af5de933b989e372014f82a4af6e746485e71055678d689acd820"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "8a858b58f6772a50b1ee3ea900bef9d66eb5636efadfeef878d9c7740011de5c"
)
EXPECTED_ENDPOINT_FIREWALL_SHA256 = (
    "9dbd3f78d2505ba96b33715e6409cd8524e9b909f4134af0020b933dff2f769f"
)
EXPECTED_DISCOVERY_MANIFEST_SHA256 = (
    "6ca39eb42629d626559618474f75aa6bb6571a38a928b3b16512b5d987b76137"
)
EXPECTED_DISCOVERY_LABEL_SHA256 = (
    "1a7c78fd87bb3f5795e59fa3c3799fbbb07a1629b90d472aef7e73740ce7f08a"
)
C_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
TERM_COUNTS = (3, 5, 8)
STABILITY_FOLDS = 5
FORMULA_NAME = "NEXT68_ODAC23_SPARSE_STABLE_CANDIDATE.json"
SEARCH_NAME = "NEXT68_ODAC23_SPARSE_STABLE_SEARCH.json"
PREDICTIONS_NAME = "next68_odac23_sparse_stable_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _strata(features: pd.DataFrame) -> np.ndarray:
    defective = np.asarray(features["defective"], dtype=bool)
    oms = np.asarray(features["open_metal_site"], dtype=bool)
    return np.asarray(
        [f"defective={int(left)}|oms={int(right)}" for left, right in zip(defective, oms, strict=True)],
        dtype=str,
    )


def _fold(material_id: object) -> int:
    payload = b"NEXT68-CV-v1\0" + str(material_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % STABILITY_FOLDS


def apply_sparse_formula(
    features: pd.DataFrame, formula: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a one-to-eight-term explicit formula with strict fail-open behavior."""

    if formula.get("kind") != "additive" or formula.get("missing_policy") != "KEEP":
        raise ValueError("NEXT68 formula kind/missing policy differs")
    if formula.get("domain_gate") != DOMAIN_GATE:
        raise ValueError("NEXT68 formula domain gate differs")
    terms = formula.get("terms")
    if not isinstance(terms, list) or not 1 <= len(terms) <= max(TERM_COUNTS):
        raise ValueError("NEXT68 formula term count differs")
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
            raise ValueError("NEXT68 formula term differs")
        feature = str(term.get("feature"))
        if feature not in features:
            raise ValueError(f"NEXT68 formula feature is missing: {feature}")
        direction = int(term.get("direction"))
        center = float(term.get("center"))
        scale = float(term.get("scale"))
        weight = float(term.get("weight"))
        if direction not in (-1, 1) or scale <= 0.0 or weight <= 0.0:
            raise ValueError("NEXT68 formula coefficient differs")
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        supported &= np.isfinite(values)
        score += weight * direction * (values - center) / scale
    threshold = float(formula.get("threshold"))
    if not math.isfinite(threshold):
        raise ValueError("NEXT68 formula threshold differs")
    score[~supported] = np.nan
    reject = supported & (score >= threshold)
    return score, supported, reject


def _fit_l1(x: np.ndarray, y: np.ndarray, c_value: float) -> tuple[np.ndarray, float, int]:
    if len(np.unique(y)) != 2:
        raise ValueError("NEXT68 fit requires both extreme classes")
    model = LogisticRegression(
        l1_ratio=1.0,
        C=float(c_value),
        solver="liblinear",
        class_weight="balanced",
        fit_intercept=True,
        max_iter=5000,
        tol=1.0e-8,
        random_state=0,
    )
    model.fit(x, y)
    return model.coef_[0].astype(float), float(model.intercept_[0]), int(model.n_iter_[0])


def search_sparse_stable_law(
    *,
    features: pd.DataFrame,
    endpoint: Sequence[float],
    candidate_features: Sequence[str] = NEXT65_FEATURE_NAMES,
    c_values: Sequence[float] = C_VALUES,
    term_counts: Sequence[int] = TERM_COUNTS,
) -> dict[str, object]:
    """Fit only extreme discovery labels, then evaluate explicit sparse formulas."""

    endpoint = np.asarray(endpoint, dtype=float)
    required = {
        "material_id",
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
        raise ValueError("NEXT68 discovery arrays differ")
    if not c_values or not term_counts or any(float(value) <= 0.0 for value in c_values):
        raise ValueError("NEXT68 fit grid differs")
    if any(int(value) < 1 or int(value) > max(TERM_COUNTS) for value in term_counts):
        raise ValueError("NEXT68 term-count grid differs")

    dimension = pd.to_numeric(features["periodic_dimension_max"], errors="coerce").to_numpy(float)
    fraction = pd.to_numeric(features["periodic_framework_fraction"], errors="coerce").to_numpy(float)
    base_supported = (
        np.asarray(features["combined_supported"], dtype=bool)
        & np.isfinite(dimension)
        & np.isfinite(fraction)
        & (dimension >= DOMAIN_GATE["periodic_dimension_max_min"])
        & (fraction >= DOMAIN_GATE["periodic_framework_fraction_min"])
    )
    names: list[str] = []
    centers: list[float] = []
    scales: list[float] = []
    columns: list[np.ndarray] = []
    for feature in candidate_features:
        if feature not in features or feature in names:
            continue
        values = pd.to_numeric(features[feature], errors="coerce").to_numpy(float)
        if not np.isfinite(values[base_supported]).all():
            continue
        center = float(np.median(values[base_supported]))
        q25, q75 = np.quantile(values[base_supported], (0.25, 0.75))
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1.0e-12:
            continue
        names.append(str(feature))
        centers.append(center)
        scales.append(scale)
        columns.append((values - center) / scale)
    if not columns:
        raise ValueError("NEXT68 has no complete nonconstant feature")
    matrix = np.column_stack(columns)
    extreme = (endpoint <= PROTECTED_MAX) | (endpoint >= SEVERE_MIN)
    truth = endpoint >= SEVERE_MIN
    fit_mask = base_supported & extreme
    x_fit = matrix[fit_mask]
    y_fit = truth[fit_mask].astype(int)
    folds = np.asarray([_fold(value) for value in features["material_id"]], dtype=int)[fit_mask]
    strata = _strata(features)

    best = None
    formula_keys: set[str] = set()
    candidate_count = 0
    model_diagnostics: list[dict[str, object]] = []
    for c_value in c_values:
        full_coef, intercept, full_iterations = _fit_l1(x_fit, y_fit, float(c_value))
        fold_coefs = []
        fold_iterations = []
        for held_out in range(STABILITY_FOLDS):
            train = folds != held_out
            coef, _fold_intercept, iterations = _fit_l1(x_fit[train], y_fit[train], float(c_value))
            fold_coefs.append(coef)
            fold_iterations.append(iterations)
        fold_matrix = np.vstack(fold_coefs)
        full_nonzero = np.abs(full_coef) > 1.0e-10
        same_sign = (np.abs(fold_matrix) > 1.0e-10) & (
            np.sign(fold_matrix) == np.sign(full_coef)[None, :]
        )
        stable = full_nonzero & (same_sign.sum(axis=0) >= 4)
        stable_indices = np.flatnonzero(stable)
        stable_indices = np.asarray(
            sorted(stable_indices.tolist(), key=lambda idx: (-abs(full_coef[idx]), names[idx])),
            dtype=int,
        )
        model_diagnostics.append(
            {
                "C": float(c_value),
                "intercept": intercept,
                "full_iterations": full_iterations,
                "fold_iterations": fold_iterations,
                "full_nonzero_features": int(full_nonzero.sum()),
                "stable_features": [names[idx] for idx in stable_indices],
                "stable_coefficients": [float(full_coef[idx]) for idx in stable_indices],
                "same_sign_fold_counts": [int(same_sign[:, idx].sum()) for idx in stable_indices],
            }
        )
        if len(stable_indices) == 0:
            continue
        for requested_k in term_counts:
            selected_indices = stable_indices[: min(int(requested_k), len(stable_indices))]
            terms = []
            for idx in sorted(selected_indices.tolist(), key=lambda value: names[value]):
                coefficient = float(full_coef[idx])
                terms.append(
                    {
                        "feature": names[idx],
                        "direction": 1 if coefficient > 0.0 else -1,
                        "center": centers[idx],
                        "scale": scales[idx],
                        "weight": abs(coefficient),
                    }
                )
            terms_key = json.dumps(terms, sort_keys=True, separators=(",", ":"))
            if terms_key in formula_keys:
                continue
            formula_keys.add(terms_key)
            provisional = {
                "kind": "additive",
                "terms": terms,
                "threshold": 0.0,
                "missing_policy": "KEEP",
                "domain_gate": dict(DOMAIN_GATE),
            }
            raw_score, supported, _reject = apply_sparse_formula(features, provisional)
            aucs = _auc_diagnostics(
                score=raw_score, supported=supported, endpoint=endpoint, strata=strata
            )
            finite_score = raw_score[supported]
            thresholds = {
                float(np.quantile(finite_score, 1.0 - rejection, method="inverted_cdf"))
                for rejection in REJECTION_FRACTIONS
            }
            for threshold in sorted(thresholds):
                formula = {**provisional, "threshold": threshold}
                score, formula_supported, reject = apply_sparse_formula(features, formula)
                metrics = _decision_metrics(
                    supported=formula_supported, reject=reject, endpoint=endpoint
                )
                rank = _gate_rank(metrics, aucs, len(terms))
                key = json.dumps(formula, sort_keys=True, separators=(",", ":"))
                candidate_count += 1
                record = (
                    rank,
                    key,
                    formula,
                    metrics,
                    aucs,
                    score,
                    formula_supported,
                    reject,
                    float(c_value),
                    int(requested_k),
                )
                if best is None or rank > best[0] or (rank == best[0] and key < best[1]):
                    best = record
    if best is None:
        raise RuntimeError("NEXT68 sparse stable catalogue is empty")
    rank, _key, formula, metrics, aucs, score, supported, reject, c_value, requested_k = best
    return {
        "selected_formula": formula,
        "selected_C": c_value,
        "selected_requested_term_count": requested_k,
        "discovery_metrics": {
            **metrics,
            **{
                key: aucs[key]
                for key in (
                    "pooled_extreme_auc",
                    "macro_stratum_auc",
                    "worst_stratum_auc",
                    "evaluable_strata",
                )
            },
        },
        "stratum_diagnostics": aucs["strata"],
        "passes_discovery_gates": bool(rank[0] == 1.0),
        "candidate_count": candidate_count,
        "unique_sparse_formula_count": len(formula_keys),
        "requested_feature_count": len(candidate_features),
        "usable_feature_count": len(names),
        "fit_extreme_rows": int(fit_mask.sum()),
        "fit_protected_rows": int((fit_mask & ~truth).sum()),
        "fit_severe_rows": int((fit_mask & truth).sum()),
        "model_diagnostics": model_diagnostics,
        "rank": list(rank),
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT68 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_sparse_stable_search(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run discovery only and immutably publish the frozen NEXT68 artifacts."""

    feature_dir = Path(feature_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / MANIFEST_NAME,
        "endpoint_firewall": Path(endpoint_firewall_manifest_path).resolve(),
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT68 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT68 frozen input hash differs")
    feature_manifest = _read_json(paths["feature_manifest"])
    endpoint_firewall = _read_json(paths["endpoint_firewall"])
    discovery_manifest = _read_json(paths["discovery_manifest"])
    feature_outputs = feature_manifest.get("outputs_sha256")
    discovery_outputs = discovery_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != SOURCE_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
        or endpoint_firewall.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_firewall.get("internal_validation_endpoint_values_summarized_or_inspected") is not False
        or endpoint_firewall.get("internal_replication_endpoint_values_summarized_or_inspected") is not False
        or discovery_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or discovery_manifest.get("partition_role") != "discovery"
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get(ROLE_LABELS_NAME) != hashes["discovery_labels"]
    ):
        raise ValueError("NEXT68 discovery-only provenance differs")

    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT68 received non-discovery labels")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise ValueError("NEXT68 robust discovery identity differs")
    result = search_sparse_stable_law(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        candidate_features=NEXT65_FEATURE_NAMES,
    )
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / robust discovery only",
        "fitting_role": "sparse coefficient discovery only; executable output is an explicit x0 formula",
        "endpoint_definition": {
            "column": ENDPOINT_COLUMN,
            "protected_max_angstrom": PROTECTED_MAX,
            "severe_min_angstrom": SEVERE_MIN,
            "minimum_adsorbate_configurations": 4,
            "common_translation_removed": True,
        },
        "gates": GATES,
        "selected_C": result["selected_C"],
        "selected_requested_term_count": result["selected_requested_term_count"],
        "candidate_feature_count": len(NEXT65_FEATURE_NAMES),
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
    endpoint = joined[ENDPOINT_COLUMN].to_numpy(float)
    predictions = pd.DataFrame(
        {
            "material_id": joined["material_id"].astype(str),
            "partition_role": "discovery",
            ENDPOINT_COLUMN: endpoint,
            "protected": endpoint <= PROTECTED_MAX,
            "severe": endpoint >= SEVERE_MIN,
            "risk_score": result["score"],
            "supported": result["supported"],
            "reject": result["reject"],
        }
    )
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "sparse_stable_explicit_robust_discovery_search",
        "robust_discovery_labels_opened": True,
        "internal_validation_labels_opened": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "executable_is_fixed_explicit_weighted_sum": True,
        "passes_discovery_gates": result["passes_discovery_gates"],
        "counts": {
            "rows": len(joined),
            "protected": int(predictions["protected"].sum()),
            "severe": int(predictions["severe"].sum()),
            "supported": int(predictions["supported"].sum()),
            "rejected": int(predictions["reject"].sum()),
            "candidate_features": len(NEXT65_FEATURE_NAMES),
            "usable_features": int(result["usable_feature_count"]),
            "candidate_formulas": int(result["candidate_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next68_odac23_sparse_stable_law.py": source_hash
        },
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
            path.name: _sha256(path) for path in (formula_path, search_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT68 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT68 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--endpoint-firewall", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_sparse_stable_search(
        feature_dir=args.feature_dir,
        endpoint_firewall_manifest_path=args.endpoint_firewall,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"passes": manifest["passes_discovery_gates"], **manifest["counts"]},
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "PROTOCOL",
    "apply_sparse_formula",
    "run_sparse_stable_search",
    "search_sparse_stable_law",
]


if __name__ == "__main__":
    main()
