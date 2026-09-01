#!/usr/bin/env python3
"""Discover a precision-targeted stable explicit ODAC23 x0 law."""

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
from src.next65_odac23_physics_couplings import NEXT65_FEATURE_NAMES
from src.next68_odac23_sparse_stable_law import apply_sparse_formula
from src.next70_odac23_metal_donor_bond_valence_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    METAL_DONOR_BV_FEATURE_NAMES,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next71-odac23-precision-targeted-stable-law-v1"
DESIGN_SHA256 = "3ef716c5c9eb16b0d8d78e9215b99671db703ec6f2cb8a6abaae84c590797f49"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "928a0bbfa1120e2c92bac2e9d3f0046a1d440c24beb72f652e477eb827874f14"
)
EXPECTED_FEATURE_SHA256 = (
    "d3684af21c70e3be18ae4aed8dd9a505209cfb2d91e9639911aae72da77ca6dc"
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
CANDIDATE_FEATURE_NAMES = tuple(NEXT65_FEATURE_NAMES) + tuple(METAL_DONOR_BV_FEATURE_NAMES)
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)
PROTECTED_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0)
TERM_COUNTS = (3, 5, 8)
STABILITY_FOLDS = 5
FORMULA_NAME = "NEXT71_ODAC23_PRECISION_TARGETED_CANDIDATE.json"
SEARCH_NAME = "NEXT71_ODAC23_PRECISION_TARGETED_SEARCH.json"
PREDICTIONS_NAME = "next71_odac23_precision_targeted_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _fold(material_id: object) -> int:
    payload = b"NEXT71-CV-v1\0" + str(material_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % STABILITY_FOLDS


def _strata(features: pd.DataFrame) -> np.ndarray:
    defective = np.asarray(features["defective"], dtype=bool)
    oms = np.asarray(features["open_metal_site"], dtype=bool)
    return np.asarray(
        [f"defective={int(left)}|oms={int(right)}" for left, right in zip(defective, oms, strict=True)],
        dtype=str,
    )


def _fit_l2(
    x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, c_value: float
) -> tuple[np.ndarray, float, int]:
    if len(np.unique(y)) != 2:
        raise ValueError("NEXT71 fit requires both severe and non-severe rows")
    model = LogisticRegression(
        l1_ratio=0.0,
        C=float(c_value),
        solver="liblinear",
        class_weight="balanced",
        fit_intercept=True,
        max_iter=5000,
        tol=1.0e-8,
        random_state=0,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model.coef_[0].astype(float), float(model.intercept_[0]), int(model.n_iter_[0])


def search_precision_targeted_stable_law(
    *,
    features: pd.DataFrame,
    endpoint: Sequence[float],
    candidate_features: Sequence[str] = CANDIDATE_FEATURE_NAMES,
    c_values: Sequence[float] = C_VALUES,
    protected_multipliers: Sequence[float] = PROTECTED_MULTIPLIERS,
    term_counts: Sequence[int] = TERM_COUNTS,
) -> dict[str, object]:
    """Fit severe versus all non-severe rows and evaluate frozen decision gates."""

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
        raise ValueError("NEXT71 discovery arrays differ")
    if (
        not c_values
        or not protected_multipliers
        or not term_counts
        or any(float(value) <= 0.0 for value in (*c_values, *protected_multipliers))
        or any(int(value) not in TERM_COUNTS for value in term_counts)
    ):
        raise ValueError("NEXT71 frozen catalogue differs")
    dimension = pd.to_numeric(features["periodic_dimension_max"], errors="coerce").to_numpy(float)
    fraction = pd.to_numeric(features["periodic_framework_fraction"], errors="coerce").to_numpy(float)
    domain = (
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
        finite = domain & np.isfinite(values)
        if int(finite.sum()) / max(1, int(domain.sum())) < 0.95:
            continue
        center = float(np.median(values[finite]))
        q25, q75 = np.quantile(values[finite], (0.25, 0.75))
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1.0e-12:
            continue
        names.append(str(feature))
        centers.append(center)
        scales.append(scale)
        columns.append((values - center) / scale)
    if not columns:
        raise ValueError("NEXT71 has no evaluable feature")
    matrix = np.column_stack(columns)
    complete = domain & np.isfinite(matrix).all(axis=1)
    if int(complete.sum()) / max(1, len(features)) < 0.95:
        raise ValueError("NEXT71 common fitting coverage is below frozen gate")
    truth = endpoint >= SEVERE_MIN
    protected = endpoint <= PROTECTED_MAX
    x_fit = matrix[complete]
    y_fit = truth[complete].astype(int)
    fit_protected = protected[complete]
    folds = np.asarray([_fold(value) for value in features["material_id"]], dtype=int)[complete]
    strata = _strata(features)

    best = None
    formula_keys: set[str] = set()
    candidate_count = 0
    model_diagnostics: list[dict[str, object]] = []
    for c_value in c_values:
        for protected_multiplier in protected_multipliers:
            weights = np.ones(len(x_fit), dtype=float)
            weights[fit_protected] *= float(protected_multiplier)
            full_coef, intercept, full_iterations = _fit_l2(
                x_fit, y_fit, weights, float(c_value)
            )
            fold_coefs = []
            fold_iterations = []
            for held_out in range(STABILITY_FOLDS):
                train = folds != held_out
                coef, _fold_intercept, iterations = _fit_l2(
                    x_fit[train], y_fit[train], weights[train], float(c_value)
                )
                fold_coefs.append(coef)
                fold_iterations.append(iterations)
            reached_limit = full_iterations >= 5000 or any(
                value >= 5000 for value in fold_iterations
            )
            fold_matrix = np.vstack(fold_coefs)
            nonzero = np.abs(full_coef) > 1.0e-12
            same_sign = (np.abs(fold_matrix) > 1.0e-12) & (
                np.sign(fold_matrix) == np.sign(full_coef)[None, :]
            )
            stable = nonzero & (same_sign.sum(axis=0) >= 4)
            stable_indices = np.asarray(
                sorted(
                    np.flatnonzero(stable).tolist(),
                    key=lambda idx: (-abs(full_coef[idx]), names[idx]),
                ),
                dtype=int,
            )
            model_diagnostics.append(
                {
                    "C": float(c_value),
                    "protected_multiplier": float(protected_multiplier),
                    "intercept": intercept,
                    "full_iterations": full_iterations,
                    "fold_iterations": fold_iterations,
                    "reached_iteration_limit": reached_limit,
                    "stable_feature_count": len(stable_indices),
                    "leading_stable_features": [names[idx] for idx in stable_indices[:20]],
                    "leading_stable_coefficients": [
                        float(full_coef[idx]) for idx in stable_indices[:20]
                    ],
                }
            )
            if reached_limit or len(stable_indices) == 0:
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
                thresholds = {
                    float(np.quantile(raw_score[supported], 1.0 - rejection, method="inverted_cdf"))
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
                        float(protected_multiplier),
                        int(requested_k),
                    )
                    if best is None or rank > best[0] or (rank == best[0] and key < best[1]):
                        best = record
    if best is None:
        raise RuntimeError("NEXT71 stable formula catalogue is empty")
    (
        rank,
        _key,
        formula,
        metrics,
        aucs,
        score,
        supported,
        reject,
        c_value,
        protected_multiplier,
        requested_k,
    ) = best
    return {
        "selected_formula": formula,
        "selected_C": c_value,
        "selected_protected_multiplier": protected_multiplier,
        "selected_requested_term_count": requested_k,
        "selected_fit_reached_iteration_limit": False,
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
        "common_fit_rows": int(complete.sum()),
        "model_diagnostics": model_diagnostics,
        "rank": list(rank),
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT71 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_precision_targeted_search(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run discovery only and atomically publish the explicit candidate."""

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
        raise FileNotFoundError("NEXT71 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "features": EXPECTED_FEATURE_SHA256,
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT71 frozen input hash differs")
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
        raise ValueError("NEXT71 discovery-only provenance differs")
    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT71 received non-discovery labels")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise ValueError("NEXT71 discovery identity differs")
    result = search_precision_targeted_stable_law(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        candidate_features=CANDIDATE_FEATURE_NAMES,
    )
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / robust discovery only",
        "fitting_role": "severe-versus-all coefficient discovery; executable is fixed x0 arithmetic",
        "endpoint_definition": {
            "column": ENDPOINT_COLUMN,
            "protected_max_angstrom": PROTECTED_MAX,
            "severe_min_angstrom": SEVERE_MIN,
            "minimum_adsorbate_configurations": 4,
            "common_translation_removed": True,
        },
        "gates": GATES,
        "selected_C": result["selected_C"],
        "selected_protected_multiplier": result["selected_protected_multiplier"],
        "selected_requested_term_count": result["selected_requested_term_count"],
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
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
        "mode": "precision_targeted_stable_explicit_robust_discovery_search",
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
            "candidate_features": len(CANDIDATE_FEATURE_NAMES),
            "usable_features": int(result["usable_feature_count"]),
            "candidate_formulas": int(result["candidate_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next71_odac23_precision_targeted_stable_law.py": source_hash
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
            raise RuntimeError("NEXT71 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT71 input changed before publication")
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
    manifest = run_precision_targeted_search(
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


__all__ = ["PROTOCOL", "run_precision_targeted_search", "search_precision_targeted_stable_law"]


if __name__ == "__main__":
    main()
