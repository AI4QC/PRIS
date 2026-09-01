#!/usr/bin/env python3
"""Bounded two-partition search for the ODAC23 SSSP framework guard."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next57_odac23_discovery_search import (
    GATES,
    _auc_diagnostics,
    _decision_metrics,
    _stratum_labels,
)
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next60_odac23_robust_scaffold_endpoint import (
    ENDPOINT_COLUMN,
    PROTOCOL as ENDPOINT_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
)
from src.next68_odac23_sparse_stable_law import apply_sparse_formula
from src.next77_odac23_analytic_electrostatic_features import (
    FEATURES_NAME as FRAMEWORK_FEATURES_NAME,
    MANIFEST_NAME as FRAMEWORK_MANIFEST_NAME,
    PROTOCOL as FRAMEWORK_PROTOCOL,
)
from src.next533_odac23_sssp_label_free_features import (
    CATALOGUE_NAME as SSSP_CATALOGUE_NAME,
    MANIFEST_NAME as SSSP_MANIFEST_NAME,
    PROTOCOL as SSSP_PROTOCOL,
    TABLE_NAME as SSSP_TABLE_NAME,
)


PROTOCOL = "2026-08-13-next534-odac23-sssp-framework-guard-v1"
DEVELOPMENT_ROLES = ("discovery", "internal_validation")
SSSP_FEATURE = "sssp_same_sign_shell_purity_q10"
SSSP_CUTOFF = 0.5231805323
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
COMBINED_PRECISION_LOWER_MIN = 0.80
NEXT79_PROTOCOL = "2026-08-03-next79-odac23-single-electrostatic-residual-guard-v1"
FORMULA_NAME = "NEXT534_ODAC23_SSSP_FRAMEWORK_GUARD.json"
SEARCH_NAME = "NEXT534_ODAC23_SSSP_FRAMEWORK_SEARCH.json"
PREDICTIONS_NAME = "next534_odac23_sssp_framework_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": "74db90a13d3eb30e392ab3b1723f8bfb92e1a08882c3577fd8e94172161c37af",
    "framework_features": "c2c6668d24c77c20d8ab2878fa5e6b7d266f26ebea85b58ae083985eec6f2ad7",
    "framework_manifest": "ac1ecd88cfdf57fcec1c5dbe331ea8373f07aac79b706ed0d2c4c2575ab9ad82",
    "next79_formula": "ad0e079405e16c482c5f77c9b846b415c833a5a153ddd1c15e1d5cf19db9f291",
    "next79_manifest": "cc226a551a53e1e16a725090c1f896e20be09e4120c275bf6816a440b1e46170",
    "endpoint_firewall": "9dbd3f78d2505ba96b33715e6409cd8524e9b909f4134af0020b933dff2f769f",
    "discovery_manifest": "6ca39eb42629d626559618474f75aa6bb6571a38a928b3b16512b5d987b76137",
    "discovery_labels": "1a7c78fd87bb3f5795e59fa3c3799fbbb07a1629b90d472aef7e73740ce7f08a",
    "validation_manifest": "cb3aa1a4914d25e8b50b626386dda34fe59c5b692e130a10faa06d1083d058b9",
    "validation_labels": "ad964615c26fff9a7fcdaea7dc9d24042fa55f61eedd690758236d08cd6e2fc8",
    "sssp_features": "f570d299be7fdbf2e6dcd5e2989dfea8e7f3c83c34fba7add82162c685ccbbb4",
    "sssp_catalogue": "f283b0a4b4a457a00f44752299958ed0028b2f0841dc07ac6133f4383a3bb379",
    "sssp_manifest": "192bdd2bb3ac018f07d0b5efcf951f9c17a070a3fb5835f58db1136d19fccb63",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"NEXT534 {path.name} must contain an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sssp_deficit(values: np.ndarray, supported: np.ndarray) -> np.ndarray:
    """Return the frozen bounded SSSP deficit, with missing support contributing zero."""

    array = np.asarray(values, dtype=float)
    support = np.asarray(supported, dtype=bool)
    if array.ndim != 1 or support.shape != array.shape:
        raise ValueError("NEXT534 SSSP arrays differ")
    if np.any(support & ~np.isfinite(array)):
        raise ValueError("NEXT534 SSSP support semantics differ")
    result = np.zeros(array.shape, dtype=float)
    result[support] = np.maximum(0.0, SSSP_CUTOFF - array[support]) / SSSP_CUTOFF
    if np.any(~np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0 + 1e-12):
        raise ValueError("NEXT534 SSSP deficit differs")
    return np.minimum(result, 1.0)


def apply_sssp_framework_guard(
    features: pd.DataFrame,
    anchor_formula: Mapping[str, object],
    *,
    weight: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact NEXT79 score plus the sole optional frozen SSSP increment."""

    if SSSP_FEATURE not in features or "sssp_supported" not in features:
        raise ValueError("NEXT534 SSSP feature is missing")
    weight = float(weight)
    threshold = float(threshold)
    if weight not in WEIGHTS or not math.isfinite(threshold):
        raise ValueError("NEXT534 grid value differs")
    base_score, base_supported, _ = apply_sparse_formula(features, anchor_formula)
    sssp_values = pd.to_numeric(features[SSSP_FEATURE], errors="coerce").to_numpy(float)
    sssp_support = features["sssp_supported"].fillna(False).to_numpy(bool)
    deficit = sssp_deficit(sssp_values, sssp_support)
    score = base_score + weight * deficit
    score[~base_supported] = np.nan
    reject = base_supported & (score >= threshold)
    return score, base_supported, reject, deficit


def _passes_partition_gates(
    metrics: Mapping[str, object], aucs: Mapping[str, object]
) -> bool:
    checks = {
        "coverage_lower_at_least": metrics["coverage_lower"],
        "protected_recall_lower_at_least": metrics["protected_recall_lower"],
        "reject_precision_lower_at_least": metrics["reject_precision_lower"],
        "savings_lower_at_least": metrics["savings_lower"],
        "pooled_extreme_auc_at_least": aucs["pooled_extreme_auc"],
        "macro_stratum_auc_at_least": aucs["macro_stratum_auc"],
        "worst_stratum_auc_at_least": aucs["worst_stratum_auc"],
    }
    return all(
        value is not None and float(value) >= float(GATES[name])
        for name, value in checks.items()
    )


def _candidate_sort_key(record: Mapping[str, object]) -> tuple[object, ...]:
    partitions = record["partition_metrics"]
    combined = record["combined_metrics"]
    weight = float(record["weight"])
    min_precision = min(
        float(partitions[role]["reject_precision_lower"]) for role in DEVELOPMENT_ROLES
    )
    min_auc = min(
        float(partitions[role]["pooled_extreme_auc"]) for role in DEVELOPMENT_ROLES
    )
    min_protected = min(
        float(partitions[role]["protected_recall_lower"]) for role in DEVELOPMENT_ROLES
    )
    severe = int(combined["severe"])
    combined_severe_recall = int(combined["severe_rejected"]) / severe if severe else 0.0
    formula_order = json.dumps(record["formula"], sort_keys=True, separators=(",", ":"))
    return (
        -min_precision,
        -min_auc,
        -min_protected,
        -combined_severe_recall,
        int(weight != 0.0),
        weight,
        formula_order,
    )


def _diagnostic_sort_key(record: Mapping[str, object]) -> tuple[object, ...]:
    partitions = record["partition_metrics"]
    ratios = []
    for role in DEVELOPMENT_ROLES:
        metrics = partitions[role]
        ratios.extend(
            [
                float(metrics["coverage_lower"]) / float(GATES["coverage_lower_at_least"]),
                float(metrics["protected_recall_lower"])
                / float(GATES["protected_recall_lower_at_least"]),
                float(metrics["reject_precision_lower"] or 0.0)
                / float(GATES["reject_precision_lower_at_least"]),
                float(metrics["savings_lower"]) / float(GATES["savings_lower_at_least"]),
                float(metrics["pooled_extreme_auc"] or 0.0)
                / float(GATES["pooled_extreme_auc_at_least"]),
                float(metrics["macro_stratum_auc"] or 0.0)
                / float(GATES["macro_stratum_auc_at_least"]),
                float(metrics["worst_stratum_auc"] or 0.0)
                / float(GATES["worst_stratum_auc_at_least"]),
            ]
        )
    combined_ratio = float(record["combined_metrics"]["reject_precision_lower"] or 0.0) / (
        COMBINED_PRECISION_LOWER_MIN
    )
    return (
        -min(ratios + [combined_ratio]),
        *_candidate_sort_key(record),
    )


def _serializable_candidate(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "weight": record["weight"],
        "threshold": record["threshold"],
        "formula": record["formula"],
        "partition_metrics": record["partition_metrics"],
        "combined_metrics": record["combined_metrics"],
        "passes_two_partition_readiness": record["passes_two_partition_readiness"],
    }


def search_two_partition_guard(
    development: pd.DataFrame,
    anchor_formula: Mapping[str, object],
    *,
    weights: Sequence[float] = WEIGHTS,
) -> dict[str, object]:
    """Search the precommitted grid using discovery and validation together only."""

    required = {
        "material_id",
        "partition_role",
        ENDPOINT_COLUMN,
        "defective",
        "open_metal_site",
        SSSP_FEATURE,
        "sssp_supported",
    }
    if required - set(development):
        raise ValueError("NEXT534 development table differs")
    roles = development["partition_role"].astype(str).to_numpy()
    if set(roles.tolist()) != set(DEVELOPMENT_ROLES):
        raise ValueError("NEXT534 development roles differ")
    if development["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT534 development identity differs")
    endpoint = pd.to_numeric(development[ENDPOINT_COLUMN], errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(endpoint)):
        raise ValueError("NEXT534 endpoint differs")
    weights = tuple(float(value) for value in weights)
    if not weights or len(weights) != len(set(weights)) or any(value not in WEIGHTS for value in weights):
        raise ValueError("NEXT534 weight grid differs")

    strata = _stratum_labels(development)
    candidate_count = 0
    eligible: list[dict[str, object]] = []
    diagnostic_best_by_weight: dict[str, dict[str, object]] = {}
    for weight in weights:
        score, supported, _, deficit = apply_sssp_framework_guard(
            development,
            anchor_formula,
            weight=weight,
            threshold=float(anchor_formula["threshold"]),
        )
        thresholds = np.unique(score[supported])
        partition_aucs: dict[str, dict[str, object]] = {}
        for role in DEVELOPMENT_ROLES:
            mask = roles == role
            partition_aucs[role] = _auc_diagnostics(
                score=score[mask],
                supported=supported[mask],
                endpoint=endpoint[mask],
                strata=strata[mask],
            )
        for threshold in thresholds.tolist():
            reject = supported & (score >= threshold)
            partition_metrics: dict[str, dict[str, object]] = {}
            passes = True
            for role in DEVELOPMENT_ROLES:
                mask = roles == role
                metrics = _decision_metrics(
                    supported=supported[mask],
                    reject=reject[mask],
                    endpoint=endpoint[mask],
                )
                aucs = partition_aucs[role]
                role_passes = _passes_partition_gates(metrics, aucs)
                partition_metrics[role] = {
                    **metrics,
                    **aucs,
                    "passes_all_gates": role_passes,
                }
                passes &= role_passes
            combined = _decision_metrics(
                supported=supported,
                reject=reject,
                endpoint=endpoint,
            )
            combined_passes = (
                combined["reject_precision_lower"] is not None
                and float(combined["reject_precision_lower"])
                >= COMBINED_PRECISION_LOWER_MIN
            )
            formula = {
                **dict(anchor_formula),
                "threshold": float(threshold),
                "sssp_deficit_feature": SSSP_FEATURE,
                "sssp_deficit_cutoff": SSSP_CUTOFF,
                "sssp_deficit_weight": weight,
                "sssp_missing_policy": "ZERO_INCREMENT",
            }
            record = {
                "weight": weight,
                "threshold": float(threshold),
                "formula": formula,
                "partition_metrics": partition_metrics,
                "combined_metrics": {
                    **combined,
                    "passes_precision_safety_margin": combined_passes,
                },
                "passes_two_partition_readiness": bool(passes and combined_passes),
            }
            candidate_count += 1
            if record["passes_two_partition_readiness"]:
                eligible.append(record)
            key = f"{weight:g}"
            previous = diagnostic_best_by_weight.get(key)
            if previous is None or _diagnostic_sort_key(record) < _diagnostic_sort_key(previous):
                diagnostic_best_by_weight[key] = record

    eligible.sort(key=_candidate_sort_key)
    selected = eligible[0] if eligible else None
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "weights": list(weights),
        "candidate_count": candidate_count,
        "eligible_candidate_count": len(eligible),
        "passes_two_partition_readiness": selected is not None,
        "diagnostic_best_by_weight": {
            key: _serializable_candidate(value)
            for key, value in sorted(diagnostic_best_by_weight.items())
        },
        "selected_formula": selected["formula"] if selected else None,
        "selected_metrics": (
            {
                **selected["partition_metrics"],
                "combined": selected["combined_metrics"],
            }
            if selected
            else None
        ),
        "selected_score": None,
        "selected_supported": None,
        "selected_reject": None,
        "selected_deficit": None,
    }
    if selected:
        score, supported, reject, deficit = apply_sssp_framework_guard(
            development,
            anchor_formula,
            weight=float(selected["weight"]),
            threshold=float(selected["threshold"]),
        )
        result.update(
            selected_score=score,
            selected_supported=supported,
            selected_reject=reject,
            selected_deficit=deficit,
        )
    return result


def run_two_partition_guard_search(
    *,
    framework_feature_dir: Path,
    sssp_feature_dir: Path,
    next79_dir: Path,
    endpoint_firewall_path: Path,
    discovery_dir: Path,
    validation_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Run NEXT534 without constructing or reading any replication endpoint path."""

    framework_dir = Path(framework_feature_dir).resolve()
    sssp_dir = Path(sssp_feature_dir).resolve()
    anchor_dir = Path(next79_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    validation_dir = Path(validation_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "framework_features": framework_dir / FRAMEWORK_FEATURES_NAME,
        "framework_manifest": framework_dir / FRAMEWORK_MANIFEST_NAME,
        "sssp_features": sssp_dir / SSSP_TABLE_NAME,
        "sssp_catalogue": sssp_dir / SSSP_CATALOGUE_NAME,
        "sssp_manifest": sssp_dir / SSSP_MANIFEST_NAME,
        "next79_formula": anchor_dir / "NEXT79_ODAC23_ELECTROSTATIC_RESIDUAL_GUARD.json",
        "next79_manifest": anchor_dir / "MANIFEST.json",
        "endpoint_firewall": Path(endpoint_firewall_path).resolve(),
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "validation_manifest": validation_dir / ROLE_MANIFEST_NAME,
        "validation_labels": validation_dir / ROLE_LABELS_NAME,
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT534 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT534 formal input identity differs: {differing}")

    framework_manifest = _read_json(paths["framework_manifest"])
    sssp_manifest = _read_json(paths["sssp_manifest"])
    sssp_catalogue = _read_json(paths["sssp_catalogue"])
    anchor_manifest = _read_json(paths["next79_manifest"])
    anchor_formula = _read_json(paths["next79_formula"])
    firewall = _read_json(paths["endpoint_firewall"])
    discovery_manifest = _read_json(paths["discovery_manifest"])
    validation_manifest = _read_json(paths["validation_manifest"])
    framework_outputs = framework_manifest.get("outputs_sha256")
    sssp_outputs = sssp_manifest.get("outputs_sha256")
    anchor_outputs = anchor_manifest.get("outputs_sha256")
    if (
        framework_manifest.get("protocol") != FRAMEWORK_PROTOCOL
        or framework_manifest.get("labels_opened") is not False
        or framework_manifest.get("internal_replication_labels_opened") is not False
        or not isinstance(framework_outputs, Mapping)
        or framework_outputs.get(FRAMEWORK_FEATURES_NAME) != hashes["framework_features"]
        or sssp_manifest.get("protocol") != SSSP_PROTOCOL
        or sssp_manifest.get("internal_replication_endpoint_values_opened") is not False
        or sssp_manifest.get("next534_two_partition_development_authorized") is not True
        or not isinstance(sssp_manifest.get("counts"), Mapping)
        or sssp_manifest["counts"].get("passes") is not True
        or not isinstance(sssp_outputs, Mapping)
        or sssp_outputs.get(SSSP_TABLE_NAME) != hashes["sssp_features"]
        or sssp_outputs.get(SSSP_CATALOGUE_NAME) != hashes["sssp_catalogue"]
        or sssp_catalogue.get("internal_replication_labels_opened") is not False
        or anchor_manifest.get("protocol") != NEXT79_PROTOCOL
        or anchor_manifest.get("internal_replication_labels_opened") is not False
        or not isinstance(anchor_outputs, Mapping)
        or anchor_outputs.get(paths["next79_formula"].name) != hashes["next79_formula"]
        or anchor_formula.get("protocol") != NEXT79_PROTOCOL
        or anchor_formula.get("kind") != "additive"
        or anchor_formula.get("missing_policy") != "KEEP"
        or len(anchor_formula.get("terms", [])) != 6
        or anchor_formula.get("gates") != GATES
        or firewall.get("protocol") != ENDPOINT_PROTOCOL
        or firewall.get("internal_replication_endpoint_values_summarized_or_inspected")
        is not False
    ):
        raise ValueError("NEXT534 zero-DFT feature/formula provenance differs")

    role_specs = (
        ("discovery", discovery_manifest, hashes["discovery_labels"]),
        ("internal_validation", validation_manifest, hashes["validation_labels"]),
    )
    for role, manifest, label_hash in role_specs:
        outputs = manifest.get("outputs_sha256")
        if (
            manifest.get("protocol") != ENDPOINT_PROTOCOL
            or manifest.get("partition_role") != role
            or not isinstance(outputs, Mapping)
            or outputs.get(ROLE_LABELS_NAME) != label_hash
        ):
            raise ValueError(f"NEXT534 {role} endpoint provenance differs")

    framework = pd.read_parquet(paths["framework_features"])
    sssp = pd.read_parquet(paths["sssp_features"])
    if (
        len(framework) != 7_815
        or len(sssp) != 7_815
        or framework["material_id"].astype(str).duplicated().any()
        or sssp["material_id"].astype(str).duplicated().any()
        or ENDPOINT_COLUMN in framework
        or ENDPOINT_COLUMN in sssp
    ):
        raise ValueError("NEXT534 label-free feature identity differs")
    sssp_view = sssp[
        ["material_id", "partition_role", SSSP_FEATURE, "sssp_supported", "sssp_failure"]
    ].rename(columns={"partition_role": "sssp_partition_role"})
    all_features = framework.merge(sssp_view, on="material_id", validate="one_to_one")
    if (
        len(all_features) != 7_815
        or not np.array_equal(
            all_features["partition_role"].astype(str).to_numpy(),
            all_features["sssp_partition_role"].astype(str).to_numpy(),
        )
    ):
        raise ValueError("NEXT534 label-free feature role differs")
    all_features = all_features.drop(columns="sssp_partition_role")

    development_parts = []
    for role, label_path in (
        ("discovery", paths["discovery_labels"]),
        ("internal_validation", paths["validation_labels"]),
    ):
        labels = pd.read_parquet(label_path)
        if (
            set(labels["partition_role"].astype(str)) != {role}
            or labels["material_id"].astype(str).duplicated().any()
            or ENDPOINT_COLUMN not in labels
        ):
            raise ValueError(f"NEXT534 {role} endpoint table differs")
        feature_part = all_features[all_features["partition_role"].astype(str).eq(role)]
        joined = feature_part.merge(
            labels[["material_id", ENDPOINT_COLUMN]],
            on="material_id",
            validate="one_to_one",
        )
        if len(joined) != len(labels):
            raise ValueError(f"NEXT534 {role} endpoint identity differs")
        development_parts.append(joined)
    development = pd.concat(development_parts, ignore_index=True)

    result = search_two_partition_guard(development, anchor_formula, weights=WEIGHTS)
    selected = bool(result["passes_two_partition_readiness"])
    search_record = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "selected_score",
            "selected_supported",
            "selected_reject",
            "selected_deficit",
        }
    }
    search_record.update(
        {
            "development_roles": list(DEVELOPMENT_ROLES),
            "internal_replication_endpoint_opened": False,
            "next535_replication_prediction_authorized": selected,
            "failure_expands_same_endpoint_grid": False,
        }
    )

    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "bounded_two_partition_sssp_framework_guard_search",
        "counts": {
            "development_rows": len(development),
            "discovery_rows": int(development["partition_role"].eq("discovery").sum()),
            "internal_validation_rows": int(
                development["partition_role"].eq("internal_validation").sum()
            ),
            "candidate_count": result["candidate_count"],
            "eligible_candidate_count": result["eligible_candidate_count"],
        },
        "passes_two_partition_readiness": selected,
        "next535_replication_prediction_authorized": selected,
        "replication_endpoint_path_constructed_or_read": False,
        "internal_replication_endpoint_values_opened": False,
        "development_offline_labels_opened": True,
        "executable_input_boundary": [
            "composition",
            "one raw initial fully periodic geometry",
        ],
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next534_odac23_sssp_framework_guard.py": source_hash
        },
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        search_path = staging / SEARCH_NAME
        search_path.write_bytes(_json_bytes(search_record))
        output_paths = [search_path]
        if selected:
            formula = {
                **result["selected_formula"],
                "protocol": PROTOCOL,
                "formula_family": "exact NEXT79 framework score plus bounded SSSP deficit",
                "training_partitions": list(DEVELOPMENT_ROLES),
                "combined_precision_lower_min": COMBINED_PRECISION_LOWER_MIN,
                "scientific_status": "advance_to_label_sealed_replication_prediction_freeze",
            }
            formula_path = staging / FORMULA_NAME
            formula_path.write_bytes(_json_bytes(formula))
            endpoint = development[ENDPOINT_COLUMN].to_numpy(float)
            predictions = pd.DataFrame(
                {
                    "material_id": development["material_id"].astype(str),
                    "partition_role": development["partition_role"].astype(str),
                    ENDPOINT_COLUMN: endpoint,
                    "protected": endpoint <= 0.05,
                    "severe": endpoint >= 0.20,
                    "sssp_deficit": result["selected_deficit"],
                    "risk_score": result["selected_score"],
                    "supported": result["selected_supported"],
                    "reject": result["selected_reject"],
                }
            )
            predictions_path = staging / PREDICTIONS_NAME
            predictions.to_parquet(predictions_path, index=False)
            output_paths.extend([formula_path, predictions_path])
        manifest["outputs_sha256"] = {
            path.name: _sha256(path) for path in output_paths
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT534 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT534 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-feature-dir", type=Path, required=True)
    parser.add_argument("--sssp-feature-dir", type=Path, required=True)
    parser.add_argument("--next79-dir", type=Path, required=True)
    parser.add_argument("--endpoint-firewall", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_two_partition_guard_search(
        framework_feature_dir=args.framework_feature_dir,
        sssp_feature_dir=args.sssp_feature_dir,
        next79_dir=args.next79_dir,
        endpoint_firewall_path=args.endpoint_firewall,
        discovery_dir=args.discovery_dir,
        validation_dir=args.validation_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "passes_two_partition_readiness": manifest[
                    "passes_two_partition_readiness"
                ],
                "next535_replication_prediction_authorized": manifest[
                    "next535_replication_prediction_authorized"
                ],
                **manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "COMBINED_PRECISION_LOWER_MIN",
    "DEVELOPMENT_ROLES",
    "PROTOCOL",
    "SSSP_CUTOFF",
    "SSSP_FEATURE",
    "WEIGHTS",
    "apply_sssp_framework_guard",
    "run_two_partition_guard_search",
    "search_two_partition_guard",
    "sssp_deficit",
]


if __name__ == "__main__":
    main()
