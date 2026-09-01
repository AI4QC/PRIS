#!/usr/bin/env python3
"""Bounded two-partition NEXT79 plus PBAAA framework-guard search."""

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

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next57_odac23_discovery_search import (
    GATES,
    _auc_diagnostics,
    _decision_metrics,
    _stratum_labels,
)
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
from src.next534_odac23_sssp_framework_guard import (
    _candidate_sort_key,
    _diagnostic_sort_key,
    _passes_partition_gates,
    _serializable_candidate,
)
from src.next539_odac23_pbaaa_label_free_features import (
    CATALOGUE_NAME as PBAAA_CATALOGUE_NAME,
    FEATURE_NAME as PBAAA_FEATURE,
    MANIFEST_NAME as PBAAA_MANIFEST_NAME,
    PROTOCOL as PBAAA_PROTOCOL,
    TABLE_NAME as PBAAA_TABLE_NAME,
)


PROTOCOL = "2026-08-13-next540-odac23-pbaaa-framework-guard-v1"
DEVELOPMENT_ROLES = ("discovery", "internal_validation")
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
COMBINED_PRECISION_LOWER_MIN = 0.80
NEXT79_PROTOCOL = "2026-08-03-next79-odac23-single-electrostatic-residual-guard-v1"
FORMULA_NAME = "NEXT540_ODAC23_PBAAA_FRAMEWORK_GUARD.json"
SEARCH_NAME = "NEXT540_ODAC23_PBAAA_FRAMEWORK_SEARCH.json"
PREDICTIONS_NAME = "next540_odac23_pbaaa_framework_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": "68bee1dea45492f1bf7349965dcd149497e91e9ed6a1aa29098ce3bc0b01ceac",
    "framework_features": "c2c6668d24c77c20d8ab2878fa5e6b7d266f26ebea85b58ae083985eec6f2ad7",
    "framework_manifest": "ac1ecd88cfdf57fcec1c5dbe331ea8373f07aac79b706ed0d2c4c2575ab9ad82",
    "pbaaa_features": "3e3a48377b5bd5862f69192339d6401f43cdd8b7aa80741944dbf9e61443ecf0",
    "pbaaa_catalogue": "49ab2015ddb5f75b6ff691f6fd99f2a795fb4de13590b324e553b1f68d633955",
    "pbaaa_manifest": "916432edc021ac43e6fdba5de0f5d601cbd4f405d60c3cb0efd057e04c954e80",
    "next79_formula": "ad0e079405e16c482c5f77c9b846b415c833a5a153ddd1c15e1d5cf19db9f291",
    "next79_manifest": "cc226a551a53e1e16a725090c1f896e20be09e4120c275bf6816a440b1e46170",
    "endpoint_firewall": "9dbd3f78d2505ba96b33715e6409cd8524e9b909f4134af0020b933dff2f769f",
    "discovery_manifest": "6ca39eb42629d626559618474f75aa6bb6571a38a928b3b16512b5d987b76137",
    "discovery_labels": "1a7c78fd87bb3f5795e59fa3c3799fbbb07a1629b90d472aef7e73740ce7f08a",
    "validation_manifest": "cb3aa1a4914d25e8b50b626386dda34fe59c5b692e130a10faa06d1083d058b9",
    "validation_labels": "ad964615c26fff9a7fcdaea7dc9d24042fa55f61eedd690758236d08cd6e2fc8",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"NEXT540 {path.name} must contain an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def pbaaa_increment(values: np.ndarray, supported: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    support = np.asarray(supported, dtype=bool)
    if array.ndim != 1 or support.shape != array.shape:
        raise ValueError("NEXT540 PBAAA arrays differ")
    if np.any(support & ~np.isfinite(array)):
        raise ValueError("NEXT540 PBAAA support semantics differ")
    if np.any(support & ((array < 0.0) | (array > 1.0))):
        raise ValueError("NEXT540 PBAAA range differs")
    result = np.zeros(array.shape, dtype=float)
    result[support] = array[support]
    return result


def apply_pbaaa_framework_guard(
    features: pd.DataFrame,
    anchor_formula: Mapping[str, object],
    *,
    weight: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if PBAAA_FEATURE not in features or "pbaaa_supported" not in features:
        raise ValueError("NEXT540 PBAAA feature is missing")
    weight = float(weight)
    threshold = float(threshold)
    if weight not in WEIGHTS or not math.isfinite(threshold):
        raise ValueError("NEXT540 grid value differs")
    base_score, base_supported, _ = apply_sparse_formula(features, anchor_formula)
    values = pd.to_numeric(features[PBAAA_FEATURE], errors="coerce").to_numpy(float)
    feature_support = features["pbaaa_supported"].fillna(False).to_numpy(bool)
    increment = pbaaa_increment(values, feature_support)
    score = base_score + weight * increment
    score[~base_supported] = np.nan
    reject = base_supported & (score >= threshold)
    return score, base_supported, reject, increment


def search_two_partition_pbaaa_guard(
    development: pd.DataFrame,
    anchor_formula: Mapping[str, object],
    *,
    weights: Sequence[float] = WEIGHTS,
) -> dict[str, object]:
    required = {
        "material_id",
        "partition_role",
        ENDPOINT_COLUMN,
        "defective",
        "open_metal_site",
        PBAAA_FEATURE,
        "pbaaa_supported",
    }
    if required - set(development):
        raise ValueError("NEXT540 development table differs")
    roles = development["partition_role"].astype(str).to_numpy()
    if set(roles.tolist()) != set(DEVELOPMENT_ROLES):
        raise ValueError("NEXT540 development roles differ")
    if development["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT540 development identity differs")
    endpoint = pd.to_numeric(development[ENDPOINT_COLUMN], errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(endpoint)):
        raise ValueError("NEXT540 endpoint differs")
    weights = tuple(float(value) for value in weights)
    if not weights or len(weights) != len(set(weights)) or any(value not in WEIGHTS for value in weights):
        raise ValueError("NEXT540 weight grid differs")
    strata = _stratum_labels(development)
    eligible = []
    diagnostic_best_by_weight = {}
    candidate_count = 0
    for weight in weights:
        score, supported, _, increment = apply_pbaaa_framework_guard(
            development,
            anchor_formula,
            weight=weight,
            threshold=float(anchor_formula["threshold"]),
        )
        thresholds = np.unique(score[supported])
        partition_aucs = {}
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
            partition_metrics = {}
            passes = True
            for role in DEVELOPMENT_ROLES:
                mask = roles == role
                metrics = _decision_metrics(
                    supported=supported[mask], reject=reject[mask], endpoint=endpoint[mask]
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
                supported=supported, reject=reject, endpoint=endpoint
            )
            combined_passes = (
                combined["reject_precision_lower"] is not None
                and float(combined["reject_precision_lower"])
                >= COMBINED_PRECISION_LOWER_MIN
            )
            formula = {
                **dict(anchor_formula),
                "threshold": float(threshold),
                "pbaaa_feature": PBAAA_FEATURE,
                "pbaaa_weight": weight,
                "pbaaa_missing_policy": "ZERO_INCREMENT",
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
    result = {
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
            {**selected["partition_metrics"], "combined": selected["combined_metrics"]}
            if selected
            else None
        ),
        "selected_score": None,
        "selected_supported": None,
        "selected_reject": None,
        "selected_increment": None,
    }
    if selected:
        score, supported, reject, increment = apply_pbaaa_framework_guard(
            development,
            anchor_formula,
            weight=float(selected["weight"]),
            threshold=float(selected["threshold"]),
        )
        result.update(
            selected_score=score,
            selected_supported=supported,
            selected_reject=reject,
            selected_increment=increment,
        )
    return result


def run_two_partition_pbaaa_search(
    *,
    framework_feature_dir: Path,
    pbaaa_feature_dir: Path,
    next79_dir: Path,
    endpoint_firewall_path: Path,
    discovery_dir: Path,
    validation_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    framework_dir = Path(framework_feature_dir).resolve()
    pbaaa_dir = Path(pbaaa_feature_dir).resolve()
    anchor_dir = Path(next79_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    validation_dir = Path(validation_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "framework_features": framework_dir / FRAMEWORK_FEATURES_NAME,
        "framework_manifest": framework_dir / FRAMEWORK_MANIFEST_NAME,
        "pbaaa_features": pbaaa_dir / PBAAA_TABLE_NAME,
        "pbaaa_catalogue": pbaaa_dir / PBAAA_CATALOGUE_NAME,
        "pbaaa_manifest": pbaaa_dir / PBAAA_MANIFEST_NAME,
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
        raise FileNotFoundError("NEXT540 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT540 formal input identity differs: {differing}")
    framework_manifest = _read_json(paths["framework_manifest"])
    pbaaa_manifest = _read_json(paths["pbaaa_manifest"])
    pbaaa_catalogue = _read_json(paths["pbaaa_catalogue"])
    anchor_manifest = _read_json(paths["next79_manifest"])
    anchor_formula = _read_json(paths["next79_formula"])
    firewall = _read_json(paths["endpoint_firewall"])
    discovery_manifest = _read_json(paths["discovery_manifest"])
    validation_manifest = _read_json(paths["validation_manifest"])
    framework_outputs = framework_manifest.get("outputs_sha256")
    pbaaa_outputs = pbaaa_manifest.get("outputs_sha256")
    anchor_outputs = anchor_manifest.get("outputs_sha256")
    if (
        framework_manifest.get("protocol") != FRAMEWORK_PROTOCOL
        or framework_manifest.get("labels_opened") is not False
        or framework_manifest.get("internal_replication_labels_opened") is not False
        or not isinstance(framework_outputs, Mapping)
        or framework_outputs.get(FRAMEWORK_FEATURES_NAME) != hashes["framework_features"]
        or pbaaa_manifest.get("protocol") != PBAAA_PROTOCOL
        or pbaaa_manifest.get("internal_replication_endpoint_values_opened") is not False
        or pbaaa_manifest.get("next540_two_partition_development_authorized") is not True
        or not isinstance(pbaaa_manifest.get("counts"), Mapping)
        or pbaaa_manifest["counts"].get("passes") is not True
        or not isinstance(pbaaa_outputs, Mapping)
        or pbaaa_outputs.get(PBAAA_TABLE_NAME) != hashes["pbaaa_features"]
        or pbaaa_outputs.get(PBAAA_CATALOGUE_NAME) != hashes["pbaaa_catalogue"]
        or pbaaa_catalogue.get("internal_replication_labels_opened") is not False
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
        raise ValueError("NEXT540 feature/formula provenance differs")
    for role, manifest, label_hash in (
        ("discovery", discovery_manifest, hashes["discovery_labels"]),
        ("internal_validation", validation_manifest, hashes["validation_labels"]),
    ):
        outputs = manifest.get("outputs_sha256")
        if (
            manifest.get("protocol") != ENDPOINT_PROTOCOL
            or manifest.get("partition_role") != role
            or not isinstance(outputs, Mapping)
            or outputs.get(ROLE_LABELS_NAME) != label_hash
        ):
            raise ValueError(f"NEXT540 {role} endpoint provenance differs")

    framework = pd.read_parquet(paths["framework_features"])
    pbaaa = pd.read_parquet(paths["pbaaa_features"])
    if (
        len(framework) != 7_815
        or len(pbaaa) != 7_815
        or framework["material_id"].astype(str).duplicated().any()
        or pbaaa["material_id"].astype(str).duplicated().any()
        or ENDPOINT_COLUMN in framework
        or ENDPOINT_COLUMN in pbaaa
    ):
        raise ValueError("NEXT540 label-free feature identity differs")
    pbaaa_view = pbaaa[
        ["material_id", "partition_role", PBAAA_FEATURE, "pbaaa_supported", "pbaaa_failure"]
    ].rename(columns={"partition_role": "pbaaa_partition_role"})
    all_features = framework.merge(pbaaa_view, on="material_id", validate="one_to_one")
    if (
        len(all_features) != 7_815
        or not np.array_equal(
            all_features["partition_role"].astype(str).to_numpy(),
            all_features["pbaaa_partition_role"].astype(str).to_numpy(),
        )
    ):
        raise ValueError("NEXT540 label-free feature role differs")
    all_features = all_features.drop(columns="pbaaa_partition_role")
    parts = []
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
            raise ValueError(f"NEXT540 {role} endpoint table differs")
        feature_part = all_features[all_features["partition_role"].astype(str).eq(role)]
        joined = feature_part.merge(
            labels[["material_id", ENDPOINT_COLUMN]], on="material_id", validate="one_to_one"
        )
        if len(joined) != len(labels):
            raise ValueError(f"NEXT540 {role} endpoint identity differs")
        parts.append(joined)
    development = pd.concat(parts, ignore_index=True)
    result = search_two_partition_pbaaa_guard(development, anchor_formula, weights=WEIGHTS)
    selected = bool(result["passes_two_partition_readiness"])
    search_record = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "selected_score",
            "selected_supported",
            "selected_reject",
            "selected_increment",
        }
    }
    search_record.update(
        development_roles=list(DEVELOPMENT_ROLES),
        internal_replication_endpoint_opened=False,
        next541_replication_prediction_authorized=selected,
        failure_expands_same_endpoint_grid=False,
    )
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest = {
        "protocol": PROTOCOL,
        "mode": "bounded_two_partition_pbaaa_framework_guard_search",
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
        "next541_replication_prediction_authorized": selected,
        "replication_endpoint_path_constructed_or_read": False,
        "internal_replication_endpoint_values_opened": False,
        "development_offline_labels_opened": True,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_or_virtual_coordinate_relaxation_executed": False,
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next540_odac23_pbaaa_framework_guard.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        search_path = staging / SEARCH_NAME
        search_path.write_bytes(_json_bytes(search_record))
        outputs = [search_path]
        if selected:
            formula = {
                **result["selected_formula"],
                "protocol": PROTOCOL,
                "formula_family": "exact NEXT79 framework score plus PBAAA",
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
                    "pbaaa_increment": result["selected_increment"],
                    "risk_score": result["selected_score"],
                    "supported": result["selected_supported"],
                    "reject": result["selected_reject"],
                }
            )
            predictions_path = staging / PREDICTIONS_NAME
            predictions.to_parquet(predictions_path, index=False)
            outputs.extend([formula_path, predictions_path])
        manifest["outputs_sha256"] = {path.name: _sha256(path) for path in outputs}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT540 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT540 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework-feature-dir", type=Path, required=True)
    parser.add_argument("--pbaaa-feature-dir", type=Path, required=True)
    parser.add_argument("--next79-dir", type=Path, required=True)
    parser.add_argument("--endpoint-firewall", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_two_partition_pbaaa_search(
        framework_feature_dir=args.framework_feature_dir,
        pbaaa_feature_dir=args.pbaaa_feature_dir,
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
                "next541_replication_prediction_authorized": manifest[
                    "next541_replication_prediction_authorized"
                ],
                **manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
