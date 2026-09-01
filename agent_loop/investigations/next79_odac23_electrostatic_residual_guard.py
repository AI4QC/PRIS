#!/usr/bin/env python3
"""Final single electrostatic-residual guard search and frozen stop decision."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next57_odac23_discovery_search import GATES, PROTECTED_MAX, SEVERE_MIN
from src.next60_odac23_robust_scaffold_endpoint import (
    ENDPOINT_COLUMN,
    PROTOCOL as ENDPOINT_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
)
from src.next72_odac23_anchored_tail_correction_search import (
    PAIR_WEIGHTS,
    PROTOCOL as ANCHOR_PROTOCOL,
    SINGLE_WEIGHTS,
    search_anchored_tail_correction,
)
from src.next76_odac23_rigidity_tail_search import (
    EXPECTED_DISCOVERY_LABEL_SHA256,
    EXPECTED_DISCOVERY_MANIFEST_SHA256,
    EXPECTED_ENDPOINT_FIREWALL_SHA256,
)
from src.next77_odac23_analytic_electrostatic_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    MANIFEST_NAME as FEATURE_MANIFEST_NAME,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)
from src.next78_odac23_electrostatic_tail_search import (
    EXPECTED_FEATURE_MANIFEST_SHA256,
    EXPECTED_FEATURE_SHA256,
)


PROTOCOL = "2026-08-03-next79-odac23-single-electrostatic-residual-guard-v1"
DESIGN_SHA256 = "cbe70fe955f0aa17a330cc89e53c7f28e3d6b1f756e6f2dbccef6471ace3701d"
EXPECTED_ANCHOR_SHA256 = (
    "3b28dddeb978cd9607ab9f35a7bcaed145f5a489e843c94443b4a81a2a5cd23d"
)
REPLICATION_PRECISION_LOWER_MIN = 0.80
CANDIDATE_FEATURE_NAMES = ("aefi_residual_q95",)
FORMULA_NAME = "NEXT79_ODAC23_ELECTROSTATIC_RESIDUAL_GUARD.json"
SEARCH_NAME = "NEXT79_ODAC23_ELECTROSTATIC_RESIDUAL_SEARCH.json"
PREDICTIONS_NAME = "next79_odac23_electrostatic_residual_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def replication_ready(metrics: Mapping[str, object]) -> bool:
    checks = {
        "coverage_lower_at_least": float(metrics["coverage_lower"]),
        "protected_recall_lower_at_least": float(metrics["protected_recall_lower"]),
        "reject_precision_lower_at_least": float(metrics["reject_precision_lower"]),
        "savings_lower_at_least": float(metrics["savings_lower"]),
        "pooled_extreme_auc_at_least": float(metrics["pooled_extreme_auc"]),
        "macro_stratum_auc_at_least": float(metrics["macro_stratum_auc"]),
        "worst_stratum_auc_at_least": float(metrics["worst_stratum_auc"]),
    }
    return all(checks[name] >= float(threshold) for name, threshold in GATES.items()) and (
        checks["reject_precision_lower_at_least"]
        >= REPLICATION_PRECISION_LOWER_MIN
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT79 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_electrostatic_residual_guard_search(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    anchor_formula_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run the final one-feature discovery test without non-discovery labels."""

    feature_dir = Path(feature_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / FEATURE_MANIFEST_NAME,
        "endpoint_firewall": Path(endpoint_firewall_manifest_path).resolve(),
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
        "anchor_formula": Path(anchor_formula_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT79 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "features": EXPECTED_FEATURE_SHA256,
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
        "anchor_formula": EXPECTED_ANCHOR_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT79 frozen input hash differs")

    feature_manifest = _read_json(paths["feature_manifest"])
    endpoint_firewall = _read_json(paths["endpoint_firewall"])
    discovery_manifest = _read_json(paths["discovery_manifest"])
    anchor = _read_json(paths["anchor_formula"])
    feature_outputs = feature_manifest.get("outputs_sha256")
    discovery_outputs = discovery_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != SOURCE_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or feature_manifest.get("opened_internal_validation_result_used") is not False
        or feature_manifest.get("internal_replication_labels_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
        or endpoint_firewall.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_firewall.get(
            "internal_replication_endpoint_values_summarized_or_inspected"
        )
        is not False
        or discovery_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or discovery_manifest.get("partition_role") != "discovery"
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get(ROLE_LABELS_NAME) != hashes["discovery_labels"]
        or anchor.get("protocol") != ANCHOR_PROTOCOL
    ):
        raise ValueError("NEXT79 discovery-only provenance differs")

    all_features = pd.read_parquet(paths["features"])
    features = all_features[all_features["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT79 received non-discovery labels")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise ValueError("NEXT79 discovery identity differs")

    result = search_anchored_tail_correction(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        anchor_formula=anchor,
        candidate_features=CANDIDATE_FEATURE_NAMES,
        single_weights=SINGLE_WEIGHTS,
        pair_weights=PAIR_WEIGHTS,
        pair_shortlist=1,
    )
    metrics = result["discovery_metrics"]
    ready = replication_ready(metrics)
    if ready:
        status = "advance_to_one_shot_internal_replication"
    elif result["passes_discovery_gates"]:
        status = "original_discovery_pass_below_adaptive_safety_margin"
    else:
        status = "stop_additive_odac23_discovery_search"
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / robust discovery only",
        "formula_family": "sealed NEXT72 formula plus at most one Ewald-residual guard",
        "anchor_formula_sha256": hashes["anchor_formula"],
        "endpoint_definition": {
            "column": ENDPOINT_COLUMN,
            "protected_max_angstrom": PROTECTED_MAX,
            "severe_min_angstrom": SEVERE_MIN,
            "minimum_adsorbate_configurations": 4,
            "common_translation_removed": True,
        },
        "gates": GATES,
        "replication_precision_lower_min": REPLICATION_PRECISION_LOWER_MIN,
        "candidate_features": list(CANDIDATE_FEATURE_NAMES),
        "feature_artifact_sha256": hashes["features"],
        "scientific_status": status,
    }
    search_record = {
        key: value
        for key, value in result.items()
        if key not in {"score", "supported", "reject", "selected_formula"}
    }
    search_record["passes_replication_readiness_margin"] = ready
    search_record["replication_precision_lower_min"] = REPLICATION_PRECISION_LOWER_MIN
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
        "mode": "single_electrostatic_residual_guard_discovery_search",
        "adaptive_final_search": True,
        "failure_is_precommitted_additive_search_stop": True,
        "robust_discovery_labels_opened": True,
        "previous_internal_validation_artifact_used": False,
        "internal_validation_labels_opened_here": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "classical_analytic_electrostatics_used": True,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "passes_original_discovery_gates": result["passes_discovery_gates"],
        "passes_replication_readiness_margin": ready,
        "counts": {
            "rows": len(joined),
            "protected": int(predictions["protected"].sum()),
            "severe": int(predictions["severe"].sum()),
            "supported": int(predictions["supported"].sum()),
            "rejected": int(predictions["reject"].sum()),
            "candidate_features": len(CANDIDATE_FEATURE_NAMES),
            "usable_guard_features": int(result["usable_guard_feature_count"]),
            "candidate_thresholds": int(result["candidate_count"]),
            "unique_term_lists": int(result["unique_term_list_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next79_odac23_electrostatic_residual_guard.py": source_hash
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
            path.name: _sha256(path)
            for path in (formula_path, search_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT79 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT79 input changed before publication")
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
    parser.add_argument("--anchor-formula", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_electrostatic_residual_guard_search(
        feature_dir=args.feature_dir,
        endpoint_firewall_manifest_path=args.endpoint_firewall,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        anchor_formula_path=args.anchor_formula,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "passes_original": manifest["passes_original_discovery_gates"],
                "passes_replication_margin": manifest[
                    "passes_replication_readiness_margin"
                ],
                **manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = ["PROTOCOL", "replication_ready", "run_electrostatic_residual_guard_search"]


if __name__ == "__main__":
    main()
