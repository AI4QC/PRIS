#!/usr/bin/env python3
"""Rigidity-augmented finite anchored tail search with a replication margin."""

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
from src.next65_odac23_physics_couplings import NEXT65_FEATURE_NAMES
from src.next67_odac23_monotone_expanded_search import PROTOCOL as ANCHOR_PROTOCOL
from src.next70_odac23_metal_donor_bond_valence_features import METAL_DONOR_BV_FEATURE_NAMES
from src.next72_odac23_anchored_tail_correction_search import (
    PAIR_SHORTLIST,
    PAIR_WEIGHTS,
    SINGLE_WEIGHTS,
    search_anchored_tail_correction,
)
from src.next75_odac23_metal_ligand_rigidity_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    METAL_LIGAND_RIGIDITY_FEATURE_NAMES,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next76-odac23-rigidity-augmented-tail-search-v1"
DESIGN_SHA256 = "91ddf46eeefaf62359e96d7b68cc552586407cc6979801255f65a64dc9b15a33"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "7ee6197a08da3fafc1aa46d6ed5f5cb82aeb55480466974892af111dc685a758"
)
EXPECTED_FEATURE_SHA256 = (
    "7d7079571d2f6be5c2e21835108ba6fb43e95cb0fae1edaacfbabe7dc71b813d"
)
EXPECTED_ANCHOR_SHA256 = (
    "9ea83c8c1f70acf619c8e9f163f4468de316972c33b71605fa75260ce013ed58"
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
REPLICATION_PRECISION_LOWER_MIN = 0.75
CANDIDATE_FEATURE_NAMES = (
    tuple(NEXT65_FEATURE_NAMES)
    + tuple(METAL_DONOR_BV_FEATURE_NAMES)
    + tuple(METAL_LIGAND_RIGIDITY_FEATURE_NAMES)
)
FORMULA_NAME = "NEXT76_ODAC23_RIGIDITY_TAIL_CANDIDATE.json"
SEARCH_NAME = "NEXT76_ODAC23_RIGIDITY_TAIL_SEARCH.json"
PREDICTIONS_NAME = "next76_odac23_rigidity_tail_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def replication_ready(metrics: Mapping[str, object]) -> bool:
    """Apply original gates plus the frozen precision safety margin."""

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
        checks["reject_precision_lower_at_least"] >= REPLICATION_PRECISION_LOWER_MIN
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT76 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_rigidity_tail_search(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    anchor_formula_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Search discovery only; never accept an opened-validation artifact path."""

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
        "anchor_formula": Path(anchor_formula_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT76 input is missing")
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
        raise ValueError("NEXT76 frozen input hash differs")
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
        or endpoint_firewall.get("internal_replication_endpoint_values_summarized_or_inspected") is not False
        or discovery_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or discovery_manifest.get("partition_role") != "discovery"
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get(ROLE_LABELS_NAME) != hashes["discovery_labels"]
        or anchor.get("protocol") != ANCHOR_PROTOCOL
    ):
        raise ValueError("NEXT76 discovery-only provenance differs")
    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT76 received non-discovery labels")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise ValueError("NEXT76 discovery identity differs")
    result = search_anchored_tail_correction(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        anchor_formula=anchor,
        candidate_features=CANDIDATE_FEATURE_NAMES,
        single_weights=SINGLE_WEIGHTS,
        pair_weights=PAIR_WEIGHTS,
        pair_shortlist=PAIR_SHORTLIST,
    )
    metrics = result["discovery_metrics"]
    ready = replication_ready(metrics)
    if ready:
        status = "advance_to_one_shot_internal_replication"
    elif result["passes_discovery_gates"]:
        status = "original_discovery_pass_below_replication_safety_margin"
    else:
        status = "discovery_failure_diagnostic_only"
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / robust discovery only",
        "formula_family": "sealed NEXT67 axis plus finite rigidity-augmented corrections",
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
        "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
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
        "mode": "finite_rigidity_augmented_tail_discovery_search",
        "robust_discovery_labels_opened": True,
        "previous_internal_validation_artifact_used": False,
        "internal_validation_labels_opened_here": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
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
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next76_odac23_rigidity_tail_search.py": source_hash
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
            raise RuntimeError("NEXT76 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT76 input changed before publication")
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
    manifest = run_rigidity_tail_search(
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
                "passes_replication_margin": manifest["passes_replication_readiness_margin"],
                **manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = ["PROTOCOL", "replication_ready", "run_rigidity_tail_search"]


if __name__ == "__main__":
    main()
