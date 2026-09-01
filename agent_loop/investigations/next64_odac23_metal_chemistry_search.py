#!/usr/bin/env python3
"""Robust discovery search over analytic, motif, and metal-chemistry x0 terms."""

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
from src.next57_odac23_discovery_search import (
    GATES,
    PROTECTED_MAX,
    SEVERE_MIN,
    search_discovery_rule,
)
from src.next60_odac23_robust_scaffold_endpoint import (
    ENDPOINT_COLUMN,
    PROTOCOL as ENDPOINT_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
)
from src.next63_odac23_metal_chemistry_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    NEXT63_FEATURE_NAMES,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next64-odac23-metal-chemistry-robust-search-v1"
DESIGN_SHA256 = "32e075a8b7f788eaa716545ce31d270248ccb84532a82ee160fdfc10df188579"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "65d5d7bfc7d2ffa28c1dae27beb46b529af2e02c0301ae7d19c72e911b7e37e0"
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
EXPECTED_SEARCH_ENGINE_SHA256 = (
    "486cabdcbd417316179cddf89625a34b14c03b209265b8decc2046eb3c797967"
)
FORMULA_NAME = "NEXT64_ODAC23_METAL_CHEMISTRY_CANDIDATE.json"
SEARCH_NAME = "NEXT64_ODAC23_METAL_CHEMISTRY_SEARCH.json"
PREDICTIONS_NAME = "next64_odac23_metal_chemistry_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


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


def run_metal_chemistry_search(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    feature_dir = Path(feature_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    target = Path(output_dir).resolve()
    search_engine_path = Path(
        __import__("src.next57_odac23_discovery_search", fromlist=["x"]).__file__
    ).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / MANIFEST_NAME,
        "endpoint_firewall": Path(endpoint_firewall_manifest_path).resolve(),
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
        "search_engine": search_engine_path,
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT64 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
        "search_engine": EXPECTED_SEARCH_ENGINE_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT64 frozen input hash differs")
    feature_manifest = _strict_json(paths["feature_manifest"], "NEXT63 manifest")
    endpoint_firewall = _strict_json(paths["endpoint_firewall"], "NEXT60 firewall")
    discovery_manifest = _strict_json(paths["discovery_manifest"], "NEXT60 discovery manifest")
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
        raise ValueError("NEXT64 robust discovery provenance differs")

    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT64 received non-discovery labels")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise ValueError("NEXT64 robust discovery identity differs")
    result = search_discovery_rule(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        candidate_features=NEXT63_FEATURE_NAMES,
    )
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / robust discovery only",
        "endpoint_definition": {
            "column": ENDPOINT_COLUMN,
            "protected_max_angstrom": PROTECTED_MAX,
            "severe_min_angstrom": SEVERE_MIN,
            "minimum_adsorbate_configurations": 4,
            "common_translation_removed": True,
        },
        "gates": GATES,
        "candidate_feature_count": len(NEXT63_FEATURE_NAMES),
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
        "mode": "metal_chemistry_augmented_robust_discovery_search",
        "robust_discovery_labels_opened": True,
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
            "candidate_features": len(NEXT63_FEATURE_NAMES),
            "candidate_formulas": int(result["candidate_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next64_odac23_metal_chemistry_search.py": source_hash,
            "src/next57_odac23_discovery_search.py": hashes["search_engine"],
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
            raise RuntimeError("NEXT64 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT64 input changed before publication")
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
    manifest = run_metal_chemistry_search(
        feature_dir=args.feature_dir,
        endpoint_firewall_manifest_path=args.endpoint_firewall,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps({"passes": manifest["passes_discovery_gates"], **manifest["counts"]}, indent=2, sort_keys=True))


__all__ = ["PROTOCOL", "run_metal_chemistry_search"]


if __name__ == "__main__":
    main()
