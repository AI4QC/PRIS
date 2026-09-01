#!/usr/bin/env python3
"""Discovery-only NEXT57 search over the expanded NEXT58 x0 feature catalogue."""

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
from src.next56_odac23_label_firewall import (
    PROTOCOL as FIREWALL_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
)
from src.next57_odac23_discovery_search import (
    ENDPOINT_COLUMN,
    EXPECTED_DISCOVERY_LABEL_SHA256,
    EXPECTED_DISCOVERY_MANIFEST_SHA256,
    EXPECTED_FIREWALL_MANIFEST_SHA256,
    GATES,
    PROTECTED_MAX,
    SEVERE_MIN,
    search_discovery_rule,
)
from src.next58_odac23_shared_motif_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    NEXT58_FEATURE_NAMES,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next59-odac23-motif-discovery-search-v1"
DESIGN_SHA256 = "5f5cf3ba355d7805cacf8e2f652da5f6c3b7254a4274ec47621a88ff53f6ea87"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "4d0c2b667ea67233444d41b4f2c8035ab5eb047fad93342e9efb568c04ec5946"
)
EXPECTED_SEARCH_ENGINE_SHA256 = (
    "486cabdcbd417316179cddf89625a34b14c03b209265b8decc2046eb3c797967"
)
FORMULA_NAME = "NEXT59_ODAC23_MOTIF_DISCOVERY_CANDIDATE.json"
SEARCH_NAME = "NEXT59_ODAC23_MOTIF_DISCOVERY_SEARCH.json"
PREDICTIONS_NAME = "next59_odac23_motif_discovery_predictions.parquet"
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


def run_motif_discovery_search(
    *,
    feature_dir: Path,
    firewall_manifest_path: Path,
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
        "firewall_manifest": Path(firewall_manifest_path).resolve(),
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
        "search_engine": search_engine_path,
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT59 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "firewall_manifest": EXPECTED_FIREWALL_MANIFEST_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
        "search_engine": EXPECTED_SEARCH_ENGINE_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT59 frozen input hash differs")
    feature_manifest = _strict_json(paths["feature_manifest"], "NEXT58 manifest")
    firewall_manifest = _strict_json(paths["firewall_manifest"], "NEXT56 firewall manifest")
    discovery_manifest = _strict_json(paths["discovery_manifest"], "NEXT56 discovery manifest")
    feature_outputs = feature_manifest.get("outputs_sha256")
    discovery_outputs = discovery_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != SOURCE_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
        or firewall_manifest.get("protocol") != FIREWALL_PROTOCOL
        or discovery_manifest.get("protocol") != FIREWALL_PROTOCOL
        or discovery_manifest.get("partition_role") != "discovery"
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get(ROLE_LABELS_NAME) != hashes["discovery_labels"]
    ):
        raise ValueError("NEXT59 discovery-only provenance differs")

    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT59 received non-discovery labels")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(features) or len(joined) != len(labels):
        raise ValueError("NEXT59 discovery identity coverage differs")
    result = search_discovery_rule(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        candidate_features=NEXT58_FEATURE_NAMES,
    )
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
        "candidate_feature_count": len(NEXT58_FEATURE_NAMES),
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
        "mode": "motif_augmented_discovery_only_finite_search",
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
            "candidate_features": len(NEXT58_FEATURE_NAMES),
            "candidate_formulas": int(result["candidate_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next59_odac23_motif_discovery_search.py": source_hash,
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
            raise RuntimeError("NEXT59 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT59 input changed before publication")
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
    manifest = run_motif_discovery_search(
        feature_dir=args.feature_dir,
        firewall_manifest_path=args.firewall_manifest,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps({"passes": manifest["passes_discovery_gates"], **manifest["counts"]}, indent=2, sort_keys=True))


__all__ = ["PROTOCOL", "run_motif_discovery_search"]


if __name__ == "__main__":
    main()
