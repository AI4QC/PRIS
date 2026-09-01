#!/usr/bin/env python3
"""Physically partition selected ODAC23 labels without summarizing endpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next54_odac23_train_selection import (
    LABELS_NAME as SOURCE_LABELS_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    METADATA_NAME as SOURCE_METADATA_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next56-odac23-train-label-firewall-v1"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
ROLES = ("discovery", "internal_validation", "internal_replication")
TOP_MANIFEST_NAME = "FIREWALL_MANIFEST.json"
ROLE_LABELS_NAME = "offline_labels.parquet"
ROLE_MANIFEST_NAME = "MANIFEST.json"


def partition_label_rows(
    metadata: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Route exact label rows using frozen metadata roles, with no summaries."""

    if not {"material_id", "partition_role"}.issubset(metadata.columns):
        raise ValueError("NEXT56 metadata schema differs")
    if "material_id" not in labels.columns:
        raise ValueError("NEXT56 label schema differs")
    if (
        metadata.empty
        or len(metadata) != len(labels)
        or metadata["material_id"].duplicated().any()
        or labels["material_id"].duplicated().any()
        or set(metadata["partition_role"]) != set(ROLES)
        or set(metadata["material_id"].astype(str))
        != set(labels["material_id"].astype(str))
    ):
        raise ValueError("NEXT56 label identity/roles differ")
    roles = metadata.set_index("material_id")["partition_role"]
    routed = labels.copy()
    routed["partition_role"] = routed["material_id"].map(roles)
    if routed["partition_role"].isna().any():
        raise RuntimeError("NEXT56 label routing failed")
    return {
        role: routed[routed["partition_role"].eq(role)]
        .sort_values("material_id", kind="mergesort")
        .reset_index(drop=True)
        for role in ROLES
    }


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid NEXT54 manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("NEXT54 manifest must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_label_firewall(*, source_dir: Path, output_dir: Path) -> dict[str, object]:
    source_dir = Path(source_dir).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": source_dir / SOURCE_METADATA_NAME,
        "labels": source_dir / SOURCE_LABELS_NAME,
        "manifest": source_dir / SOURCE_MANIFEST_NAME,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT56 source artifact is incomplete")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes["manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("NEXT56 source manifest hash differs")
    source_manifest = _strict_json(paths["manifest"])
    outputs = source_manifest.get("outputs_sha256")
    if (
        source_manifest.get("protocol") != SOURCE_PROTOCOL
        or not isinstance(outputs, dict)
        or outputs.get(paths["metadata"].name) != hashes["metadata"]
        or outputs.get(paths["labels"].name) != hashes["labels"]
    ):
        raise ValueError("NEXT56 source provenance differs")

    metadata = pd.read_parquet(paths["metadata"])
    labels = pd.read_parquet(paths["labels"])
    partitions = partition_label_rows(metadata, labels)
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    top_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "identity_only_physical_label_partition",
        "endpoint_values_summarized_or_inspected": False,
        "formula_or_feature_search_executed": False,
        "official_validation_or_test_payload_deserialized": False,
        "roles": list(ROLES),
        "counts": {role: len(partitions[role]) for role in ROLES},
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {"src/next56_odac23_label_firewall.py": source_hash},
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        role_outputs = {}
        for role in ROLES:
            role_dir = staging / role
            role_dir.mkdir()
            labels_path = role_dir / ROLE_LABELS_NAME
            partitions[role].to_parquet(labels_path, index=False)
            role_manifest = {
                "protocol": PROTOCOL,
                "partition_role": role,
                "rows": len(partitions[role]),
                "endpoint_values_summarized_or_inspected": False,
                "source_manifest_sha256": hashes["manifest"],
                "outputs_sha256": {ROLE_LABELS_NAME: _sha256(labels_path)},
            }
            role_manifest_path = role_dir / ROLE_MANIFEST_NAME
            role_manifest_path.write_bytes(_json_bytes(role_manifest))
            role_outputs[f"{role}/{ROLE_LABELS_NAME}"] = _sha256(labels_path)
            role_outputs[f"{role}/{ROLE_MANIFEST_NAME}"] = _sha256(role_manifest_path)
        top_manifest["outputs_sha256"] = role_outputs
        (staging / TOP_MANIFEST_NAME).write_bytes(_json_bytes(top_manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT56 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT56 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return top_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_label_firewall(source_dir=args.source_dir, output_dir=args.output_dir)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "PROTOCOL",
    "ROLE_LABELS_NAME",
    "ROLE_MANIFEST_NAME",
    "ROLES",
    "TOP_MANIFEST_NAME",
    "build_label_firewall",
    "partition_label_rows",
]


if __name__ == "__main__":
    main()
