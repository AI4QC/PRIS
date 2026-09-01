#!/usr/bin/env python3
"""Assemble identity-locked OMC25 endpoints for frozen NEXT28 predictions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next26_omc25 import build_endpoint_table
from src.next27_build_endpoints import select_expected_endpoints


PROTOCOL = "2026-08-03-next28-omc25-identity-locked-endpoints-v1"
ENDPOINT_NAME = "next28_dft_endpoints.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def assemble_endpoints(
    *,
    shards: Sequence[str],
    db_paths: Sequence[Path],
    metadata_paths: Sequence[Path],
    catalogue_path: Path,
    frozen_predictions_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Decode endpoints only after freeze and require the exact prediction identity set."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not shards or not (len(shards) == len(db_paths) == len(metadata_paths)):
        raise ValueError("shards, databases, and metadata paths must align")
    if len(set(map(str, shards))) != len(shards):
        raise ValueError("source shards must be unique")
    catalogue = Path(catalogue_path).resolve()
    predictions_path = Path(frozen_predictions_path).resolve()
    inputs = [catalogue, predictions_path, *map(Path, db_paths), *map(Path, metadata_paths)]
    for path in inputs:
        if not Path(path).resolve().is_file():
            raise FileNotFoundError(str(path))
    hashes = {str(Path(path).resolve()): _sha256(Path(path).resolve()) for path in inputs}
    parts: list[pd.DataFrame] = []
    counts: dict[str, dict[str, int]] = {}
    for shard, db_path, metadata_path in zip(shards, db_paths, metadata_paths, strict=True):
        full = build_endpoint_table(db_path=Path(db_path), catalogue_path=catalogue)
        selected = select_expected_endpoints(
            full, pd.read_parquet(metadata_path), source_shard=str(shard)
        )
        parts.append(selected)
        counts[str(shard)] = {"complete_trajectories": len(full), "selected": len(selected)}
        print(json.dumps({"source_shard": shard, **counts[str(shard)]}, sort_keys=True), flush=True)
    endpoints = pd.concat(parts, ignore_index=True).sort_values("material_id", kind="stable")
    predictions = pd.read_parquet(predictions_path, columns=["material_id"])
    if (
        endpoints.material_id.duplicated().any()
        or predictions.material_id.isna().any()
        or predictions.material_id.duplicated().any()
        or set(endpoints.material_id.astype(str)) != set(predictions.material_id.astype(str))
    ):
        raise ValueError("assembled endpoint identities differ from frozen predictions")
    opened_at = datetime.now(timezone.utc).isoformat()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        endpoint_path = staging / ENDPOINT_NAME
        endpoints.to_parquet(endpoint_path, index=False)
        source_hashes = {"src/next28_build_endpoints.py": _sha256(Path(__file__).resolve())}
        manifest = {
            "protocol": PROTOCOL,
            "labels_opened": True,
            "opened_at_utc": opened_at,
            "identity_lock": "exact frozen NEXT28 prediction material_id set",
            "counts": {"rows": len(endpoints), "source_shards": len(shards)},
            "shard_counts": counts,
            "inputs_sha256": hashes,
            "outputs_sha256": {ENDPOINT_NAME: _sha256(endpoint_path)},
            "executed_source_sha256": source_hashes,
            "physical_never_read_lockbox": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for path in inputs:
            resolved = Path(path).resolve()
            if _sha256(resolved) != hashes[str(resolved)]:
                raise RuntimeError(f"input changed while labels were opened: {resolved}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--databases", nargs="+", required=True, type=Path)
    parser.add_argument("--metadata", nargs="+", required=True, type=Path)
    parser.add_argument("--catalogue", required=True, type=Path)
    parser.add_argument("--frozen-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(assemble_endpoints(
        shards=args.shards,
        db_paths=args.databases,
        metadata_paths=args.metadata,
        catalogue_path=args.catalogue,
        frozen_predictions_path=args.frozen_predictions,
        output_dir=args.output_dir,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ENDPOINT_NAME", "assemble_endpoints"]
