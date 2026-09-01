#!/usr/bin/env python3
"""Publish source-agnostic DFT-free absolute contact features from frozen x0."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import pandas as pd

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256, _strict_json
from src.next32_inorganic_response_features import (
    CONTACT_FEATURE_NAMES,
    compute_periodic_contact_features,
)


PROTOCOL = "2026-08-03-next41-source-balanced-contact-features-v1"
FEATURE_NAME = "next41_contact_guard_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
_ALLOWED_INPUT_ROLES = {
    "unrelaxed_x0_geometry_only",
    "step0_unrelaxed_x0_geometry_only",
}
_FORBIDDEN_TRUE_FLAGS = (
    "labels_opened",
    "endpoint_artifacts_opened",
    "relaxed_structures_opened",
    "later_geometry_opened",
    "dft_values_read",
    "endpoint_fields_read",
    "dft_numeric_fields_parsed",
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _load_frozen_geometry(
    *, metadata_path: Path, frames_zip_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list, dict[str, object]]:
    manifest = _strict_json(manifest_path, role="frozen geometry manifest")
    if any(manifest.get(name) is True for name in _FORBIDDEN_TRUE_FLAGS):
        raise ValueError("frozen geometry manifest crossed the no-endpoint boundary")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or any(
        outputs.get(path.name) != _sha256(path)
        for path in (metadata_path, frames_zip_path)
    ):
        raise ValueError("frozen geometry artifact hash differs")
    metadata = pd.read_parquet(metadata_path)
    required = {"material_id", "natoms", "input_role"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"frozen geometry metadata lacks {sorted(required - set(metadata))}")
    metadata = metadata.copy()
    metadata["material_id"] = metadata.material_id.astype(str)
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        metadata.material_id.duplicated().any()
        or not set(metadata.input_role.astype(str)).issubset(_ALLOWED_INPUT_ROLES)
    ):
        raise ValueError("frozen geometry identity or input role differs")
    ids = tuple(metadata.material_id)
    loaded, structures = _load_archive_only(frames_zip_path, ids)
    if loaded != list(ids) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata.natoms, strict=True)
    ):
        raise ValueError("frozen geometry archive identity differs")
    return metadata, structures, manifest


def build_contact_guard_features(
    *,
    metadata_path: Path,
    frames_zip_path: Path,
    upstream_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Compute frozen-radius contact burdens before any endpoint is joined."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry": Path(frames_zip_path).resolve(),
        "upstream_manifest": Path(upstream_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT41 geometry input is missing")
    input_hashes = {name: _sha256(path) for name, path in paths.items()}
    metadata, structures, upstream = _load_frozen_geometry(
        metadata_path=paths["metadata"],
        frames_zip_path=paths["geometry"],
        manifest_path=paths["upstream_manifest"],
    )
    failures: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for upstream_row, atoms in zip(metadata.to_dict("records"), structures, strict=True):
        result = compute_periodic_contact_features(atoms)
        if not result.supported:
            failures[result.failure_reason or "unknown"] += 1
        row: dict[str, object] = {
            "material_id": str(upstream_row["material_id"]),
            "natoms": int(upstream_row["natoms"]),
            "contact_supported": bool(result.supported),
            "contact_failure": result.failure_reason,
        }
        row.update({name: result.features.get(name, float("nan")) for name in CONTACT_FEATURE_NAMES})
        rows.append(row)
    elapsed = time.perf_counter() - started
    table = pd.DataFrame(rows).sort_values("material_id", kind="stable", ignore_index=True)
    if len(table) != len(metadata) or table.material_id.duplicated().any():
        raise RuntimeError("NEXT41 contact feature identity accounting differs")

    repository = Path(__file__).resolve().parents[1]
    source_names = (
        "src/next32_inorganic_response_features.py",
        "src/next41_contact_guard_features.py",
    )
    source_hashes = {name: _sha256(repository / name) for name in source_names}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "upstream_protocol": upstream.get("protocol"),
        "input_role": "one_frozen_unrelaxed_or_step0_structure",
        "labels_opened": False,
        "later_geometry_opened": False,
        "dft_values_read": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "feature_names": list(CONTACT_FEATURE_NAMES),
        "counts": {
            "rows": len(table),
            "atoms": int(table.natoms.sum()),
            "supported": int(table.contact_supported.sum()),
            "failed": int((~table.contact_supported).sum()),
        },
        "failure_counts": dict(sorted(failures.items())),
        "execution": {"wall_time_seconds": elapsed},
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / FEATURE_NAME
        table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {FEATURE_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT41 feature input changed during publication")
        if any(_sha256(repository / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("NEXT41 feature source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--upstream-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    build_contact_guard_features(
        metadata_path=args.metadata,
        frames_zip_path=args.frames_zip,
        upstream_manifest_path=args.upstream_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FEATURE_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "build_contact_guard_features",
]
