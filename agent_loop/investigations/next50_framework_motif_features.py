#!/usr/bin/env python3
"""Combine NEXT49 periodic topology with x0-only CrystalNN motif coherence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next46_motif_coherence_features import (
    FEATURE_NAMES as MOTIF_FEATURE_NAMES,
    compute_motif_coherence_features,
)
from src.next49_framework_topology import (
    FEATURES_NAME as NEXT49_FEATURES_NAME,
    FRAMEWORK_FEATURE_NAMES,
    PROTOCOL as NEXT49_PROTOCOL,
    _environment_versions,
    compute_framework_topology_features,
)


PROTOCOL = "2026-08-03-next50-framework-motif-features-v1"
FEATURES_NAME = "next50_qmof_framework_motif_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
COMBINED_FEATURE_NAMES = tuple(FRAMEWORK_FEATURE_NAMES) + tuple(MOTIF_FEATURE_NAMES)


@dataclass(frozen=True)
class FrameworkMotifResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def compute_framework_motif_features(atoms: Atoms) -> FrameworkMotifResult:
    """Compute both pure-geometry families, failing open if either is absent."""

    topology = compute_framework_topology_features(atoms)
    if not topology.supported:
        return FrameworkMotifResult(False, topology.failure_reason, {})
    motif = compute_motif_coherence_features(atoms)
    if not motif.supported:
        return FrameworkMotifResult(False, motif.failure_reason, {})
    values = {
        **{name: float(topology.features[name]) for name in FRAMEWORK_FEATURE_NAMES},
        **{name: float(motif.features[name]) for name in MOTIF_FEATURE_NAMES},
    }
    if tuple(values) != COMBINED_FEATURE_NAMES or not np.isfinite(
        list(values.values())
    ).all():
        return FrameworkMotifResult(False, "combined feature schema differs", {})
    return FrameworkMotifResult(True, None, values)


def _motif_record(atoms: Atoms) -> dict[str, object]:
    """Pickle-safe worker returning one deterministic motif record."""

    result = compute_motif_coherence_features(atoms)
    record: dict[str, object] = {
        "motif_supported": result.supported,
        "motif_failure": result.failure_reason,
    }
    record.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in MOTIF_FEATURE_NAMES
        }
    )
    return record


def _strict_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_qmof_framework_motif_batch(
    *,
    geometry_path: Path,
    next49_features_path: Path,
    next49_manifest_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, object]:
    """Seal combined features before any QMOF endpoint is joined."""

    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT50 workers must be an integer from 1 through 64")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "geometry": Path(geometry_path).resolve(),
        "next49_features": Path(next49_features_path).resolve(),
        "next49_manifest": Path(next49_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT50 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    manifest49 = _strict_json(paths["next49_manifest"], role="NEXT49 manifest")
    outputs49 = manifest49.get("outputs_sha256")
    inputs49 = manifest49.get("inputs_sha256")
    geometry49 = inputs49.get("geometry") if isinstance(inputs49, Mapping) else None
    if (
        manifest49.get("protocol") != NEXT49_PROTOCOL
        or manifest49.get("input_role") != "unrelaxed_x0_geometry_only"
        or manifest49.get("labels_opened") is not False
        or manifest49.get("relaxed_coordinate_payloads_opened") is not False
        or manifest49.get("model_or_proxy_potential_used") is not False
        or manifest49.get("dft_or_energy_proxy_used_at_execution") is not False
        or not isinstance(outputs49, Mapping)
        or outputs49.get(NEXT49_FEATURES_NAME) != hashes["next49_features"]
        or not isinstance(geometry49, Mapping)
        or geometry49.get("sha256") != hashes["geometry"]
    ):
        raise ValueError("NEXT50 input crossed the x0-only boundary")

    topology = pd.read_parquet(paths["next49_features"])
    required = {
        "material_id",
        "source_family",
        "framework_feature_supported",
        *FRAMEWORK_FEATURE_NAMES,
    }
    if (
        topology.empty
        or not required.issubset(topology.columns)
        or topology["material_id"].isna().any()
        or topology["material_id"].duplicated().any()
    ):
        raise ValueError("NEXT50 topology table is invalid")
    material_ids = topology["material_id"].astype(str).tolist()
    archive_ids, frames = _load_archive_only(paths["geometry"], tuple(material_ids))
    if archive_ids != material_ids:
        raise ValueError("NEXT50 geometry order differs")

    motif_rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    if workers == 1:
        iterator = map(_motif_record, frames)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_motif_record, frames, chunksize=1)
    try:
        indexed_rows = enumerate(iterator, start=1)
        for index, row in indexed_rows:
            motif_rows.append(row)
            if not bool(row["motif_supported"]):
                failures[str(row["motif_failure"] or "unsupported")] += 1
            if index % 50 == 0 or index == len(frames):
                print(f"NEXT50 QMOF motif features: {index}/{len(frames)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    table = pd.concat(
        [topology.reset_index(drop=True), pd.DataFrame(motif_rows)], axis=1
    )
    table["combined_supported"] = (
        table["framework_feature_supported"].astype(bool)
        & table["motif_supported"].astype(bool)
        & np.isfinite(table.loc[:, COMBINED_FEATURE_NAMES]).all(axis=1)
    )

    source_path = Path(__file__).resolve()
    motif_source = Path(__import__(
        "src.next46_motif_coherence_features", fromlist=["x"]
    ).__file__).resolve()
    source_hashes = {
        "src/next50_framework_motif_features.py": _sha256(source_path),
        "src/next46_motif_coherence_features.py": _sha256(motif_source),
    }
    output_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "exposed_qmof_label_free_framework_motif_build",
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "model_or_proxy_potential_used": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "missing_policy": "fail_open_do_not_reject",
        "feature_columns": list(COMBINED_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "topology_supported": int(table["framework_feature_supported"].sum()),
            "motif_supported": int(table["motif_supported"].sum()),
            "combined_supported": int(table["combined_supported"].sum()),
            "motif_failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": source_hashes,
        "environment_versions": {
            **_environment_versions(),
            "matminer": importlib.metadata.version("matminer"),
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / FEATURES_NAME
        table.to_parquet(output_path, index=False)
        output_manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(output_manifest))
        for name, path in paths.items():
            if _sha256(path) != hashes[name]:
                raise RuntimeError(f"NEXT50 input {name} changed before publication")
        for name, digest in source_hashes.items():
            check_path = source_path if name.startswith("src/next50") else motif_source
            if _sha256(check_path) != digest:
                raise RuntimeError(f"NEXT50 source {name} changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_manifest


__all__ = [
    "COMBINED_FEATURE_NAMES",
    "FEATURES_NAME",
    "FrameworkMotifResult",
    "MANIFEST_NAME",
    "PROTOCOL",
    "build_qmof_framework_motif_batch",
    "compute_framework_motif_features",
]
