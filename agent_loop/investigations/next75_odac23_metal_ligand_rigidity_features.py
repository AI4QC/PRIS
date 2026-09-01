#!/usr/bin/env python3
"""Label-free metal-ligand rigidity and self-stress features for ODAC23 x0."""

from __future__ import annotations

import argparse
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

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next20_valence_rigidity import (
    FEATURE_NAMES as SIVR_ALL_FEATURE_NAMES,
    _weighted_quantile,
    rigidity_features_from_edges,
)
from src.next26_packing import _radii
from src.next37_self_stress_compatibility_features import (
    CANDIDATE_FEATURE_NAMES as SSCP_CANDIDATE_FEATURE_NAMES,
    DIAGNOSTIC_FEATURE_NAMES as SSCP_DIAGNOSTIC_FEATURE_NAMES,
    self_stress_compatibility_features,
)
from src.next49_framework_topology import (
    _canonical_covalent_edges,
    _environment_versions,
    _is_metal,
    _strict_geometry,
)
from src.next54_odac23_train_selection import (
    GEOMETRY_NAME as SOURCE_GEOMETRY_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    METADATA_NAME as SOURCE_METADATA_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)
from src.next55_odac23_analytic_features import _load_archive
from src.next70_odac23_metal_donor_bond_valence_features import (
    DONOR_VALENCE_BY_NUMBER,
    FEATURES_NAME as BASE_FEATURES_NAME,
    PROTOCOL as BASE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next75-odac23-metal-ligand-rigidity-features-v1"
DESIGN_SHA256 = "12735bbf413fd6f742948db0fff9715f527da3fcb95ec1931fcc903cd497e5d9"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
EXPECTED_BASE_MANIFEST_SHA256 = (
    "928a0bbfa1120e2c92bac2e9d3f0046a1d440c24beb72f652e477eb827874f14"
)
EXPECTED_BASE_FEATURE_SHA256 = (
    "d3684af21c70e3be18ae4aed8dd9a505209cfb2d91e9639911aae72da77ca6dc"
)
SIVR_INTENSIVE_FEATURE_NAMES = tuple(
    name
    for name in SIVR_ALL_FEATURE_NAMES
    if name not in {"sivr_stiffness_min", "sivr_edge_count", "sivr_site_count"}
)
SSCP_INTENSIVE_FEATURE_NAMES = tuple(SSCP_CANDIDATE_FEATURE_NAMES) + (
    "sscp_balanced_fraction",
    "sscp_cokernel_dimension_fraction",
)
GRAPH_FEATURE_NAMES = (
    "mlr_active_site_fraction",
    "mlr_metal_fraction_of_active_sites",
    "mlr_edge_per_active_site",
)
METAL_LIGAND_RIGIDITY_FEATURE_NAMES = (
    tuple(f"mlr_{name}" for name in SIVR_INTENSIVE_FEATURE_NAMES)
    + tuple(f"mlr_{name}" for name in SSCP_INTENSIVE_FEATURE_NAMES)
    + GRAPH_FEATURE_NAMES
)
FEATURES_NAME = "next75_odac23_metal_ligand_rigidity_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class MetalLigandRigidityResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def compute_metal_ligand_rigidity_features(atoms: Atoms) -> MetalLigandRigidityResult:
    """Compute compact central-force and compatibility features from raw x0."""

    try:
        numbers, _positions, _cell = _strict_geometry(atoms)
        covalent, _vdw = _radii(numbers)
        covalent = np.asarray(covalent, dtype=float)
        edges = _canonical_covalent_edges(atoms, covalent)
        metal = np.asarray([_is_metal(number) for number in numbers], dtype=bool)
        if not metal.any():
            raise ValueError("framework has no recognized metal")
        selected = []
        active = set()
        for edge in edges:
            left_metal = bool(metal[edge.first])
            right_metal = bool(metal[edge.second])
            if left_metal == right_metal:
                continue
            donor_index = edge.second if left_metal else edge.first
            if int(numbers[donor_index]) not in DONOR_VALENCE_BY_NUMBER:
                continue
            selected.append(edge)
            active.add(int(edge.first))
            active.add(int(edge.second))
        if not selected or len(active) < 2:
            raise ValueError("framework has no metal-donor constraint graph")
        ordered = sorted(active)
        remap = {old: new for new, old in enumerate(ordered)}
        endpoints = np.asarray(
            [(remap[int(edge.first)], remap[int(edge.second)]) for edge in selected],
            dtype=int,
        )
        vectors = np.asarray([edge.vector for edge in selected], dtype=float)
        radius_sums = np.asarray(
            [covalent[edge.first] + covalent[edge.second] for edge in selected], dtype=float
        )
        weights = np.ones(len(selected), dtype=float)
        sivr = rigidity_features_from_edges(
            n_sites=len(ordered),
            endpoints=endpoints,
            vectors=vectors,
            radius_sums=radius_sums,
            weights=weights,
        )
        if not sivr.supported:
            raise ValueError(sivr.failure_reason or "metal-ligand rigidity unsupported")
        distance = np.linalg.norm(vectors, axis=1)
        log_ratio = np.log(distance / radius_sums)
        scale = _weighted_quantile(log_ratio, weights, 0.5)
        residual = log_ratio - scale
        sscp = self_stress_compatibility_features(
            n_sites=len(ordered),
            endpoints=endpoints,
            vectors=vectors,
            residuals=residual,
            weights=weights,
        )
        if not sscp.supported:
            raise ValueError(sscp.failure_reason or "metal-ligand compatibility unsupported")
        active_metal = int(np.sum(metal[ordered]))
        values = {
            **{
                f"mlr_{name}": float(sivr.features[name])
                for name in SIVR_INTENSIVE_FEATURE_NAMES
            },
            **{
                f"mlr_{name}": float(sscp.features[name])
                for name in SSCP_INTENSIVE_FEATURE_NAMES
            },
            "mlr_active_site_fraction": float(len(ordered) / len(atoms)),
            "mlr_metal_fraction_of_active_sites": float(active_metal / len(ordered)),
            "mlr_edge_per_active_site": float(len(selected) / len(ordered)),
        }
        if tuple(values) != METAL_LIGAND_RIGIDITY_FEATURE_NAMES or not np.isfinite(
            list(values.values())
        ).all():
            raise ValueError("metal-ligand rigidity feature schema differs")
        return MetalLigandRigidityResult(True, None, values)
    except Exception as exc:
        return MetalLigandRigidityResult(False, f"{type(exc).__name__}: {exc}", {})


def _feature_record(atoms: Atoms) -> dict[str, object]:
    result = compute_metal_ligand_rigidity_features(atoms)
    row: dict[str, object] = {
        "metal_ligand_rigidity_supported": result.supported,
        "metal_ligand_rigidity_failure": result.failure_reason,
    }
    row.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in METAL_LIGAND_RIGIDITY_FEATURE_NAMES
        }
    )
    return row


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT75 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_metal_ligand_rigidity_batch(
    *,
    source_dir: Path,
    base_feature_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, object]:
    """Build all partitions without any endpoint or opened validation input."""

    source_dir = Path(source_dir).resolve()
    base_feature_dir = Path(base_feature_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT75 workers must be 1 through 64")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": source_dir / SOURCE_METADATA_NAME,
        "geometry": source_dir / SOURCE_GEOMETRY_NAME,
        "source_manifest": source_dir / SOURCE_MANIFEST_NAME,
        "base_features": base_feature_dir / BASE_FEATURES_NAME,
        "base_manifest": base_feature_dir / MANIFEST_NAME,
        "design": design_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT75 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["source_manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256
        or hashes["base_manifest"] != EXPECTED_BASE_MANIFEST_SHA256
        or hashes["base_features"] != EXPECTED_BASE_FEATURE_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT75 frozen input hash differs")
    source_manifest = _read_json(paths["source_manifest"])
    base_manifest = _read_json(paths["base_manifest"])
    source_outputs = source_manifest.get("outputs_sha256")
    base_outputs = base_manifest.get("outputs_sha256")
    if (
        source_manifest.get("protocol") != SOURCE_PROTOCOL
        or source_manifest.get("selection_frozen_before_row_labels_opened") is not True
        or source_manifest.get("validation_or_test_payload_deserialized") is not False
        or not isinstance(source_outputs, Mapping)
        or source_outputs.get(paths["metadata"].name) != hashes["metadata"]
        or source_outputs.get(paths["geometry"].name) != hashes["geometry"]
        or base_manifest.get("protocol") != BASE_FEATURE_PROTOCOL
        or base_manifest.get("labels_opened") is not False
        or not isinstance(base_outputs, Mapping)
        or base_outputs.get(BASE_FEATURES_NAME) != hashes["base_features"]
    ):
        raise ValueError("NEXT75 label-free provenance differs")
    metadata = pd.read_parquet(paths["metadata"])
    base = pd.read_parquet(paths["base_features"])
    material_ids = tuple(metadata["material_id"].astype(str))
    if (
        len(metadata) != len(base)
        or metadata["material_id"].duplicated().any()
        or base["material_id"].duplicated().any()
        or tuple(base["material_id"].astype(str)) != material_ids
    ):
        raise ValueError("NEXT75 base feature identity differs")
    structures = _load_archive(paths["geometry"], material_ids)
    rows = []
    failures: Counter[str] = Counter()
    if workers == 1:
        iterator = map(_feature_record, structures)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_feature_record, structures, chunksize=2)
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if not bool(row["metal_ligand_rigidity_supported"]):
                failures[str(row["metal_ligand_rigidity_failure"])] += 1
            if index % 100 == 0 or index == len(structures):
                print(f"NEXT75 ODAC23 metal-ligand rigidity: {index}/{len(structures)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    additions = pd.DataFrame(rows)
    if tuple(additions.loc[:, METAL_LIGAND_RIGIDITY_FEATURE_NAMES].columns) != (
        METAL_LIGAND_RIGIDITY_FEATURE_NAMES
    ):
        raise ValueError("NEXT75 output feature schema differs")
    table = pd.concat([base.reset_index(drop=True), additions], axis=1)
    if not table["combined_supported"].equals(base["combined_supported"]):
        raise RuntimeError("NEXT75 changed pre-existing support")
    source_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "selected_odac23_all_partitions_label_free_metal_ligand_rigidity",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "opened_internal_validation_result_used": False,
        "internal_replication_labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "dft_calculation_or_value_used": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "missing_policy": "optional_family_fail_open_keep",
        "feature_columns": list(METAL_LIGAND_RIGIDITY_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "base_supported": int(table["combined_supported"].sum()),
            "metal_ligand_rigidity_supported": int(
                table["metal_ligand_rigidity_supported"].sum()
            ),
            "failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next75_odac23_metal_ligand_rigidity_features.py": _sha256(source_path)
        },
        "environment_versions": {
            **_environment_versions(),
            "ase": importlib.metadata.version("ase"),
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURES_NAME
        table.to_parquet(feature_path, index=False)
        manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(feature_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != manifest["executed_source_sha256"][
            "src/next75_odac23_metal_ligand_rigidity_features.py"
        ]:
            raise RuntimeError("NEXT75 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT75 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-feature-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_metal_ligand_rigidity_batch(
        source_dir=args.source_dir,
        base_feature_dir=args.base_feature_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "METAL_LIGAND_RIGIDITY_FEATURE_NAMES",
    "PROTOCOL",
    "build_metal_ligand_rigidity_batch",
    "compute_metal_ligand_rigidity_features",
]


if __name__ == "__main__":
    main()
