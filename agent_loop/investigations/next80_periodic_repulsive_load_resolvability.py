#!/usr/bin/env python3
"""Periodic repulsive-load resolvability features from one unchanged raw x0."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
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
from ase.neighborlist import neighbor_list
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsqr

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next26_packing import _radii
from src.next49_framework_topology import (
    _canonical_covalent_edges,
    _environment_versions,
    _strict_geometry,
)
from src.next54_odac23_train_selection import (
    GEOMETRY_NAME as SOURCE_GEOMETRY_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    METADATA_NAME as SOURCE_METADATA_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)
from src.next55_odac23_analytic_features import _load_archive
from src.next77_odac23_analytic_electrostatic_features import (
    FEATURES_NAME as BASE_FEATURES_NAME,
    MANIFEST_NAME as BASE_MANIFEST_NAME,
    PROTOCOL as BASE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next80-periodic-repulsive-load-resolvability-v1"
DESIGN_SHA256 = "4b7a8b5abf047b1897f0b80d6a16cc8be87483990bcf43762c6fb9117121be29"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
EXPECTED_BASE_MANIFEST_SHA256 = (
    "ac1ecd88cfdf57fcec1c5dbe331ea8373f07aac79b706ed0d2c4c2575ab9ad82"
)
EXPECTED_BASE_FEATURE_SHA256 = (
    "c2c6668d24c77c20d8ab2878fa5e6b7d266f26ebea85b58ae083985eec6f2ad7"
)
PRLR_FEATURE_NAMES = (
    "prlr_residual_fraction",
    "prlr_atomic_residual_fraction",
    "prlr_cell_residual_fraction",
    "prlr_site_residual_q95",
    "prlr_bar_stress_rms",
    "prlr_bar_stress_amplification",
    "prlr_bar_stress_localization",
    "prlr_contact_weight_rms",
    "prlr_contact_weight_max",
    "prlr_contact_edges_per_atom",
    "prlr_contact_active_site_fraction",
    "prlr_covalent_edges_per_atom",
    "prlr_risk",
)
FEATURES_NAME = "next80_odac23_repulsive_load_resolvability_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class RepulsiveLoadResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> RepulsiveLoadResult:
    return RepulsiveLoadResult(False, reason, {})


def _equilibrium_rows(
    *,
    n_sites: int,
    endpoints: np.ndarray,
    vectors: np.ndarray,
    characteristic_length: float,
) -> csr_matrix:
    n_edges = len(endpoints)
    if n_edges == 0:
        return csr_matrix((0, 3 * n_sites + 6), dtype=float)
    distances = np.linalg.norm(vectors, axis=1)
    directions = vectors / distances[:, None]
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for edge_index, ((left, right), direction, distance) in enumerate(
        zip(endpoints, directions, distances, strict=True)
    ):
        for axis in range(3):
            rows.extend((edge_index, edge_index))
            columns.extend((3 * int(left) + axis, 3 * int(right) + axis))
            data.extend((-float(direction[axis]), float(direction[axis])))
        x, y, z = (float(value) for value in direction)
        factor = float(distance / characteristic_length)
        affine = factor * np.asarray(
            [x * x, y * y, z * z, math.sqrt(2.0) * y * z,
             math.sqrt(2.0) * x * z, math.sqrt(2.0) * x * y],
            dtype=float,
        )
        for offset, value in enumerate(affine):
            rows.append(edge_index)
            columns.append(3 * n_sites + offset)
            data.append(float(value))
    return coo_matrix(
        (data, (rows, columns)),
        shape=(n_edges, 3 * n_sites + 6),
        dtype=float,
    ).tocsr()


def repulsive_load_resolvability_features(
    *,
    n_sites: int,
    covalent_endpoints: Sequence[Sequence[int]] | np.ndarray,
    covalent_vectors: Sequence[Sequence[float]] | np.ndarray,
    contact_endpoints: Sequence[Sequence[int]] | np.ndarray,
    contact_vectors: Sequence[Sequence[float]] | np.ndarray,
    contact_weights: Sequence[float] | np.ndarray,
    characteristic_length: float,
) -> RepulsiveLoadResult:
    """Resolve fixed compressive contact load through periodic bar stresses."""

    try:
        if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
            return _failure("periodic framework needs at least two sites")
        n_sites = int(n_sites)
        cov_pair = np.asarray(covalent_endpoints, dtype=int)
        cov_vector = np.asarray(covalent_vectors, dtype=float)
        contact_pair = np.asarray(contact_endpoints, dtype=int)
        contact_vector = np.asarray(contact_vectors, dtype=float)
        weight = np.asarray(contact_weights, dtype=float)
        if cov_pair.ndim != 2 or cov_pair.shape[1:] != (2,) or len(cov_pair) < 1:
            return _failure("covalent endpoints must have nonempty shape (E,2)")
        if cov_vector.shape != (len(cov_pair), 3):
            return _failure("covalent vectors must have shape (E,3)")
        if contact_pair.shape != (len(weight), 2):
            return _failure("contact endpoints must have shape (C,2)")
        if contact_vector.shape != (len(weight), 3):
            return _failure("contact vectors must have shape (C,3)")
        if any(
            np.any(pair < 0) or np.any(pair >= n_sites)
            for pair in (cov_pair, contact_pair)
        ):
            return _failure("edge endpoints contain invalid site indices")
        if (
            not np.isfinite(cov_vector).all()
            or not np.isfinite(contact_vector).all()
            or not np.isfinite(weight).all()
            or np.any(weight <= 0.0)
        ):
            return _failure("edge vectors and contact weights must be finite and positive")
        cov_distance = np.linalg.norm(cov_vector, axis=1)
        contact_distance = np.linalg.norm(contact_vector, axis=1)
        if np.any(cov_distance <= 1.0e-12) or np.any(contact_distance <= 1.0e-12):
            return _failure("edge distances must be positive")
        length = float(characteristic_length)
        if not math.isfinite(length) or length <= 1.0e-12:
            return _failure("characteristic length must be finite and positive")

        n_covalent = len(cov_pair)
        n_contact = len(contact_pair)
        if n_contact == 0:
            values = {
                "prlr_residual_fraction": 0.0,
                "prlr_atomic_residual_fraction": 0.0,
                "prlr_cell_residual_fraction": 0.0,
                "prlr_site_residual_q95": 0.0,
                "prlr_bar_stress_rms": 0.0,
                "prlr_bar_stress_amplification": 0.0,
                "prlr_bar_stress_localization": 0.0,
                "prlr_contact_weight_rms": 0.0,
                "prlr_contact_weight_max": 0.0,
                "prlr_contact_edges_per_atom": 0.0,
                "prlr_contact_active_site_fraction": 0.0,
                "prlr_covalent_edges_per_atom": float(n_covalent / n_sites),
                "prlr_risk": 0.0,
            }
            return RepulsiveLoadResult(True, None, values)

        cov_rows = _equilibrium_rows(
            n_sites=n_sites,
            endpoints=cov_pair,
            vectors=cov_vector,
            characteristic_length=length,
        )
        contact_rows = _equilibrium_rows(
            n_sites=n_sites,
            endpoints=contact_pair,
            vectors=contact_vector,
            characteristic_length=length,
        )
        compression = weight / float(np.mean(weight))
        generalized_load = np.asarray(contact_rows.T @ (-compression), dtype=float).reshape(-1)
        load_norm = float(np.linalg.norm(generalized_load))
        if not math.isfinite(load_norm) or load_norm <= 1.0e-14:
            return _failure("nonzero contacts generated no finite generalized load")
        equilibrium = cov_rows.T.tocsr()
        solution = lsqr(
            equilibrium,
            -generalized_load,
            atol=1.0e-12,
            btol=1.0e-12,
            iter_lim=max(100, min(5000, 10 * n_covalent)),
            show=False,
        )
        bar_stress = np.asarray(solution[0], dtype=float)
        residual = np.asarray(equilibrium @ bar_stress, dtype=float).reshape(-1)
        residual += generalized_load
        if not np.isfinite(bar_stress).all() or not np.isfinite(residual).all():
            return _failure("sparse equilibrium solve returned non-finite values")

        atomic_load = generalized_load[: 3 * n_sites]
        cell_load = generalized_load[3 * n_sites :]
        atomic_residual = residual[: 3 * n_sites]
        cell_residual = residual[3 * n_sites :]
        residual_fraction = float(np.clip(np.linalg.norm(residual) / load_norm, 0.0, 1.0))
        atomic_load_norm = float(np.linalg.norm(atomic_load))
        cell_load_norm = float(np.linalg.norm(cell_load))
        atomic_fraction = (
            float(np.clip(np.linalg.norm(atomic_residual) / atomic_load_norm, 0.0, 1.0))
            if atomic_load_norm > 1.0e-14
            else 0.0
        )
        cell_fraction = (
            float(np.clip(np.linalg.norm(cell_residual) / cell_load_norm, 0.0, 1.0))
            if cell_load_norm > 1.0e-14
            else 0.0
        )
        site_residual = np.linalg.norm(atomic_residual.reshape(n_sites, 3), axis=1)
        site_scale = atomic_load_norm / math.sqrt(n_sites)
        site_q95 = (
            float(
                np.quantile(
                    site_residual / site_scale,
                    0.95,
                    method="inverted_cdf",
                )
            )
            if site_scale > 1.0e-14
            else 0.0
        )
        bar_rms = float(np.sqrt(np.mean(bar_stress**2)))
        contact_rms_normalized = float(np.sqrt(np.mean(compression**2)))
        amplification = bar_rms / contact_rms_normalized
        bar_square = bar_stress**2
        localization = (
            float(n_covalent * np.sum(bar_square**2) / float(np.sum(bar_square)) ** 2)
            if float(np.sum(bar_square)) > 1.0e-28
            else 0.0
        )
        raw_weight_rms = float(np.sqrt(np.mean(weight**2)))
        risk = float(residual_fraction * math.log1p(raw_weight_rms))
        active_sites = len(set(int(value) for value in contact_pair.reshape(-1)))
        values = {
            "prlr_residual_fraction": residual_fraction,
            "prlr_atomic_residual_fraction": atomic_fraction,
            "prlr_cell_residual_fraction": cell_fraction,
            "prlr_site_residual_q95": site_q95,
            "prlr_bar_stress_rms": bar_rms,
            "prlr_bar_stress_amplification": float(amplification),
            "prlr_bar_stress_localization": localization,
            "prlr_contact_weight_rms": raw_weight_rms,
            "prlr_contact_weight_max": float(np.max(weight)),
            "prlr_contact_edges_per_atom": float(n_contact / n_sites),
            "prlr_contact_active_site_fraction": float(active_sites / n_sites),
            "prlr_covalent_edges_per_atom": float(n_covalent / n_sites),
            "prlr_risk": risk,
        }
        if tuple(values) != PRLR_FEATURE_NAMES or not np.isfinite(list(values.values())).all():
            return _failure("repulsive-load feature schema or values differ")
        return RepulsiveLoadResult(True, None, values)
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _canonical_contact_edges(
    atoms: Atoms,
    *,
    vdw: np.ndarray,
    covalent_keys: set[tuple[int, int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second, shifts, vectors, distances = neighbor_list(
        "ijSDd",
        atoms,
        np.asarray(vdw, dtype=float),
        self_interaction=True,
    )
    rows: list[tuple[int, int, tuple[int, int, int], np.ndarray, float]] = []
    for raw_i, raw_j, raw_shift, raw_vector, raw_distance in zip(
        first, second, shifts, vectors, distances, strict=True
    ):
        i = int(raw_i)
        j = int(raw_j)
        shift = tuple(int(value) for value in raw_shift)
        if i == j and shift == (0, 0, 0):
            continue
        reverse = tuple(-value for value in shift)
        if (i, j, *shift) >= (j, i, *reverse):
            continue
        key = (i, j, *shift)
        if key in covalent_keys:
            continue
        distance = float(raw_distance)
        ratio = distance / float(vdw[i] + vdw[j])
        if distance <= 1.0e-12 or not math.isfinite(ratio) or ratio >= 1.0:
            continue
        weight = max(ratio, 0.45) ** -12 - 1.0
        if weight > 0.0 and math.isfinite(weight):
            rows.append((i, j, shift, np.asarray(raw_vector, dtype=float), float(weight)))
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[4]))
    return (
        np.asarray([(row[0], row[1]) for row in rows], dtype=int).reshape(-1, 2),
        np.asarray([row[3] for row in rows], dtype=float).reshape(-1, 3),
        np.asarray([row[4] for row in rows], dtype=float),
    )


def compute_periodic_repulsive_load_resolvability(atoms: Atoms) -> RepulsiveLoadResult:
    """Build bars and compression contacts directly from one periodic x0."""

    try:
        numbers, _positions, _cell = _strict_geometry(atoms)
        if len(atoms) < 2 or not np.all(atoms.pbc):
            raise ValueError("periodic structure must contain at least two atoms")
        covalent, vdw = _radii(numbers)
        covalent = np.asarray(covalent, dtype=float)
        vdw = np.asarray(vdw, dtype=float)
        covalent_edges = _canonical_covalent_edges(atoms, covalent)
        covalent_keys = {
            (int(edge.first), int(edge.second), *tuple(int(value) for value in edge.shift))
            for edge in covalent_edges
        }
        contact_endpoints, contact_vectors, contact_weights = _canonical_contact_edges(
            atoms,
            vdw=vdw,
            covalent_keys=covalent_keys,
        )
        volume = float(atoms.get_volume())
        length = float((volume / len(atoms)) ** (1.0 / 3.0))
        return repulsive_load_resolvability_features(
            n_sites=len(atoms),
            covalent_endpoints=np.asarray(
                [(edge.first, edge.second) for edge in covalent_edges], dtype=int
            ),
            covalent_vectors=np.asarray([edge.vector for edge in covalent_edges], dtype=float),
            contact_endpoints=contact_endpoints,
            contact_vectors=contact_vectors,
            contact_weights=contact_weights,
            characteristic_length=length,
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _feature_record(atoms: Atoms) -> dict[str, object]:
    result = compute_periodic_repulsive_load_resolvability(atoms)
    row: dict[str, object] = {
        "repulsive_load_supported": result.supported,
        "repulsive_load_failure": result.failure_reason,
    }
    row.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in PRLR_FEATURE_NAMES
        }
    )
    return row


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT80 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_repulsive_load_resolvability_batch(
    *,
    source_dir: Path,
    base_feature_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, object]:
    """Build all partitions without reading any endpoint labels."""

    source_dir = Path(source_dir).resolve()
    base_feature_dir = Path(base_feature_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT80 workers must be 1 through 64")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": source_dir / SOURCE_METADATA_NAME,
        "geometry": source_dir / SOURCE_GEOMETRY_NAME,
        "source_manifest": source_dir / SOURCE_MANIFEST_NAME,
        "base_features": base_feature_dir / BASE_FEATURES_NAME,
        "base_manifest": base_feature_dir / BASE_MANIFEST_NAME,
        "design": design_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT80 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["source_manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256
        or hashes["base_manifest"] != EXPECTED_BASE_MANIFEST_SHA256
        or hashes["base_features"] != EXPECTED_BASE_FEATURE_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT80 frozen input hash differs")
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
        or base_manifest.get("opened_internal_validation_result_used") is not False
        or base_manifest.get("internal_replication_labels_opened") is not False
        or not isinstance(base_outputs, Mapping)
        or base_outputs.get(BASE_FEATURES_NAME) != hashes["base_features"]
    ):
        raise ValueError("NEXT80 label-free provenance differs")
    metadata = pd.read_parquet(paths["metadata"])
    base = pd.read_parquet(paths["base_features"])
    material_ids = tuple(metadata["material_id"].astype(str))
    if (
        len(metadata) != len(base)
        or metadata["material_id"].duplicated().any()
        or base["material_id"].duplicated().any()
        or tuple(base["material_id"].astype(str)) != material_ids
    ):
        raise ValueError("NEXT80 base feature identity differs")
    structures = _load_archive(paths["geometry"], material_ids)
    rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    if workers == 1:
        iterator = map(_feature_record, structures)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_feature_record, structures, chunksize=1)
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if not bool(row["repulsive_load_supported"]):
                failures[str(row["repulsive_load_failure"])] += 1
            if index % 100 == 0 or index == len(structures):
                print(f"NEXT80 repulsive-load certificate: {index}/{len(structures)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    additions = pd.DataFrame(rows)
    if tuple(additions.loc[:, PRLR_FEATURE_NAMES].columns) != PRLR_FEATURE_NAMES:
        raise ValueError("NEXT80 output feature schema differs")
    table = pd.concat([base.reset_index(drop=True), additions], axis=1)
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "all_partitions_label_free_periodic_repulsive_load_resolvability",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "opened_internal_validation_result_used": False,
        "internal_replication_labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "periodic_tensegrity_equilibrium_used": True,
        "electronic_structure_calculation_used": False,
        "dft_calculation_or_value_used": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "missing_policy": "optional_family_fail_open_keep",
        "feature_columns": list(PRLR_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "base_supported": int(table["combined_supported"].sum()),
            "repulsive_load_supported": int(table["repulsive_load_supported"].sum()),
            "failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next80_periodic_repulsive_load_resolvability.py": source_hash
        },
        "environment_versions": {
            **_environment_versions(),
            "ase": importlib.metadata.version("ase"),
            "scipy": importlib.metadata.version("scipy"),
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
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT80 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT80 input changed before publication")
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
    manifest = build_repulsive_load_resolvability_batch(
        source_dir=args.source_dir,
        base_feature_dir=args.base_feature_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "FEATURES_NAME",
    "PRLR_FEATURE_NAMES",
    "PROTOCOL",
    "build_repulsive_load_resolvability_batch",
    "compute_periodic_repulsive_load_resolvability",
    "repulsive_load_resolvability_features",
]


if __name__ == "__main__":
    main()
