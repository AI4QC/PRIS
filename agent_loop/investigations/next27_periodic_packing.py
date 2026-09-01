#!/usr/bin/env python3
"""Exact periodic intermolecular contact-pressure descriptors for NEXT27."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next26_omc25 import PROTOCOL as COHORT_PROTOCOL
from src.next26_packing import _radii


PROTOCOL = "2026-08-03-next27-periodic-contact-pressure-features-v1"
FEATURES_NAME = "next27_periodic_packing_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
NEXT27_FEATURE_COLUMNS = (
    "periodic_nonbond_vdw_min",
    "periodic_nonbond_vdw_q01",
    "periodic_nonbond_vdw_q05",
    "periodic_overlap2_pa",
    "periodic_overlap3_pa",
    "periodic_repulsion12_pa",
    "periodic_contact_coord100",
    "periodic_contact_coord105",
    "periodic_contact_coord110",
    "periodic_nearest_mean",
    "periodic_nearest_q10",
    "periodic_pairs_pa",
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _strict_periodic_geometry(atoms: Atoms) -> np.ndarray:
    if atoms.calc is not None or bool(atoms.info) or set(atoms.arrays) != {
        "numbers",
        "positions",
    }:
        raise ValueError("NEXT27 features require exact geometry-only Atoms")
    numbers = np.asarray(atoms.numbers, dtype=int)
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    if (
        len(numbers) < 1
        or positions.shape != (len(numbers), 3)
        or cell.shape != (3, 3)
        or not np.all(atoms.pbc)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(cell))
        or abs(float(np.linalg.det(cell))) < 1e-12
    ):
        raise ValueError("invalid periodic geometry-only Atoms")
    return numbers


def _reachable_covalent_states(
    n_atoms: int,
    first: np.ndarray,
    second: np.ndarray,
    shifts: np.ndarray,
    bonded: np.ndarray,
) -> list[set[tuple[int, tuple[int, int, int]]]]:
    edges: dict[int, set[tuple[int, tuple[int, int, int]]]] = defaultdict(set)
    for i, j, shift in zip(first[bonded], second[bonded], shifts[bonded], strict=True):
        lattice_shift = tuple(int(value) for value in shift)
        edges[int(i)].add((int(j), lattice_shift))
        edges[int(j)].add((int(i), tuple(-value for value in lattice_shift)))
    reachable: list[set[tuple[int, tuple[int, int, int]]]] = []
    for start in range(n_atoms):
        origin = (start, (0, 0, 0))
        seen = {origin}
        frontier = {origin}
        for _depth in range(3):
            following: set[tuple[int, tuple[int, int, int]]] = set()
            for atom, base in frontier:
                for neighbour, edge_shift in edges.get(atom, ()):
                    state = (
                        neighbour,
                        tuple(base[axis] + edge_shift[axis] for axis in range(3)),
                    )
                    if state not in seen:
                        seen.add(state)
                        following.add(state)
            frontier = following
        reachable.append(seen)
    return reachable


def periodic_nonbonded_contacts(
    atoms: Atoms,
) -> list[tuple[int, int, tuple[int, int, int], float]]:
    """Return canonical `(i,j,shift,q_vdw)` contacts after exact path exclusion."""

    numbers = _strict_periodic_geometry(atoms)
    covalent, van_der_waals = _radii(numbers)
    cutoff = max(6.0, 3.2 * float(van_der_waals.max()), 2.5 * float(covalent.max()))
    first, second, shifts, distances = neighbor_list(
        "ijSd", atoms, cutoff, self_interaction=True
    )
    nonzero = (first != second) | np.any(shifts != 0, axis=1)
    first = np.asarray(first[nonzero], dtype=int)
    second = np.asarray(second[nonzero], dtype=int)
    shifts = np.asarray(shifts[nonzero], dtype=int)
    distances = np.asarray(distances[nonzero], dtype=float)
    covalent_ratio = distances / (covalent[first] + covalent[second])
    reachable = _reachable_covalent_states(
        len(numbers), first, second, shifts, covalent_ratio <= 1.25
    )
    contacts: list[tuple[int, int, tuple[int, int, int], float]] = []
    for i, j, shift_array, distance in zip(
        first, second, shifts, distances, strict=True
    ):
        shift = tuple(int(value) for value in shift_array)
        reverse_shift = tuple(-value for value in shift)
        left = (int(i), int(j), *shift)
        right = (int(j), int(i), *reverse_shift)
        if left >= right:
            continue
        if (int(j), shift) in reachable[int(i)]:
            continue
        ratio = float(distance / (van_der_waals[i] + van_der_waals[j]))
        if ratio <= 1.6:
            contacts.append((int(i), int(j), shift, ratio))
    contacts.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    if not contacts:
        raise ValueError("periodic geometry has no eligible nonbonded contact")
    return contacts


def compute_periodic_features(atoms: Atoms) -> dict[str, float]:
    contacts = periodic_nonbonded_contacts(atoms)
    n_atoms = len(atoms)
    ratios = np.asarray([row[3] for row in contacts], dtype=float)
    per_atom: list[list[float]] = [[] for _ in range(n_atoms)]
    for first, second, _shift, ratio in contacts:
        per_atom[first].append(ratio)
        per_atom[second].append(ratio)
    nearest = np.asarray(
        [min(values) if values else 1.6 for values in per_atom], dtype=float
    )
    overlap = np.maximum(0.0, 1.0 - ratios)
    repulsion = np.maximum(0.0, np.clip(ratios, 0.5, None) ** -12 - 1.0)
    values = {
        "periodic_nonbond_vdw_min": float(ratios.min()),
        "periodic_nonbond_vdw_q01": float(np.quantile(ratios, 0.01)),
        "periodic_nonbond_vdw_q05": float(np.quantile(ratios, 0.05)),
        "periodic_overlap2_pa": float(np.sum(overlap**2) / n_atoms),
        "periodic_overlap3_pa": float(np.sum(overlap**3) / n_atoms),
        "periodic_repulsion12_pa": float(np.sum(repulsion) / n_atoms),
        "periodic_contact_coord100": float(2.0 * np.sum(ratios < 1.00) / n_atoms),
        "periodic_contact_coord105": float(2.0 * np.sum(ratios < 1.05) / n_atoms),
        "periodic_contact_coord110": float(2.0 * np.sum(ratios < 1.10) / n_atoms),
        "periodic_nearest_mean": float(nearest.mean()),
        "periodic_nearest_q10": float(np.quantile(nearest, 0.10)),
        "periodic_pairs_pa": float(len(ratios) / n_atoms),
    }
    if tuple(values) != NEXT27_FEATURE_COLUMNS or not np.all(
        np.isfinite(list(values.values()))
    ):
        raise ValueError("NEXT27 periodic feature schema or values are invalid")
    return values


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid x0 manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("x0 manifest must be an object")
    return value


def build_periodic_features(
    *,
    metadata_path: Path,
    geometry_path: Path,
    cohort_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry": Path(geometry_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    hashes = {role: _sha256(path) for role, path in paths.items()}
    cohort = _read_manifest(paths["cohort_manifest"])
    outputs = cohort.get("outputs_sha256")
    if (
        cohort.get("protocol") != COHORT_PROTOCOL
        or cohort.get("labels_opened") is not False
        or cohort.get("endpoint_numeric_fields_parsed") is not False
        or cohort.get("model_or_proxy_potential_used") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(paths["metadata"].name) != hashes["metadata"]
        or outputs.get(paths["geometry"].name) != hashes["geometry"]
    ):
        raise ValueError("x0 cohort crossed the no-DFT boundary")
    metadata = pd.read_parquet(paths["metadata"])
    if (
        "material_id" not in metadata
        or metadata["material_id"].isna().any()
        or metadata["material_id"].duplicated().any()
        or not metadata["input_role"].eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("invalid x0 metadata")
    ids, frames = _load_archive_only(
        paths["geometry"], tuple(metadata["material_id"].astype(str))
    )
    if ids != metadata["material_id"].astype(str).tolist():
        raise ValueError("geometry and metadata order differ")
    features = pd.DataFrame(
        [
            {"material_id": material_id, **compute_periodic_features(atoms)}
            for material_id, atoms in zip(ids, frames, strict=True)
        ]
    )
    features["analytic_supported"] = np.isfinite(
        features.loc[:, NEXT27_FEATURE_COLUMNS]
    ).all(axis=1)
    source_hashes = {
        "src/next27_periodic_packing.py": _sha256(Path(__file__).resolve())
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "relaxed_structures_opened": False,
        "model_or_proxy_potential_used": False,
        "same_composition_candidates_used": False,
        "feature_columns": list(NEXT27_FEATURE_COLUMNS),
        "counts": {
            "rows": len(features),
            "supported": int(features["analytic_supported"].sum()),
        },
        "inputs_sha256": {
            role: {"path": str(path), "sha256": hashes[role]}
            for role, path in paths.items()
        },
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        output_path = staging / FEATURES_NAME
        features.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in paths.items():
            if _sha256(path) != hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        if _sha256(Path(__file__).resolve()) != source_hashes[
            "src/next27_periodic_packing.py"
        ]:
            raise RuntimeError("NEXT27 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_periodic_features(
        metadata_path=args.metadata,
        geometry_path=args.geometry,
        cohort_manifest_path=args.cohort_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FEATURES_NAME",
    "NEXT27_FEATURE_COLUMNS",
    "build_periodic_features",
    "compute_periodic_features",
    "periodic_nonbonded_contacts",
]
