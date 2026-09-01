#!/usr/bin/env python3
"""Deterministic, x0-only molecular-crystal packing descriptors for NEXT26."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
from ase.data import atomic_masses, covalent_radii, vdw_radii
from ase.geometry import find_mic
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next26_omc25 import GEOMETRY_NAME, MANIFEST_NAME as COHORT_MANIFEST_NAME, METADATA_NAME, PROTOCOL as COHORT_PROTOCOL


PROTOCOL = "2026-08-03-next26-molecular-packing-features-v1"
FEATURES_NAME = "next26_packing_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
FEATURE_COLUMNS = (
    "volume_pa",
    "density_proxy",
    "cov_packing",
    "vdw_packing",
    "cell_anisotropy",
    "cell_angle_dev",
    "cov_ratio_min",
    "cov_ratio_q01",
    "bond_ratio_sd",
    "bond_ratio_absdev",
    "nonbond_vdw_min",
    "nonbond_vdw_q01",
    "nonbond_vdw_q05",
    "nonbond_clash_frac075",
    "nonbond_clash_frac085",
)
FORBIDDEN_TOKENS = ("energy", "force", "stress", "relax", "dft", "label", "endpoint", "mlip")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _strict_geometry(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if atoms.calc is not None or bool(atoms.info) or set(atoms.arrays) != {"numbers", "positions"}:
        raise ValueError("NEXT26 features require exact geometry-only Atoms")
    numbers = np.asarray(atoms.numbers, dtype=int)
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    if (
        len(numbers) < 2
        or positions.shape != (len(numbers), 3)
        or cell.shape != (3, 3)
        or not np.all(atoms.pbc)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(cell))
        or abs(float(np.linalg.det(cell))) < 1e-12
    ):
        raise ValueError("invalid periodic geometry-only Atoms")
    return numbers, positions, cell


def _radii(numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covalent = np.asarray(covalent_radii[numbers], dtype=float)
    van_der_waals = np.asarray(
        [
            vdw_radii[number]
            if number < len(vdw_radii) and np.isfinite(vdw_radii[number])
            else 1.70
            for number in numbers
        ],
        dtype=float,
    )
    if np.any(covalent <= 0) or np.any(van_der_waals <= 0):
        raise ValueError("tabulated elemental radii are unavailable")
    return covalent, van_der_waals


def compute_packing_features(atoms: Atoms) -> dict[str, float]:
    """Compute permutation- and wrapping-invariant analytic x0 descriptors."""

    numbers, positions, cell = _strict_geometry(atoms)
    n = len(numbers)
    volume = abs(float(np.linalg.det(cell)))
    covalent, van_der_waals = _radii(numbers)
    first, second = np.triu_indices(n, 1)
    vectors = positions[second] - positions[first]
    minimum_image, _ = find_mic(vectors, cell, pbc=True)
    distances = np.linalg.norm(minimum_image, axis=1)
    if np.any(~np.isfinite(distances)) or np.any(distances <= 0):
        raise ValueError("geometry contains coincident or invalid atoms")
    cov_ratio = distances / (covalent[first] + covalent[second])
    vdw_ratio = distances / (van_der_waals[first] + van_der_waals[second])

    bonded = cov_ratio <= 1.25
    adjacency = np.zeros((n, n), dtype=np.int8)
    adjacency[first[bonded], second[bonded]] = 1
    adjacency[second[bonded], first[bonded]] = 1
    graph_distance = shortest_path(csr_matrix(adjacency), directed=False, unweighted=True)
    remote = graph_distance[first, second] > 3
    nonbonded = vdw_ratio[remote]
    bonded_ratio = cov_ratio[bonded]
    if not len(bonded_ratio) or not len(nonbonded):
        raise ValueError("packing geometry lacks bonded or nonbonded pairs")

    lengths = np.linalg.norm(cell, axis=1)
    cosine = np.clip(
        np.asarray(
            [
                np.dot(cell[1], cell[2]) / (lengths[1] * lengths[2]),
                np.dot(cell[0], cell[2]) / (lengths[0] * lengths[2]),
                np.dot(cell[0], cell[1]) / (lengths[0] * lengths[1]),
            ]
        ),
        -1.0,
        1.0,
    )
    angles = np.degrees(np.arccos(cosine))
    values = {
        "volume_pa": volume / n,
        "density_proxy": float(atomic_masses[numbers].sum() / volume),
        "cov_packing": float((4.0 * np.pi / 3.0) * np.sum(covalent**3) / volume),
        "vdw_packing": float((4.0 * np.pi / 3.0) * np.sum(van_der_waals**3) / volume),
        "cell_anisotropy": float(lengths.max() / lengths.min()),
        "cell_angle_dev": float(np.max(np.abs(angles - 90.0))),
        "cov_ratio_min": float(cov_ratio.min()),
        "cov_ratio_q01": float(np.quantile(cov_ratio, 0.01)),
        "bond_ratio_sd": float(bonded_ratio.std()),
        "bond_ratio_absdev": float(np.mean(np.abs(bonded_ratio - 1.0))),
        "nonbond_vdw_min": float(nonbonded.min()),
        "nonbond_vdw_q01": float(np.quantile(nonbonded, 0.01)),
        "nonbond_vdw_q05": float(np.quantile(nonbonded, 0.05)),
        "nonbond_clash_frac075": float(np.mean(nonbonded < 0.75)),
        "nonbond_clash_frac085": float(np.mean(nonbonded < 0.85)),
    }
    if tuple(values) != FEATURE_COLUMNS or not np.all(np.isfinite(list(values.values()))):
        raise ValueError("packing features are nonfinite or have the wrong schema")
    return values


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid NEXT26 cohort manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("cohort manifest must be an object")
    return value


def build_packing_features(
    *,
    metadata_path: Path,
    geometry_path: Path,
    cohort_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build and checksum-lock the exact DFT-free feature table."""

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
    cohort_manifest = _read_manifest(paths["cohort_manifest"])
    outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("labels_opened") is not False
        or cohort_manifest.get("endpoint_numeric_fields_parsed") is not False
        or cohort_manifest.get("model_or_proxy_potential_used") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(paths["metadata"].name) != hashes["metadata"]
        or outputs.get(paths["geometry"].name) != hashes["geometry"]
    ):
        raise ValueError("cohort manifest crossed the no-DFT boundary")
    metadata = pd.read_parquet(paths["metadata"])
    required = {"material_id", "natoms", "input_role"}
    if (
        not required.issubset(metadata.columns)
        or metadata.empty
        or metadata["material_id"].isna().any()
        or metadata["material_id"].duplicated().any()
        or not metadata["input_role"].eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("invalid NEXT26 x0 metadata")
    ids, atoms = _load_archive_only(paths["geometry"], tuple(metadata["material_id"].astype(str)))
    if ids != sorted(ids) or ids != metadata["material_id"].astype(str).tolist():
        raise ValueError("geometry and metadata identity/order differ")
    rows = [{"material_id": material_id, **compute_packing_features(frame)} for material_id, frame in zip(ids, atoms, strict=True)]
    features = pd.DataFrame(rows)
    features["analytic_supported"] = np.isfinite(features.loc[:, FEATURE_COLUMNS]).all(axis=1)
    source_hashes = {"src/next26_packing.py": _sha256(Path(__file__).resolve())}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "relaxed_structures_opened": False,
        "model_or_proxy_potential_used": False,
        "same_composition_candidates_used": False,
        "feature_columns": list(FEATURE_COLUMNS),
        "counts": {"rows": len(features), "supported": int(features["analytic_supported"].sum())},
        "inputs_sha256": {role: {"path": str(path), "sha256": hashes[role]} for role, path in paths.items()},
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / FEATURES_NAME
        features.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in paths.items():
            if _sha256(path) != hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        if _sha256(Path(__file__).resolve()) != source_hashes["src/next26_packing.py"]:
            raise RuntimeError("feature source changed before publication")
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
    result = build_packing_features(
        metadata_path=args.metadata,
        geometry_path=args.geometry,
        cohort_manifest_path=args.cohort_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FEATURE_COLUMNS", "FEATURES_NAME", "build_packing_features", "compute_packing_features"]
