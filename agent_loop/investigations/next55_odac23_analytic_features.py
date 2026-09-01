#!/usr/bin/env python3
"""Label-free analytic mechanics descriptors for selected ODAC23 framework x0."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from ase import Atoms
from ase.data import chemical_symbols
import numpy as np
import pandas as pd
from pymatgen.core import Element

from src.next11_geometry_only_frames import _parse_frame
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next26_packing import _radii
from src.next49_framework_topology import (
    FRAMEWORK_FEATURE_NAMES,
    _canonical_covalent_edges,
    _directed_adjacency,
    _environment_versions,
    _is_metal,
    _strict_geometry,
    compute_framework_topology_features,
)
from src.next54_odac23_train_selection import (
    GEOMETRY_NAME as SOURCE_GEOMETRY_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    METADATA_NAME as SOURCE_METADATA_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next55-odac23-analytic-mechanics-features-v1"
DESIGN_SHA256 = "f938c9bebf39191e3e758d9a2bb12355fbed696fafca521acd60887b32041378"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
FEATURES_NAME = "next55_odac23_analytic_mechanics_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
DONOR_NUMBERS = frozenset((7, 8, 9, 15, 16, 17, 35, 53))
ANALYTIC_FEATURE_NAMES = (
    "hydrogen_fraction",
    "carbon_fraction",
    "donor_fraction",
    "heavy_nonmetal_fraction",
    "atomic_number_mean",
    "atomic_number_std",
    "electronegativity_mean",
    "electronegativity_std",
    "atom_density",
    "volume_per_atom",
    "degree_std",
    "degree_q10",
    "degree_q90",
    "degree_one_fraction",
    "degree_two_fraction",
    "low_degree_heavy_fraction",
    "organic_degree_two_fraction",
    "edge_excess_per_atom",
    "degree2_bend_mean",
    "degree2_bend_q95",
    "degree2_bent_fraction",
    "heteroatomic_edge_fraction",
    "metal_donor_edge_fraction",
    "organic_organic_edge_fraction",
    "donor_metal_contact_fraction",
    "metal_neighbor_diversity_mean",
    "electronegativity_edge_difference_mean",
    "electronegativity_edge_difference_q95",
    "bond_ratio_mean",
    "bond_ratio_std",
    "bond_ratio_max",
    "short_bond_fraction",
    "long_bond_fraction",
    "bond_orientation_lambda_min",
    "bond_orientation_lambda_max",
    "bond_orientation_anisotropy",
)
NEXT55_FEATURE_NAMES = tuple(FRAMEWORK_FEATURE_NAMES) + ANALYTIC_FEATURE_NAMES
_FORBIDDEN_TOKENS = ("energy", "force", "stress", "relax", "dft", "label", "target")


@dataclass(frozen=True)
class ODACAnalyticResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _quantile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    return float(np.quantile(values, q)) if len(values) else float(fallback)


def _electronegativities(numbers: np.ndarray) -> np.ndarray:
    values = []
    for number in numbers:
        value = Element(chemical_symbols[int(number)]).X
        values.append(float(value) if value is not None and math.isfinite(value) else 0.0)
    return np.asarray(values, dtype=float)


def compute_odac23_analytic_features(atoms: Atoms) -> ODACAnalyticResult:
    """Compute topology plus finite analytic mechanics descriptors from one x0."""

    try:
        topology = compute_framework_topology_features(atoms)
        if not topology.supported:
            raise ValueError(topology.failure_reason or "topology unsupported")
        numbers, _positions, cell = _strict_geometry(atoms)
        covalent, _vdw = _radii(numbers)
        covalent = np.asarray(covalent, dtype=float)
        edges = _canonical_covalent_edges(atoms, covalent)
        adjacency = _directed_adjacency(len(atoms), edges)
        degrees = np.asarray([len(row) for row in adjacency], dtype=float)
        metal = np.asarray([_is_metal(number) for number in numbers], dtype=bool)
        donor = np.asarray([int(number) in DONOR_NUMBERS for number in numbers], dtype=bool)
        hydrogen = numbers == 1
        carbon = numbers == 6
        heavy_nonmetal = (~metal) & (~hydrogen)
        electronegativity = _electronegativities(numbers)
        volume = abs(float(np.linalg.det(cell)))

        ratios = np.asarray([edge.ratio for edge in edges], dtype=float)
        hetero = np.asarray(
            [numbers[edge.first] != numbers[edge.second] for edge in edges], dtype=bool
        )
        metal_donor = np.asarray(
            [
                (metal[edge.first] and donor[edge.second])
                or (metal[edge.second] and donor[edge.first])
                for edge in edges
            ],
            dtype=bool,
        )
        organic_organic = np.asarray(
            [heavy_nonmetal[edge.first] and heavy_nonmetal[edge.second] for edge in edges],
            dtype=bool,
        )
        en_differences = np.asarray(
            [
                abs(electronegativity[edge.first] - electronegativity[edge.second])
                for edge in edges
            ],
            dtype=float,
        )

        donor_contacts = []
        for index in np.flatnonzero(donor):
            donor_contacts.append(
                any(metal[neighbour] for neighbour, _shift, _vector, _ratio in adjacency[index])
            )
        metal_diversity = []
        for index in np.flatnonzero(metal):
            neighbours = adjacency[index]
            metal_diversity.append(
                len({int(numbers[row[0]]) for row in neighbours}) / len(neighbours)
                if neighbours
                else 0.0
            )

        degree2_bends = []
        for index in np.flatnonzero(heavy_nonmetal & (degrees == 2.0)):
            vectors = [row[2] for row in adjacency[index]]
            units = [vector / np.linalg.norm(vector) for vector in vectors]
            cosine = float(np.clip(np.dot(units[0], units[1]), -1.0, 1.0))
            degree2_bends.append(0.5 * (1.0 + cosine))
        bends = np.asarray(degree2_bends, dtype=float)

        directions = np.asarray(
            [edge.vector / np.linalg.norm(edge.vector) for edge in edges], dtype=float
        )
        orientation = np.mean(
            np.einsum("ni,nj->nij", directions, directions), axis=0
        )
        eigenvalues = np.linalg.eigvalsh(orientation)

        analytic = {
            "hydrogen_fraction": float(np.mean(hydrogen)),
            "carbon_fraction": float(np.mean(carbon)),
            "donor_fraction": float(np.mean(donor)),
            "heavy_nonmetal_fraction": float(np.mean(heavy_nonmetal)),
            "atomic_number_mean": float(np.mean(numbers)),
            "atomic_number_std": float(np.std(numbers)),
            "electronegativity_mean": float(np.mean(electronegativity)),
            "electronegativity_std": float(np.std(electronegativity)),
            "atom_density": float(len(atoms) / volume),
            "volume_per_atom": float(volume / len(atoms)),
            "degree_std": float(np.std(degrees)),
            "degree_q10": _quantile(degrees, 0.10),
            "degree_q90": _quantile(degrees, 0.90),
            "degree_one_fraction": float(np.mean(degrees == 1.0)),
            "degree_two_fraction": float(np.mean(degrees == 2.0)),
            "low_degree_heavy_fraction": float(np.mean(heavy_nonmetal & (degrees <= 1.0))),
            "organic_degree_two_fraction": float(np.mean(heavy_nonmetal & (degrees == 2.0))),
            "edge_excess_per_atom": float(len(edges) / len(atoms) - 1.0),
            "degree2_bend_mean": float(np.mean(bends)) if len(bends) else 0.0,
            "degree2_bend_q95": _quantile(bends, 0.95),
            "degree2_bent_fraction": float(np.mean(bends > 0.15)) if len(bends) else 0.0,
            "heteroatomic_edge_fraction": float(np.mean(hetero)),
            "metal_donor_edge_fraction": float(np.mean(metal_donor)),
            "organic_organic_edge_fraction": float(np.mean(organic_organic)),
            "donor_metal_contact_fraction": float(np.mean(donor_contacts))
            if donor_contacts
            else 0.0,
            "metal_neighbor_diversity_mean": float(np.mean(metal_diversity))
            if metal_diversity
            else 0.0,
            "electronegativity_edge_difference_mean": float(np.mean(en_differences)),
            "electronegativity_edge_difference_q95": _quantile(en_differences, 0.95),
            "bond_ratio_mean": float(np.mean(ratios)),
            "bond_ratio_std": float(np.std(ratios)),
            "bond_ratio_max": float(np.max(ratios)),
            "short_bond_fraction": float(np.mean(ratios < 0.75)),
            "long_bond_fraction": float(np.mean(ratios > 1.15)),
            "bond_orientation_lambda_min": float(eigenvalues[0]),
            "bond_orientation_lambda_max": float(eigenvalues[-1]),
            "bond_orientation_anisotropy": float(eigenvalues[-1] - eigenvalues[0]),
        }
        values = {
            **{name: float(topology.features[name]) for name in FRAMEWORK_FEATURE_NAMES},
            **analytic,
        }
        if (
            tuple(analytic) != ANALYTIC_FEATURE_NAMES
            or tuple(values) != NEXT55_FEATURE_NAMES
            or any(token in name for name in NEXT55_FEATURE_NAMES for token in _FORBIDDEN_TOKENS)
            or not np.isfinite(list(values.values())).all()
        ):
            raise ValueError("NEXT55 analytic feature schema differs")
        return ODACAnalyticResult(True, None, values)
    except Exception as exc:
        return ODACAnalyticResult(False, f"{type(exc).__name__}: {exc}", {})


def _feature_record(atoms: Atoms) -> dict[str, object]:
    result = compute_odac23_analytic_features(atoms)
    row: dict[str, object] = {
        "analytic_supported": result.supported,
        "analytic_failure": result.failure_reason,
    }
    row.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in NEXT55_FEATURE_NAMES
        }
    )
    return row


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


def _load_archive(path: Path, material_ids: tuple[str, ...]) -> list[Atoms]:
    structures = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        expected_names = [f"{material_id}.extxyz" for material_id in material_ids]
        if [info.filename for info in infos] != expected_names:
            raise ValueError("NEXT55 geometry exact identity/order differs")
        for info in infos:
            if PurePosixPath(info.filename).parent != PurePosixPath("."):
                raise ValueError("NEXT55 geometry member is not root-level")
            parsed = _parse_frame(archive.read(info), strict_output=True)
            if parsed.atoms.calc is not None or parsed.atoms.info or set(parsed.atoms.arrays) != {
                "numbers",
                "positions",
            }:
                raise ValueError("NEXT55 geometry retained forbidden metadata")
            structures.append(parsed.atoms)
    return structures


def build_odac23_analytic_feature_batch(
    *, source_dir: Path, design_path: Path, output_dir: Path, workers: int = 1
) -> dict[str, object]:
    """Seal NEXT55 x0-only features without opening the selected label table."""

    source_dir = Path(source_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT55 workers must be 1 through 64")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": source_dir / SOURCE_METADATA_NAME,
        "geometry": source_dir / SOURCE_GEOMETRY_NAME,
        "manifest": source_dir / SOURCE_MANIFEST_NAME,
        "design": design_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT55 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes["manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256 or hashes["design"] != DESIGN_SHA256:
        raise ValueError("NEXT55 frozen input hash differs")
    source_manifest = _strict_json(paths["manifest"])
    source_outputs = source_manifest.get("outputs_sha256")
    if (
        source_manifest.get("protocol") != SOURCE_PROTOCOL
        or source_manifest.get("selection_frozen_before_row_labels_opened") is not True
        or source_manifest.get("validation_or_test_payload_deserialized") is not False
        or not isinstance(source_outputs, Mapping)
        or source_outputs.get(paths["metadata"].name) != hashes["metadata"]
        or source_outputs.get(paths["geometry"].name) != hashes["geometry"]
    ):
        raise ValueError("NEXT55 source provenance differs")
    metadata = pd.read_parquet(paths["metadata"])
    required = {"material_id", "framework_name", "partition_role", "input_role"}
    if (
        metadata.empty
        or not required.issubset(metadata.columns)
        or metadata["material_id"].duplicated().any()
        or metadata["framework_name"].duplicated().any()
        or set(metadata["partition_role"]) != {
            "discovery",
            "internal_validation",
            "internal_replication",
        }
    ):
        raise ValueError("NEXT55 selected metadata differs")
    material_ids = tuple(metadata["material_id"].astype(str))
    structures = _load_archive(paths["geometry"], material_ids)

    rows = []
    failures: Counter[str] = Counter()
    if workers == 1:
        iterator = map(_feature_record, structures)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_feature_record, structures, chunksize=4)
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if not bool(row["analytic_supported"]):
                failures[str(row["analytic_failure"] or "unsupported")] += 1
            if index % 100 == 0 or index == len(structures):
                print(f"NEXT55 ODAC23 analytic features: {index}/{len(structures)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    table = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    table["combined_supported"] = table["analytic_supported"].astype(bool) & np.isfinite(
        table.loc[:, NEXT55_FEATURE_NAMES]
    ).all(axis=1)

    source_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "selected_odac23_train_label_free_analytic_mechanics",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "model_or_proxy_potential_used": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "missing_policy": "fail_open_keep",
        "feature_columns": list(NEXT55_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "supported": int(table["combined_supported"].sum()),
            "failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next55_odac23_analytic_features.py": _sha256(source_path),
            "src/next49_framework_topology.py": _sha256(
                Path(__import__("src.next49_framework_topology", fromlist=["x"]).__file__)
            ),
        },
        "environment_versions": {
            **_environment_versions(),
            "matminer": importlib.metadata.version("matminer"),
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
            "src/next55_odac23_analytic_features.py"
        ]:
            raise RuntimeError("NEXT55 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT55 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_odac23_analytic_feature_batch(
        source_dir=args.source_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "ANALYTIC_FEATURE_NAMES",
    "FEATURES_NAME",
    "MANIFEST_NAME",
    "NEXT55_FEATURE_NAMES",
    "ODACAnalyticResult",
    "PROTOCOL",
    "build_odac23_analytic_feature_batch",
    "compute_odac23_analytic_features",
]


if __name__ == "__main__":
    main()
