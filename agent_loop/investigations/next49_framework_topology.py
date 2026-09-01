#!/usr/bin/env python3
"""Pure-x0 periodic-topology and local-coordination features for NEXT49.

The module accepts one unmodified periodic structure and frozen elemental
radii.  It never reads calculated properties, alternative structures, or a
relaxation endpoint.  Unsupported inputs fail open.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile

from ase import Atoms
from ase.data import chemical_symbols
from ase.neighborlist import neighbor_list
import numpy as np
import pandas as pd
from pymatgen.core import Element

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next26_packing import _radii


PROTOCOL = "2026-08-03-next49-framework-topology-features-v1"
FEATURES_NAME = "next49_qmof_framework_topology_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
COVALENT_EDGE_RATIO_MAX = 1.25
FRAMEWORK_FEATURE_NAMES = (
    "periodic_dimension_max",
    "periodic_dimension_atom_mean",
    "periodic_framework_fraction",
    "covalent_edges_pa",
    "covalent_ratio_q05",
    "covalent_ratio_q95",
    "covalent_ratio_iqr",
    "metal_fraction",
    "metal_coord_mean",
    "metal_coord_q10",
    "metal_coord_q90",
    "metal_undercoord_fraction",
    "metal_ligand_ratio_q05",
    "metal_ligand_ratio_q95",
    "metal_ligand_ratio_iqr",
    "metal_vector_imbalance_mean",
    "metal_vector_imbalance_q95",
    "metal_bond_spread_q95",
    "covalent_packing_fraction",
    "vdw_packing_fraction",
)
_FORBIDDEN_FEATURE_TOKENS = (
    "energy",
    "force",
    "stress",
    "relax",
    "dft",
    "endpoint",
    "label",
    "target",
    "mattersim",
)


@dataclass(frozen=True)
class FrameworkTopologyResult:
    """Fail-open result for one raw periodic geometry."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


@dataclass(frozen=True)
class _Edge:
    first: int
    second: int
    shift: tuple[int, int, int]
    vector: np.ndarray
    ratio: float


def _environment_versions() -> dict[str, str]:
    """Return the exact geometry-stack versions bound into sealed artifacts."""

    return {
        name: importlib.metadata.version(name)
        for name in ("ase", "numpy", "pandas", "pymatgen")
    }


def _strict_geometry(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if atoms.calc is not None or bool(atoms.info) or set(atoms.arrays) != {
        "numbers",
        "positions",
    }:
        raise ValueError("NEXT49 requires exact geometry-only Atoms")
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
        or abs(float(np.linalg.det(cell))) <= 1e-12
    ):
        raise ValueError("invalid periodic geometry-only Atoms")
    return numbers, positions, cell


def _is_metal(number: int) -> bool:
    try:
        return bool(Element(chemical_symbols[int(number)]).is_metal)
    except (ValueError, KeyError, IndexError):
        return False


def _canonical_covalent_edges(
    atoms: Atoms, covalent: np.ndarray
) -> list[_Edge]:
    first, second, shifts, vectors = neighbor_list(
        "ijSD",
        atoms,
        COVALENT_EDGE_RATIO_MAX * covalent,
        self_interaction=True,
    )
    rows: list[_Edge] = []
    for raw_i, raw_j, raw_shift, raw_vector in zip(
        first, second, shifts, vectors, strict=True
    ):
        i = int(raw_i)
        j = int(raw_j)
        shift = tuple(int(value) for value in raw_shift)
        if i == j and shift == (0, 0, 0):
            continue
        reverse = tuple(-value for value in shift)
        if (i, j, *shift) >= (j, i, *reverse):
            continue
        vector = np.asarray(raw_vector, dtype=float)
        distance = float(np.linalg.norm(vector))
        denominator = float(covalent[i] + covalent[j])
        ratio = distance / denominator
        if (
            distance <= 1e-10
            or not math.isfinite(ratio)
            or ratio > COVALENT_EDGE_RATIO_MAX + 1e-10
        ):
            continue
        rows.append(_Edge(i, j, shift, vector, ratio))
    rows.sort(key=lambda edge: (edge.first, edge.second, edge.shift, edge.ratio))
    if not rows:
        raise ValueError("periodic geometry has no covalent-radius graph edge")
    return rows


def _directed_adjacency(
    n_atoms: int, edges: list[_Edge]
) -> list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]]:
    adjacency: list[
        list[tuple[int, tuple[int, int, int], np.ndarray, float]]
    ] = [[] for _ in range(n_atoms)]
    for edge in edges:
        reverse_shift = tuple(-value for value in edge.shift)
        adjacency[edge.first].append(
            (edge.second, edge.shift, edge.vector, edge.ratio)
        )
        adjacency[edge.second].append(
            (edge.first, reverse_shift, -edge.vector, edge.ratio)
        )
    for neighbours in adjacency:
        neighbours.sort(key=lambda row: (row[0], row[1], row[3]))
    return adjacency


def _component_dimensions(
    adjacency: list[list[tuple[int, tuple[int, int, int], np.ndarray, float]]],
) -> list[tuple[tuple[int, ...], int]]:
    """Return quotient components and ranks of their translation subspaces."""

    assigned: dict[int, np.ndarray] = {}
    result: list[tuple[tuple[int, ...], int]] = []
    for start in range(len(adjacency)):
        if start in assigned:
            continue
        assigned[start] = np.zeros(3, dtype=int)
        queue: deque[int] = deque([start])
        members: list[int] = []
        residuals: list[np.ndarray] = []
        while queue:
            atom = queue.popleft()
            members.append(atom)
            base = assigned[atom]
            for neighbour, shift, _vector, _ratio in adjacency[atom]:
                candidate = base + np.asarray(shift, dtype=int)
                if neighbour not in assigned:
                    assigned[neighbour] = candidate
                    queue.append(neighbour)
                else:
                    residual = candidate - assigned[neighbour]
                    if np.any(residual):
                        residuals.append(residual)
        dimension = (
            int(np.linalg.matrix_rank(np.asarray(residuals, dtype=float), tol=1e-9))
            if residuals
            else 0
        )
        result.append((tuple(sorted(members)), dimension))
    return result


def _quantile(values: np.ndarray, probability: float, fallback: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else float(fallback)


def compute_framework_topology_features(atoms: Atoms) -> FrameworkTopologyResult:
    """Compute deterministic periodic graph and local coordination descriptors."""

    try:
        numbers, _positions, cell = _strict_geometry(atoms)
        covalent, van_der_waals = _radii(numbers)
        covalent = np.asarray(covalent, dtype=float)
        van_der_waals = np.asarray(van_der_waals, dtype=float)
        if (
            covalent.shape != (len(atoms),)
            or van_der_waals.shape != (len(atoms),)
            or not np.all(np.isfinite(covalent))
            or not np.all(np.isfinite(van_der_waals))
            or np.any(covalent <= 0)
            or np.any(van_der_waals <= 0)
        ):
            raise ValueError("frozen elemental radii are unavailable")

        edges = _canonical_covalent_edges(atoms, covalent)
        adjacency = _directed_adjacency(len(atoms), edges)
        components = _component_dimensions(adjacency)
        component_dimensions = np.asarray(
            [dimension for _members, dimension in components], dtype=float
        )
        component_sizes = np.asarray(
            [len(members) for members, _dimension in components], dtype=float
        )
        periodic_mask = component_dimensions > 0

        ratios = np.asarray([edge.ratio for edge in edges], dtype=float)
        metal_mask = np.asarray([_is_metal(number) for number in numbers], dtype=bool)
        metal_indices = np.flatnonzero(metal_mask)
        metal_coord = np.asarray(
            [len(adjacency[int(index)]) for index in metal_indices], dtype=float
        )

        metal_ligand_ratios = np.asarray(
            [
                edge.ratio
                for edge in edges
                if bool(metal_mask[edge.first]) != bool(metal_mask[edge.second])
            ],
            dtype=float,
        )
        vector_imbalances: list[float] = []
        bond_spreads: list[float] = []
        for raw_index in metal_indices:
            neighbours = adjacency[int(raw_index)]
            if not neighbours:
                vector_imbalances.append(1.0)
                bond_spreads.append(0.0)
                continue
            units = np.asarray(
                [vector / np.linalg.norm(vector) for _j, _s, vector, _r in neighbours]
            )
            vector_imbalances.append(float(np.linalg.norm(units.sum(axis=0)) / len(units)))
            site_ratios = np.asarray([ratio for _j, _s, _v, ratio in neighbours])
            bond_spreads.append(
                float(np.std(site_ratios) / np.mean(site_ratios))
                if len(site_ratios) >= 2
                else 0.0
            )
        imbalance = np.asarray(vector_imbalances, dtype=float)
        spreads = np.asarray(bond_spreads, dtype=float)

        volume = abs(float(np.linalg.det(cell)))
        sphere_factor = 4.0 * math.pi / 3.0
        values = {
            "periodic_dimension_max": float(component_dimensions.max()),
            "periodic_dimension_atom_mean": float(
                np.sum(component_dimensions * component_sizes) / len(atoms)
            ),
            "periodic_framework_fraction": float(
                np.sum(component_sizes[periodic_mask]) / len(atoms)
            ),
            "covalent_edges_pa": float(2.0 * len(edges) / len(atoms)),
            "covalent_ratio_q05": _quantile(ratios, 0.05, 1.0),
            "covalent_ratio_q95": _quantile(ratios, 0.95, 1.0),
            "covalent_ratio_iqr": _quantile(ratios, 0.75, 1.0)
            - _quantile(ratios, 0.25, 1.0),
            "metal_fraction": float(np.mean(metal_mask)),
            "metal_coord_mean": float(np.mean(metal_coord)) if len(metal_coord) else 0.0,
            "metal_coord_q10": _quantile(metal_coord, 0.10, 0.0),
            "metal_coord_q90": _quantile(metal_coord, 0.90, 0.0),
            "metal_undercoord_fraction": float(np.mean(metal_coord < 2.0))
            if len(metal_coord)
            else 0.0,
            "metal_ligand_ratio_q05": _quantile(metal_ligand_ratios, 0.05, 1.0),
            "metal_ligand_ratio_q95": _quantile(metal_ligand_ratios, 0.95, 1.0),
            "metal_ligand_ratio_iqr": _quantile(metal_ligand_ratios, 0.75, 1.0)
            - _quantile(metal_ligand_ratios, 0.25, 1.0),
            "metal_vector_imbalance_mean": float(np.mean(imbalance))
            if len(imbalance)
            else 0.0,
            "metal_vector_imbalance_q95": _quantile(imbalance, 0.95, 0.0),
            "metal_bond_spread_q95": _quantile(spreads, 0.95, 0.0),
            "covalent_packing_fraction": float(
                sphere_factor * np.sum(covalent**3) / volume
            ),
            "vdw_packing_fraction": float(
                sphere_factor * np.sum(van_der_waals**3) / volume
            ),
        }
        if (
            tuple(values) != FRAMEWORK_FEATURE_NAMES
            or any(
                token in name
                for name in FRAMEWORK_FEATURE_NAMES
                for token in _FORBIDDEN_FEATURE_TOKENS
            )
            or not np.all(np.isfinite(list(values.values())))
        ):
            raise ValueError("NEXT49 framework feature schema or values are invalid")
        return FrameworkTopologyResult(True, None, values)
    except Exception as exc:
        return FrameworkTopologyResult(False, f"{type(exc).__name__}: {exc}", {})


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


def build_qmof_framework_feature_batch(
    *,
    geometry_path: Path,
    next48_features_path: Path,
    next48_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build and atomically publish label-free NEXT49 features from sealed x0."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "geometry": Path(geometry_path).resolve(),
        "next48_features": Path(next48_features_path).resolve(),
        "next48_manifest": Path(next48_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT49 sealed x0 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    manifest48 = _strict_json(paths["next48_manifest"], role="NEXT48 manifest")
    outputs48 = manifest48.get("outputs_sha256")
    if (
        manifest48.get("input_role") != "unrelaxed_x0_geometry_only"
        or manifest48.get("labels_opened") is not False
        or manifest48.get("relaxed_coordinate_payloads_opened") is not False
        or manifest48.get("model_or_proxy_potential_used") is not False
        or manifest48.get("dft_or_energy_proxy_used_at_execution") is not False
        or not isinstance(outputs48, Mapping)
        or outputs48.get(paths["geometry"].name) != hashes["geometry"]
        or outputs48.get(paths["next48_features"].name) != hashes["next48_features"]
    ):
        raise ValueError("NEXT49 input crossed the geometry-only boundary")

    identities = pd.read_parquet(
        paths["next48_features"], columns=["material_id", "source_family"]
    )
    if (
        identities.empty
        or identities["material_id"].isna().any()
        or identities["material_id"].duplicated().any()
        or identities["source_family"].isna().any()
    ):
        raise ValueError("NEXT49 identities are invalid")
    material_ids = identities["material_id"].astype(str).tolist()
    archive_ids, frames = _load_archive_only(paths["geometry"], tuple(material_ids))
    if archive_ids != material_ids:
        raise ValueError("NEXT49 geometry and identities differ")

    records: list[dict[str, object]] = []
    for material_id, source_family, atoms in zip(
        material_ids, identities["source_family"].astype(str), frames, strict=True
    ):
        result = compute_framework_topology_features(atoms)
        record: dict[str, object] = {
            "material_id": material_id,
            "source_family": source_family,
            "framework_feature_supported": result.supported,
            "framework_feature_error": result.failure_reason,
        }
        record.update(
            {
                name: float(result.features[name]) if result.supported else math.nan
                for name in FRAMEWORK_FEATURE_NAMES
            }
        )
        records.append(record)
    table = pd.DataFrame(records)
    if tuple(table.columns[4:]) != FRAMEWORK_FEATURE_NAMES:
        raise ValueError("NEXT49 batch feature schema differs")

    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    output_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "exposed_qmof_label_free_framework_feature_build",
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "model_or_proxy_potential_used": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "missing_policy": "fail_open_do_not_reject",
        "feature_columns": list(FRAMEWORK_FEATURE_NAMES),
        "counts": {
            "rows": len(table),
            "supported": int(table["framework_feature_supported"].sum()),
            "source_families": {
                str(key): int(value)
                for key, value in table["source_family"].value_counts().sort_index().items()
            },
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next49_framework_topology.py": source_hash
        },
        "environment_versions": _environment_versions(),
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / FEATURES_NAME
        table.to_parquet(output_path, index=False)
        output_manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(output_manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT49 source changed before publication")
        for name, path in paths.items():
            if _sha256(path) != hashes[name]:
                raise RuntimeError(f"NEXT49 input {name} changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    return output_manifest


__all__ = [
    "FEATURES_NAME",
    "FRAMEWORK_FEATURE_NAMES",
    "FrameworkTopologyResult",
    "MANIFEST_NAME",
    "PROTOCOL",
    "build_qmof_framework_feature_batch",
    "compute_framework_topology_features",
]
