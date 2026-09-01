#!/usr/bin/env python3
"""Build radius-weighted periodic power-cell features from raw geometry only."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import minkowski_reduce
from ase.neighborlist import neighbor_list
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

import src.next85_scigen_label_free_features as n85
import src.next94_wyformer_label_free_features as n94
import src.next263_local_angular_persistent_homology as n263
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next267-periodic-radical-voronoi-packing-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT267_PRV_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next267_scigen_periodic_radical_voronoi_features.parquet",
    "wyformer": "next267_wyformer_periodic_radical_voronoi_features.parquet",
}
FEATURE_NAMES = (
    "prv_empty_cell_fraction",
    "prv_generator_excluded_fraction",
    "prv_sphere_crossing_fraction",
    "prv_allocation_total_variation",
    "prv_volume_ratio_q10",
    "prv_volume_ratio_q90",
    "prv_volume_ratio_cv",
    "prv_chebyshev_ratio_q10",
    "prv_chebyshev_ratio_q90",
    "prv_chebyshev_ratio_cv",
    "prv_centroid_offset_mean",
    "prv_centroid_offset_q90",
    "prv_vertex_anisotropy_mean",
    "prv_vertex_anisotropy_q90",
    "prv_facet_count_mean",
    "prv_facet_count_cv",
)
EXPECTED_DESIGN_SHA256 = (
    "258aade1020fa7911b293a4201dc4f72f428f5df6f1870fe584d64aa3b7b154a"
)
EXPECTED_INPUT_SHA256 = {
    **n263.EXPECTED_INPUT_SHA256,
    "design": EXPECTED_DESIGN_SHA256,
}
EXPECTED_UPSTREAM_SOURCE_SHA256 = n263.EXPECTED_UPSTREAM_SOURCE_SHA256
BOUNDARY_FLAGS = n263.BOUNDARY_FLAGS
TRANSLATION_RANGE = 2
PLANE_DISTANCE_TOLERANCE = 1.0e-8
INTERIOR_TOLERANCE = 1.0e-10
VERTEX_FEASIBILITY_TOLERANCE = 1.0e-7
VOLUME_TILING_RELATIVE_TOLERANCE = 1.0e-6
OUTPUT_GRID = 10_000_000_000
MAX_NEIGHBOR_IMAGES = 2_000_000


@dataclass(frozen=True)
class PowerCell:
    empty: bool
    volume: float
    chebyshev_radius: float
    generator_margin: float
    centroid_offset: float
    vertex_anisotropy: float
    facet_count: int
    vertex_count: int


@dataclass(frozen=True)
class PRVFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    empty_cell_count: int
    min_facet_count: int
    max_facet_count: int
    volume_tiling_relative_error: float
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PRVFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PRVFeatureResult(False, reason, 0, 0, 0, 0, math.nan, {})


@lru_cache(maxsize=None)
def _tabulated_radius(symbol: str) -> float:
    from pymatgen.core.periodic_table import Element

    element = Element(symbol)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No data available for atomic_radius_calculated.*",
            category=UserWarning,
        )
        value = element.atomic_radius_calculated
    if value is None:
        value = element.atomic_radius
    if value is None:
        raise ValueError(f"tabulated radius is missing for {symbol}")
    radius = float(value)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"tabulated radius differs for {symbol}")
    return radius


def _validated_reduced_atoms(atoms: Atoms) -> Atoms:
    if not isinstance(atoms, Atoms) or len(atoms) < 1:
        raise ValueError("PRV requires at least one ASE site")
    if np.asarray(atoms.pbc, dtype=bool).shape != (3,) or not np.all(atoms.pbc):
        raise ValueError("PRV requires fully periodic three-dimensional geometry")
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    if cell.shape != (3, 3) or positions.shape != (len(atoms), 3):
        raise ValueError("PRV cell or positions differ")
    if not np.isfinite(cell).all() or not np.isfinite(positions).all():
        raise ValueError("PRV cell or positions are non-finite")
    volume = abs(float(np.linalg.det(cell)))
    if not math.isfinite(volume) or volume <= 1.0e-12:
        raise ValueError("PRV cell volume differs")
    reduced, _ = minkowski_reduce(cell, pbc=True)
    reduced = np.asarray(reduced, dtype=float)
    if reduced.shape != (3, 3) or abs(float(np.linalg.det(reduced))) <= 1.0e-12:
        raise ValueError("PRV Minkowski-reduced cell differs")
    work = atoms.copy()
    work.set_cell(reduced, scale_atoms=False)
    work.wrap()
    return work


def _lattice_wigner_seitz(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    translations = np.asarray(
        [
            value
            for value in itertools.product(
                range(-TRANSLATION_RANGE, TRANSLATION_RANGE + 1), repeat=3
            )
            if value != (0, 0, 0)
        ],
        dtype=float,
    )
    vectors = translations @ cell
    distances = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(distances).all() or np.any(distances <= 1.0e-12):
        raise ValueError("PRV lattice translation differs")
    normals = vectors / distances[:, None]
    offsets = distances / 2.0
    intersections = HalfspaceIntersection(
        np.column_stack((normals, -offsets)),
        np.zeros(3, dtype=float),
        qhull_options="Qx",
    ).intersections
    if intersections.ndim != 2 or intersections.shape[1:] != (3,) or len(intersections) < 4:
        raise ValueError("PRV lattice Wigner-Seitz cell differs")
    hull = ConvexHull(intersections, qhull_options="Qx")
    radius = float(np.max(np.linalg.norm(intersections[hull.vertices], axis=1)))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("PRV lattice Wigner-Seitz radius differs")
    return normals, offsets, radius


def _active_facet_count(
    *, vertices: np.ndarray, normals: np.ndarray, offsets: np.ndarray, scale: float
) -> int:
    tolerance = PLANE_DISTANCE_TOLERANCE * max(1.0, float(scale))
    unique: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    for normal, offset in zip(normals, offsets, strict=True):
        key = (
            *np.rint(np.asarray(normal) * OUTPUT_GRID).astype(np.int64).tolist(),
            int(np.rint(float(offset) * OUTPUT_GRID)),
        )
        unique.setdefault(key, (normal, float(offset)))
    count = 0
    for normal, offset in unique.values():
        on_plane = np.abs(vertices @ normal - offset) <= tolerance
        points = vertices[on_plane]
        if len(points) < 3:
            continue
        singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
        if int(np.sum(singular > tolerance)) >= 2:
            count += 1
    return count


def _polyhedron_centroid(
    *, vertices: np.ndarray, simplices: np.ndarray, interior: np.ndarray, volume: float
) -> np.ndarray:
    tetrahedral_volumes: list[float] = []
    tetrahedral_centroids: list[np.ndarray] = []
    for simplex in simplices:
        left, middle, right = vertices[np.asarray(simplex, dtype=int)]
        tetra_volume = abs(
            float(
                np.dot(
                    left - interior,
                    np.cross(middle - interior, right - interior),
                )
            )
        ) / 6.0
        if tetra_volume > 0.0:
            tetrahedral_volumes.append(tetra_volume)
            tetrahedral_centroids.append((interior + left + middle + right) / 4.0)
    summed = math.fsum(tetrahedral_volumes)
    if not math.isfinite(summed) or abs(summed - volume) > 1.0e-7 * max(1.0, volume):
        raise ValueError("PRV tetrahedral volume certificate differs")
    return np.average(
        np.asarray(tetrahedral_centroids, dtype=float),
        axis=0,
        weights=np.asarray(tetrahedral_volumes, dtype=float),
    )


def _empty_cell(generator_margin: float) -> PowerCell:
    return PowerCell(True, 0.0, 0.0, generator_margin, 0.0, 0.0, 0, 0)


def _power_cell(
    *, normals: np.ndarray, offsets: np.ndarray, generator_margin: float, scale: float
) -> PowerCell:
    program = linprog(
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.column_stack((normals, np.ones(len(normals), dtype=float))),
        b_ub=offsets,
        bounds=[(None, None)] * 4,
        method="highs",
    )
    if program.status == 2:
        return _empty_cell(generator_margin)
    if not program.success or program.x.shape != (4,) or not np.isfinite(program.x).all():
        raise ValueError(f"PRV Chebyshev linear program differs: {program.message}")
    interior = np.asarray(program.x[:3], dtype=float)
    chebyshev_radius = float(program.x[3])
    if chebyshev_radius <= INTERIOR_TOLERANCE * max(1.0, scale):
        return _empty_cell(generator_margin)
    vertices = HalfspaceIntersection(
        np.column_stack((normals, -offsets)),
        interior,
        qhull_options="Qx",
    ).intersections
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
        raise ValueError("PRV power-cell vertex array differs")
    if not np.isfinite(vertices).all():
        raise ValueError("PRV power-cell vertices are non-finite")
    violation = float(np.max(normals @ vertices.T - offsets[:, None]))
    if violation > VERTEX_FEASIBILITY_TOLERANCE * max(1.0, scale):
        raise ValueError("PRV power-cell vertex feasibility differs")
    hull = ConvexHull(vertices, qhull_options="Qx")
    unique_vertices = vertices[np.asarray(hull.vertices, dtype=int)]
    volume = float(hull.volume)
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("PRV power-cell volume differs")
    centroid = _polyhedron_centroid(
        vertices=vertices,
        simplices=np.asarray(hull.simplices, dtype=int),
        interior=interior,
        volume=volume,
    )
    equal_volume_radius = float((3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0))
    centroid_offset = float(np.linalg.norm(centroid) / equal_volume_radius)
    covariance = np.cov((unique_vertices - centroid).T, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if (
        not np.isfinite(eigenvalues).all()
        or eigenvalues[-1] <= 0.0
        or eigenvalues[0] < -1.0e-10 * eigenvalues[-1]
    ):
        raise ValueError("PRV vertex-covariance spectrum differs")
    anisotropy = float(np.clip(1.0 - max(0.0, eigenvalues[0]) / eigenvalues[-1], 0.0, 1.0))
    facet_count = _active_facet_count(
        vertices=unique_vertices,
        normals=normals,
        offsets=offsets,
        scale=scale,
    )
    if facet_count < 4:
        raise ValueError("PRV active facet count differs")
    return PowerCell(
        False,
        volume,
        chebyshev_radius,
        generator_margin,
        centroid_offset,
        anisotropy,
        facet_count,
        int(len(unique_vertices)),
    )


def periodic_radical_cells(
    atoms: Atoms, *, radii: Sequence[float] | np.ndarray
) -> tuple[PowerCell, ...]:
    """Return one labelled periodic radical cell per raw site."""

    work = _validated_reduced_atoms(atoms)
    radius = np.asarray(radii, dtype=float)
    if radius.shape != (len(work),) or not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("PRV radii must be finite, positive, and site aligned")
    cell = np.asarray(work.cell.array, dtype=float)
    lattice_normals, lattice_offsets, wigner_seitz_radius = _lattice_wigner_seitz(cell)
    cutoff = (
        wigner_seitz_radius
        + math.sqrt(wigner_seitz_radius**2 + float(np.max(radius)) ** 2)
        + PLANE_DISTANCE_TOLERANCE
    )
    left, right, shift = neighbor_list(
        "ijS",
        work,
        cutoff,
        self_interaction=False,
        max_nbins=1_000_000,
    )
    if len(left) > MAX_NEIGHBOR_IMAGES:
        raise ValueError("PRV periodic neighbor-image count exceeds frozen guard")
    positions = np.asarray(work.positions, dtype=float)
    cells: list[PowerCell] = []
    for site_index in range(len(work)):
        selected = np.asarray(left == site_index) & np.asarray(right != site_index)
        neighbor_index = np.asarray(right[selected], dtype=int)
        image = np.asarray(shift[selected], dtype=float)
        displacement = positions[neighbor_index] + image @ cell - positions[site_index]
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(distance <= 1.0e-12):
            raise ValueError("PRV zero-distance periodic sites differ")
        candidate_offset = (
            distance**2
            + radius[site_index] ** 2
            - radius[neighbor_index] ** 2
        ) / (2.0 * distance)
        relevant = candidate_offset <= (
            wigner_seitz_radius + PLANE_DISTANCE_TOLERANCE
        )
        candidate_normals = displacement[relevant] / distance[relevant, None]
        candidate_offsets = candidate_offset[relevant]
        normals = np.vstack((lattice_normals, candidate_normals))
        offsets = np.concatenate((lattice_offsets, candidate_offsets))
        if not np.isfinite(normals).all() or not np.isfinite(offsets).all():
            raise ValueError("PRV power half-spaces are non-finite")
        generator_margin = float(np.min(offsets))
        cells.append(
            _power_cell(
                normals=normals,
                offsets=offsets,
                generator_margin=generator_margin,
                scale=wigner_seitz_radius,
            )
        )
    volume = abs(float(np.linalg.det(cell)))
    tiled = math.fsum(value.volume for value in cells)
    relative_error = abs(tiled - volume) / volume
    if not math.isfinite(relative_error) or relative_error > VOLUME_TILING_RELATIVE_TOLERANCE:
        raise ValueError(f"PRV volume-tiling certificate differs: {relative_error:.12g}")
    return tuple(cells)


def _quantized(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("PRV aggregate values differ")
    return np.rint(array * OUTPUT_GRID) / OUTPUT_GRID


def _mean(values: Sequence[float] | np.ndarray) -> float:
    array = _quantized(values)
    return float(math.fsum(float(value) for value in array) / len(array))


def _quantile(values: Sequence[float] | np.ndarray, probability: float) -> float:
    return float(np.quantile(_quantized(values), probability, method="inverted_cdf"))


def _cv(values: Sequence[float] | np.ndarray) -> float:
    array = _quantized(values)
    center = float(math.fsum(float(value) for value in array) / len(array))
    if abs(center) <= 1.0 / OUTPUT_GRID:
        return 0.0
    variance = math.fsum((float(value) - center) ** 2 for value in array) / len(array)
    return float(math.sqrt(max(0.0, variance)) / abs(center))


def compute_periodic_radical_voronoi_features(atoms: Atoms) -> PRVFeatureResult:
    """Compute frozen PRV summaries without labels, endpoints, or relaxation."""

    try:
        radii = np.asarray(
            [_tabulated_radius(str(symbol)) for symbol in atoms.get_chemical_symbols()],
            dtype=float,
        )
        cells = periodic_radical_cells(atoms, radii=radii)
        volume = abs(float(np.linalg.det(np.asarray(atoms.cell.array, dtype=float))))
        cell_volumes = np.asarray([value.volume for value in cells], dtype=float)
        relative_error = abs(float(cell_volumes.sum()) - volume) / volume
        empty = np.asarray([value.empty for value in cells], dtype=bool)
        nonempty = ~empty
        if not nonempty.any():
            raise ValueError("PRV structure has no positive-volume labelled cell")
        margins = np.asarray(
            [value.generator_margin for value in cells], dtype=float
        ) / radii
        observed_allocation = cell_volumes / volume
        expected_allocation = radii**3 / float(np.sum(radii**3))
        allocation_tv = 0.5 * float(
            np.sum(np.abs(observed_allocation - expected_allocation))
        )
        volume_ratio = cell_volumes[nonempty] / (
            (4.0 * math.pi / 3.0) * radii[nonempty] ** 3
        )
        chebyshev_ratio = np.asarray(
            [value.chebyshev_radius for value in cells], dtype=float
        )[nonempty] / radii[nonempty]
        centroid_offset = np.asarray(
            [value.centroid_offset for value in cells], dtype=float
        )[nonempty]
        anisotropy = np.asarray(
            [value.vertex_anisotropy for value in cells], dtype=float
        )[nonempty]
        facets = np.asarray([value.facet_count for value in cells], dtype=float)[nonempty]
        features = {
            "prv_empty_cell_fraction": _mean(empty.astype(float)),
            "prv_generator_excluded_fraction": _mean((margins <= 0.0).astype(float)),
            "prv_sphere_crossing_fraction": _mean((margins < 1.0).astype(float)),
            "prv_allocation_total_variation": float(
                np.rint(allocation_tv * OUTPUT_GRID) / OUTPUT_GRID
            ),
            "prv_volume_ratio_q10": _quantile(volume_ratio, 0.10),
            "prv_volume_ratio_q90": _quantile(volume_ratio, 0.90),
            "prv_volume_ratio_cv": _cv(volume_ratio),
            "prv_chebyshev_ratio_q10": _quantile(chebyshev_ratio, 0.10),
            "prv_chebyshev_ratio_q90": _quantile(chebyshev_ratio, 0.90),
            "prv_chebyshev_ratio_cv": _cv(chebyshev_ratio),
            "prv_centroid_offset_mean": _mean(centroid_offset),
            "prv_centroid_offset_q90": _quantile(centroid_offset, 0.90),
            "prv_vertex_anisotropy_mean": _mean(anisotropy),
            "prv_vertex_anisotropy_q90": _quantile(anisotropy, 0.90),
            "prv_facet_count_mean": _mean(facets),
            "prv_facet_count_cv": _cv(facets),
        }
        if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
            raise ValueError("PRV feature schema or values differ")
        facet_values = np.asarray([value.facet_count for value in cells if not value.empty])
        return PRVFeatureResult(
            True,
            None,
            len(cells),
            int(empty.sum()),
            int(facet_values.min()),
            int(facet_values.max()),
            float(relative_error),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_prv_row(atoms: Atoms) -> dict[str, object]:
    result = compute_periodic_radical_voronoi_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "prv_supported": bool(result.supported),
            "prv_failure": result.failure_reason,
            "prv_site_count": result.site_count,
            "prv_empty_cell_count": result.empty_cell_count,
            "prv_min_facet_count": result.min_facet_count,
            "prv_max_facet_count": result.max_facet_count,
            "prv_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "prv_supported": False,
        "prv_failure": f"{type(exc).__name__}: {exc}",
        "prv_site_count": 0,
        "prv_empty_cell_count": 0,
        "prv_min_facet_count": 0,
        "prv_max_facet_count": 0,
        "prv_volume_tiling_relative_error": math.nan,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_prv_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_prv_row(AseAtomsAdaptor.get_atoms(structure))
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(
    payloads: Sequence[tuple[str, bytes]] | Sequence[tuple[str, str]],
    *,
    source: str,
    workers: int,
) -> list[tuple[str, dict[str, object]]]:
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]  # type: ignore[arg-type]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=4))  # type: ignore[arg-type]


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def build_cross_source_prv_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT267 from physically isolated discovery geometry only."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "scigen_manifest": scigen / n85.COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / n85.COHORT_METADATA_NAME,
        "scigen_discovery_geometry": scigen / n85.GEOMETRY_NAMES["discovery"],
        "wyformer_manifest": wyformer / n94.COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / n94.COHORT_METADATA_NAME,
        "wyformer_discovery_geometry": wyformer / n94.GEOMETRY_NAMES["discovery"],
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT267 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT267 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT267 formal input identity differs: {differing}")
    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name) for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT267 frozen upstream source differs")
    scigen_manifest = _read_manifest(paths["scigen_manifest"])
    wyformer_manifest = _read_manifest(paths["wyformer_manifest"])
    if (
        scigen_manifest.get("protocol") != n85.COHORT_PROTOCOL
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or scigen_manifest.get("relaxed_structures_opened") is not False
        or wyformer_manifest.get("protocol") != n94.COHORT_PROTOCOL
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
        or wyformer_manifest.get("relaxed_structures_published") is not False
    ):
        raise ValueError("NEXT267 discovery geometry provenance differs")
    metadata = {
        "scigen": pd.read_parquet(paths["scigen_metadata"]),
        "wyformer": pd.read_parquet(paths["wyformer_metadata"]),
    }
    discovery: dict[str, pd.DataFrame] = {}
    for source, frame in metadata.items():
        required = {
            "material_id",
            "reduced_formula",
            "chemical_system",
            "natoms",
            "partition_role",
            "input_role",
        }
        if required - set(frame.columns) or frame["material_id"].astype(str).duplicated().any():
            raise ValueError(f"NEXT267 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT267 {source} discovery identity differs")
        discovery[source] = selected
    payloads = {
        "scigen": n85._archive_payloads(
            paths["scigen_discovery_geometry"],
            discovery["scigen"]["material_id"].astype(str).tolist(),
        ),
        "wyformer": n94._payloads(
            paths["wyformer_discovery_geometry"],
            discovery["wyformer"]["material_id"].astype(str).tolist(),
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    started = time.perf_counter()
    try:
        computed = {
            source: _compute_many(payloads[source], source=source, workers=workers)
            for source in ("scigen", "wyformer")
        }
        counts: dict[str, object] = {}
        output_paths: list[Path] = []
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed[source]]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT267 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery[source]):
                raise RuntimeError(f"NEXT267 {source} row accounting differs")
            supported = table["prv_supported"].fillna(False).astype(bool)
            finite_matrix = np.column_stack(
                [
                    np.isfinite(pd.to_numeric(table[name], errors="coerce").to_numpy(float))
                    for name in FEATURE_NAMES
                ]
            )
            if not finite_matrix[supported].all() or finite_matrix[~supported].any():
                raise RuntimeError(f"NEXT267 {source} support/finite contract differs")
            tiling = pd.to_numeric(
                table["prv_volume_tiling_relative_error"], errors="coerce"
            )
            sites = pd.to_numeric(table["prv_site_count"], errors="coerce")
            minimum = pd.to_numeric(table["prv_min_facet_count"], errors="coerce")
            maximum = pd.to_numeric(table["prv_max_facet_count"], errors="coerce")
            if (
                not (sites[supported] > 0).all()
                or not (minimum[supported] >= 4).all()
                or not (maximum[supported] >= minimum[supported]).all()
                or not (tiling[supported] <= VOLUME_TILING_RELATIVE_TOLERANCE).all()
            ):
                raise RuntimeError(f"NEXT267 {source} diagnostics differ")
            failures = Counter(table.loc[~supported, "prv_failure"].astype(str))
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": dict(sorted(failures.items())),
                "finite_feature_counts": {
                    name: int(finite_matrix[:, index].sum())
                    for index, name in enumerate(FEATURE_NAMES)
                },
                "site_count": int(sites[supported].sum()),
                "empty_cell_count": int(
                    pd.to_numeric(table.loc[supported, "prv_empty_cell_count"]).sum()
                ),
                "maximum_volume_tiling_relative_error": float(tiling[supported].max()),
            }
        if counts["scigen"]["rows"] != 13_470 or counts["wyformer"]["rows"] != 5_232:  # type: ignore[index]
            raise RuntimeError("NEXT267 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "radius_policy": ["atomic_radius_calculated", "atomic_radius_fallback"],
            "power_distance": "squared_euclidean_minus_radius_squared",
            "lattice_reduction": "ase_minkowski_reduce",
            "lattice_translation_range": TRANSLATION_RANGE,
            "neighbor_cutoff": "R_WS+sqrt(R_WS^2+r_max^2)+1e-8_angstrom",
            "chebyshev_solver": "scipy_linprog_highs",
            "halfspace_qhull_options": "Qx",
            "output_grid": OUTPUT_GRID,
            "quantile_method": "inverted_cdf",
            "volume_tiling_relative_tolerance": VOLUME_TILING_RELATIVE_TOLERANCE,
            "maximum_neighbor_images": MAX_NEIGHBOR_IMAGES,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_periodic_radical_voronoi_feature_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "upstream_source_sha256": upstream_hashes,
            "executed_source_sha256": {
                "src/next267_periodic_radical_voronoi_packing.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT267 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT267 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_cross_source_prv_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
