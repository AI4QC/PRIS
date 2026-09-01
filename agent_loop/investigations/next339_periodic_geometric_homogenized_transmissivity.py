#!/usr/bin/env python3
"""Periodic finite-volume corrector on raw radius-aware power facets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from ase import Atoms
import numpy as np

import src.next267_periodic_radical_voronoi_packing as n267
import src.next331_radical_facet_minimum_participation as n331


PROTOCOL = "2026-08-13-next339-periodic-geometric-homogenized-transmissivity-v1"
FEATURE_NAMES = ("pght_affine_retention_floor",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
RECIPROCAL_AREA_RELATIVE_TOLERANCE = 1.0e-7
SURFACE_AREA_RELATIVE_TOLERANCE = 1.0e-7
CORRECTOR_RESIDUAL_TOLERANCE = 1.0e-8
SPECTRUM_TOLERANCE = 1.0e-8
AFFINE_EIGENVALUE_RELATIVE_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = dict(n331.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class HomogenizedRetentionResult:
    affine_retention_floor: float
    generalized_eigenvalues: tuple[float, float, float]
    affine_tensor: np.ndarray
    homogenized_tensor: np.ndarray
    maximum_corrector_residual: float


@dataclass(frozen=True)
class DirectedPowerFacet:
    center: int
    neighbor: int
    displacement: tuple[float, float, float]
    area: float


@dataclass(frozen=True)
class PowerFacetGraph:
    site_count: int
    endpoints: np.ndarray
    displacements: np.ndarray
    conductances: np.ndarray
    volume: float
    minimum_facet_area: float
    maximum_reciprocal_area_relative_error: float
    volume_tiling_relative_error: float


@dataclass(frozen=True)
class PGHTFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    minimum_facet_area: float
    maximum_reciprocal_area_relative_error: float
    maximum_corrector_residual: float
    volume_tiling_relative_error: float
    features: dict[str, float]


def _failure(exc: Exception | str) -> PGHTFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PGHTFeatureResult(False, reason, 0, 0, math.nan, math.nan, math.nan, math.nan, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def periodic_homogenized_retention(
    *,
    n_sites: int,
    endpoints: object,
    displacements: object,
    conductances: object,
    volume: float,
) -> HomogenizedRetentionResult:
    """Solve three periodic scalar correctors and return affine retention."""

    if type(n_sites) is not int or n_sites < 1:
        raise ValueError("NEXT339 site count differs")
    pair_raw = np.asarray(endpoints)
    vector = np.asarray(displacements, dtype=float)
    conductance = np.asarray(conductances, dtype=float)
    volume = float(volume)
    if pair_raw.ndim != 2 or pair_raw.shape[1:] != (2,) or len(pair_raw) < 1:
        raise ValueError("NEXT339 edge population differs")
    try:
        pair_numeric = np.asarray(pair_raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NEXT339 edge population differs") from exc
    if (
        not np.isfinite(pair_numeric).all()
        or not np.equal(pair_numeric, np.rint(pair_numeric)).all()
    ):
        raise ValueError("NEXT339 edge population differs")
    pair = pair_numeric.astype(int)
    if (
        vector.shape != (len(pair), 3)
        or conductance.shape != (len(pair),)
        or not np.isfinite(vector).all()
        or not np.isfinite(conductance).all()
        or np.any(conductance <= 0.0)
        or np.any(pair < 0)
        or np.any(pair >= n_sites)
        or np.any(np.linalg.norm(vector, axis=1) <= 1.0e-12)
        or not math.isfinite(volume)
        or volume <= 0.0
    ):
        raise ValueError("NEXT339 edge population differs")

    incidence = np.zeros((len(pair), n_sites), dtype=float)
    rows = np.arange(len(pair), dtype=int)
    nonself = pair[:, 0] != pair[:, 1]
    incidence[rows[nonself], pair[nonself, 0]] = -1.0
    incidence[rows[nonself], pair[nonself, 1]] = 1.0
    weighted_vector = conductance[:, None] * vector
    affine = vector.T @ weighted_vector / volume
    affine = 0.5 * (affine + affine.T)
    affine_values, affine_vectors = np.linalg.eigh(affine)
    affine_scale = max(1.0, float(affine_values[-1]))
    if (
        not np.isfinite(affine_values).all()
        or float(affine_values[0]) <= AFFINE_EIGENVALUE_RELATIVE_TOLERANCE * affine_scale
    ):
        raise ValueError("NEXT339 affine tensor is not positive definite")

    laplacian = incidence.T @ (conductance[:, None] * incidence)
    right_hand_side = -(incidence.T @ weighted_vector)
    corrector = np.linalg.pinv(laplacian, rcond=1.0e-12, hermitian=True) @ right_hand_side
    corrected = vector + incidence @ corrector
    residual_matrix = incidence.T @ (conductance[:, None] * corrected)
    forcing_scale = max(
        1.0,
        float(np.max(np.abs(incidence.T @ weighted_vector), initial=0.0)),
    )
    residual = float(np.max(np.abs(residual_matrix), initial=0.0) / forcing_scale)
    if not math.isfinite(residual) or residual > CORRECTOR_RESIDUAL_TOLERANCE:
        raise ValueError(f"NEXT339 corrector residual differs: {residual:.12g}")
    homogenized = corrected.T @ (conductance[:, None] * corrected) / volume
    homogenized = 0.5 * (homogenized + homogenized.T)
    inverse_sqrt = (
        affine_vectors
        @ np.diag(1.0 / np.sqrt(affine_values))
        @ affine_vectors.T
    )
    normalized = inverse_sqrt @ homogenized @ inverse_sqrt
    normalized = 0.5 * (normalized + normalized.T)
    spectrum = np.linalg.eigvalsh(normalized)
    if (
        not np.isfinite(spectrum).all()
        or float(spectrum[0]) < -SPECTRUM_TOLERANCE
        or float(spectrum[-1]) > 1.0 + SPECTRUM_TOLERANCE
    ):
        raise ValueError("NEXT339 generalized spectrum differs")
    spectrum = np.clip(spectrum, 0.0, 1.0)
    floor = float(spectrum[0])
    return HomogenizedRetentionResult(
        floor,
        tuple(float(value) for value in spectrum),
        affine,
        homogenized,
        residual,
    )


def _cell_facets(
    *,
    center: int,
    normals: np.ndarray,
    offsets: np.ndarray,
    labels: Sequence[tuple[int, np.ndarray]],
    scale: float,
) -> tuple[float, tuple[DirectedPowerFacet, ...]]:
    """Return volume and exactly one labelled record per geometric facet."""

    program = n267.linprog(
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.column_stack((normals, np.ones(len(normals), dtype=float))),
        b_ub=offsets,
        bounds=[(None, None)] * 4,
        method="highs",
    )
    if program.status == 2 or (
        program.success
        and program.x.shape == (4,)
        and float(program.x[3]) <= n267.INTERIOR_TOLERANCE * max(1.0, scale)
    ):
        return 0.0, ()
    if not program.success or program.x.shape != (4,) or not np.isfinite(program.x).all():
        raise ValueError(f"NEXT339 Chebyshev linear program differs: {program.message}")
    interior = np.asarray(program.x[:3], dtype=float)
    vertices = n267.HalfspaceIntersection(
        np.column_stack((normals, -offsets)), interior, qhull_options="Qx"
    ).intersections
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("NEXT339 power-cell vertex array differs")
    violation = float(np.max(normals @ vertices.T - offsets[:, None]))
    if violation > n267.VERTEX_FEASIBILITY_TOLERANCE * max(1.0, scale):
        raise ValueError("NEXT339 power-cell vertex feasibility differs")
    hull = n267.ConvexHull(vertices, qhull_options="Qx")
    unique_vertices = vertices[np.asarray(hull.vertices, dtype=int)]
    volume = float(hull.volume)
    hull_area = float(hull.area)
    if volume <= 0.0 or hull_area <= 0.0 or not math.isfinite(volume + hull_area):
        raise ValueError("NEXT339 power-cell measure differs")

    grouped: dict[
        tuple[int, ...],
        tuple[np.ndarray, float, dict[tuple[int, tuple[int, int, int]], tuple[int, np.ndarray]]],
    ] = {}
    for normal, offset, label in zip(normals, offsets, labels, strict=True):
        key = n331._plane_key(normal, float(offset))
        if key not in grouped:
            grouped[key] = (np.asarray(normal, dtype=float), float(offset), {})
        displacement_key = tuple(
            int(value)
            for value in np.rint(np.asarray(label[1], dtype=float) * OUTPUT_GRID).astype(np.int64)
        )
        grouped[key][2].setdefault(
            (int(label[0]), displacement_key),
            (int(label[0]), np.asarray(label[1], dtype=float)),
        )
    tolerance = n267.PLANE_DISTANCE_TOLERANCE * max(1.0, scale)
    records: list[DirectedPowerFacet] = []
    area_total = 0.0
    for normal, offset, plane_labels in grouped.values():
        points = unique_vertices[np.abs(unique_vertices @ normal - offset) <= tolerance]
        if len(points) < 3:
            continue
        singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
        if int(np.sum(singular > tolerance)) < 2:
            continue
        if len(plane_labels) != 1:
            raise ValueError("NEXT339 active facet generator identity is ambiguous")
        neighbor, displacement = next(iter(plane_labels.values()))
        area = n331._polygon_area(points, normal, tolerance)
        area_total += area
        records.append(
            DirectedPowerFacet(
                center,
                neighbor,
                tuple(float(value) for value in displacement),
                float(area),
            )
        )
    if len(records) < 4:
        raise ValueError("NEXT339 active facet count differs")
    relative_error = abs(area_total - hull_area) / hull_area
    if relative_error > SURFACE_AREA_RELATIVE_TOLERANCE:
        raise ValueError(
            f"NEXT339 facet surface-area certificate differs: {relative_error:.12g}"
        )
    return volume, tuple(records)


def _directed_key(facet: DirectedPowerFacet) -> tuple[int, int, tuple[int, int, int]]:
    displacement_key = tuple(
        int(value)
        for value in np.rint(np.asarray(facet.displacement) * OUTPUT_GRID).astype(np.int64)
    )
    return facet.center, facet.neighbor, displacement_key


def periodic_power_facet_graph(
    atoms: Atoms, *, radii: Sequence[float] | np.ndarray
) -> PowerFacetGraph:
    """Build one reciprocal finite-volume edge per shared periodic facet."""

    work = n267._validated_reduced_atoms(atoms)
    radius = np.asarray(radii, dtype=float)
    if radius.shape != (len(work),) or not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("NEXT339 radii must be finite, positive, and site aligned")
    cell_matrix = np.asarray(work.cell.array, dtype=float)
    lattice_normals, lattice_offsets, wigner_seitz_radius = n267._lattice_wigner_seitz(cell_matrix)
    cutoff = (
        wigner_seitz_radius
        + math.sqrt(wigner_seitz_radius**2 + float(np.max(radius)) ** 2)
        + n267.PLANE_DISTANCE_TOLERANCE
    )
    left, right, shift = n267.neighbor_list(
        "ijS", work, cutoff, self_interaction=False, max_nbins=1_000_000
    )
    if len(left) > n267.MAX_NEIGHBOR_IMAGES:
        raise ValueError("NEXT339 periodic neighbor-image count exceeds frozen guard")
    positions = np.asarray(work.positions, dtype=float)
    lattice_displacements = 2.0 * lattice_offsets[:, None] * lattice_normals
    directed: list[DirectedPowerFacet] = []
    volumes: list[float] = []
    for site_index in range(len(work)):
        selected = np.asarray(left == site_index) & np.asarray(right != site_index)
        neighbor_index = np.asarray(right[selected], dtype=int)
        image = np.asarray(shift[selected], dtype=float)
        displacement = positions[neighbor_index] + image @ cell_matrix - positions[site_index]
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(~np.isfinite(distance)) or np.any(distance <= 1.0e-12):
            raise ValueError("NEXT339 zero-distance periodic sites differ")
        candidate_offset = (
            distance**2 + radius[site_index] ** 2 - radius[neighbor_index] ** 2
        ) / (2.0 * distance)
        relevant = candidate_offset <= wigner_seitz_radius + n267.PLANE_DISTANCE_TOLERANCE
        normals = np.vstack((lattice_normals, displacement[relevant] / distance[relevant, None]))
        offsets = np.concatenate((lattice_offsets, candidate_offset[relevant]))
        labels: list[tuple[int, np.ndarray]] = [
            (site_index, np.asarray(value, dtype=float)) for value in lattice_displacements
        ]
        labels.extend(
            (int(neighbor), np.asarray(value, dtype=float))
            for neighbor, value in zip(neighbor_index[relevant], displacement[relevant], strict=True)
        )
        volume, facets = _cell_facets(
            center=site_index,
            normals=normals,
            offsets=offsets,
            labels=labels,
            scale=wigner_seitz_radius,
        )
        volumes.append(volume)
        directed.extend(facets)
    raw_volume = abs(float(np.linalg.det(cell_matrix)))
    tiling_error = abs(math.fsum(volumes) - raw_volume) / raw_volume
    if tiling_error > n267.VOLUME_TILING_RELATIVE_TOLERANCE or any(value <= 0.0 for value in volumes):
        raise ValueError(f"NEXT339 volume-tiling certificate differs: {tiling_error:.12g}")

    paired: dict[
        tuple[int, int, tuple[int, int, int]], list[DirectedPowerFacet]
    ] = {}
    for facet in directed:
        forward = _directed_key(facet)
        reverse = (forward[1], forward[0], tuple(-value for value in forward[2]))
        paired.setdefault(min(forward, reverse), []).append(facet)
    endpoints: list[tuple[int, int]] = []
    displacements: list[tuple[float, float, float]] = []
    conductances: list[float] = []
    areas: list[float] = []
    reciprocal_errors: list[float] = []
    for key in sorted(paired):
        facets = paired[key]
        if len(facets) != 2:
            raise ValueError("NEXT339 active facets are not exactly reciprocal")
        first, second = facets
        first_key = _directed_key(first)
        second_key = _directed_key(second)
        if (
            first_key[0] != second_key[1]
            or first_key[1] != second_key[0]
            or first_key[2] != tuple(-value for value in second_key[2])
        ):
            raise ValueError("NEXT339 reciprocal facet identity differs")
        area_scale = max(first.area, second.area)
        area_error = abs(first.area - second.area) / area_scale
        if area_error > RECIPROCAL_AREA_RELATIVE_TOLERANCE:
            raise ValueError(
                f"NEXT339 reciprocal facet area differs: {area_error:.12g}"
            )
        chosen = first if first_key == key else second
        area = 0.5 * (first.area + second.area)
        distance = float(np.linalg.norm(chosen.displacement))
        endpoints.append((chosen.center, chosen.neighbor))
        displacements.append(chosen.displacement)
        conductances.append(area / distance)
        areas.append(area)
        reciprocal_errors.append(area_error)
    if len(endpoints) < 3:
        raise ValueError("NEXT339 reciprocal edge population differs")
    return PowerFacetGraph(
        len(work),
        np.asarray(endpoints, dtype=int),
        np.asarray(displacements, dtype=float),
        np.asarray(conductances, dtype=float),
        raw_volume,
        float(min(areas)),
        float(max(reciprocal_errors, default=0.0)),
        float(tiling_error),
    )


def compute_pght_features(atoms: Atoms) -> PGHTFeatureResult:
    try:
        work = n331._geometry_only_atoms(atoms)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        graph = periodic_power_facet_graph(work, radii=radii)
        result = periodic_homogenized_retention(
            n_sites=graph.site_count,
            endpoints=graph.endpoints,
            displacements=graph.displacements,
            conductances=graph.conductances,
            volume=graph.volume,
        )
        value = _quantize(result.affine_retention_floor)
        if value <= 0.0:
            raise ValueError("NEXT339 positive affine retention quantized to zero")
        return PGHTFeatureResult(
            True,
            None,
            graph.site_count,
            len(graph.endpoints),
            graph.minimum_facet_area,
            graph.maximum_reciprocal_area_relative_error,
            result.maximum_corrector_residual,
            graph.volume_tiling_relative_error,
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pght_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pght_features(atoms)
    row: dict[str, object] = {FEATURE_NAMES[0]: math.nan}
    row.update(
        {
            "pght_supported": bool(result.supported),
            "pght_failure": result.failure_reason,
            "pght_site_count": result.site_count,
            "pght_edge_count": result.edge_count,
            "pght_minimum_facet_area": result.minimum_facet_area,
            "pght_maximum_reciprocal_area_relative_error": result.maximum_reciprocal_area_relative_error,
            "pght_maximum_corrector_residual": result.maximum_corrector_residual,
            "pght_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row[FEATURE_NAMES[0]] = float(result.features[FEATURE_NAMES[0]])
    return row


def build_cross_source_pght_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    raise NotImplementedError("NEXT339 formal builder awaits a passing frozen probe")


__all__ = [
    "BOUNDARY_FLAGS",
    "CORRECTOR_RESIDUAL_TOLERANCE",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "PROTOCOL",
    "RECIPROCAL_AREA_RELATIVE_TOLERANCE",
    "HomogenizedRetentionResult",
    "PGHTFeatureResult",
    "PowerFacetGraph",
    "build_cross_source_pght_features",
    "compute_pght_features",
    "compute_pght_row",
    "periodic_homogenized_retention",
    "periodic_power_facet_graph",
]
