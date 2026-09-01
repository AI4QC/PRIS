#!/usr/bin/env python3
"""Minimum individual-facet participation of raw periodic radical cells."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from ase import Atoms
import numpy as np

import src.next267_periodic_radical_voronoi_packing as n267


PROTOCOL = "2026-08-13-next331-radical-facet-minimum-participation-v1"
FEATURE_NAMES = ("rfmp_minimum_area_participation_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
NORMAL_TOLERANCE = 1.0e-7
AREA_CERTIFICATE_RELATIVE_TOLERANCE = 1.0e-7
BOUNDARY_FLAGS = {
    "dft_calculation_executed": False,
    "dft_values_used_by_executable_formula": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
    "opened_validation_outputs_used": False,
    "scigen_replication_endpoint_opened": False,
    "wyformer_replication_endpoint_opened": False,
}


@dataclass(frozen=True)
class FacetAreaCell:
    empty: bool
    volume: float
    surface_area: float
    facet_count: int
    vertex_count: int
    minimum_participation: float
    facet_areas: tuple[float, ...]


@dataclass(frozen=True)
class RFMPFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    empty_cell_count: int
    minimum_facet_count: int
    maximum_facet_count: int
    minimum_facet_area: float
    volume_tiling_relative_error: float
    features: dict[str, float]


def _failure(exc: Exception | str) -> RFMPFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return RFMPFeatureResult(False, reason, 0, 0, 0, 0, math.nan, math.nan, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def facet_minimum_participation(areas: object) -> float:
    """Return ``K min(A_f) / sum(A_f)`` for one genuine facet population."""

    values = np.asarray(areas, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 4
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
    ):
        raise ValueError("NEXT331 facet areas differ")
    result = float(len(values) * float(np.min(values)) / math.fsum(values.tolist()))
    if not math.isfinite(result) or result <= 0.0 or result > 1.0 + 1.0e-10:
        raise ValueError("NEXT331 facet areas differ")
    return float(min(1.0, result))


def structure_minimum_participation_q10(values: object) -> float:
    """Return the frozen quantized sitewise q10, refusing grid collapse."""

    population = np.asarray(values, dtype=float)
    if (
        population.ndim != 1
        or len(population) < 1
        or not np.isfinite(population).all()
        or np.any(population <= 0.0)
        or np.any(population > 1.0)
    ):
        raise ValueError("NEXT331 site participation population differs")
    raw = float(np.quantile(population, 0.10, method="inverted_cdf"))
    quantized = _quantize(raw)
    if quantized <= 0.0:
        raise ValueError("NEXT331 positive site q10 quantized to zero")
    return quantized


def _empty_cell() -> FacetAreaCell:
    return FacetAreaCell(True, 0.0, 0.0, 0, 0, 0.0, ())


def _plane_key(normal: np.ndarray, offset: float) -> tuple[int, ...]:
    return (
        *np.rint(np.asarray(normal, dtype=float) * OUTPUT_GRID)
        .astype(np.int64)
        .tolist(),
        int(np.rint(float(offset) * OUTPUT_GRID)),
    )


def _polygon_area(points: np.ndarray, normal: np.ndarray, tolerance: float) -> float:
    centered = points - np.mean(points, axis=0)
    lengths = np.linalg.norm(centered, axis=1)
    pivot = int(np.argmax(lengths))
    if not math.isfinite(float(lengths[pivot])) or lengths[pivot] <= tolerance:
        raise ValueError("NEXT331 active facet polygon differs")
    first = centered[pivot] / lengths[pivot]
    second = np.cross(normal, first)
    second_norm = float(np.linalg.norm(second))
    if not math.isfinite(second_norm) or second_norm <= NORMAL_TOLERANCE:
        raise ValueError("NEXT331 active facet polygon differs")
    second /= second_norm
    projected = np.column_stack((centered @ first, centered @ second))
    hull = n267.ConvexHull(projected, qhull_options="Qx")
    area = float(hull.volume)
    if not math.isfinite(area) or area <= tolerance**2:
        raise ValueError("NEXT331 active facet area differs")
    return area


def power_cell_facet_areas(
    *, normals: object, offsets: object, scale: float
) -> FacetAreaCell:
    """Reconstruct one bounded power cell and every distinct facet area."""

    direction = np.asarray(normals, dtype=float)
    boundary = np.asarray(offsets, dtype=float)
    scale = float(scale)
    if (
        direction.ndim != 2
        or direction.shape[1:] != (3,)
        or len(direction) < 4
        or boundary.shape != (len(direction),)
        or not np.isfinite(direction).all()
        or not np.isfinite(boundary).all()
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError("NEXT331 power half-spaces differ")
    norms = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > NORMAL_TOLERANCE):
        raise ValueError("NEXT331 power half-space normals differ")
    direction = direction / norms[:, None]
    boundary = boundary / norms

    program = n267.linprog(
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.column_stack((direction, np.ones(len(direction), dtype=float))),
        b_ub=boundary,
        bounds=[(None, None)] * 4,
        method="highs",
    )
    if program.status == 2:
        return _empty_cell()
    if (
        not program.success
        or program.x.shape != (4,)
        or not np.isfinite(program.x).all()
    ):
        raise ValueError(f"NEXT331 Chebyshev linear program differs: {program.message}")
    interior = np.asarray(program.x[:3], dtype=float)
    chebyshev_radius = float(program.x[3])
    if chebyshev_radius <= n267.INTERIOR_TOLERANCE * max(1.0, scale):
        return _empty_cell()

    vertices = n267.HalfspaceIntersection(
        np.column_stack((direction, -boundary)), interior, qhull_options="Qx"
    ).intersections
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("NEXT331 power-cell vertex array differs")
    violation = float(np.max(direction @ vertices.T - boundary[:, None]))
    if violation > n267.VERTEX_FEASIBILITY_TOLERANCE * max(1.0, scale):
        raise ValueError("NEXT331 power-cell vertex feasibility differs")
    hull = n267.ConvexHull(vertices, qhull_options="Qx")
    unique_vertices = vertices[np.asarray(hull.vertices, dtype=int)]
    volume = float(hull.volume)
    hull_area = float(hull.area)
    if (
        not math.isfinite(volume)
        or volume <= 0.0
        or not math.isfinite(hull_area)
        or hull_area <= 0.0
    ):
        raise ValueError("NEXT331 power-cell measure differs")

    tolerance = n267.PLANE_DISTANCE_TOLERANCE * max(1.0, scale)
    unique: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    for normal, offset in zip(direction, boundary, strict=True):
        unique.setdefault(_plane_key(normal, float(offset)), (normal, float(offset)))
    facet_areas: list[float] = []
    for normal, offset in unique.values():
        points = unique_vertices[
            np.abs(unique_vertices @ normal - offset) <= tolerance
        ]
        if len(points) < 3:
            continue
        singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
        if int(np.sum(singular > tolerance)) < 2:
            continue
        facet_areas.append(_polygon_area(points, normal, tolerance))
    if len(facet_areas) < 4:
        raise ValueError("NEXT331 active facet count differs")
    summed_area = math.fsum(facet_areas)
    relative_error = abs(summed_area - hull_area) / hull_area
    if (
        not math.isfinite(relative_error)
        or relative_error > AREA_CERTIFICATE_RELATIVE_TOLERANCE
    ):
        raise ValueError(
            f"NEXT331 facet surface-area certificate differs: {relative_error:.12g}"
        )
    ordered = tuple(sorted(float(value) for value in facet_areas))
    return FacetAreaCell(
        False,
        volume,
        summed_area,
        len(ordered),
        int(len(unique_vertices)),
        facet_minimum_participation(ordered),
        ordered,
    )


def _geometry_only_atoms(atoms: Atoms) -> Atoms:
    try:
        pbc = np.asarray(atoms.pbc, dtype=bool)
        cell = np.asarray(atoms.cell.array, dtype=float)
        positions = np.asarray(atoms.positions, dtype=float)
        numbers = np.asarray(atoms.numbers)
    except Exception as exc:
        raise ValueError("NEXT331 features require exact periodic geometry-only Atoms") from exc
    if (
        not isinstance(atoms, Atoms)
        or len(atoms) < 1
        or atoms.calc is not None
        or bool(atoms.info)
        or set(atoms.arrays) != {"numbers", "positions"}
        or pbc.shape != (3,)
        or not np.all(pbc)
        or cell.shape != (3, 3)
        or positions.shape != (len(atoms), 3)
        or numbers.shape != (len(atoms),)
        or not np.isfinite(cell).all()
        or not np.isfinite(positions).all()
        or not np.isfinite(numbers).all()
        or abs(float(np.linalg.det(cell))) <= 1.0e-12
    ):
        raise ValueError("NEXT331 features require exact periodic geometry-only Atoms")
    return atoms.copy()


def periodic_radical_facet_area_cells(
    atoms: Atoms, *, radii: Sequence[float] | np.ndarray
) -> tuple[FacetAreaCell, ...]:
    """Return site-aligned periodic power cells with individual facet areas."""

    work = n267._validated_reduced_atoms(atoms)
    radius = np.asarray(radii, dtype=float)
    if (
        radius.shape != (len(work),)
        or not np.isfinite(radius).all()
        or np.any(radius <= 0.0)
    ):
        raise ValueError("NEXT331 radii must be finite, positive, and site aligned")
    cell_matrix = np.asarray(work.cell.array, dtype=float)
    lattice_normals, lattice_offsets, wigner_seitz_radius = n267._lattice_wigner_seitz(
        cell_matrix
    )
    cutoff = (
        wigner_seitz_radius
        + math.sqrt(wigner_seitz_radius**2 + float(np.max(radius)) ** 2)
        + n267.PLANE_DISTANCE_TOLERANCE
    )
    left, right, shift = n267.neighbor_list(
        "ijS", work, cutoff, self_interaction=False, max_nbins=1_000_000
    )
    if len(left) > n267.MAX_NEIGHBOR_IMAGES:
        raise ValueError("NEXT331 periodic neighbor-image count exceeds frozen guard")
    positions = np.asarray(work.positions, dtype=float)
    cells: list[FacetAreaCell] = []
    for site_index in range(len(work)):
        selected = np.asarray(left == site_index) & np.asarray(right != site_index)
        neighbor_index = np.asarray(right[selected], dtype=int)
        image = np.asarray(shift[selected], dtype=float)
        displacement = (
            positions[neighbor_index] + image @ cell_matrix - positions[site_index]
        )
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(~np.isfinite(distance)) or np.any(distance <= 1.0e-12):
            raise ValueError("NEXT331 zero-distance periodic sites differ")
        candidate_offset = (
            distance**2
            + radius[site_index] ** 2
            - radius[neighbor_index] ** 2
        ) / (2.0 * distance)
        relevant = candidate_offset <= (
            wigner_seitz_radius + n267.PLANE_DISTANCE_TOLERANCE
        )
        candidate_normals = displacement[relevant] / distance[relevant, None]
        candidate_offsets = candidate_offset[relevant]
        normals = np.vstack((lattice_normals, candidate_normals))
        offsets = np.concatenate((lattice_offsets, candidate_offsets))
        cells.append(
            power_cell_facet_areas(
                normals=normals,
                offsets=offsets,
                scale=wigner_seitz_radius,
            )
        )
    volume = abs(float(np.linalg.det(cell_matrix)))
    tiled = math.fsum(value.volume for value in cells)
    relative_error = abs(tiled - volume) / volume
    if (
        not math.isfinite(relative_error)
        or relative_error > n267.VOLUME_TILING_RELATIVE_TOLERANCE
    ):
        raise ValueError(
            f"NEXT331 volume-tiling certificate differs: {relative_error:.12g}"
        )
    return tuple(cells)


def compute_rfmp_features(atoms: Atoms) -> RFMPFeatureResult:
    """Compute RFMP from element identities and one raw periodic geometry."""

    try:
        work = _geometry_only_atoms(atoms)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        cells = periodic_radical_facet_area_cells(work, radii=radii)
        empty = np.asarray([value.empty for value in cells], dtype=bool)
        if len(cells) != len(work) or empty.any():
            raise ValueError("NEXT331 requires every labelled power cell to be nonempty")
        participation = np.asarray(
            [value.minimum_participation for value in cells], dtype=float
        )
        q10 = structure_minimum_participation_q10(participation)
        cell_volume = math.fsum(value.volume for value in cells)
        raw_volume = abs(float(np.linalg.det(np.asarray(work.cell.array, dtype=float))))
        tiling_error = abs(cell_volume - raw_volume) / raw_volume
        facets = np.asarray([value.facet_count for value in cells], dtype=int)
        minimum_area = min(min(value.facet_areas) for value in cells)
        return RFMPFeatureResult(
            True,
            None,
            len(cells),
            0,
            int(facets.min()),
            int(facets.max()),
            float(minimum_area),
            float(tiling_error),
            {FEATURE_NAMES[0]: q10},
        )
    except Exception as exc:
        return _failure(exc)


def compute_rfmp_row(atoms: Atoms) -> dict[str, object]:
    result = compute_rfmp_features(atoms)
    row: dict[str, object] = {FEATURE_NAMES[0]: math.nan}
    row.update(
        {
            "rfmp_supported": bool(result.supported),
            "rfmp_failure": result.failure_reason,
            "rfmp_site_count": result.site_count,
            "rfmp_empty_cell_count": result.empty_cell_count,
            "rfmp_minimum_facet_count": result.minimum_facet_count,
            "rfmp_maximum_facet_count": result.maximum_facet_count,
            "rfmp_minimum_facet_area": result.minimum_facet_area,
            "rfmp_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row[FEATURE_NAMES[0]] = float(result.features[FEATURE_NAMES[0]])
    return row


def build_cross_source_rfmp_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build formal RFMP tables only after the frozen probe authorizes it."""

    raise NotImplementedError("NEXT331 formal builder awaits a passing frozen probe")


__all__ = [
    "BOUNDARY_FLAGS",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "PROTOCOL",
    "FacetAreaCell",
    "RFMPFeatureResult",
    "build_cross_source_rfmp_features",
    "compute_rfmp_features",
    "compute_rfmp_row",
    "facet_minimum_participation",
    "periodic_radical_facet_area_cells",
    "power_cell_facet_areas",
    "structure_minimum_participation_q10",
]
