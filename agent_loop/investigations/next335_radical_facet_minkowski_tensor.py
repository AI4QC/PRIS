#!/usr/bin/env python3
"""Surface-normal Minkowski anisotropy of raw periodic radical cells."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from ase import Atoms
import numpy as np

import src.next267_periodic_radical_voronoi_packing as n267
import src.next331_radical_facet_minimum_participation as n331


PROTOCOL = "2026-08-13-next335-radical-facet-minkowski-tensor-v1"
FEATURE_NAMES = ("rfmt_surface_normal_beta_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
NORMAL_TOLERANCE = 1.0e-7
EIGENVALUE_TOLERANCE = 1.0e-10
AREA_CERTIFICATE_RELATIVE_TOLERANCE = 1.0e-7
BOUNDARY_FLAGS = dict(n331.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class MinkowskiTensorCell:
    empty: bool
    volume: float
    surface_area: float
    facet_count: int
    vertex_count: int
    surface_normal_beta: float
    tensor_eigenvalues: tuple[float, float, float]


@dataclass(frozen=True)
class RFMTFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    empty_cell_count: int
    minimum_facet_count: int
    maximum_facet_count: int
    minimum_site_beta: float
    maximum_site_beta: float
    volume_tiling_relative_error: float
    features: dict[str, float]


def _failure(exc: Exception | str) -> RFMTFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return RFMTFeatureResult(False, reason, 0, 0, 0, 0, math.nan, math.nan, math.nan, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _surface_normal_spectrum(*, normals: object, areas: object) -> np.ndarray:
    direction = np.asarray(normals, dtype=float)
    measure = np.asarray(areas, dtype=float)
    if (
        direction.ndim != 2
        or direction.shape[1:] != (3,)
        or len(direction) < 4
        or measure.shape != (len(direction),)
        or not np.isfinite(direction).all()
        or not np.isfinite(measure).all()
        or np.any(measure <= 0.0)
    ):
        raise ValueError("NEXT335 surface-normal population differs")
    norms = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > NORMAL_TOLERANCE):
        raise ValueError("NEXT335 surface-normal population differs")
    unit = direction / norms[:, None]
    tensor = np.einsum("i,ij,ik->jk", measure, unit, unit) / math.fsum(measure.tolist())
    eigenvalues = np.linalg.eigvalsh(tensor)
    if (
        not np.isfinite(eigenvalues).all()
        or float(eigenvalues[0]) <= 0.0
        or float(eigenvalues[-1]) > 1.0 + EIGENVALUE_TOLERANCE
        or not math.isclose(float(eigenvalues.sum()), 1.0, abs_tol=EIGENVALUE_TOLERANCE)
    ):
        raise ValueError("NEXT335 surface-normal tensor spectrum differs")
    return np.clip(eigenvalues, 0.0, 1.0)


def surface_normal_beta(*, normals: object, areas: object) -> float:
    """Return the standard minimum/maximum eigenvalue anisotropy ratio."""

    eigenvalues = _surface_normal_spectrum(normals=normals, areas=areas)
    value = float(eigenvalues[0] / eigenvalues[-1])
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("NEXT335 surface-normal tensor spectrum differs")
    return value


def _empty_cell() -> MinkowskiTensorCell:
    return MinkowskiTensorCell(True, 0.0, 0.0, 0, 0, 0.0, (0.0, 0.0, 0.0))


def power_cell_minkowski_tensor(
    *, normals: object, offsets: object, scale: float
) -> MinkowskiTensorCell:
    """Reconstruct one bounded power cell and its normal Minkowski tensor."""

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
        raise ValueError("NEXT335 power half-spaces differ")
    norms = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > NORMAL_TOLERANCE):
        raise ValueError("NEXT335 power half-space normals differ")
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
    if not program.success or program.x.shape != (4,) or not np.isfinite(program.x).all():
        raise ValueError(f"NEXT335 Chebyshev linear program differs: {program.message}")
    interior = np.asarray(program.x[:3], dtype=float)
    if float(program.x[3]) <= n267.INTERIOR_TOLERANCE * max(1.0, scale):
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
        raise ValueError("NEXT335 power-cell vertex array differs")
    violation = float(np.max(direction @ vertices.T - boundary[:, None]))
    if violation > n267.VERTEX_FEASIBILITY_TOLERANCE * max(1.0, scale):
        raise ValueError("NEXT335 power-cell vertex feasibility differs")
    hull = n267.ConvexHull(vertices, qhull_options="Qx")
    unique_vertices = vertices[np.asarray(hull.vertices, dtype=int)]
    volume = float(hull.volume)
    hull_area = float(hull.area)
    if volume <= 0.0 or hull_area <= 0.0 or not math.isfinite(volume + hull_area):
        raise ValueError("NEXT335 power-cell measure differs")

    tolerance = n267.PLANE_DISTANCE_TOLERANCE * max(1.0, scale)
    unique: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    for normal, offset in zip(direction, boundary, strict=True):
        unique.setdefault(n331._plane_key(normal, float(offset)), (normal, float(offset)))
    active_normals: list[np.ndarray] = []
    active_areas: list[float] = []
    for normal, offset in unique.values():
        points = unique_vertices[np.abs(unique_vertices @ normal - offset) <= tolerance]
        if len(points) < 3:
            continue
        singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
        if int(np.sum(singular > tolerance)) < 2:
            continue
        active_normals.append(normal)
        active_areas.append(n331._polygon_area(points, normal, tolerance))
    if len(active_areas) < 4:
        raise ValueError("NEXT335 active facet count differs")
    summed_area = math.fsum(active_areas)
    relative_error = abs(summed_area - hull_area) / hull_area
    if relative_error > AREA_CERTIFICATE_RELATIVE_TOLERANCE:
        raise ValueError(
            f"NEXT335 facet surface-area certificate differs: {relative_error:.12g}"
        )
    spectrum = _surface_normal_spectrum(normals=active_normals, areas=active_areas)
    beta = float(spectrum[0] / spectrum[-1])
    return MinkowskiTensorCell(
        False,
        volume,
        summed_area,
        len(active_areas),
        int(len(unique_vertices)),
        beta,
        tuple(float(value) for value in spectrum),
    )


def periodic_radical_minkowski_cells(
    atoms: Atoms, *, radii: Sequence[float] | np.ndarray
) -> tuple[MinkowskiTensorCell, ...]:
    work = n267._validated_reduced_atoms(atoms)
    radius = np.asarray(radii, dtype=float)
    if radius.shape != (len(work),) or not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("NEXT335 radii must be finite, positive, and site aligned")
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
        raise ValueError("NEXT335 periodic neighbor-image count exceeds frozen guard")
    positions = np.asarray(work.positions, dtype=float)
    cells: list[MinkowskiTensorCell] = []
    for site_index in range(len(work)):
        selected = np.asarray(left == site_index) & np.asarray(right != site_index)
        neighbor_index = np.asarray(right[selected], dtype=int)
        image = np.asarray(shift[selected], dtype=float)
        displacement = positions[neighbor_index] + image @ cell_matrix - positions[site_index]
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(~np.isfinite(distance)) or np.any(distance <= 1.0e-12):
            raise ValueError("NEXT335 zero-distance periodic sites differ")
        candidate_offset = (
            distance**2 + radius[site_index] ** 2 - radius[neighbor_index] ** 2
        ) / (2.0 * distance)
        relevant = candidate_offset <= wigner_seitz_radius + n267.PLANE_DISTANCE_TOLERANCE
        cells.append(
            power_cell_minkowski_tensor(
                normals=np.vstack((lattice_normals, displacement[relevant] / distance[relevant, None])),
                offsets=np.concatenate((lattice_offsets, candidate_offset[relevant])),
                scale=wigner_seitz_radius,
            )
        )
    volume = abs(float(np.linalg.det(cell_matrix)))
    relative_error = abs(math.fsum(cell.volume for cell in cells) - volume) / volume
    if relative_error > n267.VOLUME_TILING_RELATIVE_TOLERANCE:
        raise ValueError(f"NEXT335 volume-tiling certificate differs: {relative_error:.12g}")
    return tuple(cells)


def _geometry_only_atoms(atoms: Atoms) -> Atoms:
    try:
        pbc = np.asarray(atoms.pbc, dtype=bool)
        cell = np.asarray(atoms.cell.array, dtype=float)
        positions = np.asarray(atoms.positions, dtype=float)
        numbers = np.asarray(atoms.numbers)
    except Exception as exc:
        raise ValueError("NEXT335 features require exact periodic geometry-only Atoms") from exc
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
        raise ValueError("NEXT335 features require exact periodic geometry-only Atoms")
    return atoms.copy()


def compute_rfmt_features(atoms: Atoms) -> RFMTFeatureResult:
    try:
        work = _geometry_only_atoms(atoms)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        cells = periodic_radical_minkowski_cells(work, radii=radii)
        empty = np.asarray([cell.empty for cell in cells], dtype=bool)
        if len(cells) != len(work) or empty.any():
            raise ValueError("NEXT335 requires every labelled power cell to be nonempty")
        beta = np.asarray([cell.surface_normal_beta for cell in cells], dtype=float)
        if not np.isfinite(beta).all() or np.any(beta <= 0.0) or np.any(beta > 1.0):
            raise ValueError("NEXT335 site beta population differs")
        q10 = _quantize(float(np.quantile(beta, 0.10, method="inverted_cdf")))
        if q10 <= 0.0:
            raise ValueError("NEXT335 positive site q10 quantized to zero")
        volume = abs(float(np.linalg.det(np.asarray(work.cell.array, dtype=float))))
        tiling = abs(math.fsum(cell.volume for cell in cells) - volume) / volume
        facets = np.asarray([cell.facet_count for cell in cells], dtype=int)
        return RFMTFeatureResult(
            True,
            None,
            len(cells),
            0,
            int(facets.min()),
            int(facets.max()),
            float(beta.min()),
            float(beta.max()),
            float(tiling),
            {FEATURE_NAMES[0]: q10},
        )
    except Exception as exc:
        return _failure(exc)


def compute_rfmt_row(atoms: Atoms) -> dict[str, object]:
    result = compute_rfmt_features(atoms)
    row: dict[str, object] = {FEATURE_NAMES[0]: math.nan}
    row.update(
        {
            "rfmt_supported": bool(result.supported),
            "rfmt_failure": result.failure_reason,
            "rfmt_site_count": result.site_count,
            "rfmt_empty_cell_count": result.empty_cell_count,
            "rfmt_minimum_facet_count": result.minimum_facet_count,
            "rfmt_maximum_facet_count": result.maximum_facet_count,
            "rfmt_minimum_site_beta": result.minimum_site_beta,
            "rfmt_maximum_site_beta": result.maximum_site_beta,
            "rfmt_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row[FEATURE_NAMES[0]] = float(result.features[FEATURE_NAMES[0]])
    return row


def build_cross_source_rfmt_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    raise NotImplementedError("NEXT335 formal builder awaits a passing frozen probe")


__all__ = [
    "BOUNDARY_FLAGS",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "PROTOCOL",
    "MinkowskiTensorCell",
    "RFMTFeatureResult",
    "build_cross_source_rfmt_features",
    "compute_rfmt_features",
    "compute_rfmt_row",
    "periodic_radical_minkowski_cells",
    "power_cell_minkowski_tensor",
    "surface_normal_beta",
]
