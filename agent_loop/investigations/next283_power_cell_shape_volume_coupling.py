#!/usr/bin/env python3
"""Materialize power-cell shape--volume coupling descriptors from raw geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import src.next267_periodic_radical_voronoi_packing as n267
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next283-power-cell-shape-volume-coupling-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT283_PSVC_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next283_scigen_power_cell_shape_volume_coupling.parquet",
    "wyformer": "next283_wyformer_power_cell_shape_volume_coupling.parquet",
}
FEATURE_NAMES = (
    "psvc_sphericity_mean",
    "psvc_sphericity_q10",
    "psvc_log_volume_asphericity_correlation",
    "psvc_inflated_asphericity_mean",
    "psvc_inflated_asphericity_q90",
    "psvc_small_inflated_asphericity_mean",
)
FEATURE_DIRECTIONS = {
    "psvc_sphericity_mean": "protected_high",
    "psvc_sphericity_q10": "protected_high",
    "psvc_log_volume_asphericity_correlation": "protected_low",
    "psvc_inflated_asphericity_mean": "protected_low",
    "psvc_inflated_asphericity_q90": "protected_low",
    "psvc_small_inflated_asphericity_mean": "protected_low",
}
EXPECTED_DESIGN_SHA256 = (
    "b895a074f6ca5c90e7173df1bfbb39eb463c68ede7c0004e8544243a8be726aa"
)
EXPECTED_AMENDMENT_SHA256 = (
    "43f99af8101b5fdf91a46574697703f4729f21da887cfc633d00c7380333bfcd"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n267.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "design": EXPECTED_DESIGN_SHA256,
    "amendment": EXPECTED_AMENDMENT_SHA256,
}
EXPECTED_NEXT267_SOURCE_SHA256 = (
    "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1"
)
BOUNDARY_FLAGS = n267.BOUNDARY_FLAGS
OUTPUT_GRID = n267.OUTPUT_GRID
SPHERICITY_TOLERANCE = 1.0e-7


@dataclass(frozen=True)
class ShapeVolumeCell:
    empty: bool
    volume: float
    surface_area: float
    sphericity: float
    facet_count: int
    vertex_count: int


@dataclass(frozen=True)
class PSVCFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    empty_cell_count: int
    min_facet_count: int
    max_facet_count: int
    minimum_surface_area: float
    minimum_sphericity: float
    maximum_sphericity: float
    volume_tiling_relative_error: float
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PSVCFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PSVCFeatureResult(
        False, reason, 0, 0, 0, 0, math.nan, math.nan, math.nan, math.nan, {}
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def cell_sphericity(volume: float, surface_area: float) -> float:
    """Return the dimensionless sphericity of one positive-volume convex cell."""

    volume = float(volume)
    surface_area = float(surface_area)
    if (
        not math.isfinite(volume)
        or not math.isfinite(surface_area)
        or volume <= 0.0
        or surface_area <= 0.0
    ):
        raise ValueError("NEXT283 cell volume and area differ")
    value = math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / surface_area
    if not math.isfinite(value) or value <= 0.0 or value > 1.0 + SPHERICITY_TOLERANCE:
        raise ValueError("NEXT283 cell sphericity differs")
    return float(min(1.0, value))


def _empty_cell() -> ShapeVolumeCell:
    return ShapeVolumeCell(True, 0.0, 0.0, 0.0, 0, 0)


def _shape_volume_cell(
    *, normals: np.ndarray, offsets: np.ndarray, scale: float
) -> ShapeVolumeCell:
    program = n267.linprog(
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.column_stack((normals, np.ones(len(normals), dtype=float))),
        b_ub=offsets,
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
        raise ValueError(f"NEXT283 Chebyshev linear program differs: {program.message}")
    interior = np.asarray(program.x[:3], dtype=float)
    chebyshev_radius = float(program.x[3])
    if chebyshev_radius <= n267.INTERIOR_TOLERANCE * max(1.0, scale):
        return _empty_cell()
    vertices = n267.HalfspaceIntersection(
        np.column_stack((normals, -offsets)), interior, qhull_options="Qx"
    ).intersections
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
        raise ValueError("NEXT283 power-cell vertex array differs")
    if not np.isfinite(vertices).all():
        raise ValueError("NEXT283 power-cell vertices are non-finite")
    violation = float(np.max(normals @ vertices.T - offsets[:, None]))
    if violation > n267.VERTEX_FEASIBILITY_TOLERANCE * max(1.0, scale):
        raise ValueError("NEXT283 power-cell vertex feasibility differs")
    hull = n267.ConvexHull(vertices, qhull_options="Qx")
    unique_vertices = vertices[np.asarray(hull.vertices, dtype=int)]
    volume = float(hull.volume)
    surface_area = float(hull.area)
    sphericity = cell_sphericity(volume, surface_area)
    facet_count = n267._active_facet_count(
        vertices=unique_vertices,
        normals=normals,
        offsets=offsets,
        scale=scale,
    )
    if facet_count < 4:
        raise ValueError("NEXT283 active facet count differs")
    return ShapeVolumeCell(
        False,
        volume,
        surface_area,
        sphericity,
        facet_count,
        int(len(unique_vertices)),
    )


def periodic_shape_volume_cells(
    atoms, *, radii: Sequence[float] | np.ndarray
) -> tuple[ShapeVolumeCell, ...]:
    """Return exact NEXT267-equivalent power cells with area certificates."""

    work = n267._validated_reduced_atoms(atoms)
    radius = np.asarray(radii, dtype=float)
    if (
        radius.shape != (len(work),)
        or not np.isfinite(radius).all()
        or np.any(radius <= 0.0)
    ):
        raise ValueError("NEXT283 radii must be finite, positive, and site aligned")
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
        raise ValueError("NEXT283 periodic neighbor-image count exceeds frozen guard")
    positions = np.asarray(work.positions, dtype=float)
    cells: list[ShapeVolumeCell] = []
    for site_index in range(len(work)):
        selected = np.asarray(left == site_index) & np.asarray(right != site_index)
        neighbor_index = np.asarray(right[selected], dtype=int)
        image = np.asarray(shift[selected], dtype=float)
        displacement = (
            positions[neighbor_index] + image @ cell_matrix - positions[site_index]
        )
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(distance <= 1.0e-12):
            raise ValueError("NEXT283 zero-distance periodic sites differ")
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
        if not np.isfinite(normals).all() or not np.isfinite(offsets).all():
            raise ValueError("NEXT283 power half-spaces are non-finite")
        cells.append(
            _shape_volume_cell(
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
            f"NEXT283 volume-tiling certificate differs: {relative_error:.12g}"
        )
    return tuple(cells)


def shape_volume_summaries(
    *, volumes: object, surface_areas: object, radii: object
) -> dict[str, float]:
    """Return the six frozen shape--volume summaries for a nonempty population."""

    volume = np.asarray(volumes, dtype=float)
    area = np.asarray(surface_areas, dtype=float)
    radius = np.asarray(radii, dtype=float)
    if (
        volume.ndim != 1
        or area.shape != volume.shape
        or radius.shape != volume.shape
        or not len(volume)
        or not np.isfinite(volume).all()
        or not np.isfinite(area).all()
        or not np.isfinite(radius).all()
        or np.any(volume <= 0.0)
        or np.any(area <= 0.0)
        or np.any(radius <= 0.0)
    ):
        raise ValueError("NEXT283 shape-volume population differs")
    sphericity = n267._quantized(
        [cell_sphericity(v, a) for v, a in zip(volume, area, strict=True)]
    )
    asphericity = n267._quantized(1.0 - sphericity)
    volume_ratio = n267._quantized(
        volume / ((4.0 * math.pi / 3.0) * radius**3)
    )
    if np.any(volume_ratio <= 0.0):
        raise ValueError("NEXT283 radius-normalized volume differs")
    log_volume = n267._quantized(np.log(volume_ratio))
    median = float(np.median(log_volume))
    inflation = n267._quantized(np.maximum(log_volume - median, 0.0))
    burden = n267._quantized(asphericity * inflation)

    centered_volume = log_volume - (
        math.fsum(float(value) for value in log_volume) / len(log_volume)
    )
    centered_shape = asphericity - (
        math.fsum(float(value) for value in asphericity) / len(asphericity)
    )
    volume_ss = math.fsum(float(value) ** 2 for value in centered_volume)
    shape_ss = math.fsum(float(value) ** 2 for value in centered_shape)
    correlation = 0.0
    threshold = (1.0 / OUTPUT_GRID) ** 2
    if volume_ss > threshold and shape_ss > threshold:
        numerator = math.fsum(
            float(left) * float(right)
            for left, right in zip(centered_volume, centered_shape, strict=True)
        )
        correlation = float(
            np.clip(numerator / math.sqrt(volume_ss * shape_ss), -1.0, 1.0)
        )
    small = radius < float(np.mean(radius))
    small_burden = n267._mean(burden[small]) if small.any() else 0.0
    features = {
        "psvc_sphericity_mean": n267._mean(sphericity),
        "psvc_sphericity_q10": n267._quantile(sphericity, 0.10),
        "psvc_log_volume_asphericity_correlation": _quantize(correlation),
        "psvc_inflated_asphericity_mean": n267._mean(burden),
        "psvc_inflated_asphericity_q90": n267._quantile(burden, 0.90),
        "psvc_small_inflated_asphericity_mean": _quantize(small_burden),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        raise ValueError("NEXT283 feature schema or values differ")
    return features


def compute_power_cell_shape_volume_features(atoms) -> PSVCFeatureResult:
    """Compute the frozen PSVC features from composition and initial geometry."""

    try:
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in atoms.get_chemical_symbols()],
            dtype=float,
        )
        cells = periodic_shape_volume_cells(atoms, radii=radii)
        empty = np.asarray([value.empty for value in cells], dtype=bool)
        nonempty = ~empty
        if not nonempty.any():
            raise ValueError("NEXT283 structure has no positive-volume labelled cell")
        cell_volumes = np.asarray([value.volume for value in cells], dtype=float)
        volume = abs(float(np.linalg.det(np.asarray(atoms.cell.array, dtype=float))))
        relative_error = abs(float(cell_volumes.sum()) - volume) / volume
        active_cells = [value for value in cells if not value.empty]
        surface_areas = np.asarray(
            [value.surface_area for value in active_cells], dtype=float
        )
        sphericities = np.asarray(
            [value.sphericity for value in active_cells], dtype=float
        )
        facets = np.asarray([value.facet_count for value in active_cells], dtype=int)
        features = shape_volume_summaries(
            volumes=cell_volumes[nonempty],
            surface_areas=surface_areas,
            radii=radii[nonempty],
        )
        return PSVCFeatureResult(
            True,
            None,
            len(cells),
            int(empty.sum()),
            int(facets.min()),
            int(facets.max()),
            float(surface_areas.min()),
            float(sphericities.min()),
            float(sphericities.max()),
            float(relative_error),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_psvc_row(atoms) -> dict[str, object]:
    result = compute_power_cell_shape_volume_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "psvc_supported": bool(result.supported),
            "psvc_failure": result.failure_reason,
            "psvc_site_count": result.site_count,
            "psvc_empty_cell_count": result.empty_cell_count,
            "psvc_min_facet_count": result.min_facet_count,
            "psvc_max_facet_count": result.max_facet_count,
            "psvc_minimum_surface_area": result.minimum_surface_area,
            "psvc_minimum_sphericity": result.minimum_sphericity,
            "psvc_maximum_sphericity": result.maximum_sphericity,
            "psvc_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "psvc_supported": False,
        "psvc_failure": f"{type(exc).__name__}: {exc}",
        "psvc_site_count": 0,
        "psvc_empty_cell_count": 0,
        "psvc_min_facet_count": 0,
        "psvc_max_facet_count": 0,
        "psvc_minimum_surface_area": math.nan,
        "psvc_minimum_sphericity": math.nan,
        "psvc_maximum_sphericity": math.nan,
        "psvc_volume_tiling_relative_error": math.nan,
    }


def _compute_scigen_payload(item: tuple[str, bytes]):
    material_id, payload = item
    try:
        parsed = n267.n85._parse_frame(payload, strict_output=True)
        return material_id, compute_psvc_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]):
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_psvc_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=4))


def build_power_cell_shape_volume_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    amendment_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT283 from physically isolated discovery geometry only."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "scigen_manifest": scigen / n267.n85.COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / n267.n85.COHORT_METADATA_NAME,
        "scigen_discovery_geometry": scigen / n267.n85.GEOMETRY_NAMES["discovery"],
        "wyformer_manifest": wyformer / n267.n94.COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / n267.n94.COHORT_METADATA_NAME,
        "wyformer_discovery_geometry": wyformer / n267.n94.GEOMETRY_NAMES["discovery"],
        "design": Path(design_path).resolve(),
        "amendment": Path(amendment_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT283 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT283 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT283 formal input identity differs: {differing}")
    if _sha256_file(Path(n267.__file__).resolve()) != EXPECTED_NEXT267_SOURCE_SHA256:
        raise ValueError("NEXT283 frozen NEXT267 source differs")
    scigen_manifest = n267._read_manifest(paths["scigen_manifest"])
    wyformer_manifest = n267._read_manifest(paths["wyformer_manifest"])
    if (
        scigen_manifest.get("protocol") != n267.n85.COHORT_PROTOCOL
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or scigen_manifest.get("relaxed_structures_opened") is not False
        or wyformer_manifest.get("protocol") != n267.n94.COHORT_PROTOCOL
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
        or wyformer_manifest.get("relaxed_structures_published") is not False
    ):
        raise ValueError("NEXT283 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT283 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT283 {source} discovery identity differs")
        discovery[source] = selected
    payloads = {
        "scigen": n267.n85._archive_payloads(
            paths["scigen_discovery_geometry"],
            discovery["scigen"]["material_id"].astype(str).tolist(),
        ),
        "wyformer": n267.n94._payloads(
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
                [
                    {"material_id": material_id, **row}
                    for material_id, row in computed[source]
                ]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT283 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["psvc_supported"].fillna(False).astype(bool)
            finite = np.column_stack(
                [
                    np.isfinite(
                        pd.to_numeric(table[name], errors="coerce").to_numpy(float)
                    )
                    for name in FEATURE_NAMES
                ]
            )
            sites = pd.to_numeric(table["psvc_site_count"], errors="coerce")
            empty = pd.to_numeric(table["psvc_empty_cell_count"], errors="coerce")
            minimum = pd.to_numeric(table["psvc_min_facet_count"], errors="coerce")
            maximum = pd.to_numeric(table["psvc_max_facet_count"], errors="coerce")
            surface = pd.to_numeric(table["psvc_minimum_surface_area"], errors="coerce")
            min_sphericity = pd.to_numeric(table["psvc_minimum_sphericity"], errors="coerce")
            max_sphericity = pd.to_numeric(table["psvc_maximum_sphericity"], errors="coerce")
            tiling = pd.to_numeric(
                table["psvc_volume_tiling_relative_error"], errors="coerce"
            )
            if (
                len(table) != len(discovery[source])
                or not finite[supported].all()
                or finite[~supported].any()
                or not (sites[supported] > 0).all()
                or not (empty[supported] >= 0).all()
                or not (minimum[supported] >= 4).all()
                or not (maximum[supported] >= minimum[supported]).all()
                or not (surface[supported] > 0.0).all()
                or not (min_sphericity[supported] > 0.0).all()
                or not (max_sphericity[supported] >= min_sphericity[supported]).all()
                or not (max_sphericity[supported] <= 1.0).all()
                or not (
                    tiling[supported] <= n267.VOLUME_TILING_RELATIVE_TOLERANCE
                ).all()
            ):
                raise RuntimeError(f"NEXT283 {source} support certificate differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            failures = Counter(table.loc[~supported, "psvc_failure"].astype(str))
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": dict(sorted(failures.items())),
                "finite_feature_counts": {
                    name: int(finite[:, index].sum())
                    for index, name in enumerate(FEATURE_NAMES)
                },
                "site_count": int(sites[supported].sum()),
                "empty_cell_count": int(empty[supported].sum()),
                "minimum_surface_area": float(surface[supported].min()),
                "minimum_sphericity": float(min_sphericity[supported].min()),
                "maximum_sphericity": float(max_sphericity[supported].max()),
                "maximum_volume_tiling_relative_error": float(tiling[supported].max()),
            }
        if (
            counts["scigen"]["rows"] != 13_470  # type: ignore[index]
            or counts["wyformer"]["rows"] != 5_232  # type: ignore[index]
        ):
            raise RuntimeError("NEXT283 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "feature_count": len(FEATURE_NAMES),
            "base_geometry_protocol": n267.PROTOCOL,
            "cell_geometry": ["convex_hull_volume", "convex_hull_surface_area", "sphericity"],
            "coupling_geometry": [
                "log_radius_normalized_volume",
                "asphericity",
                "positive_median_centered_inflation",
                "small_generator_radius_below_structure_mean",
            ],
            "radius_policy": ["atomic_radius_calculated", "atomic_radius_fallback"],
            "power_distance": "squared_euclidean_minus_radius_squared",
            "median_method": "numpy_linear",
            "quantile_method": "inverted_cdf",
            "output_grid": OUTPUT_GRID,
            "sphericity_tolerance": SPHERICITY_TOLERANCE,
            "volume_tiling_relative_tolerance": n267.VOLUME_TILING_RELATIVE_TOLERANCE,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_power_cell_shape_volume_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "next284_audit_authorized": True,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "upstream_source_sha256": {
                "src/next267_periodic_radical_voronoi_packing.py": EXPECTED_NEXT267_SOURCE_SHA256
            },
            "executed_source_sha256": {
                "src/next283_power_cell_shape_volume_coupling.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT283 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT283 source changed before publication")
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
    parser.add_argument("--amendment-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_power_cell_shape_volume_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        amendment_path=args.amendment_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
