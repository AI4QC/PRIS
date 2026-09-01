#!/usr/bin/env python3
"""Materialize periodic radical-packing autocorrelation descriptors."""

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


PROTOCOL = "2026-08-09-next279-radical-packing-autocorrelation-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT279_PRPA_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next279_scigen_radical_packing_autocorrelation.parquet",
    "wyformer": "next279_wyformer_radical_packing_autocorrelation.parquet",
}
FEATURE_NAMES = (
    "prpa_volume_moran",
    "prpa_volume_geary",
    "prpa_volume_absolute_moran",
    "prpa_volume_extreme_edge_fraction",
    "prpa_chebyshev_moran",
    "prpa_chebyshev_geary",
    "prpa_chebyshev_absolute_moran",
    "prpa_chebyshev_extreme_edge_fraction",
)
FEATURE_DIRECTIONS = {
    "prpa_volume_moran": "protected_low",
    "prpa_volume_geary": "protected_high",
    "prpa_volume_absolute_moran": "protected_low",
    "prpa_volume_extreme_edge_fraction": "protected_low",
    "prpa_chebyshev_moran": "protected_low",
    "prpa_chebyshev_geary": "protected_high",
    "prpa_chebyshev_absolute_moran": "protected_low",
    "prpa_chebyshev_extreme_edge_fraction": "protected_low",
}
EXPECTED_DESIGN_SHA256 = (
    "36b6e1e67ce81e15802b3604e80509d9a9c49aa83b3d9ee3c9d77a5713e72fd5"
)
EXPECTED_AMENDMENT_SHA256 = (
    "9fe2d1e03693e170e14881d7a7c93949962f7de6f2eeb2796f3c2a8fc6e09d10"
)
EXPECTED_SECOND_AMENDMENT_SHA256 = (
    "cb5d767c0cc2af263993d2395103698880ade16fe1ba04a2f24973dababe7db2"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n267.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "design": EXPECTED_DESIGN_SHA256,
    "amendment": EXPECTED_AMENDMENT_SHA256,
    "second_amendment": EXPECTED_SECOND_AMENDMENT_SHA256,
}
EXPECTED_NEXT267_SOURCE_SHA256 = (
    "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1"
)
BOUNDARY_FLAGS = n267.BOUNDARY_FLAGS
OUTPUT_GRID = n267.OUTPUT_GRID


@dataclass(frozen=True)
class ContactIncidence:
    center: int
    neighbor: int
    displacement: tuple[float, float, float]


@dataclass(frozen=True)
class PRPAFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    empty_cell_count: int
    min_facet_count: int
    max_facet_count: int
    contact_count: int
    volume_tiling_relative_error: float
    features: Mapping[str, float]


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def packing_autocorrelation_features(
    *, values: object, contacts: Sequence[Sequence[int]]
) -> dict[str, float]:
    """Return the four frozen periodic graph autocorrelation summaries."""

    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or len(array) < 2
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
        or len(contacts) == 0
    ):
        raise ValueError("NEXT279 autocorrelation population differs")
    pairs = np.asarray(contacts, dtype=object)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("NEXT279 autocorrelation population differs")
    try:
        pair_index = pairs.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NEXT279 autocorrelation population differs") from exc
    if (
        not np.equal(pairs, pair_index).all()
        or np.any(pair_index < 0)
        or np.any(pair_index >= len(array))
    ):
        raise ValueError("NEXT279 autocorrelation population differs")
    residual = np.log(array)
    residual -= math.fsum(float(value) for value in residual) / len(residual)
    total = math.fsum(float(value) ** 2 for value in residual)
    if total <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(residual))) ** 2):
        return {
            "moran": 0.0,
            "geary": 0.0,
            "absolute_moran": 0.0,
            "extreme_edge_fraction": 0.0,
        }
    left = pair_index[:, 0]
    right = pair_index[:, 1]
    count = len(pair_index)
    moran = (
        len(array)
        / count
        * math.fsum(float(residual[i]) * float(residual[j]) for i, j in pair_index)
        / total
    )
    geary = (
        len(array)
        / (2.0 * count)
        * math.fsum((float(residual[i]) - float(residual[j])) ** 2 for i, j in pair_index)
        / total
    )
    magnitude = np.abs(residual)
    centered_magnitude = magnitude - (
        math.fsum(float(value) for value in magnitude) / len(magnitude)
    )
    magnitude_total = math.fsum(float(value) ** 2 for value in centered_magnitude)
    absolute_moran = 0.0
    if magnitude_total > np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(centered_magnitude))) ** 2
    ):
        absolute_moran = (
            len(array)
            / count
            * math.fsum(
                float(centered_magnitude[i]) * float(centered_magnitude[j])
                for i, j in pair_index
            )
            / magnitude_total
        )
    threshold = float(np.quantile(magnitude, 0.75, method="inverted_cdf"))
    extreme = magnitude >= threshold
    extreme_fraction = float(np.mean(extreme[left] & extreme[right]))
    result = {
        "moran": moran,
        "geary": geary,
        "absolute_moran": absolute_moran,
        "extreme_edge_fraction": extreme_fraction,
    }
    if not np.isfinite(list(result.values())).all():
        raise RuntimeError("NEXT279 autocorrelation result is non-finite")
    return result


def _plane_key(normal: np.ndarray, offset: float) -> tuple[int, ...]:
    return (
        *np.rint(np.asarray(normal, dtype=float) * OUTPUT_GRID)
        .astype(np.int64)
        .tolist(),
        int(np.rint(float(offset) * OUTPUT_GRID)),
    )


def _displacement_key(displacement: np.ndarray) -> tuple[int, int, int]:
    values = np.rint(np.asarray(displacement, dtype=float) * OUTPUT_GRID).astype(
        np.int64
    )
    if values.shape != (3,):
        raise ValueError("NEXT279 contact displacement differs")
    return tuple(int(value) for value in values)


def _active_plane_labels(
    *,
    vertices: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    labels: Sequence[tuple[int, np.ndarray]],
    scale: float,
) -> tuple[tuple[tuple[int, np.ndarray], ...], int]:
    tolerance = n267.PLANE_DISTANCE_TOLERANCE * max(1.0, float(scale))
    grouped: dict[
        tuple[int, ...],
        tuple[np.ndarray, float, dict[tuple[int, tuple[int, int, int]], tuple[int, np.ndarray]]],
    ] = {}
    for normal, offset, label in zip(normals, offsets, labels, strict=True):
        key = _plane_key(normal, float(offset))
        if key not in grouped:
            grouped[key] = (np.asarray(normal), float(offset), {})
        label_key = (int(label[0]), _displacement_key(label[1]))
        grouped[key][2].setdefault(
            label_key, (int(label[0]), np.asarray(label[1], dtype=float))
        )
    active: list[tuple[int, np.ndarray]] = []
    plane_count = 0
    for normal, offset, plane_labels in grouped.values():
        points = vertices[np.abs(vertices @ normal - offset) <= tolerance]
        if len(points) < 3:
            continue
        singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
        if int(np.sum(singular > tolerance)) < 2:
            continue
        plane_count += 1
        active.extend(plane_labels[key] for key in sorted(plane_labels))
    return tuple(active), plane_count


def _power_cell_with_contacts(
    *,
    normals: np.ndarray,
    offsets: np.ndarray,
    labels: Sequence[tuple[int, np.ndarray]],
    generator_margin: float,
    scale: float,
) -> tuple[n267.PowerCell, tuple[tuple[int, np.ndarray], ...]]:
    program = n267.linprog(
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.column_stack((normals, np.ones(len(normals), dtype=float))),
        b_ub=offsets,
        bounds=[(None, None)] * 4,
        method="highs",
    )
    if program.status == 2:
        return n267._empty_cell(generator_margin), ()
    if not program.success or program.x.shape != (4,) or not np.isfinite(program.x).all():
        raise ValueError(f"NEXT279 Chebyshev linear program differs: {program.message}")
    interior = np.asarray(program.x[:3], dtype=float)
    chebyshev_radius = float(program.x[3])
    if chebyshev_radius <= n267.INTERIOR_TOLERANCE * max(1.0, scale):
        return n267._empty_cell(generator_margin), ()
    vertices = n267.HalfspaceIntersection(
        np.column_stack((normals, -offsets)), interior, qhull_options="Qx"
    ).intersections
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
        raise ValueError("NEXT279 power-cell vertex array differs")
    if not np.isfinite(vertices).all():
        raise ValueError("NEXT279 power-cell vertices are non-finite")
    violation = float(np.max(normals @ vertices.T - offsets[:, None]))
    if violation > n267.VERTEX_FEASIBILITY_TOLERANCE * max(1.0, scale):
        raise ValueError("NEXT279 power-cell vertex feasibility differs")
    hull = n267.ConvexHull(vertices, qhull_options="Qx")
    unique_vertices = vertices[np.asarray(hull.vertices, dtype=int)]
    volume = float(hull.volume)
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("NEXT279 power-cell volume differs")
    centroid = n267._polyhedron_centroid(
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
        raise ValueError("NEXT279 vertex-covariance spectrum differs")
    anisotropy = float(
        np.clip(1.0 - max(0.0, eigenvalues[0]) / eigenvalues[-1], 0.0, 1.0)
    )
    active, facet_count = _active_plane_labels(
        vertices=unique_vertices,
        normals=normals,
        offsets=offsets,
        labels=labels,
        scale=scale,
    )
    if facet_count < 4:
        raise ValueError("NEXT279 active facet count differs")
    cell = n267.PowerCell(
        False,
        volume,
        chebyshev_radius,
        generator_margin,
        centroid_offset,
        anisotropy,
        facet_count,
        int(len(unique_vertices)),
    )
    return cell, active


def periodic_radical_cells_and_contacts(
    atoms, *, radii: Sequence[float] | np.ndarray
) -> tuple[tuple[n267.PowerCell, ...], tuple[ContactIncidence, ...]]:
    """Return NEXT267-equivalent cells and active-facet contact incidences."""

    work = n267._validated_reduced_atoms(atoms)
    radius = np.asarray(radii, dtype=float)
    if radius.shape != (len(work),) or not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("NEXT279 radii must be finite, positive, and site aligned")
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
        raise ValueError("NEXT279 periodic neighbor-image count exceeds frozen guard")
    positions = np.asarray(work.positions, dtype=float)
    cells: list[n267.PowerCell] = []
    contacts: list[ContactIncidence] = []
    lattice_displacement = 2.0 * lattice_offsets[:, None] * lattice_normals
    for site_index in range(len(work)):
        selected = np.asarray(left == site_index) & np.asarray(right != site_index)
        neighbor_index = np.asarray(right[selected], dtype=int)
        image = np.asarray(shift[selected], dtype=float)
        displacement = (
            positions[neighbor_index] + image @ cell_matrix - positions[site_index]
        )
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(distance <= 1.0e-12):
            raise ValueError("NEXT279 zero-distance periodic sites differ")
        candidate_offset = (
            distance**2 + radius[site_index] ** 2 - radius[neighbor_index] ** 2
        ) / (2.0 * distance)
        relevant = candidate_offset <= (
            wigner_seitz_radius + n267.PLANE_DISTANCE_TOLERANCE
        )
        candidate_normals = displacement[relevant] / distance[relevant, None]
        candidate_offsets = candidate_offset[relevant]
        normals = np.vstack((lattice_normals, candidate_normals))
        offsets = np.concatenate((lattice_offsets, candidate_offsets))
        labels = [
            (site_index, np.asarray(value, dtype=float))
            for value in lattice_displacement
        ]
        labels.extend(
            (int(other), np.asarray(vector, dtype=float))
            for other, vector in zip(
                neighbor_index[relevant], displacement[relevant], strict=True
            )
        )
        if not np.isfinite(normals).all() or not np.isfinite(offsets).all():
            raise ValueError("NEXT279 power half-spaces are non-finite")
        generator_margin = float(np.min(offsets))
        power_cell, active = _power_cell_with_contacts(
            normals=normals,
            offsets=offsets,
            labels=labels,
            generator_margin=generator_margin,
            scale=wigner_seitz_radius,
        )
        cells.append(power_cell)
        contacts.extend(
            ContactIncidence(
                center=site_index,
                neighbor=int(other),
                displacement=tuple(float(value) for value in vector),
            )
            for other, vector in active
        )
    volume = abs(float(np.linalg.det(cell_matrix)))
    tiled = math.fsum(value.volume for value in cells)
    relative_error = abs(tiled - volume) / volume
    if (
        not math.isfinite(relative_error)
        or relative_error > n267.VOLUME_TILING_RELATIVE_TOLERANCE
    ):
        raise ValueError(
            f"NEXT279 volume-tiling certificate differs: {relative_error:.12g}"
        )
    return tuple(cells), tuple(contacts)


def _contact_counter(contacts: Sequence[ContactIncidence]) -> Counter:
    return Counter(
        (
            int(contact.center),
            int(contact.neighbor),
            _displacement_key(np.asarray(contact.displacement, dtype=float)),
        )
        for contact in contacts
    )


def contacts_are_reciprocal(contacts: Sequence[ContactIncidence]) -> bool:
    counts = _contact_counter(contacts)
    reversed_counts = Counter(
        {
            (neighbor, center, tuple(-value for value in displacement)): count
            for (center, neighbor, displacement), count in counts.items()
        }
    )
    return counts == reversed_counts


def _failure(exc: Exception) -> PRPAFeatureResult:
    return PRPAFeatureResult(
        False,
        f"{type(exc).__name__}: {exc}",
        0,
        0,
        0,
        0,
        0,
        math.nan,
        {name: math.nan for name in FEATURE_NAMES},
    )


def compute_radical_packing_autocorrelation_features(atoms) -> PRPAFeatureResult:
    """Compute the frozen PRPA features from initial geometry only."""

    try:
        if len(atoms) < 2:
            raise ValueError("NEXT279 requires at least two sites")
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
        radii = np.asarray([n267._tabulated_radius(str(value)) for value in symbols])
        cells, contacts = periodic_radical_cells_and_contacts(atoms, radii=radii)
        empty = np.asarray([value.empty for value in cells], dtype=bool)
        if empty.any():
            raise ValueError("NEXT279 graph requires every labelled cell to be nonempty")
        if len(contacts) == 0:
            raise ValueError("NEXT279 graph has no active-facet contact incidence")
        if not contacts_are_reciprocal(contacts):
            raise ValueError("NEXT279 active-facet contacts are not reciprocal")
        cell_volumes = np.asarray([value.volume for value in cells], dtype=float)
        volume = abs(float(np.linalg.det(np.asarray(atoms.cell.array, dtype=float))))
        relative_error = abs(float(cell_volumes.sum()) - volume) / volume
        volume_ratio = cell_volumes / ((4.0 * math.pi / 3.0) * radii**3)
        chebyshev_ratio = np.asarray(
            [value.chebyshev_radius for value in cells], dtype=float
        ) / radii
        pairs = [(value.center, value.neighbor) for value in contacts]
        volume_stats = packing_autocorrelation_features(
            values=volume_ratio, contacts=pairs
        )
        chebyshev_stats = packing_autocorrelation_features(
            values=chebyshev_ratio, contacts=pairs
        )
        features = {
            "prpa_volume_moran": _quantize(volume_stats["moran"]),
            "prpa_volume_geary": _quantize(volume_stats["geary"]),
            "prpa_volume_absolute_moran": _quantize(
                volume_stats["absolute_moran"]
            ),
            "prpa_volume_extreme_edge_fraction": _quantize(
                volume_stats["extreme_edge_fraction"]
            ),
            "prpa_chebyshev_moran": _quantize(chebyshev_stats["moran"]),
            "prpa_chebyshev_geary": _quantize(chebyshev_stats["geary"]),
            "prpa_chebyshev_absolute_moran": _quantize(
                chebyshev_stats["absolute_moran"]
            ),
            "prpa_chebyshev_extreme_edge_fraction": _quantize(
                chebyshev_stats["extreme_edge_fraction"]
            ),
        }
        if tuple(features) != FEATURE_NAMES or not np.isfinite(
            list(features.values())
        ).all():
            raise ValueError("NEXT279 feature schema or values differ")
        facets = np.asarray([value.facet_count for value in cells], dtype=int)
        return PRPAFeatureResult(
            True,
            None,
            len(cells),
            0,
            int(facets.min()),
            int(facets.max()),
            len(contacts),
            float(relative_error),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_prpa_row(atoms) -> dict[str, object]:
    result = compute_radical_packing_autocorrelation_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "prpa_supported": bool(result.supported),
            "prpa_failure": result.failure_reason,
            "prpa_site_count": result.site_count,
            "prpa_empty_cell_count": result.empty_cell_count,
            "prpa_min_facet_count": result.min_facet_count,
            "prpa_max_facet_count": result.max_facet_count,
            "prpa_contact_count": result.contact_count,
            "prpa_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "prpa_supported": False,
        "prpa_failure": f"{type(exc).__name__}: {exc}",
        "prpa_site_count": 0,
        "prpa_empty_cell_count": 0,
        "prpa_min_facet_count": 0,
        "prpa_max_facet_count": 0,
        "prpa_contact_count": 0,
        "prpa_volume_tiling_relative_error": math.nan,
    }


def _compute_scigen_payload(item: tuple[str, bytes]):
    material_id, payload = item
    try:
        parsed = n267.n85._parse_frame(payload, strict_output=True)
        return material_id, compute_prpa_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]):
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_prpa_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=4))


def build_radical_packing_autocorrelation_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    amendment_path: Path,
    second_amendment_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT279 from physically isolated discovery geometry only."""

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
        "second_amendment": Path(second_amendment_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT279 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT279 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT279 formal input identity differs: {differing}")
    if _sha256_file(Path(n267.__file__).resolve()) != EXPECTED_NEXT267_SOURCE_SHA256:
        raise ValueError("NEXT279 frozen NEXT267 source differs")
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
        raise ValueError("NEXT279 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT279 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT279 {source} discovery identity differs")
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
                raise RuntimeError(f"NEXT279 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["prpa_supported"].fillna(False).astype(bool)
            finite = np.column_stack(
                [
                    np.isfinite(
                        pd.to_numeric(table[name], errors="coerce").to_numpy(float)
                    )
                    for name in FEATURE_NAMES
                ]
            )
            tiling = pd.to_numeric(
                table["prpa_volume_tiling_relative_error"], errors="coerce"
            )
            sites = pd.to_numeric(table["prpa_site_count"], errors="coerce")
            contacts = pd.to_numeric(table["prpa_contact_count"], errors="coerce")
            minimum = pd.to_numeric(table["prpa_min_facet_count"], errors="coerce")
            maximum = pd.to_numeric(table["prpa_max_facet_count"], errors="coerce")
            empty = pd.to_numeric(table["prpa_empty_cell_count"], errors="coerce")
            if (
                len(table) != len(discovery[source])
                or not finite[supported].all()
                or finite[~supported].any()
                or not (sites[supported] >= 2).all()
                or not (contacts[supported] > 0).all()
                or not (contacts[supported] % 2 == 0).all()
                or not (empty[supported] == 0).all()
                or not (minimum[supported] >= 4).all()
                or not (maximum[supported] >= minimum[supported]).all()
                or not (
                    tiling[supported] <= n267.VOLUME_TILING_RELATIVE_TOLERANCE
                ).all()
            ):
                raise RuntimeError(f"NEXT279 {source} support certificate differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            failures = Counter(table.loc[~supported, "prpa_failure"].astype(str))
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
                "directed_contact_count": int(contacts[supported].sum()),
                "maximum_volume_tiling_relative_error": float(
                    tiling[supported].max()
                ),
            }
        if (
            counts["scigen"]["rows"] != 13_470  # type: ignore[index]
            or counts["wyformer"]["rows"] != 5_232  # type: ignore[index]
        ):
            raise RuntimeError("NEXT279 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "feature_count": len(FEATURE_NAMES),
            "base_geometry_protocol": n267.PROTOCOL,
            "contact_graph": "reciprocal_active_periodic_radical_facet_incidences",
            "contact_multiplicity": "periodic_generator_image_preserving",
            "site_values": ["log_radius_normalized_volume", "log_radius_normalized_chebyshev"],
            "autocorrelation": [
                "moran_N_over_W",
                "periodic_geary_N_over_2W",
                "absolute_residual_moran_N_over_W",
                "inverted_cdf_q75_extreme_edge_fraction",
            ],
            "radius_policy": ["atomic_radius_calculated", "atomic_radius_fallback"],
            "power_distance": "squared_euclidean_minus_radius_squared",
            "output_grid": OUTPUT_GRID,
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
            "mode": "physically_isolated_discovery_x0_radical_packing_autocorrelation_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "next280_audit_authorized": True,
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
                "src/next279_radical_packing_autocorrelation.py": source_hash
            },
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in output_paths
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT279 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT279 source changed before publication")
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
    parser.add_argument("--second-amendment-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_radical_packing_autocorrelation_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        amendment_path=args.amendment_path,
        second_amendment_path=args.second_amendment_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
