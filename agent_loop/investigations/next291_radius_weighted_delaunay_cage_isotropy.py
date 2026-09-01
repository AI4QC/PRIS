#!/usr/bin/env python3
"""Build radius-weighted regular-Delaunay cage features from raw geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
import pandas as pd

import src.next267_periodic_radical_voronoi_packing as n267
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next291-radius-weighted-delaunay-cage-isotropy-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT291_RWDCI_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next291_scigen_radius_weighted_delaunay_cage_features.parquet",
    "wyformer": "next291_wyformer_radius_weighted_delaunay_cage_features.parquet",
}
METRIC_NAMES = ("tightness", "volume", "eigenratio", "closure")
STATISTIC_NAMES = ("mean", "q10", "q25", "lower_quartile_mean")
FEATURE_NAMES = tuple(
    f"rwdci_{metric}_{statistic}"
    for metric in METRIC_NAMES
    for statistic in STATISTIC_NAMES
)
FEATURE_DIRECTIONS = {name: "protected_high" for name in FEATURE_NAMES}
EXPECTED_DESIGN_SHA256 = (
    "6f477c750a27c3b5dd6c53d4637b94c8f18792c8aaecf0f3d448122b85a4cbf1"
)
EXPECTED_INPUT_SHA256 = {
    **{
        key: value
        for key, value in n267.EXPECTED_INPUT_SHA256.items()
        if key != "design"
    },
    "design": EXPECTED_DESIGN_SHA256,
}
EXPECTED_NEXT267_SOURCE_SHA256 = (
    "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1"
)
BOUNDARY_FLAGS = n267.BOUNDARY_FLAGS
ACTIVE_PLANE_RELATIVE_TOLERANCE = 1.0e-8
ACTIVE_NORMAL_RANK_TOLERANCE = 1.0e-10
POWER_EQUALITY_RELATIVE_TOLERANCE = 1.0e-7
CAGE_DISTANCE_TOLERANCE = 1.0e-12
NUMERICAL_TOLERANCE = 1.0e-12
LOWER_QUARTILE_INCLUSION_TOLERANCE = 1.0e-12
MINIMUM_FORMAL_COVERAGE = 0.95


@dataclass(frozen=True)
class RWDCIFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    empty_cell_count: int
    incidence_count: int
    min_cage_size: int
    max_cage_size: int
    volume_tiling_relative_error: float
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> RWDCIFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return RWDCIFeatureResult(False, reason, 0, 0, 0, 0, 0, math.nan, {})


def _bounded(value: float, *, label: str) -> float:
    value = float(value)
    if (
        not math.isfinite(value)
        or value < -NUMERICAL_TOLERANCE
        or value > 1.0 + NUMERICAL_TOLERANCE
    ):
        raise ValueError(f"NEXT291 {label} metric differs")
    return float(np.clip(value, 0.0, 1.0))


def weighted_cage_metrics(*, vectors: object, radii: object) -> dict[str, float]:
    """Return four bounded invariants of one radius-weighted cage."""

    vector = np.asarray(vectors, dtype=float)
    radius = np.asarray(radii, dtype=float)
    if (
        vector.ndim != 2
        or vector.shape[1:] != (3,)
        or len(vector) < 4
        or not np.isfinite(vector).all()
    ):
        raise ValueError("NEXT291 cage population differs")
    if (
        radius.shape != (len(vector),)
        or not np.isfinite(radius).all()
        or np.any(radius <= 0.0)
    ):
        raise ValueError("NEXT291 cage radii differ")
    distance = np.linalg.norm(vector, axis=1)
    if not np.isfinite(distance).all() or np.any(distance <= CAGE_DISTANCE_TOLERANCE):
        raise ValueError("NEXT291 cage distance differs")
    direction = vector / distance[:, None]
    raw_weight = (radius / distance) ** 2
    total = math.fsum(float(value) for value in raw_weight)
    if not math.isfinite(total) or total <= 0.0 or not np.isfinite(raw_weight).all():
        raise ValueError("NEXT291 cage angular weight differs")
    weight = raw_weight / total
    tensor = np.einsum("i,ij,ik->jk", weight, direction, direction)
    tensor = 0.5 * (tensor + tensor.T)
    eigenvalues = np.linalg.eigvalsh(tensor)
    if (
        eigenvalues.shape != (3,)
        or not np.isfinite(eigenvalues).all()
        or eigenvalues[0] < -NUMERICAL_TOLERANCE
        or eigenvalues[-1] <= 0.0
        or abs(float(eigenvalues.sum()) - 1.0) > NUMERICAL_TOLERANCE
    ):
        raise ValueError("NEXT291 weighted cage tensor spectrum differs")
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    resultant = np.einsum("i,ij->j", weight, direction)
    result = {
        "tightness": _bounded(3.0 * eigenvalues[0], label="tightness"),
        "volume": _bounded(27.0 * float(np.prod(eigenvalues)), label="volume"),
        "eigenratio": _bounded(
            float(eigenvalues[0] / eigenvalues[-1]), label="eigenratio"
        ),
        "closure": _bounded(
            1.0 - float(np.linalg.norm(resultant)), label="closure"
        ),
    }
    if tuple(result) != METRIC_NAMES:
        raise RuntimeError("NEXT291 weighted cage metric schema differs")
    return result


def _population_mean(values: np.ndarray) -> float:
    return float(math.fsum(float(value) for value in values) / len(values))


def aggregate_rwdci_features(
    metric_populations: Mapping[str, object],
) -> dict[str, float]:
    """Aggregate incidence metrics into the frozen sixteen-feature schema."""

    if tuple(metric_populations) != METRIC_NAMES:
        raise ValueError("NEXT291 metric population schema differs")
    features: dict[str, float] = {}
    expected_count: int | None = None
    for metric in METRIC_NAMES:
        values = np.asarray(metric_populations[metric], dtype=float)
        if (
            values.ndim != 1
            or len(values) == 0
            or not np.isfinite(values).all()
            or np.any(values < -NUMERICAL_TOLERANCE)
            or np.any(values > 1.0 + NUMERICAL_TOLERANCE)
        ):
            raise ValueError("NEXT291 metric population differs")
        if expected_count is None:
            expected_count = len(values)
        elif len(values) != expected_count:
            raise ValueError("NEXT291 incidence accounting differs")
        values = n267._quantized(np.clip(values, 0.0, 1.0))
        q10 = float(np.quantile(values, 0.10, method="inverted_cdf"))
        q25 = float(np.quantile(values, 0.25, method="inverted_cdf"))
        lower = values[values <= q25 + LOWER_QUARTILE_INCLUSION_TOLERANCE]
        if not len(lower):
            raise RuntimeError("NEXT291 lower-quartile population differs")
        features[f"rwdci_{metric}_mean"] = _population_mean(values)
        features[f"rwdci_{metric}_q10"] = q10
        features[f"rwdci_{metric}_q25"] = q25
        features[f"rwdci_{metric}_lower_quartile_mean"] = _population_mean(lower)
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        raise ValueError("NEXT291 aggregate feature schema differs")
    return features


def _geometry_only_reduced_atoms(atoms: Atoms) -> Atoms:
    if (
        not isinstance(atoms, Atoms)
        or len(atoms) < 1
        or atoms.calc is not None
        or bool(atoms.info)
        or set(atoms.arrays) != {"numbers", "positions"}
        or np.asarray(atoms.pbc, dtype=bool).shape != (3,)
        or not np.all(atoms.pbc)
    ):
        raise ValueError("NEXT291 features require exact periodic geometry-only Atoms")
    return n267._validated_reduced_atoms(atoms)


def _lattice_translations() -> np.ndarray:
    return np.asarray(
        [
            value
            for value in itertools.product(
                range(-n267.TRANSLATION_RANGE, n267.TRANSLATION_RANGE + 1),
                repeat=3,
            )
            if value != (0, 0, 0)
        ],
        dtype=int,
    )


def _power_cell_incidence_metrics(
    *,
    site_index: int,
    radius: np.ndarray,
    lattice_normals: np.ndarray,
    lattice_offsets: np.ndarray,
    lattice_translations: np.ndarray,
    cell_matrix: np.ndarray,
    positions: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    shift: np.ndarray,
    wigner_seitz_radius: float,
) -> tuple[float, list[dict[str, float]], list[int], bool]:
    selected = np.asarray(left == site_index) & np.asarray(right != site_index)
    neighbor_index = np.asarray(right[selected], dtype=int)
    image = np.asarray(shift[selected], dtype=int)
    displacement = (
        positions[neighbor_index] + image @ cell_matrix - positions[site_index]
    )
    distance = np.linalg.norm(displacement, axis=1)
    if np.any(~np.isfinite(distance)) or np.any(distance <= CAGE_DISTANCE_TOLERANCE):
        raise ValueError("NEXT291 zero-distance periodic sites differ")
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
        raise ValueError("NEXT291 power half-spaces are non-finite")

    lattice_displacements = lattice_translations @ cell_matrix
    records: list[tuple[int, tuple[int, int, int], np.ndarray, float]] = [
        (
            site_index,
            tuple(int(value) for value in translation),
            np.asarray(disp, dtype=float),
            float(radius[site_index]),
        )
        for translation, disp in zip(
            lattice_translations, lattice_displacements, strict=True
        )
    ]
    records.extend(
        (
            int(neighbor),
            tuple(int(value) for value in neighbor_image),
            np.asarray(disp, dtype=float),
            float(radius[int(neighbor)]),
        )
        for neighbor, neighbor_image, disp in zip(
            neighbor_index[relevant], image[relevant], displacement[relevant], strict=True
        )
    )
    if len(records) != len(normals):
        raise RuntimeError("NEXT291 plane-generator accounting differs")

    program = n267.linprog(
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.column_stack((normals, np.ones(len(normals), dtype=float))),
        b_ub=offsets,
        bounds=[(None, None)] * 4,
        method="highs",
    )
    if program.status == 2:
        return 0.0, [], [], True
    if (
        not program.success
        or program.x.shape != (4,)
        or not np.isfinite(program.x).all()
    ):
        raise ValueError(f"NEXT291 Chebyshev linear program differs: {program.message}")
    interior = np.asarray(program.x[:3], dtype=float)
    chebyshev_radius = float(program.x[3])
    if chebyshev_radius <= n267.INTERIOR_TOLERANCE * max(1.0, wigner_seitz_radius):
        return 0.0, [], [], True
    vertices = n267.HalfspaceIntersection(
        np.column_stack((normals, -offsets)), interior, qhull_options="Qx"
    ).intersections
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 4
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("NEXT291 power-cell vertex array differs")
    violation = float(np.max(normals @ vertices.T - offsets[:, None]))
    if violation > n267.VERTEX_FEASIBILITY_TOLERANCE * max(
        1.0, wigner_seitz_radius
    ):
        raise ValueError("NEXT291 power-cell vertex feasibility differs")
    hull = n267.ConvexHull(vertices, qhull_options="Qx")
    unique_vertices = vertices[np.asarray(hull.vertices, dtype=int)]
    volume = float(hull.volume)
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("NEXT291 power-cell volume differs")

    active_tolerance = ACTIVE_PLANE_RELATIVE_TOLERANCE * max(
        1.0, wigner_seitz_radius
    )
    equality_tolerance = POWER_EQUALITY_RELATIVE_TOLERANCE * max(
        1.0, wigner_seitz_radius**2, float(np.max(radius)) ** 2
    )
    metrics: list[dict[str, float]] = []
    cage_sizes: list[int] = []
    for vertex in unique_vertices:
        active = np.abs(normals @ vertex - offsets) <= active_tolerance
        if np.linalg.matrix_rank(
            normals[active], tol=ACTIVE_NORMAL_RANK_TOLERANCE
        ) != 3:
            raise ValueError("NEXT291 active plane rank differs")
        generators: dict[
            tuple[int, tuple[int, int, int]], tuple[np.ndarray, float]
        ] = {}
        for is_active, record in zip(active, records, strict=True):
            if not is_active:
                continue
            neighbor, neighbor_image, disp, neighbor_radius = record
            key = (neighbor, neighbor_image)
            previous = generators.get(key)
            if previous is not None:
                if (
                    not np.allclose(previous[0], disp, rtol=0.0, atol=1.0e-10)
                    or previous[1] != neighbor_radius
                ):
                    raise ValueError("NEXT291 duplicate generator plane differs")
                continue
            generators[key] = (disp, neighbor_radius)
        if len(generators) < 3:
            raise ValueError("NEXT291 active generator population differs")
        vectors = [-np.asarray(vertex, dtype=float)]
        cage_radii = [float(radius[site_index])]
        for key in sorted(generators):
            disp, neighbor_radius = generators[key]
            vectors.append(np.asarray(disp, dtype=float) - vertex)
            cage_radii.append(float(neighbor_radius))
        vector_array = np.asarray(vectors, dtype=float)
        radius_array = np.asarray(cage_radii, dtype=float)
        power = np.sum(vector_array**2, axis=1) - radius_array**2
        if (
            not np.isfinite(power).all()
            or float(np.max(np.abs(power - power[0]))) > equality_tolerance
        ):
            raise ValueError("NEXT291 common power distance differs")
        metrics.append(
            weighted_cage_metrics(vectors=vector_array, radii=radius_array)
        )
        cage_sizes.append(len(vector_array))
    if not metrics:
        raise ValueError("NEXT291 positive power cell has no cage incidence")
    return volume, metrics, cage_sizes, False


def compute_rwdci_features(atoms: Atoms) -> RWDCIFeatureResult:
    """Compute frozen radius-weighted cage invariants without any outcome."""

    try:
        work = _geometry_only_reduced_atoms(atoms)
        radius = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        cell_matrix = np.asarray(work.cell.array, dtype=float)
        lattice_normals, lattice_offsets, wigner_seitz_radius = (
            n267._lattice_wigner_seitz(cell_matrix)
        )
        lattice_translations = _lattice_translations()
        if len(lattice_translations) != len(lattice_normals):
            raise RuntimeError("NEXT291 lattice-plane accounting differs")
        cutoff = (
            wigner_seitz_radius
            + math.sqrt(wigner_seitz_radius**2 + float(np.max(radius)) ** 2)
            + n267.PLANE_DISTANCE_TOLERANCE
        )
        left, right, shift = n267.neighbor_list(
            "ijS", work, cutoff, self_interaction=False, max_nbins=1_000_000
        )
        if len(left) > n267.MAX_NEIGHBOR_IMAGES:
            raise ValueError("NEXT291 periodic neighbor-image count exceeds frozen guard")
        positions = np.asarray(work.positions, dtype=float)
        populations: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        cage_sizes: list[int] = []
        volumes: list[float] = []
        empty_cell_count = 0
        for site_index in range(len(work)):
            volume, site_metrics, sizes, empty = _power_cell_incidence_metrics(
                site_index=site_index,
                radius=radius,
                lattice_normals=lattice_normals,
                lattice_offsets=lattice_offsets,
                lattice_translations=lattice_translations,
                cell_matrix=cell_matrix,
                positions=positions,
                left=np.asarray(left),
                right=np.asarray(right),
                shift=np.asarray(shift),
                wigner_seitz_radius=wigner_seitz_radius,
            )
            volumes.append(volume)
            empty_cell_count += int(empty)
            cage_sizes.extend(sizes)
            for item in site_metrics:
                for name in METRIC_NAMES:
                    populations[name].append(item[name])
        cell_volume = abs(float(np.linalg.det(cell_matrix)))
        relative_error = abs(math.fsum(volumes) - cell_volume) / cell_volume
        if (
            not math.isfinite(relative_error)
            or relative_error > n267.VOLUME_TILING_RELATIVE_TOLERANCE
        ):
            raise ValueError(
                f"NEXT291 volume-tiling certificate differs: {relative_error:.12g}"
            )
        if not cage_sizes:
            raise ValueError("NEXT291 structure has no regular-Delaunay cage")
        features = aggregate_rwdci_features(populations)
        return RWDCIFeatureResult(
            True,
            None,
            len(work),
            empty_cell_count,
            len(cage_sizes),
            min(cage_sizes),
            max(cage_sizes),
            float(relative_error),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_rwdci_row(atoms: Atoms) -> dict[str, object]:
    result = compute_rwdci_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "rwdci_supported": bool(result.supported),
            "rwdci_failure": result.failure_reason,
            "rwdci_site_count": result.site_count,
            "rwdci_empty_cell_count": result.empty_cell_count,
            "rwdci_incidence_count": result.incidence_count,
            "rwdci_min_cage_size": result.min_cage_size,
            "rwdci_max_cage_size": result.max_cage_size,
            "rwdci_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "rwdci_supported": False,
        "rwdci_failure": f"{type(exc).__name__}: {exc}",
        "rwdci_site_count": 0,
        "rwdci_empty_cell_count": 0,
        "rwdci_incidence_count": 0,
        "rwdci_min_cage_size": 0,
        "rwdci_max_cage_size": 0,
        "rwdci_volume_tiling_relative_error": math.nan,
    }


def _compute_scigen_payload(item: tuple[str, bytes]):
    material_id, payload = item
    try:
        parsed = n267.n85._parse_frame(payload, strict_output=True)
        return material_id, compute_rwdci_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]):
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_rwdci_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=4))


def build_cross_source_rwdci_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT291 from physically isolated discovery geometry only."""

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
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT291 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT291 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT291 formal input identity differs: {differing}")
    if _sha256_file(Path(n267.__file__).resolve()) != EXPECTED_NEXT267_SOURCE_SHA256:
        raise ValueError("NEXT291 frozen NEXT267 source differs")
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
        raise ValueError("NEXT291 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT291 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT291 {source} discovery identity differs")
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
                [{"material_id": material_id, **row} for material_id, row in computed[source]]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT291 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["rwdci_supported"].fillna(False).astype(bool)
            finite = np.column_stack(
                [
                    np.isfinite(
                        pd.to_numeric(table[name], errors="coerce").to_numpy(float)
                    )
                    for name in FEATURE_NAMES
                ]
            )
            sites = pd.to_numeric(table["rwdci_site_count"], errors="coerce")
            empty = pd.to_numeric(table["rwdci_empty_cell_count"], errors="coerce")
            incidence = pd.to_numeric(table["rwdci_incidence_count"], errors="coerce")
            minimum = pd.to_numeric(table["rwdci_min_cage_size"], errors="coerce")
            maximum = pd.to_numeric(table["rwdci_max_cage_size"], errors="coerce")
            tiling = pd.to_numeric(
                table["rwdci_volume_tiling_relative_error"], errors="coerce"
            )
            values = np.column_stack(
                [
                    pd.to_numeric(table[name], errors="coerce").to_numpy(float)
                    for name in FEATURE_NAMES
                ]
            )
            coverage = float(supported.mean())
            if (
                len(table) != len(discovery[source])
                or coverage < MINIMUM_FORMAL_COVERAGE
                or not finite[supported].all()
                or finite[~supported].any()
                or not ((values[supported] >= 0.0) & (values[supported] <= 1.0)).all()
                or not (sites[supported] > 0).all()
                or not (empty[supported] >= 0).all()
                or not (incidence[supported] > 0).all()
                or not (minimum[supported] >= 4).all()
                or not (maximum[supported] >= minimum[supported]).all()
                or not (
                    tiling[supported] <= n267.VOLUME_TILING_RELATIVE_TOLERANCE
                ).all()
            ):
                raise RuntimeError(f"NEXT291 {source} support certificate differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            failures = Counter(table.loc[~supported, "rwdci_failure"].astype(str))
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "coverage": coverage,
                "failure_counts": dict(sorted(failures.items())),
                "finite_feature_counts": {
                    name: int(finite[:, index].sum())
                    for index, name in enumerate(FEATURE_NAMES)
                },
                "site_count": int(sites[supported].sum()),
                "empty_cell_count": int(empty[supported].sum()),
                "incidence_count": int(incidence[supported].sum()),
                "minimum_cage_size": int(minimum[supported].min()),
                "maximum_cage_size": int(maximum[supported].max()),
                "maximum_volume_tiling_relative_error": float(tiling[supported].max()),
            }
        if (
            counts["scigen"]["rows"] != 13_470  # type: ignore[index]
            or counts["wyformer"]["rows"] != 5_232  # type: ignore[index]
        ):
            raise RuntimeError("NEXT291 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "feature_count": len(FEATURE_NAMES),
            "metric_names": list(METRIC_NAMES),
            "statistic_names": list(STATISTIC_NAMES),
            "base_geometry_protocol": n267.PROTOCOL,
            "dual_geometry": "periodic_regular_delaunay_complete_power_vertex_cage",
            "radius_policy": ["atomic_radius_calculated", "atomic_radius_fallback"],
            "angular_weight": "normalized_(radius/distance)^2",
            "active_plane_relative_tolerance": ACTIVE_PLANE_RELATIVE_TOLERANCE,
            "active_normal_rank_tolerance": ACTIVE_NORMAL_RANK_TOLERANCE,
            "power_equality_relative_tolerance": POWER_EQUALITY_RELATIVE_TOLERANCE,
            "quantile_method": "inverted_cdf",
            "output_grid": n267.OUTPUT_GRID,
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE,
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
            "mode": "physically_isolated_discovery_x0_radius_weighted_delaunay_cage_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "next292_audit_authorized": True,
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
                "src/next291_radius_weighted_delaunay_cage_isotropy.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT291 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT291 source changed before publication")
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
    manifest = build_cross_source_rwdci_features(
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


__all__ = [
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "METRIC_NAMES",
    "PROTOCOL",
    "aggregate_rwdci_features",
    "build_cross_source_rwdci_features",
    "compute_rwdci_features",
    "weighted_cage_metrics",
]
