#!/usr/bin/env python3
"""Build dimension-minimal opposite-sign periodic-cage features from raw x0."""

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

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next299-minimal-opposite-sign-periodic-cage-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT299_MOSPC_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next299_scigen_minimal_opposite_sign_periodic_cage_features.parquet",
    "wyformer": "next299_wyformer_minimal_opposite_sign_periodic_cage_features.parquet",
}
METRIC_NAMES = (
    "uniform_closure",
    "inverse_square_closure",
    "uniform_equilibrium",
    "inverse_square_equilibrium",
)
STATISTIC_NAMES = ("min", "q10", "mean")
FEATURE_NAMES = tuple(
    [
        f"mospc_{metric}_{statistic}"
        for metric in METRIC_NAMES
        for statistic in STATISTIC_NAMES
    ]
    + ["mospc_locally_enclosed_fraction"]
)
FEATURE_DIRECTIONS = {name: "protected_high" for name in FEATURE_NAMES}
EXPECTED_ROWS = {"scigen": 13_470, "wyformer": 5_232}
EXPECTED_DESIGN_SHA256 = (
    "76e02d357e34be30160ded46b9ed3d96ee538fdaace62cb25124d7cfc5f2bd18"
)
EXPECTED_INPUT_SHA256 = {
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
    "design": EXPECTED_DESIGN_SHA256,
    "next19_source": "f1195a7ef519827f8da1704b9abe773bcee105eff1bdf6dfd5b8eabba1b94712",
    "next267_source": "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1",
    "next295_source": "4b92811e7f3c7ac60c1506104a18d2bd9d0fe06c6202e7f34cb996b32cd649a3",
}
BOUNDARY_FLAGS = n267.BOUNDARY_FLAGS
MINIMUM_FORMAL_COVERAGE = 0.97
MAX_TRANSLATION_RANGE = 8
MAX_ENUMERATED_CANDIDATES = 2_000_000
MAX_RETAINED_CAGE_SIZE = 256
DISTANCE_TOLERANCE = 1.0e-12
TIE_RELATIVE_TOLERANCE = 1.0e-8
LINEAR_RESIDUAL_TOLERANCE = 1.0e-9
ENCLOSURE_TOLERANCE = 1.0e-9
OUTPUT_GRID = 10_000_000_000


@dataclass(frozen=True)
class MinimalOppositeSignCage:
    vectors: np.ndarray
    distances: np.ndarray
    certified_range: int
    fourth_distance: float
    tie_tolerance: float
    outside_lower_bound: float


@dataclass(frozen=True)
class MOSPCFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    min_cage_size: int
    max_cage_size: int
    max_translation_range: int
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> MOSPCFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return MOSPCFeatureResult(False, reason, 0, 0, 0, 0, {})


def _quantized(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _bounded(value: float, *, tolerance: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
        raise ValueError(f"NEXT299 {label} differs")
    return _quantized(float(np.clip(value, 0.0, 1.0)))


@lru_cache(maxsize=None)
def _translation_cube(radius: int) -> np.ndarray:
    if type(radius) is not int or radius < 1 or radius > MAX_TRANSLATION_RANGE:
        raise ValueError("NEXT299 translation range differs")
    return np.asarray(
        list(itertools.product(range(-radius, radius + 1), repeat=3)),
        dtype=float,
    )


def _validated_direction_prior(
    directions: object, prior: object
) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(directions, dtype=float)
    q = np.asarray(prior, dtype=float)
    if (
        direction.shape != (4, 3)
        or not np.isfinite(direction).all()
        or q.shape != (4,)
        or not np.isfinite(q).all()
        or np.any(q <= 0.0)
        or not math.isclose(float(q.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError("NEXT299 four-direction inputs differ")
    norm = np.linalg.norm(direction, axis=1)
    if not np.isfinite(norm).all() or np.any(norm <= DISTANCE_TOLERANCE):
        raise ValueError("NEXT299 four-direction inputs differ")
    return direction / norm[:, None], q / q.sum()


def four_direction_equilibrium(directions: object, prior: object) -> float:
    """Solve the unique normalized balance for exactly four directions."""

    direction, q = _validated_direction_prior(directions, prior)
    singular = np.linalg.svd(direction, compute_uv=False)
    if singular.shape != (3,) or singular[0] <= 0.0 or int(
        np.count_nonzero(singular > singular[0] * n295.RANK_RELATIVE_TOLERANCE)
    ) < 3:
        return 0.0
    system = np.vstack((direction.T, np.ones(4, dtype=float)))
    system_singular = np.linalg.svd(system, compute_uv=False)
    if (
        system_singular.shape != (4,)
        or system_singular[0] <= 0.0
        or system_singular[-1]
        <= system_singular[0] * n295.RANK_RELATIVE_TOLERANCE
    ):
        return 0.0
    target = np.asarray([0.0, 0.0, 0.0, 1.0])
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        return 0.0
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise ValueError("NEXT299 four-direction balance residual differs")
    if float(np.min(coefficients)) < -LINEAR_RESIDUAL_TOLERANCE:
        return 0.0
    if (
        float(np.max(np.abs(system @ coefficients - target)))
        > LINEAR_RESIDUAL_TOLERANCE
    ):
        raise ValueError("NEXT299 four-direction balance residual differs")
    coefficients = np.clip(coefficients, 0.0, None)
    alpha = float(np.min(coefficients / q))
    return _bounded(
        alpha, tolerance=LINEAR_RESIDUAL_TOLERANCE, label="four-direction equilibrium"
    )


def minimal_opposite_sign_cage_for_site(
    *,
    structure,
    formal_valences: Sequence[float] | np.ndarray,
    site_index: int,
    max_translation_range: int = MAX_TRANSLATION_RANGE,
) -> MinimalOppositeSignCage:
    """Return the certified fourth-nearest opposite-sign periodic cage."""

    charges = np.asarray(formal_valences, dtype=float)
    if (
        charges.shape != (len(structure),)
        or not np.isfinite(charges).all()
        or type(site_index) is not int
        or site_index < 0
        or site_index >= len(structure)
    ):
        raise ValueError("NEXT299 site-cage inputs differ")
    if (
        type(max_translation_range) is not int
        or max_translation_range < 1
        or max_translation_range > MAX_TRANSLATION_RANGE
    ):
        raise ValueError("NEXT299 translation range differs")
    opposite = np.flatnonzero(charges[site_index] * charges < 0.0)
    if not len(opposite):
        raise ValueError("NEXT299 site has no formal opposite sign")
    lattice = structure.lattice
    matrix = np.asarray(lattice.matrix, dtype=float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.shape != (3,) or not np.isfinite(singular).all() or singular[-1] <= 0.0:
        raise ValueError("NEXT299 lattice singular value differs")
    center = np.asarray(structure[site_index].frac_coords, dtype=float)
    deltas = []
    for neighbor in opposite:
        fractional = np.asarray(structure[int(neighbor)].frac_coords, dtype=float)
        _, image = lattice.get_distance_and_image(center, fractional)
        image = np.asarray(image, dtype=float)
        if image.shape != (3,) or not np.isfinite(image).all():
            raise ValueError("NEXT299 lattice nearest image differs")
        deltas.append(fractional + image - center)
    delta = np.asarray(deltas, dtype=float)
    if delta.shape != (len(opposite), 3) or not np.isfinite(delta).all():
        raise ValueError("NEXT299 opposite-sign displacement differs")
    maximum_delta = float(np.max(np.abs(delta)))
    for radius in range(1, max_translation_range + 1):
        translations = _translation_cube(radius)
        candidate_count = len(delta) * len(translations)
        if candidate_count > MAX_ENUMERATED_CANDIDATES:
            raise ValueError("NEXT299 enumerated candidate guard differs")
        fractional = delta[:, None, :] + translations[None, :, :]
        vectors = fractional.reshape((-1, 3)) @ matrix
        distances = np.linalg.norm(vectors, axis=1)
        if (
            distances.shape != (candidate_count,)
            or not np.isfinite(distances).all()
            or np.any(distances <= DISTANCE_TOLERANCE)
        ):
            raise ValueError("NEXT299 opposite-sign periodic distance differs")
        fourth_distance = float(np.partition(distances, 3)[3])
        tie_tolerance = TIE_RELATIVE_TOLERANCE * max(1.0, fourth_distance)
        outside_lower_bound = float(singular[-1]) * max(
            0.0, radius + 1.0 - maximum_delta
        )
        if outside_lower_bound > fourth_distance + tie_tolerance:
            break
    else:
        raise ValueError("NEXT299 translation range certification differs")
    retained = distances <= fourth_distance + tie_tolerance
    retained_count = int(retained.sum())
    if retained_count < 4 or retained_count > MAX_RETAINED_CAGE_SIZE:
        raise ValueError("NEXT299 retained cage population differs")
    return MinimalOppositeSignCage(
        vectors=np.asarray(vectors[retained], dtype=float),
        distances=np.asarray(distances[retained], dtype=float),
        certified_range=radius,
        fourth_distance=fourth_distance,
        tie_tolerance=tie_tolerance,
        outside_lower_bound=outside_lower_bound,
    )


def _site_metrics(cage: MinimalOppositeSignCage) -> dict[str, float]:
    vector = np.asarray(cage.vectors, dtype=float)
    distance = np.asarray(cage.distances, dtype=float)
    if (
        vector.ndim != 2
        or vector.shape[1:] != (3,)
        or len(vector) < 4
        or distance.shape != (len(vector),)
        or not np.isfinite(vector).all()
        or not np.isfinite(distance).all()
        or np.any(distance <= DISTANCE_TOLERANCE)
    ):
        raise ValueError("NEXT299 retained cage metric inputs differ")
    measured = np.linalg.norm(vector, axis=1)
    if not np.allclose(measured, distance, rtol=0.0, atol=1.0e-10):
        raise ValueError("NEXT299 retained cage distance accounting differs")
    direction = vector / distance[:, None]
    uniform = np.full(len(direction), 1.0 / len(direction), dtype=float)
    raw_weight = (float(distance.min()) / distance) ** 2
    inverse_square = raw_weight / math.fsum(float(value) for value in raw_weight)

    def closure(prior: np.ndarray) -> float:
        return _bounded(
            1.0 - float(np.linalg.norm(prior @ direction)),
            tolerance=n295.CLOSURE_NUMERICAL_TOLERANCE,
            label="directional closure",
        )

    if len(direction) == 4:
        uniform_equilibrium = four_direction_equilibrium(direction, uniform)
        inverse_square_equilibrium = four_direction_equilibrium(
            direction, inverse_square
        )
    else:
        uniform_equilibrium = n295.positive_equilibrium_fraction(direction, uniform)
        inverse_square_equilibrium = n295.positive_equilibrium_fraction(
            direction, inverse_square
        )
    result = {
        "uniform_closure": closure(uniform),
        "inverse_square_closure": closure(inverse_square),
        "uniform_equilibrium": uniform_equilibrium,
        "inverse_square_equilibrium": inverse_square_equilibrium,
        "locally_enclosed": float(uniform_equilibrium > ENCLOSURE_TOLERANCE),
    }
    if tuple(result) != (*METRIC_NAMES, "locally_enclosed"):
        raise RuntimeError("NEXT299 site metric schema differs")
    return result


def _mean(values: np.ndarray) -> float:
    return float(math.fsum(float(value) for value in values) / len(values))


def _aggregate(site_metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not site_metrics:
        raise ValueError("NEXT299 site metric population differs")
    features: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in site_metrics], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("NEXT299 site metric population differs")
        features[f"mospc_{name}_min"] = _quantized(float(values.min()))
        features[f"mospc_{name}_q10"] = _quantized(
            float(np.quantile(values, 0.10, method="inverted_cdf"))
        )
        features[f"mospc_{name}_mean"] = _quantized(_mean(values))
    enclosed = np.asarray(
        [float(row["locally_enclosed"]) for row in site_metrics], dtype=float
    )
    features["mospc_locally_enclosed_fraction"] = _quantized(_mean(enclosed))
    values = np.asarray(list(features.values()), dtype=float)
    if (
        tuple(features) != FEATURE_NAMES
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise RuntimeError("NEXT299 aggregate feature schema or bounds differ")
    return features


def compute_mospc_features(atoms: Atoms) -> MOSPCFeatureResult:
    """Compute MOSPC from element identities and one raw periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT299 formal valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        if not np.any(charges > 0.0) or not np.any(charges < 0.0):
            raise ValueError("NEXT299 formal assignment has no opposite signs")
        site_metrics = []
        cage_sizes = []
        ranges = []
        for site_index in range(len(structure)):
            cage = minimal_opposite_sign_cage_for_site(
                structure=structure,
                formal_valences=charges,
                site_index=site_index,
            )
            site_metrics.append(_site_metrics(cage))
            cage_sizes.append(len(cage.vectors))
            ranges.append(cage.certified_range)
        features = _aggregate(site_metrics)
        return MOSPCFeatureResult(
            True,
            None,
            len(structure),
            min(cage_sizes),
            max(cage_sizes),
            max(ranges),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_mospc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_mospc_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "mospc_supported": bool(result.supported),
            "mospc_failure": result.failure_reason,
            "mospc_site_count": result.site_count,
            "mospc_min_cage_size": result.min_cage_size,
            "mospc_max_cage_size": result.max_cage_size,
            "mospc_max_translation_range": result.max_translation_range,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "mospc_supported": False,
        "mospc_failure": f"{type(exc).__name__}: {exc}",
        "mospc_site_count": 0,
        "mospc_min_cage_size": 0,
        "mospc_max_cage_size": 0,
        "mospc_max_translation_range": 0,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = n267.n85._parse_frame(payload, strict_output=True).atoms
        return material_id, compute_mospc_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_mospc_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=8))


def _label_free_statistics(table: pd.DataFrame) -> dict[str, object]:
    supported = table["mospc_supported"].fillna(False).astype(bool)
    statistics: dict[str, object] = {}
    for name in FEATURE_NAMES:
        values = pd.to_numeric(table.loc[supported, name], errors="coerce").to_numpy(float)
        statistics[name] = {
            "unique_rounded_10": int(len(np.unique(np.round(values, 10)))),
            "minimum": float(values.min()),
            "q10": float(np.quantile(values, 0.10, method="inverted_cdf")),
            "median": float(np.quantile(values, 0.50, method="inverted_cdf")),
            "q90": float(np.quantile(values, 0.90, method="inverted_cdf")),
            "maximum": float(values.max()),
        }
    return statistics


def build_cross_source_mospc_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT299 from physically isolated discovery geometry only."""

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
        "next19_source": Path(n19.__file__).resolve(),
        "next267_source": Path(n267.__file__).resolve(),
        "next295_source": Path(n295.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT299 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT299 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT299 formal input identity differs: {differing}")
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
        raise ValueError("NEXT299 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT299 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT299 {source} discovery identity differs")
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
    test_path = source_path.parent.parent / "tests/test_next299_minimal_opposite_sign_periodic_cage.py"
    executed_hashes = {
        "src/next299_minimal_opposite_sign_periodic_cage.py": _sha256_file(source_path),
        "tests/test_next299_minimal_opposite_sign_periodic_cage.py": _sha256_file(test_path),
    }
    started = time.perf_counter()
    try:
        computed = {
            source: _compute_many(payloads[source], source=source, workers=workers)
            for source in ("scigen", "wyformer")
        }
        counts: dict[str, object] = {}
        outputs: list[Path] = []
        statistics: dict[str, object] = {}
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed[source]]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT299 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["mospc_supported"].fillna(False).astype(bool)
            values = np.column_stack(
                [
                    pd.to_numeric(table[name], errors="coerce").to_numpy(float)
                    for name in FEATURE_NAMES
                ]
            )
            finite = np.isfinite(values)
            sites = pd.to_numeric(table["mospc_site_count"], errors="coerce")
            minimum = pd.to_numeric(table["mospc_min_cage_size"], errors="coerce")
            maximum = pd.to_numeric(table["mospc_max_cage_size"], errors="coerce")
            translation = pd.to_numeric(
                table["mospc_max_translation_range"], errors="coerce"
            )
            coverage = float(supported.mean())
            if (
                len(table) != EXPECTED_ROWS[source]
                or coverage < MINIMUM_FORMAL_COVERAGE
                or not finite[supported].all()
                or finite[~supported].any()
                or not ((values[supported] >= 0.0) & (values[supported] <= 1.0)).all()
                or not (sites[supported] >= 2).all()
                or not (minimum[supported] >= 4).all()
                or not (maximum[supported] >= minimum[supported]).all()
                or not (maximum[supported] <= MAX_RETAINED_CAGE_SIZE).all()
                or not (translation[supported] >= 1).all()
                or not (translation[supported] <= MAX_TRANSLATION_RANGE).all()
            ):
                raise RuntimeError(f"NEXT299 {source} support certificate differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            outputs.append(output)
            failures = Counter(table.loc[~supported, "mospc_failure"].astype(str))
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
                "minimum_cage_size": int(minimum[supported].min()),
                "maximum_cage_size": int(maximum[supported].max()),
                "maximum_translation_range": int(translation[supported].max()),
            }
            statistics[source] = _label_free_statistics(table)
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "feature_count": len(FEATURE_NAMES),
            "metric_names": list(METRIC_NAMES),
            "statistic_names": list(STATISTIC_NAMES),
            "cage_order": 4,
            "fourth_distance_ties_included": True,
            "tie_relative_tolerance": TIE_RELATIVE_TOLERANCE,
            "maximum_translation_range": MAX_TRANSLATION_RANGE,
            "maximum_enumerated_candidates": MAX_ENUMERATED_CANDIDATES,
            "maximum_retained_cage_size": MAX_RETAINED_CAGE_SIZE,
            "outside_certificate": "sigma_min_times_max_0_R_plus_1_minus_D_infinity",
            "inverse_square_prior": "normalized_(minimum_distance/distance)^2",
            "four_direction_solver": "verified_unique_4x4_normalized_balance",
            "quantile_method": "inverted_cdf",
            "output_grid": OUTPUT_GRID,
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE,
            "label_free_statistics": statistics,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        outputs.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_minimal_opposite_sign_periodic_cage_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "next300_audit_authorized": True,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": executed_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT299 input changed before publication")
        if (
            _sha256_file(source_path)
            != executed_hashes["src/next299_minimal_opposite_sign_periodic_cage.py"]
            or _sha256_file(test_path)
            != executed_hashes["tests/test_next299_minimal_opposite_sign_periodic_cage.py"]
        ):
            raise RuntimeError("NEXT299 executed artifact changed before publication")
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
    manifest = build_cross_source_mospc_features(
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
    "build_cross_source_mospc_features",
    "compute_mospc_features",
    "four_direction_equilibrium",
    "minimal_opposite_sign_cage_for_site",
]
