#!/usr/bin/env python3
"""Build periodic reciprocal cage-balance features from raw x0 geometry."""

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

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next299_minimal_opposite_sign_periodic_cage as n299
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next303-periodic-reciprocal-cage-balance-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT303_PRCB_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next303_scigen_periodic_reciprocal_cage_balance_features.parquet",
    "wyformer": "next303_wyformer_periodic_reciprocal_cage_balance_features.parquet",
}
PRIOR_NAMES = ("uniform", "inverse_square", "charge_inverse_square")
STATISTIC_NAMES = ("min", "q10", "mean")
FEATURE_NAMES = tuple(
    [
        f"prcb_{prior}_closure_{statistic}"
        for prior in PRIOR_NAMES
        for statistic in STATISTIC_NAMES
    ]
    + [
        "prcb_mutual_site_fraction_min",
        "prcb_mutual_site_fraction_q10",
        "prcb_mutual_site_fraction_mean",
        "prcb_mutual_edge_fraction",
    ]
)
FEATURE_DIRECTIONS = {name: "protected_high" for name in FEATURE_NAMES}
EXPECTED_ROWS = n299.EXPECTED_ROWS
EXPECTED_DESIGN_SHA256 = (
    "db5a0b8829afc7f1883333beb6fda5a6c20b89e5c04ccd94e0d678b81bc99555"
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
    "next299_source": "dca4a9683af6c4141d792f39a895d9c5f9e4e88399113d28defd7eb2788c064e",
}
BOUNDARY_FLAGS = n299.BOUNDARY_FLAGS
MINIMUM_FORMAL_COVERAGE = 0.97
EDGE_VECTOR_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class PRCBFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    mutual_edge_count: int
    max_translation_range: int
    features: Mapping[str, float]


@dataclass
class _EdgeRecord:
    vector: np.ndarray
    distance: float
    voters: set[int]


def _failure(exc: Exception | str) -> PRCBFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PRCBFeatureResult(False, reason, 0, 0, 0, 0, {})


def _quantized(value: float) -> float:
    return n299._quantized(value)


def _bounded(value: float, *, label: str) -> float:
    return n299._bounded(
        value,
        tolerance=n295.CLOSURE_NUMERICAL_TOLERANCE,
        label=f"PRCB {label}",
    )


def _mean(values: Sequence[float] | np.ndarray) -> float:
    if not len(values):
        raise ValueError("NEXT303 mean population differs")
    return float(math.fsum(float(value) for value in values) / len(values))


def _statistics(values: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or not len(array)
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or np.any(array > 1.0)
    ):
        raise ValueError("NEXT303 site statistic population differs")
    return (
        _quantized(float(array.min())),
        _quantized(float(np.quantile(array, 0.10, method="inverted_cdf"))),
        _quantized(_mean(array)),
    )


def _site_periodic_images(
    *, structure, charges: np.ndarray, site_index: int
) -> tuple[list[tuple[int, tuple[int, int, int], np.ndarray, float]], int]:
    opposite = np.flatnonzero(charges[site_index] * charges < 0.0)
    if not len(opposite):
        raise ValueError("NEXT303 site has no formal opposite sign")
    lattice = structure.lattice
    matrix = np.asarray(lattice.matrix, dtype=float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.shape != (3,) or not np.isfinite(singular).all() or singular[-1] <= 0.0:
        raise ValueError("NEXT303 lattice singular value differs")
    center = np.asarray(structure[site_index].frac_coords, dtype=float)
    nearest_images = []
    deltas = []
    for neighbor in opposite:
        fractional = np.asarray(structure[int(neighbor)].frac_coords, dtype=float)
        _, raw_image = lattice.get_distance_and_image(center, fractional)
        raw_image = np.asarray(raw_image, dtype=float)
        image = np.rint(raw_image).astype(int)
        if (
            raw_image.shape != (3,)
            or not np.isfinite(raw_image).all()
            or not np.allclose(raw_image, image, rtol=0.0, atol=1.0e-10)
        ):
            raise ValueError("NEXT303 lattice nearest image differs")
        nearest_images.append(image)
        deltas.append(fractional + image - center)
    base_images = np.asarray(nearest_images, dtype=int)
    delta = np.asarray(deltas, dtype=float)
    maximum_delta = float(np.max(np.abs(delta)))
    for radius in range(1, n299.MAX_TRANSLATION_RANGE + 1):
        translations = np.asarray(n299._translation_cube(radius), dtype=int)
        candidate_count = len(delta) * len(translations)
        if candidate_count > n299.MAX_ENUMERATED_CANDIDATES:
            raise ValueError("NEXT303 enumerated candidate guard differs")
        fractional = delta[:, None, :] + translations[None, :, :]
        vectors = fractional.reshape((-1, 3)) @ matrix
        distances = np.linalg.norm(vectors, axis=1)
        if (
            distances.shape != (candidate_count,)
            or not np.isfinite(distances).all()
            or np.any(distances <= n299.DISTANCE_TOLERANCE)
        ):
            raise ValueError("NEXT303 opposite-sign periodic distance differs")
        fourth_distance = float(np.partition(distances, 3)[3])
        tie_tolerance = n299.TIE_RELATIVE_TOLERANCE * max(1.0, fourth_distance)
        outside_lower_bound = float(singular[-1]) * max(
            0.0, radius + 1.0 - maximum_delta
        )
        if outside_lower_bound > fourth_distance + tie_tolerance:
            break
    else:
        raise ValueError("NEXT303 translation range certification differs")
    retained = distances <= fourth_distance + tie_tolerance
    count = int(retained.sum())
    if count < 4 or count > n299.MAX_RETAINED_CAGE_SIZE:
        raise ValueError("NEXT303 retained cage population differs")
    neighbor_ids = np.repeat(opposite, len(translations))[retained]
    shifts = (
        base_images[:, None, :] + translations[None, :, :]
    ).reshape((-1, 3))[retained]
    result = []
    for neighbor, shift, vector, distance in zip(
        neighbor_ids,
        shifts,
        vectors[retained],
        distances[retained],
        strict=True,
    ):
        result.append(
            (
                int(neighbor),
                tuple(int(value) for value in shift),
                np.asarray(vector, dtype=float),
                float(distance),
            )
        )
    return result, radius


def _canonical_edge(
    *, site_index: int, neighbor_index: int, shift: tuple[int, int, int]
) -> tuple[tuple[int, int, int, int, int], bool]:
    forward = (site_index, neighbor_index, *shift)
    reverse = (neighbor_index, site_index, *(-value for value in shift))
    return (forward, False) if forward <= reverse else (reverse, True)


def _reciprocal_features(structure, charges: np.ndarray):
    edges: dict[tuple[int, int, int, int, int], _EdgeRecord] = {}
    maximum_range = 0
    for site_index in range(len(structure)):
        images, radius = _site_periodic_images(
            structure=structure, charges=charges, site_index=site_index
        )
        maximum_range = max(maximum_range, radius)
        for neighbor, shift, vector, distance in images:
            key, reversed_direction = _canonical_edge(
                site_index=site_index,
                neighbor_index=neighbor,
                shift=shift,
            )
            canonical_vector = -vector if reversed_direction else vector
            record = edges.get(key)
            if record is None:
                edges[key] = _EdgeRecord(
                    np.asarray(canonical_vector, dtype=float),
                    float(distance),
                    {site_index},
                )
            else:
                if (
                    not np.allclose(
                        record.vector,
                        canonical_vector,
                        rtol=0.0,
                        atol=EDGE_VECTOR_TOLERANCE,
                    )
                    or not math.isclose(
                        record.distance,
                        distance,
                        rel_tol=0.0,
                        abs_tol=EDGE_VECTOR_TOLERANCE,
                    )
                ):
                    raise ValueError("NEXT303 reciprocal edge identity differs")
                record.voters.add(site_index)
    if not edges:
        raise ValueError("NEXT303 reciprocal edge population differs")

    keys = sorted(edges)
    incident: list[list[tuple[int, float]]] = [[] for _ in structure]
    distances = np.empty(len(keys), dtype=float)
    charge_products = np.empty(len(keys), dtype=float)
    directions = np.empty((len(keys), 3), dtype=float)
    mutual = np.empty(len(keys), dtype=bool)
    for edge_index, key in enumerate(keys):
        first, second = key[:2]
        record = edges[key]
        if record.voters - {first, second} or len(record.voters) not in (1, 2):
            raise ValueError("NEXT303 reciprocal edge voters differ")
        direction = record.vector / record.distance
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or not math.isclose(
            norm, 1.0, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError("NEXT303 reciprocal edge direction differs")
        directions[edge_index] = direction
        distances[edge_index] = record.distance
        charge_products[edge_index] = abs(charges[first] * charges[second])
        mutual[edge_index] = len(record.voters) == 2
        incident[first].append((edge_index, 1.0))
        incident[second].append((edge_index, -1.0))
    if any(not values for values in incident) or np.any(charge_products <= 0.0):
        raise ValueError("NEXT303 reciprocal edge incidence differs")

    priors = {
        "uniform": np.ones(len(keys), dtype=float),
        "inverse_square": 1.0 / np.square(distances),
        "charge_inverse_square": charge_products / np.square(distances),
    }
    features: dict[str, float] = {}
    for prior_name in PRIOR_NAMES:
        weight = priors[prior_name]
        closures = []
        for site_edges in incident:
            indices = np.asarray([value[0] for value in site_edges], dtype=int)
            signs = np.asarray([value[1] for value in site_edges], dtype=float)
            selected_weight = weight[indices]
            denominator = math.fsum(float(value) for value in selected_weight)
            net = (signs * selected_weight) @ directions[indices]
            closures.append(
                _bounded(
                    1.0 - float(np.linalg.norm(net)) / denominator,
                    label=f"{prior_name} reciprocal closure",
                )
            )
        for statistic, value in zip(
            STATISTIC_NAMES, _statistics(closures), strict=True
        ):
            features[f"prcb_{prior_name}_closure_{statistic}"] = value

    mutual_site = []
    for site_edges in incident:
        values = [float(mutual[index]) for index, _ in site_edges]
        mutual_site.append(_bounded(_mean(values), label="mutual site fraction"))
    for statistic, value in zip(
        STATISTIC_NAMES, _statistics(mutual_site), strict=True
    ):
        features[f"prcb_mutual_site_fraction_{statistic}"] = value
    features["prcb_mutual_edge_fraction"] = _quantized(
        _bounded(_mean(mutual.astype(float)), label="mutual edge fraction")
    )
    values = np.asarray(list(features.values()), dtype=float)
    if (
        tuple(features) != FEATURE_NAMES
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise RuntimeError("NEXT303 aggregate feature schema or bounds differ")
    return features, len(keys), int(mutual.sum()), maximum_range


def compute_prcb_features(atoms: Atoms) -> PRCBFeatureResult:
    """Compute PRCB from element identities and one raw periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT303 formal valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        if not np.any(charges > 0.0) or not np.any(charges < 0.0):
            raise ValueError("NEXT303 formal assignment has no opposite signs")
        features, edge_count, mutual_edge_count, maximum_range = _reciprocal_features(
            structure, charges
        )
        return PRCBFeatureResult(
            True,
            None,
            len(structure),
            edge_count,
            mutual_edge_count,
            maximum_range,
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_prcb_row(atoms: Atoms) -> dict[str, object]:
    result = compute_prcb_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "prcb_supported": bool(result.supported),
            "prcb_failure": result.failure_reason,
            "prcb_site_count": result.site_count,
            "prcb_edge_count": result.edge_count,
            "prcb_mutual_edge_count": result.mutual_edge_count,
            "prcb_max_translation_range": result.max_translation_range,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "prcb_supported": False,
        "prcb_failure": f"{type(exc).__name__}: {exc}",
        "prcb_site_count": 0,
        "prcb_edge_count": 0,
        "prcb_mutual_edge_count": 0,
        "prcb_max_translation_range": 0,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = n267.n85._parse_frame(payload, strict_output=True).atoms
        return material_id, compute_prcb_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_prcb_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=8))


def _label_free_statistics(table: pd.DataFrame) -> dict[str, object]:
    supported = table["prcb_supported"].fillna(False).astype(bool)
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


def build_cross_source_prcb_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT303 from physically isolated discovery geometry only."""

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
        "next299_source": Path(n299.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT303 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT303 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT303 formal input identity differs: {differing}")
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
        raise ValueError("NEXT303 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT303 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT303 {source} discovery identity differs")
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
    test_path = source_path.parent.parent / "tests/test_next303_periodic_reciprocal_cage_balance.py"
    executed_hashes = {
        "src/next303_periodic_reciprocal_cage_balance.py": _sha256_file(source_path),
        "tests/test_next303_periodic_reciprocal_cage_balance.py": _sha256_file(test_path),
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
                raise RuntimeError(f"NEXT303 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["prcb_supported"].fillna(False).astype(bool)
            values = np.column_stack(
                [pd.to_numeric(table[name], errors="coerce").to_numpy(float) for name in FEATURE_NAMES]
            )
            finite = np.isfinite(values)
            sites = pd.to_numeric(table["prcb_site_count"], errors="coerce")
            edges = pd.to_numeric(table["prcb_edge_count"], errors="coerce")
            mutual_edges = pd.to_numeric(table["prcb_mutual_edge_count"], errors="coerce")
            translation = pd.to_numeric(table["prcb_max_translation_range"], errors="coerce")
            coverage = float(supported.mean())
            if (
                len(table) != EXPECTED_ROWS[source]
                or coverage < MINIMUM_FORMAL_COVERAGE
                or not finite[supported].all()
                or finite[~supported].any()
                or not ((values[supported] >= 0.0) & (values[supported] <= 1.0)).all()
                or not (sites[supported] >= 2).all()
                or not (edges[supported] >= 1).all()
                or not (mutual_edges[supported] >= 0).all()
                or not (mutual_edges[supported] <= edges[supported]).all()
                or not (translation[supported] >= 1).all()
                or not (translation[supported] <= n299.MAX_TRANSLATION_RANGE).all()
                or not (sites[~supported] == 0).all()
                or not (edges[~supported] == 0).all()
                or not (mutual_edges[~supported] == 0).all()
                or not (translation[~supported] == 0).all()
            ):
                raise RuntimeError(f"NEXT303 {source} support certificate differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            outputs.append(output)
            failures = Counter(table.loc[~supported, "prcb_failure"].astype(str))
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
                "edge_count": int(edges[supported].sum()),
                "mutual_edge_count": int(mutual_edges[supported].sum()),
                "maximum_translation_range": int(translation[supported].max()),
            }
            statistics[source] = _label_free_statistics(table)
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "feature_count": len(FEATURE_NAMES),
            "prior_names": list(PRIOR_NAMES),
            "statistic_names": list(STATISTIC_NAMES),
            "cage_order": 4,
            "fourth_distance_ties_included": True,
            "periodic_edge_orbit": "lexicographic_min_i_j_T_or_j_i_minus_T",
            "reciprocal_endpoint_signs": [1, -1],
            "shared_edge_priors": {
                "uniform": "1",
                "inverse_square": "1_over_distance_squared",
                "charge_inverse_square": "absolute_formal_charge_product_over_distance_squared",
            },
            "exact_positive_self_stress_floor_excluded_as_degenerate": True,
            "quantile_method": "inverted_cdf",
            "output_grid": n299.OUTPUT_GRID,
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
            "mode": "physically_isolated_discovery_x0_periodic_reciprocal_cage_balance_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "next304_audit_authorized": True,
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
            raise RuntimeError("NEXT303 input changed before publication")
        if (
            _sha256_file(source_path)
            != executed_hashes["src/next303_periodic_reciprocal_cage_balance.py"]
            or _sha256_file(test_path)
            != executed_hashes["tests/test_next303_periodic_reciprocal_cage_balance.py"]
        ):
            raise RuntimeError("NEXT303 executed artifact changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = build_cross_source_prcb_features(
        scigen_cohort_dir=arguments.scigen_cohort_dir,
        wyformer_cohort_dir=arguments.wyformer_cohort_dir,
        design_path=arguments.design_path,
        output_dir=arguments.output_dir,
        workers=arguments.workers,
        require_formal_inputs=not arguments.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOGUE_NAME",
    "FEATURE_DIRECTIONS",
    "FEATURE_FILES",
    "FEATURE_NAMES",
    "MANIFEST_NAME",
    "MINIMUM_FORMAL_COVERAGE",
    "PRCBFeatureResult",
    "PRIOR_NAMES",
    "PROTOCOL",
    "build_cross_source_prcb_features",
    "compute_prcb_features",
]
