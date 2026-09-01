#!/usr/bin/env python3
"""Build discovery-only local angular persistent-homology features."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence
import warnings

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import src.next85_scigen_label_free_features as n85
import src.next94_wyformer_label_free_features as n94
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next263-local-angular-persistent-homology-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT263_LAPH_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next263_scigen_discovery_laph_features.parquet",
    "wyformer": "next263_wyformer_discovery_laph_features.parquet",
}
SITE_QUANTITIES = (
    "h0_death_mean",
    "h0_death_cv",
    "h1_persistence_density",
    "h1_persistence_max",
)
AGGREGATES = ("mean", "q10", "q90", "std")
FEATURE_NAMES = tuple(
    f"laph_{quantity}_{aggregate}"
    for quantity in SITE_QUANTITIES
    for aggregate in AGGREGATES
)
EXPECTED_DESIGN_SHA256 = (
    "c1c6d02733cfe447135a51afa6971bb7189e370c12b15a685d61a616abddbe09"
)
EXPECTED_INPUT_SHA256 = {
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
    "design": EXPECTED_DESIGN_SHA256,
}
EXPECTED_UPSTREAM_SOURCE_SHA256 = {
    "src/next85_scigen_label_free_features.py": (
        "2caf0fa0aafe6df6732c3b8ed02cd19d96076314273331f32a449b6bd3b41335"
    ),
    "src/next94_wyformer_label_free_features.py": (
        "ccb04a9387b4fad9ea3b8e7e7cd54fb69965f98a3c44342c198a8511b17702a9"
    ),
}
DISTANCE_GRID = 100_000_000_000
AREA_FRACTION_DENOMINATOR = 32
MIN_RETAINED_FACETS = 4
MAX_RETAINED_FACETS = 32
NUMERICAL_TOLERANCE = 1.0e-10
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
class LAPHFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    min_retained_facets: int
    max_retained_facets: int
    h1_interval_count: int
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> LAPHFeatureResult:
    reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    return LAPHFeatureResult(False, reason, 0, 0, 0, 0, {})


def _grid_integer(value: float, *, label: str) -> int:
    if not math.isfinite(value) or value < -NUMERICAL_TOLERANCE:
        raise ValueError(f"NEXT263 {label} differs")
    return max(0, int(round(float(value) * DISTANCE_GRID)))


def vietoris_rips_intervals(points: object) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return finite H0 deaths and positive H1 persistences over F2."""

    cloud = np.asarray(points, dtype=float)
    if (
        cloud.ndim != 2
        or cloud.shape[1:] != (3,)
        or len(cloud) < 2
        or len(cloud) > MAX_RETAINED_FACETS
        or np.any(~np.isfinite(cloud))
    ):
        raise ValueError("NEXT263 angular point cloud differs")
    norms = np.linalg.norm(cloud, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > 1.0e-7):
        raise ValueError("NEXT263 angular direction norm differs")
    unit = cloud / norms[:, None]
    edge_filtration = {
        (i, j): _grid_integer(
            float(np.linalg.norm(unit[i] - unit[j])), label="edge filtration"
        )
        for i, j in combinations(range(len(unit)), 2)
    }
    simplices: list[tuple[int, int, tuple[int, ...]]] = [
        (0, 0, (i,)) for i in range(len(unit))
    ]
    simplices.extend(
        (value, 1, edge) for edge, value in edge_filtration.items()
    )
    for triangle in combinations(range(len(unit)), 3):
        i, j, k = triangle
        filtration = max(
            edge_filtration[tuple(sorted((i, j)))],
            edge_filtration[tuple(sorted((i, k)))],
            edge_filtration[tuple(sorted((j, k)))],
        )
        simplices.append((filtration, 2, triangle))
    simplices.sort(key=lambda value: (value[0], value[1], value[2]))
    simplex_index = {simplex[2]: index for index, simplex in enumerate(simplices)}
    reduced: dict[int, set[int]] = {}
    pivot_owner: dict[int, int] = {}
    h0: list[int] = []
    h1: list[int] = []
    for column_index, (filtration, dimension, vertices) in enumerate(simplices):
        if dimension == 0:
            column: set[int] = set()
        else:
            column = {
                simplex_index[face]
                for face in combinations(vertices, dimension)
            }
        while column and max(column) in pivot_owner:
            column ^= reduced[pivot_owner[max(column)]]
        reduced[column_index] = column
        if not column:
            continue
        pivot = max(column)
        pivot_owner[pivot] = column_index
        birth_filtration, birth_dimension, _ = simplices[pivot]
        persistence = filtration - birth_filtration
        if birth_dimension == 0:
            h0.append(persistence)
        elif birth_dimension == 1 and persistence > 0:
            h1.append(persistence)
    if len(h0) != len(unit) - 1:
        raise RuntimeError("NEXT263 finite H0 barcode differs")
    return (
        tuple(value / DISTANCE_GRID for value in sorted(h0)),
        tuple(value / DISTANCE_GRID for value in sorted(h1)),
    )


def _exact_moments(values: object, *, label: str) -> tuple[float, float]:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or len(raw) == 0 or np.any(~np.isfinite(raw)):
        raise ValueError(f"NEXT263 {label} population differs")
    integers = [_grid_integer(float(value), label=label) for value in raw]
    mean = Fraction(sum(integers), len(integers))
    variance = sum((Fraction(value) - mean) ** 2 for value in integers) / len(integers)
    return float(mean / DISTANCE_GRID), math.sqrt(float(variance)) / DISTANCE_GRID


def _inverse_cdf(values: Sequence[int], probability: Fraction) -> float:
    if not values or probability <= 0 or probability > 1:
        raise ValueError("NEXT263 inverse-CDF inputs differ")
    ordered = sorted(values)
    numerator = probability.numerator * len(ordered)
    index = max(0, (numerator + probability.denominator - 1) // probability.denominator - 1)
    return ordered[index] / DISTANCE_GRID


def aggregate_laph_features(
    *,
    h0_death_mean: object,
    h0_death_cv: object,
    h1_persistence_density: object,
    h1_persistence_max: object,
) -> dict[str, float]:
    """Aggregate four quantized site quantities with replication-stable rules."""

    populations = {
        "h0_death_mean": h0_death_mean,
        "h0_death_cv": h0_death_cv,
        "h1_persistence_density": h1_persistence_density,
        "h1_persistence_max": h1_persistence_max,
    }
    features: dict[str, float] = {}
    for quantity in SITE_QUANTITIES:
        raw = np.asarray(populations[quantity], dtype=float)
        mean, standard_deviation = _exact_moments(raw, label=quantity)
        integers = [_grid_integer(float(value), label=quantity) for value in raw]
        features[f"laph_{quantity}_mean"] = mean
        features[f"laph_{quantity}_q10"] = _inverse_cdf(integers, Fraction(1, 10))
        features[f"laph_{quantity}_q90"] = _inverse_cdf(integers, Fraction(9, 10))
        features[f"laph_{quantity}_std"] = standard_deviation
    if tuple(features) != FEATURE_NAMES or any(
        not math.isfinite(value) or value < -NUMERICAL_TOLERANCE
        for value in features.values()
    ):
        raise RuntimeError("NEXT263 feature schema differs")
    return features


def _site_quantities(normals: np.ndarray) -> tuple[float, float, float, float, int]:
    h0, h1 = vietoris_rips_intervals(normals)
    h0_mean, h0_std = _exact_moments(h0, label="site H0 death")
    if h0_mean <= 0.0:
        raise ValueError("NEXT263 site H0 mean differs")
    h1_integers = [_grid_integer(value, label="site H1 persistence") for value in h1]
    density = Fraction(sum(h1_integers), len(normals) * DISTANCE_GRID)
    maximum = max(h1_integers, default=0) / DISTANCE_GRID
    values = (
        h0_mean,
        h0_std / h0_mean,
        float(density),
        maximum,
    )
    quantized = tuple(
        _grid_integer(value, label="site quantity") / DISTANCE_GRID
        for value in values
    )
    return (*quantized, len(h1))


def compute_laph_features(atoms: Atoms) -> LAPHFeatureResult:
    """Compute frozen local angular persistence from one raw x0 structure."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("LAPH features require exact periodic geometry-only Atoms")
        structure = AseAtomsAdaptor.get_structure(atoms)
        finder = VoronoiNN(weight="solid_angle", tol=0, cutoff=13)
        site_rows: list[tuple[float, float, float, float, int]] = []
        retained_counts: list[int] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for center in range(len(structure)):
                info = finder.get_nn_info(structure, center)
                facets: dict[
                    tuple[int, tuple[int, int, int]], tuple[float, np.ndarray]
                ] = {}
                for item in info:
                    try:
                        site_index = int(item["site_index"])
                        image = tuple(
                            int(round(float(value))) for value in item["image"]
                        )
                        area = float(item["poly_info"]["area"])
                        normal = np.asarray(item["poly_info"]["normal"], dtype=float)
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (
                        len(image) != 3
                        or site_index < 0
                        or site_index >= len(structure)
                        or not math.isfinite(area)
                        or area <= 0.0
                        or normal.shape != (3,)
                        or np.any(~np.isfinite(normal))
                    ):
                        continue
                    key = (site_index, image)
                    previous = facets.get(key)
                    if previous is None or area > previous[0]:
                        facets[key] = (area, normal)
                keys = sorted(facets)
                if not keys:
                    raise ValueError("site has no valid Voronoi facet")
                total_area = math.fsum(facets[key][0] for key in keys)
                retained = [
                    key
                    for key in keys
                    if facets[key][0] / total_area
                    >= 1.0 / AREA_FRACTION_DENOMINATOR
                ]
                if not MIN_RETAINED_FACETS <= len(retained) <= MAX_RETAINED_FACETS:
                    raise ValueError("retained Voronoi facet count differs")
                normals = np.asarray([facets[key][1] for key in retained], dtype=float)
                norms = np.linalg.norm(normals, axis=1)
                if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
                    raise ValueError("Voronoi facet normal norm differs")
                retained_counts.append(len(retained))
                site_rows.append(_site_quantities(normals / norms[:, None]))
        site_values = np.asarray(site_rows, dtype=float)
        features = aggregate_laph_features(
            h0_death_mean=site_values[:, 0],
            h0_death_cv=site_values[:, 1],
            h1_persistence_density=site_values[:, 2],
            h1_persistence_max=site_values[:, 3],
        )
        return LAPHFeatureResult(
            True,
            None,
            len(site_rows),
            min(retained_counts),
            max(retained_counts),
            int(site_values[:, 4].sum()),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_laph_row(atoms: Atoms) -> dict[str, object]:
    result = compute_laph_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "laph_supported": bool(result.supported),
            "laph_failure": result.failure_reason,
            "laph_site_count": result.site_count,
            "laph_min_retained_facets": result.min_retained_facets,
            "laph_max_retained_facets": result.max_retained_facets,
            "laph_h1_interval_count": result.h1_interval_count,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "laph_supported": False,
        "laph_failure": f"{type(exc).__name__}: {exc}",
        "laph_site_count": 0,
        "laph_min_retained_facets": 0,
        "laph_max_retained_facets": 0,
        "laph_h1_interval_count": 0,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_laph_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_laph_row(AseAtomsAdaptor.get_atoms(structure))
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
        return list(executor.map(worker, payloads, chunksize=8))  # type: ignore[arg-type]


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def build_cross_source_laph_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT263 from physically isolated discovery geometry only."""

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
        raise ValueError("NEXT263 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT263 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT263 formal input identity differs: {differing}")
    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name)
        for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT263 frozen upstream source differs")
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
        raise ValueError("NEXT263 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT263 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT263 {source} discovery identity differs")
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
                raise RuntimeError(f"NEXT263 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery[source]):
                raise RuntimeError(f"NEXT263 {source} row accounting differs")
            supported = table["laph_supported"].fillna(False).astype(bool)
            finite_counts = {
                name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
                for name in FEATURE_NAMES
            }
            finite_matrix = np.column_stack(
                [
                    np.isfinite(pd.to_numeric(table[name], errors="coerce").to_numpy(float))
                    for name in FEATURE_NAMES
                ]
            )
            if not finite_matrix[supported].all() or finite_matrix[~supported].any():
                raise RuntimeError(f"NEXT263 {source} support/finite contract differs")
            diagnostics = {
                name: pd.to_numeric(table[name], errors="coerce")
                for name in (
                    "laph_site_count",
                    "laph_min_retained_facets",
                    "laph_max_retained_facets",
                    "laph_h1_interval_count",
                )
            }
            if (
                not (diagnostics["laph_site_count"][supported] > 0).all()
                or not (
                    diagnostics["laph_min_retained_facets"][supported]
                    >= MIN_RETAINED_FACETS
                ).all()
                or not (
                    diagnostics["laph_max_retained_facets"][supported]
                    <= MAX_RETAINED_FACETS
                ).all()
            ):
                raise RuntimeError(f"NEXT263 {source} diagnostics differ")
            failures = Counter(table.loc[~supported, "laph_failure"].astype(str))
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": dict(sorted(failures.items())),
                "finite_feature_counts": finite_counts,
                "site_count": int(diagnostics["laph_site_count"][supported].sum()),
                "h1_interval_count": int(
                    diagnostics["laph_h1_interval_count"][supported].sum()
                ),
            }
        if counts["scigen"]["rows"] != 13_470 or counts["wyformer"]["rows"] != 5_232:  # type: ignore[index]
            raise RuntimeError("NEXT263 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "site_quantities": list(SITE_QUANTITIES),
            "aggregates": list(AGGREGATES),
            "voronoi": {"weight": "solid_angle", "tol": 0, "cutoff": 13},
            "facet_identity": "site_index_integer_image_keep_largest_area",
            "minimum_facet_area_fraction": 1.0 / AREA_FRACTION_DENOMINATOR,
            "retained_facet_count_range": [MIN_RETAINED_FACETS, MAX_RETAINED_FACETS],
            "point_cloud": "unit_voronoi_facet_normals",
            "complex": "complete_vietoris_rips_through_dimension_2",
            "coefficient_field": "F2",
            "distance_grid": DISTANCE_GRID,
            "zero_persistence_h1_included": False,
            "quantile_method": "inverted_cdf",
            "aggregation_arithmetic": "quantized_exact_rational_then_float",
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_laph_feature_freeze",
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
                "src/next263_local_angular_persistent_homology.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT263 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT263 source changed before publication")
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
    manifest = build_cross_source_laph_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FEATURE_NAMES",
    "LAPHFeatureResult",
    "aggregate_laph_features",
    "build_cross_source_laph_features",
    "compute_laph_features",
    "vietoris_rips_intervals",
]


if __name__ == "__main__":
    raise SystemExit(main())
