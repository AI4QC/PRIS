#!/usr/bin/env python3
"""Build discovery-only coarse-grained Voronoi bond-order features."""

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
import warnings

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.special import sph_harm_y

import src.next85_scigen_label_free_features as n85
import src.next94_wyformer_label_free_features as n94
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next243-coarse-grained-voronoi-bond-order-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT243_CMVBO_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next243_scigen_discovery_cmvbo_features.parquet",
    "wyformer": "next243_wyformer_discovery_cmvbo_features.parquet",
}
FEATURE_NAMES = (
    "cmvbo_bar_q4_mean",
    "cmvbo_bar_q4_q10",
    "cmvbo_bar_q4_std",
    "cmvbo_bar_q6_mean",
    "cmvbo_bar_q6_q10",
    "cmvbo_bar_q6_std",
    "cmvbo_coherence_q4_mean",
    "cmvbo_coherence_q4_q10",
    "cmvbo_coherence_q6_mean",
    "cmvbo_coherence_q6_q10",
    "cmvbo_neighbor_corr_q4_mean",
    "cmvbo_neighbor_corr_q4_q10",
    "cmvbo_neighbor_corr_q6_mean",
    "cmvbo_neighbor_corr_q6_q10",
    "cmvbo_neighbor_corr_joint_q10",
)
EXPECTED_DESIGN_SHA256 = (
    "cf3127076d00304ca5258262a3aa08abaecfb65362c191e5d8dd58ae2a8dd09c"
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
ANGULAR_ORDERS = (4, 6)
EPSILON = 1.0e-14
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
class CMVBOFeatureResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> CMVBOFeatureResult:
    reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    return CMVBOFeatureResult(False, reason, {})


def _bounded(value: float, *, low: float, high: float, label: str) -> float:
    if (
        not math.isfinite(value)
        or value < low - NUMERICAL_TOLERANCE
        or value > high + NUMERICAL_TOLERANCE
    ):
        raise ValueError(f"{label} is outside the frozen [{low},{high}] guard")
    return float(np.clip(value, low, high))


def weighted_spherical_harmonics(
    *, normals: object, areas: object, order: int
) -> np.ndarray:
    """Return one frozen facet-area-weighted complex q_lm vector."""

    if type(order) is not int or order not in ANGULAR_ORDERS:
        raise ValueError("NEXT243 angular order differs")
    direction = np.asarray(normals, dtype=float)
    raw_area = np.asarray(areas, dtype=float)
    if (
        direction.ndim != 2
        or direction.shape[1:] != (3,)
        or len(direction) == 0
        or raw_area.shape != (len(direction),)
        or np.any(~np.isfinite(direction))
        or np.any(~np.isfinite(raw_area))
        or np.any(raw_area <= 0.0)
    ):
        raise ValueError("NEXT243 site facet population differs")
    norms = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > 1.0e-7):
        raise ValueError("NEXT243 Voronoi facet normal differs")
    unit = direction / norms[:, None]
    weights = raw_area / float(np.sum(raw_area))
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    vector = np.asarray(
        [
            np.sum(weights * sph_harm_y(order, m, theta, phi))
            for m in range(-order, order + 1)
        ],
        dtype=complex,
    )
    if vector.shape != (2 * order + 1,) or np.any(~np.isfinite(vector)):
        raise ValueError("NEXT243 q_lm schema differs")
    return vector


def bond_order_magnitude(vector: object, *, order: int) -> float:
    """Map a complex q_lm vector to its rotationally invariant magnitude."""

    values = np.asarray(vector, dtype=complex)
    if (
        type(order) is not int
        or order not in ANGULAR_ORDERS
        or values.shape != (2 * order + 1,)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("NEXT243 bond-order vector differs")
    squared = 4.0 * np.pi / (2 * order + 1) * float(np.vdot(values, values).real)
    if squared < -NUMERICAL_TOLERANCE or squared > 1.0 + NUMERICAL_TOLERANCE:
        raise ValueError(f"NEXT243 q{order} squared differs")
    return _bounded(
        math.sqrt(max(0.0, squared)), low=0.0, high=1.0, label=f"bar q{order}"
    )


def coarse_grained_site_values(
    *,
    qlm: object,
    neighbor_indices: Sequence[object],
    neighbor_weights: Sequence[object],
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-site bar-q, coherence, and directed neighbor correlation."""

    vectors = np.asarray(qlm, dtype=complex)
    if (
        type(order) is not int
        or order not in ANGULAR_ORDERS
        or vectors.ndim != 2
        or vectors.shape[1:] != (2 * order + 1,)
        or len(vectors) == 0
        or len(neighbor_indices) != len(vectors)
        or len(neighbor_weights) != len(vectors)
        or np.any(~np.isfinite(vectors))
    ):
        raise ValueError("NEXT243 coarse-grained population differs")
    site_norms = np.linalg.norm(vectors, axis=1)
    bar_values: list[float] = []
    coherence_values: list[float] = []
    correlation_values: list[float] = []
    for center in range(len(vectors)):
        indices = np.asarray(neighbor_indices[center], dtype=int)
        weights = np.asarray(neighbor_weights[center], dtype=float)
        if (
            indices.ndim != 1
            or len(indices) == 0
            or weights.shape != indices.shape
            or np.any(indices < 0)
            or np.any(indices >= len(vectors))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1.0e-12)
        ):
            raise ValueError("NEXT243 directed neighbor population differs")
        neighbor_average = np.sum(weights[:, None] * vectors[indices], axis=0)
        coarse = 0.5 * (vectors[center] + neighbor_average)
        bar_values.append(bond_order_magnitude(coarse, order=order))
        denominator = 0.5 * (
            site_norms[center] + float(weights @ site_norms[indices])
        )
        coherence = 0.0 if denominator <= EPSILON else np.linalg.norm(coarse) / denominator
        coherence_values.append(
            _bounded(float(coherence), low=0.0, high=1.0, label=f"coherence q{order}")
        )
        pair_denominators = site_norms[center] * site_norms[indices]
        pair = np.zeros(len(indices), dtype=float)
        valid = pair_denominators > EPSILON
        if np.any(valid):
            numerators = np.real(
                np.sum(vectors[center][None, :] * np.conj(vectors[indices]), axis=1)
            )
            pair[valid] = numerators[valid] / pair_denominators[valid]
        if np.any(pair < -1.0 - NUMERICAL_TOLERANCE) or np.any(
            pair > 1.0 + NUMERICAL_TOLERANCE
        ):
            raise ValueError(f"NEXT243 neighbor correlation q{order} differs")
        correlation_values.append(float(weights @ np.clip(pair, -1.0, 1.0)))
    outputs = tuple(
        np.asarray(values, dtype=float)
        for values in (bar_values, coherence_values, correlation_values)
    )
    if any(values.shape != (len(vectors),) for values in outputs) or any(
        np.any(~np.isfinite(values)) for values in outputs
    ):
        raise ValueError("NEXT243 coarse-grained output differs")
    return outputs  # type: ignore[return-value]


def aggregate_cmvbo_features(
    *,
    bar_q4: object,
    bar_q6: object,
    coherence_q4: object,
    coherence_q6: object,
    neighbor_corr_q4: object,
    neighbor_corr_q6: object,
) -> dict[str, float]:
    """Aggregate site values into the frozen fifteen-feature schema."""

    arrays = {
        "bar_q4": np.asarray(bar_q4, dtype=float),
        "bar_q6": np.asarray(bar_q6, dtype=float),
        "coherence_q4": np.asarray(coherence_q4, dtype=float),
        "coherence_q6": np.asarray(coherence_q6, dtype=float),
        "neighbor_corr_q4": np.asarray(neighbor_corr_q4, dtype=float),
        "neighbor_corr_q6": np.asarray(neighbor_corr_q6, dtype=float),
    }
    shapes = {values.shape for values in arrays.values()}
    if (
        len(shapes) != 1
        or next(iter(shapes), ()) == ()
        or len(next(iter(shapes))) != 1
        or next(iter(shapes))[0] == 0
        or any(np.any(~np.isfinite(values)) for values in arrays.values())
    ):
        raise ValueError("NEXT243 aggregate population differs")
    for name in ("bar_q4", "bar_q6", "coherence_q4", "coherence_q6"):
        values = arrays[name]
        if np.any(values < -NUMERICAL_TOLERANCE) or np.any(
            values > 1.0 + NUMERICAL_TOLERANCE
        ):
            raise ValueError(f"NEXT243 aggregate {name} bounds differ")
    for name in ("neighbor_corr_q4", "neighbor_corr_q6"):
        values = arrays[name]
        if np.any(values < -1.0 - NUMERICAL_TOLERANCE) or np.any(
            values > 1.0 + NUMERICAL_TOLERANCE
        ):
            raise ValueError(f"NEXT243 aggregate {name} bounds differ")
    q4 = arrays["bar_q4"]
    q6 = arrays["bar_q6"]
    c4 = arrays["coherence_q4"]
    c6 = arrays["coherence_q6"]
    r4 = arrays["neighbor_corr_q4"]
    r6 = arrays["neighbor_corr_q6"]
    joint = 0.5 * (r4 + r6)
    q10 = lambda values: float(np.quantile(values, 0.10, method="linear"))
    features = {
        "cmvbo_bar_q4_mean": float(np.mean(q4)),
        "cmvbo_bar_q4_q10": q10(q4),
        "cmvbo_bar_q4_std": float(np.std(q4)),
        "cmvbo_bar_q6_mean": float(np.mean(q6)),
        "cmvbo_bar_q6_q10": q10(q6),
        "cmvbo_bar_q6_std": float(np.std(q6)),
        "cmvbo_coherence_q4_mean": float(np.mean(c4)),
        "cmvbo_coherence_q4_q10": q10(c4),
        "cmvbo_coherence_q6_mean": float(np.mean(c6)),
        "cmvbo_coherence_q6_q10": q10(c6),
        "cmvbo_neighbor_corr_q4_mean": float(np.mean(r4)),
        "cmvbo_neighbor_corr_q4_q10": q10(r4),
        "cmvbo_neighbor_corr_q6_mean": float(np.mean(r6)),
        "cmvbo_neighbor_corr_q6_q10": q10(r6),
        "cmvbo_neighbor_corr_joint_q10": q10(joint),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        raise ValueError("NEXT243 aggregate feature schema differs")
    return features


def compute_cmvbo_features(atoms: Atoms) -> CMVBOFeatureResult:
    """Compute frozen coarse-grained bond order from one raw x0 structure."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("CMVBO features require exact periodic geometry-only Atoms")
        structure = AseAtomsAdaptor.get_structure(atoms)
        finder = VoronoiNN(weight="solid_angle", tol=0, cutoff=13)
        neighbor_indices: list[np.ndarray] = []
        neighbor_weights: list[np.ndarray] = []
        qlm = {order: [] for order in ANGULAR_ORDERS}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for center in range(len(structure)):
                info = finder.get_nn_info(structure, center)
                facets: dict[
                    tuple[int, tuple[int, int, int]], tuple[np.ndarray, float]
                ] = {}
                for item in info:
                    try:
                        site_index = int(item["site_index"])
                        image = tuple(
                            int(round(float(value))) for value in item["image"]
                        )
                        poly = item["poly_info"]
                        normal = np.asarray(poly["normal"], dtype=float)
                        area = float(poly["area"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (
                        len(image) != 3
                        or normal.shape != (3,)
                        or site_index < 0
                        or site_index >= len(structure)
                        or not math.isfinite(area)
                        or area <= 0.0
                    ):
                        continue
                    key = (site_index, image)
                    previous = facets.get(key)
                    if previous is None or area > previous[1]:
                        facets[key] = (normal, area)
                keys = sorted(facets)
                if not keys:
                    raise ValueError("site has no valid Voronoi facet")
                normals = np.asarray([facets[key][0] for key in keys])
                areas = np.asarray([facets[key][1] for key in keys])
                weights = areas / float(np.sum(areas))
                neighbor_indices.append(
                    np.asarray([key[0] for key in keys], dtype=int)
                )
                neighbor_weights.append(weights)
                for order in ANGULAR_ORDERS:
                    qlm[order].append(
                        weighted_spherical_harmonics(
                            normals=normals, areas=areas, order=order
                        )
                    )
        site_values = {
            order: coarse_grained_site_values(
                qlm=np.asarray(qlm[order]),
                neighbor_indices=neighbor_indices,
                neighbor_weights=neighbor_weights,
                order=order,
            )
            for order in ANGULAR_ORDERS
        }
        features = aggregate_cmvbo_features(
            bar_q4=site_values[4][0],
            bar_q6=site_values[6][0],
            coherence_q4=site_values[4][1],
            coherence_q6=site_values[6][1],
            neighbor_corr_q4=site_values[4][2],
            neighbor_corr_q6=site_values[6][2],
        )
        return CMVBOFeatureResult(True, None, features)
    except Exception as exc:
        return _failure(exc)


def compute_cmvbo_row(atoms: Atoms) -> dict[str, object]:
    result = compute_cmvbo_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row["cmvbo_supported"] = bool(result.supported)
    row["cmvbo_failure"] = result.failure_reason
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "cmvbo_supported": False,
        "cmvbo_failure": f"{type(exc).__name__}: {exc}",
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_cmvbo_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_cmvbo_row(AseAtomsAdaptor.get_atoms(structure))
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


def build_cross_source_cmvbo_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT243 from physically isolated discovery geometry only."""

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
        raise ValueError("NEXT243 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT243 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT243 formal input identity differs: {differing}")
    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name)
        for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT243 frozen upstream source differs")
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
        raise ValueError("NEXT243 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT243 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT243 {source} discovery identity differs")
        discovery[source] = selected
    scigen_ids = discovery["scigen"]["material_id"].astype(str).tolist()
    wyformer_ids = discovery["wyformer"]["material_id"].astype(str).tolist()
    payloads = {
        "scigen": n85._archive_payloads(paths["scigen_discovery_geometry"], scigen_ids),
        "wyformer": n94._payloads(paths["wyformer_discovery_geometry"], wyformer_ids),
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
        counts = {}
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
                raise RuntimeError(f"NEXT243 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery[source]):
                raise RuntimeError(f"NEXT243 {source} row accounting differs")
            supported = table["cmvbo_supported"].fillna(False).astype(bool)
            finite_counts = {
                name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
                for name in FEATURE_NAMES
            }
            if not supported.all() or any(count != len(table) for count in finite_counts.values()):
                failures = Counter(table.loc[~supported, "cmvbo_failure"].astype(str))
                raise RuntimeError(
                    f"NEXT243 {source} formal support differs: {dict(sorted(failures.items()))}"
                )
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": {},
                "finite_feature_counts": finite_counts,
            }
        if counts["scigen"]["rows"] != 13_470 or counts["wyformer"]["rows"] != 5_232:
            raise RuntimeError("NEXT243 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "angular_orders": list(ANGULAR_ORDERS),
            "voronoi": {"weight": "solid_angle", "tol": 0, "cutoff": 13},
            "facet_weight": "area_over_site_total_area",
            "central_weight": 0.5,
            "neighbor_weight": 0.5,
            "quantile_method": "linear",
            "source_partitions_read": {
                "scigen": ["discovery"],
                "wyformer": ["discovery"],
            },
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_cmvbo_feature_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "source_partitions_read": {
                "scigen": ["discovery"],
                "wyformer": ["discovery"],
            },
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "upstream_source_sha256": upstream_hashes,
            "executed_source_sha256": {
                "src/next243_coarse_grained_voronoi_bond_order.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT243 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT243 source changed before publication")
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
    manifest = build_cross_source_cmvbo_features(
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
    "CMVBOFeatureResult",
    "aggregate_cmvbo_features",
    "bond_order_magnitude",
    "build_cross_source_cmvbo_features",
    "coarse_grained_site_values",
    "compute_cmvbo_features",
    "weighted_spherical_harmonics",
]


if __name__ == "__main__":
    raise SystemExit(main())
