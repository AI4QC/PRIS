#!/usr/bin/env python3
"""Build discovery-only third-order Voronoi bond-order features."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
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


PROTOCOL = "2026-08-09-next247-third-order-voronoi-bond-order-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT247_TVBO_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next247_scigen_discovery_tvbo_features.parquet",
    "wyformer": "next247_wyformer_discovery_tvbo_features.parquet",
}
FEATURE_NAMES = (
    "tvbo_w4_abs_mean",
    "tvbo_w4_abs_q10",
    "tvbo_w4_abs_std",
    "tvbo_w6_abs_mean",
    "tvbo_w6_abs_q10",
    "tvbo_w6_abs_std",
    "tvbo_bar_w4_abs_mean",
    "tvbo_bar_w4_abs_q10",
    "tvbo_bar_w4_abs_std",
    "tvbo_bar_w6_abs_mean",
    "tvbo_bar_w6_abs_q10",
    "tvbo_bar_w6_abs_std",
    "tvbo_w4_coarse_delta_mean",
    "tvbo_w4_coarse_delta_q90",
    "tvbo_w6_coarse_delta_mean",
    "tvbo_w6_coarse_delta_q90",
)
EXPECTED_DESIGN_SHA256 = (
    "3bbd7c0024623a52a0fb1bee2d9bd36aff1910070ef7d2181d75f0162e1c8d7a"
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
class TVBOFeatureResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> TVBOFeatureResult:
    reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    return TVBOFeatureResult(False, reason, {})


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
        raise ValueError("NEXT247 angular order differs")
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
        raise ValueError("NEXT247 site facet population differs")
    norms = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > 1.0e-7):
        raise ValueError("NEXT247 Voronoi facet normal differs")
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
        raise ValueError("NEXT247 q_lm schema differs")
    return vector


def wigner_3j_equal_order(order: int, m1: int, m2: int, m3: int) -> float:
    """Evaluate `(l l l; m1 m2 m3)` by the frozen integer Racah sum."""

    if (
        type(order) is not int
        or order not in ANGULAR_ORDERS
        or any(type(value) is not int for value in (m1, m2, m3))
        or any(abs(value) > order for value in (m1, m2, m3))
    ):
        raise ValueError("NEXT247 Wigner-3j inputs differ")
    if m1 + m2 + m3 != 0:
        return 0.0
    factorial = math.factorial
    triangle = factorial(order) ** 3 / factorial(3 * order + 1)
    magnetic = math.prod(
        factorial(order + value) * factorial(order - value)
        for value in (m1, m2, m3)
    )
    sign = -1.0 if m3 % 2 else 1.0
    prefactor = sign * math.sqrt(triangle * magnetic)
    lower = max(0, -m1, m2)
    upper = min(order, order - m1, order + m2)
    total = 0.0
    for z in range(lower, upper + 1):
        denominator = (
            factorial(z)
            * factorial(order - z)
            * factorial(order - m1 - z)
            * factorial(order + m2 - z)
            * factorial(m1 + z)
            * factorial(-m2 + z)
        )
        total += (-1.0 if z % 2 else 1.0) / denominator
    result = prefactor * total
    if not math.isfinite(result):
        raise ValueError("NEXT247 Wigner-3j coefficient differs")
    return float(result)


@lru_cache(maxsize=2)
def wigner_3j_terms(order: int) -> tuple[tuple[int, int, int, float], ...]:
    """Return all nonzero frozen equal-order Wigner-3j coefficients."""

    if type(order) is not int or order not in ANGULAR_ORDERS:
        raise ValueError("NEXT247 Wigner-3j order differs")
    terms = []
    for m1 in range(-order, order + 1):
        for m2 in range(-order, order + 1):
            m3 = -m1 - m2
            if -order <= m3 <= order:
                coefficient = wigner_3j_equal_order(order, m1, m2, m3)
                if coefficient != 0.0:
                    terms.append((m1, m2, m3, coefficient))
    if not terms:
        raise RuntimeError("NEXT247 Wigner-3j term universe differs")
    return tuple(terms)


def normalized_third_order_invariant(vector: object, *, order: int) -> float:
    """Return the real normalized Steinhardt third-order invariant."""

    values = np.asarray(vector, dtype=complex)
    if (
        type(order) is not int
        or order not in ANGULAR_ORDERS
        or values.shape != (2 * order + 1,)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("NEXT247 third-order vector differs")
    norm_squared = float(np.vdot(values, values).real)
    denominator = norm_squared**1.5
    if denominator <= EPSILON:
        return 0.0
    total = sum(
        coefficient
        * values[m1 + order]
        * values[m2 + order]
        * values[m3 + order]
        for m1, m2, m3, coefficient in wigner_3j_terms(order)
    )
    if not (math.isfinite(total.real) and math.isfinite(total.imag)):
        raise ValueError("NEXT247 third-order contraction differs")
    if abs(float(total.imag)) > 1.0e-12:
        raise ValueError("NEXT247 third-order invariant is not real")
    return _bounded(
        float(total.real / denominator),
        low=-1.0,
        high=1.0,
        label=f"hat W{order}",
    )


def third_order_site_values(
    *,
    qlm: object,
    neighbor_indices: Sequence[object],
    neighbor_weights: Sequence[object],
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw, coarse-grained, and absolute-delta third-order values."""

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
        raise ValueError("NEXT247 coarse-grained population differs")
    raw_values: list[float] = []
    bar_values: list[float] = []
    delta_values: list[float] = []
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
            raise ValueError("NEXT247 directed neighbor population differs")
        neighbor_average = np.sum(weights[:, None] * vectors[indices], axis=0)
        coarse = 0.5 * (vectors[center] + neighbor_average)
        raw = normalized_third_order_invariant(vectors[center], order=order)
        bar = normalized_third_order_invariant(coarse, order=order)
        raw_values.append(raw)
        bar_values.append(bar)
        delta_values.append(abs(bar - raw))
    outputs = tuple(
        np.asarray(values, dtype=float)
        for values in (raw_values, bar_values, delta_values)
    )
    if any(values.shape != (len(vectors),) for values in outputs) or any(
        np.any(~np.isfinite(values)) for values in outputs
    ):
        raise ValueError("NEXT247 coarse-grained output differs")
    return outputs  # type: ignore[return-value]


def aggregate_tvbo_features(
    *,
    w4: object,
    w6: object,
    bar_w4: object,
    bar_w6: object,
    delta_w4: object,
    delta_w6: object,
) -> dict[str, float]:
    """Aggregate site values into the frozen sixteen-feature schema."""

    arrays = {
        "w4": np.asarray(w4, dtype=float),
        "w6": np.asarray(w6, dtype=float),
        "bar_w4": np.asarray(bar_w4, dtype=float),
        "bar_w6": np.asarray(bar_w6, dtype=float),
        "delta_w4": np.asarray(delta_w4, dtype=float),
        "delta_w6": np.asarray(delta_w6, dtype=float),
    }
    shapes = {values.shape for values in arrays.values()}
    if (
        len(shapes) != 1
        or next(iter(shapes), ()) == ()
        or len(next(iter(shapes))) != 1
        or next(iter(shapes))[0] == 0
        or any(np.any(~np.isfinite(values)) for values in arrays.values())
    ):
        raise ValueError("NEXT247 aggregate population differs")
    for name in ("w4", "w6", "bar_w4", "bar_w6"):
        values = arrays[name]
        if np.any(values < -1.0 - NUMERICAL_TOLERANCE) or np.any(
            values > 1.0 + NUMERICAL_TOLERANCE
        ):
            raise ValueError(f"NEXT247 aggregate {name} bounds differ")
    for name in ("delta_w4", "delta_w6"):
        values = arrays[name]
        if np.any(values < -NUMERICAL_TOLERANCE) or np.any(
            values > 2.0 + NUMERICAL_TOLERANCE
        ):
            raise ValueError(f"NEXT247 aggregate {name} bounds differ")
    w4_abs = np.abs(arrays["w4"])
    w6_abs = np.abs(arrays["w6"])
    bar_w4_abs = np.abs(arrays["bar_w4"])
    bar_w6_abs = np.abs(arrays["bar_w6"])
    delta4 = arrays["delta_w4"]
    delta6 = arrays["delta_w6"]
    q10 = lambda values: float(np.quantile(values, 0.10, method="linear"))
    q90 = lambda values: float(np.quantile(values, 0.90, method="linear"))
    features = {
        "tvbo_w4_abs_mean": float(np.mean(w4_abs)),
        "tvbo_w4_abs_q10": q10(w4_abs),
        "tvbo_w4_abs_std": float(np.std(w4_abs)),
        "tvbo_w6_abs_mean": float(np.mean(w6_abs)),
        "tvbo_w6_abs_q10": q10(w6_abs),
        "tvbo_w6_abs_std": float(np.std(w6_abs)),
        "tvbo_bar_w4_abs_mean": float(np.mean(bar_w4_abs)),
        "tvbo_bar_w4_abs_q10": q10(bar_w4_abs),
        "tvbo_bar_w4_abs_std": float(np.std(bar_w4_abs)),
        "tvbo_bar_w6_abs_mean": float(np.mean(bar_w6_abs)),
        "tvbo_bar_w6_abs_q10": q10(bar_w6_abs),
        "tvbo_bar_w6_abs_std": float(np.std(bar_w6_abs)),
        "tvbo_w4_coarse_delta_mean": float(np.mean(delta4)),
        "tvbo_w4_coarse_delta_q90": q90(delta4),
        "tvbo_w6_coarse_delta_mean": float(np.mean(delta6)),
        "tvbo_w6_coarse_delta_q90": q90(delta6),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        raise ValueError("NEXT247 aggregate feature schema differs")
    return features


def compute_tvbo_features(atoms: Atoms) -> TVBOFeatureResult:
    """Compute frozen third-order Voronoi bond order from one raw x0 structure."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("TVBO features require exact periodic geometry-only Atoms")
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
            order: third_order_site_values(
                qlm=np.asarray(qlm[order]),
                neighbor_indices=neighbor_indices,
                neighbor_weights=neighbor_weights,
                order=order,
            )
            for order in ANGULAR_ORDERS
        }
        features = aggregate_tvbo_features(
            w4=site_values[4][0],
            w6=site_values[6][0],
            bar_w4=site_values[4][1],
            bar_w6=site_values[6][1],
            delta_w4=site_values[4][2],
            delta_w6=site_values[6][2],
        )
        return TVBOFeatureResult(True, None, features)
    except Exception as exc:
        return _failure(exc)


def compute_tvbo_row(atoms: Atoms) -> dict[str, object]:
    result = compute_tvbo_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row["tvbo_supported"] = bool(result.supported)
    row["tvbo_failure"] = result.failure_reason
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "tvbo_supported": False,
        "tvbo_failure": f"{type(exc).__name__}: {exc}",
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_tvbo_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_tvbo_row(AseAtomsAdaptor.get_atoms(structure))
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


def build_cross_source_tvbo_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT247 from physically isolated discovery geometry only."""

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
        raise ValueError("NEXT247 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT247 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT247 formal input identity differs: {differing}")
    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name)
        for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT247 frozen upstream source differs")
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
        raise ValueError("NEXT247 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT247 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT247 {source} discovery identity differs")
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
                raise RuntimeError(f"NEXT247 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery[source]):
                raise RuntimeError(f"NEXT247 {source} row accounting differs")
            supported = table["tvbo_supported"].fillna(False).astype(bool)
            finite_counts = {
                name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
                for name in FEATURE_NAMES
            }
            if not supported.all() or any(count != len(table) for count in finite_counts.values()):
                failures = Counter(table.loc[~supported, "tvbo_failure"].astype(str))
                raise RuntimeError(
                    f"NEXT247 {source} formal support differs: {dict(sorted(failures.items()))}"
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
            raise RuntimeError("NEXT247 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "angular_orders": list(ANGULAR_ORDERS),
            "voronoi": {"weight": "solid_angle", "tol": 0, "cutoff": 13},
            "facet_weight": "area_over_site_total_area",
            "central_weight": 0.5,
            "neighbor_weight": 0.5,
            "third_order_invariant": "normalized_equal_l_wigner_3j_contraction",
            "wigner_3j_implementation": "integer_factorial_racah_sum",
            "zero_denominator_epsilon": EPSILON,
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
            "mode": "physically_isolated_discovery_x0_tvbo_feature_freeze",
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
                "src/next247_third_order_voronoi_bond_order.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT247 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT247 source changed before publication")
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
    manifest = build_cross_source_tvbo_features(
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
    "TVBOFeatureResult",
    "aggregate_tvbo_features",
    "build_cross_source_tvbo_features",
    "compute_tvbo_features",
    "normalized_third_order_invariant",
    "third_order_site_values",
    "weighted_spherical_harmonics",
    "wigner_3j_equal_order",
    "wigner_3j_terms",
]


if __name__ == "__main__":
    raise SystemExit(main())
