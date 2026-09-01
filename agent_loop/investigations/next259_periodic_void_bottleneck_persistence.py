#!/usr/bin/env python3
"""Build discovery-only periodic void-bottleneck persistence features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
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


PROTOCOL = "2026-08-09-next259-periodic-void-bottleneck-persistence-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT259_PVBP_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next259_scigen_discovery_pvbp_features.parquet",
    "wyformer": "next259_wyformer_discovery_pvbp_features.parquet",
}
METRIC_NAMES = ("isolation_any", "isolation_3d", "prominence_any", "radius")
STATISTIC_NAMES = ("mean", "q75", "q90", "upper_quartile_mean")
FEATURE_NAMES = tuple(
    f"pvbp_{metric}_{statistic}"
    for metric in METRIC_NAMES
    for statistic in STATISTIC_NAMES
)
EXPECTED_DESIGN_SHA256 = (
    "283a51d84284edd317e20886b06c61b6800f4806c46f6e9311b82ed04701845d"
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
FRACTIONAL_QUANTUM = 1_000_000_000
BISECTOR_RANK_TOLERANCE = 1.0e-10
BISECTOR_RESIDUAL_TOLERANCE = 1.0e-9
DUPLICATE_TOLERANCE = 5.0e-8
ROUND_DECIMALS = 11
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
class PVBPFeatureResult:
    supported: bool
    failure_reason: str | None
    node_count: int
    edge_count: int
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PVBPFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PVBPFeatureResult(False, reason, 0, 0, {})


def _integer_vector(value: object, *, label: str) -> tuple[int, int, int]:
    array = np.asarray(value)
    if array.shape != (3,) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} translation must contain three exact integers")
    numeric = np.asarray(array, dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError(f"{label} translation must contain three exact integers")
    return tuple(int(item) for item in np.rint(numeric))


def translation_rank(vectors: Sequence[Sequence[int]] | np.ndarray) -> int:
    """Return the exact rank of integer translation vectors in Z^3."""

    array = np.asarray(vectors)
    if array.size == 0:
        return 0
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("translation vectors must have shape (n, 3)")
    parsed = [_integer_vector(row, label="cycle") for row in array]
    nonzero = [np.asarray(row, dtype=np.int64) for row in parsed if any(row)]
    if not nonzero:
        return 0
    first = nonzero[0]
    second = next(
        (row for row in nonzero[1:] if np.any(np.cross(first, row))), None
    )
    if second is None:
        return 1
    for row in nonzero[1:]:
        if int(np.dot(first, np.cross(second, row))) != 0:
            return 3
    return 2


def _reduced_translation_basis(
    vectors: Sequence[Sequence[int]] | np.ndarray,
) -> list[tuple[int, int, int]]:
    basis: list[tuple[int, int, int]] = []
    rank = 0
    for raw in vectors:
        vector = _integer_vector(raw, label="basis")
        if not any(vector):
            continue
        candidate_rank = translation_rank([*basis, vector])
        if candidate_rank > rank:
            basis.append(vector)
            rank = candidate_rank
            if rank == 3:
                break
    return basis


class _TranslationDSU:
    def __init__(self, node_count: int) -> None:
        self.parent = np.arange(node_count, dtype=int)
        self.size = np.ones(node_count, dtype=int)
        self.offset = np.zeros((node_count, 3), dtype=np.int64)
        self.basis: list[list[tuple[int, int, int]]] = [[] for _ in range(node_count)]
        self.pending_any: list[list[int]] = [[index] for index in range(node_count)]
        self.pending_3d: list[list[int]] = [[index] for index in range(node_count)]

    def find(self, node: int) -> tuple[int, np.ndarray]:
        parent = int(self.parent[node])
        if parent == node:
            return node, np.zeros(3, dtype=np.int64)
        root, parent_offset = self.find(parent)
        total = self.offset[node] + parent_offset
        self.parent[node] = root
        self.offset[node] = total
        return root, total.copy()

    def add_edge(
        self, left: int, right: int, translation: tuple[int, int, int]
    ) -> int:
        left_root, left_offset = self.find(left)
        right_root, right_offset = self.find(right)
        delta = np.asarray(translation, dtype=np.int64)
        if left_root == right_root:
            residual = left_offset + delta - right_offset
            self.basis[left_root] = _reduced_translation_basis(
                [*self.basis[left_root], residual]
            )
            return left_root

        if self.size[left_root] < self.size[right_root]:
            self.parent[left_root] = right_root
            self.offset[left_root] = right_offset - left_offset - delta
            root, child = right_root, left_root
        else:
            self.parent[right_root] = left_root
            self.offset[right_root] = left_offset + delta - right_offset
            root, child = left_root, right_root
        self.size[root] += self.size[child]
        self.basis[root] = _reduced_translation_basis(
            [*self.basis[root], *self.basis[child]]
        )
        self.pending_any[root].extend(self.pending_any[child])
        self.pending_any[child] = []
        self.pending_3d[root].extend(self.pending_3d[child])
        self.pending_3d[child] = []
        return root


def annotate_periodic_bottlenecks(
    node_count: int,
    edges: Sequence[tuple[int, int, Sequence[int], float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Annotate first rank>=1 and rank==3 capacities in a quotient graph."""

    if type(node_count) is not int or node_count <= 0:
        raise ValueError("node_count must be a positive exact integer")
    parsed: list[tuple[int, int, tuple[int, int, int], float]] = []
    for item in edges:
        if len(item) != 4:
            raise ValueError("each void edge must contain four fields")
        left, right, raw_translation, raw_capacity = item
        if (
            not isinstance(left, (int, np.integer))
            or not isinstance(right, (int, np.integer))
            or isinstance(left, (bool, np.bool_))
            or isinstance(right, (bool, np.bool_))
            or int(left) < 0
            or int(right) < 0
            or int(left) >= node_count
            or int(right) >= node_count
        ):
            raise ValueError("void edge endpoint is outside the quotient graph")
        translation = _integer_vector(raw_translation, label="edge")
        capacity = float(raw_capacity)
        if not math.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("void edge capacity must be finite and positive")
        if int(left) == int(right) and not any(translation):
            raise ValueError("zero-translation self edge is invalid")
        parsed.append((int(left), int(right), translation, capacity))
    if not parsed:
        raise ValueError("periodic void graph has no edges")
    parsed.sort(key=lambda item: (-item[3], item[0], item[1], item[2]))
    any_rank = np.full(node_count, np.nan, dtype=float)
    rank3 = np.full(node_count, np.nan, dtype=float)
    dsu = _TranslationDSU(node_count)
    for left, right, translation, capacity in parsed:
        root = dsu.add_edge(left, right, translation)
        root, _ = dsu.find(root)
        rank = len(dsu.basis[root])
        if rank >= 1 and dsu.pending_any[root]:
            any_rank[dsu.pending_any[root]] = capacity
            dsu.pending_any[root] = []
        if rank >= 3 and dsu.pending_3d[root]:
            rank3[dsu.pending_3d[root]] = capacity
            dsu.pending_3d[root] = []
    return np.nan_to_num(any_rank, nan=0.0), np.nan_to_num(rank3, nan=0.0)


def aggregate_pvbp_features(
    metric_populations: Mapping[str, object],
) -> dict[str, float]:
    """Aggregate node populations with replication-invariant upper tails."""

    if set(metric_populations) != set(METRIC_NAMES):
        raise ValueError("PVBP metric population schema differs")
    features: dict[str, float] = {}
    for metric in METRIC_NAMES:
        values = np.asarray(metric_populations[metric], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"PVBP {metric} population differs")
        values = np.round(values, ROUND_DECIMALS)
        if np.any(values < 0.0):
            raise ValueError(f"PVBP {metric} population differs")
        if metric.startswith("isolation_") and np.any(values > 1.0):
            raise ValueError(f"PVBP {metric} population differs")
        integer_scale = 10**ROUND_DECIMALS
        integers = sorted(int(np.rint(value * integer_scale)) for value in values)
        q75_integer = integers[math.ceil(0.75 * len(integers)) - 1]
        q90_integer = integers[math.ceil(0.90 * len(integers)) - 1]
        upper = [value for value in integers if value >= q75_integer]
        summaries = {
            "mean": float(Fraction(sum(integers), len(integers) * integer_scale)),
            "q75": float(Fraction(q75_integer, integer_scale)),
            "q90": float(Fraction(q90_integer, integer_scale)),
            "upper_quartile_mean": float(
                Fraction(sum(upper), len(upper) * integer_scale)
            ),
        }
        for statistic in STATISTIC_NAMES:
            features[f"pvbp_{metric}_{statistic}"] = summaries[statistic]
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        raise ValueError("PVBP aggregate schema differs")
    return features


def _winning_faces(
    *, structure: Structure, center: int, info: Sequence[Mapping[str, object]]
) -> list[tuple[tuple[int, ...], np.ndarray]]:
    best: dict[
        tuple[int, tuple[int, int, int]],
        tuple[float, tuple[tuple[int, ...], tuple[float, ...]], tuple[int, ...], np.ndarray],
    ] = {}
    center_coords = np.asarray(structure[center].coords, dtype=float)
    for item in info:
        try:
            site_index = int(item["site_index"])
            image = _integer_vector(item["image"], label="face")
            poly = item["poly_info"]
            if not isinstance(poly, Mapping):
                continue
            area = float(poly["area"])
            vertices = tuple(int(value) for value in poly["verts"])
            neighbor = item["site"]
            displacement = np.asarray(neighbor.coords, dtype=float) - center_coords
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if (
            site_index < 0
            or site_index >= len(structure)
            or not math.isfinite(area)
            or area <= 0.0
            or len(vertices) < 3
            or len(set(vertices)) != len(vertices)
            or displacement.shape != (3,)
            or np.any(~np.isfinite(displacement))
        ):
            continue
        key = (site_index, image)
        tie = (
            tuple(sorted(vertices)),
            tuple(float(value) for value in np.round(displacement, 12)),
        )
        previous = best.get(key)
        if previous is None or area > previous[0] or (
            area == previous[0] and tie < previous[1]
        ):
            best[key] = (area, tie, vertices, displacement)
    if not best:
        raise ValueError("PVBP site has no valid Voronoi face")
    return [(best[key][2], best[key][3]) for key in sorted(best)]


def _node_identity(
    coordinate: np.ndarray, lattice: np.ndarray
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    fractional = np.linalg.solve(lattice.T, coordinate)
    if fractional.shape != (3,) or not np.isfinite(fractional).all():
        raise ValueError("PVBP node fractional coordinate differs")
    scaled = fractional * FRACTIONAL_QUANTUM
    limit = float(np.iinfo(np.int64).max) - FRACTIONAL_QUANTUM
    if np.any(np.abs(scaled) > limit):
        raise ValueError("PVBP node fractional coordinate overflows identity grid")
    quantized = np.rint(scaled).astype(np.int64)
    base = np.mod(quantized, FRACTIONAL_QUANTUM)
    shift = (quantized - base) // FRACTIONAL_QUANTUM
    return tuple(int(value) for value in base), tuple(int(value) for value in shift)


def _canonical_edge(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    translation: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    forward = (left, right, translation)
    reverse = (right, left, tuple(-value for value in translation))
    return forward if forward <= reverse else reverse


def _check_duplicate(values: Sequence[float], *, label: str) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"PVBP {label} observation differs")
    if float(np.max(array) - np.min(array)) > DUPLICATE_TOLERANCE:
        raise ValueError(f"PVBP duplicate {label} disagreement exceeds tolerance")
    value = float(np.median(array))
    if value <= 0.0:
        raise ValueError(f"PVBP {label} must be positive")
    return round(value, ROUND_DECIMALS)


def compute_pvbp_features(atoms: Atoms) -> PVBPFeatureResult:
    """Compute frozen periodic void-bottleneck persistence from raw x0."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("PVBP features require exact periodic geometry-only Atoms")
        lattice = np.asarray(atoms.cell.array, dtype=float)
        volume = abs(float(np.linalg.det(lattice)))
        if lattice.shape != (3, 3) or not np.isfinite(lattice).all() or volume <= 0.0:
            raise ValueError("PVBP lattice differs")
        length_scale = (volume / len(atoms)) ** (1.0 / 3.0)
        structure = AseAtomsAdaptor.get_structure(atoms)
        finder = VoronoiNN(
            weight="solid_angle", tol=0, cutoff=13, compute_adj_neighbors=True
        )
        node_observations: dict[tuple[int, int, int], list[float]] = defaultdict(list)
        edge_observations: dict[
            tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
            list[float],
        ] = defaultdict(list)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for center in range(len(structure)):
                center_coords = np.asarray(structure[center].coords, dtype=float)
                faces = _winning_faces(
                    structure=structure,
                    center=center,
                    info=finder.get_nn_info(structure, center),
                )
                incident: dict[int, list[np.ndarray]] = defaultdict(list)
                for vertices, displacement in faces:
                    for vertex in vertices:
                        incident[vertex].append(displacement)
                if not incident:
                    raise ValueError("PVBP site has no valid Voronoi vertex")
                local: dict[
                    int,
                    tuple[
                        np.ndarray,
                        tuple[int, int, int],
                        tuple[int, int, int],
                        float,
                    ],
                ] = {}
                for vertex in sorted(incident):
                    matrix = np.asarray(incident[vertex], dtype=float)
                    rhs = np.einsum("ij,ij->i", matrix, matrix) / 2.0
                    if len(matrix) < 3 or np.linalg.matrix_rank(
                        matrix, tol=BISECTOR_RANK_TOLERANCE
                    ) < 3:
                        raise ValueError("PVBP void bisector rank differs")
                    relative, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
                    residual = float(np.max(np.abs(matrix @ relative - rhs)))
                    scale = max(1.0, float(np.max(np.abs(rhs))))
                    if residual > BISECTOR_RESIDUAL_TOLERANCE * scale:
                        raise ValueError("PVBP void bisector residual exceeds tolerance")
                    radius = float(np.linalg.norm(relative) / length_scale)
                    if not math.isfinite(radius) or radius <= 0.0:
                        raise ValueError("PVBP normalized node radius differs")
                    coordinate = center_coords + relative
                    node_key, image_shift = _node_identity(coordinate, lattice)
                    local[vertex] = (coordinate, node_key, image_shift, radius)
                    node_observations[node_key].append(radius)
                for vertices, _ in faces:
                    if any(vertex not in local for vertex in vertices):
                        raise ValueError("PVBP face has an unreconstructed vertex")
                    for left_vertex, right_vertex in zip(
                        vertices, (*vertices[1:], vertices[0])
                    ):
                        left_coordinate, left_key, left_shift, _ = local[left_vertex]
                        right_coordinate, right_key, right_shift, _ = local[right_vertex]
                        segment = right_coordinate - left_coordinate
                        denominator = float(np.dot(segment, segment))
                        if not math.isfinite(denominator) or denominator <= 0.0:
                            raise ValueError("PVBP void edge length differs")
                        relative = left_coordinate - center_coords
                        position = float(
                            np.clip(-np.dot(relative, segment) / denominator, 0.0, 1.0)
                        )
                        capacity = float(
                            np.linalg.norm(relative + position * segment) / length_scale
                        )
                        translation = tuple(
                            int(right_shift[index] - left_shift[index])
                            for index in range(3)
                        )
                        edge_key = _canonical_edge(left_key, right_key, translation)
                        if edge_key[0] == edge_key[1] and not any(edge_key[2]):
                            raise ValueError("PVBP zero-translation self edge differs")
                        edge_observations[edge_key].append(capacity)
        if not node_observations or not edge_observations:
            raise ValueError("PVBP quotient graph is empty")
        keys = sorted(node_observations)
        node_index = {key: index for index, key in enumerate(keys)}
        radii = np.asarray(
            [_check_duplicate(node_observations[key], label="node radius") for key in keys],
            dtype=float,
        )
        edges: list[tuple[int, int, tuple[int, int, int], float]] = []
        for (left_key, right_key, translation), observations in sorted(
            edge_observations.items()
        ):
            edges.append(
                (
                    node_index[left_key],
                    node_index[right_key],
                    translation,
                    _check_duplicate(observations, label="edge capacity"),
                )
            )
        any_rank, rank3 = annotate_periodic_bottlenecks(len(keys), edges)
        if (
            np.any(rank3 < -DUPLICATE_TOLERANCE)
            or np.any(any_rank < rank3 - DUPLICATE_TOLERANCE)
            or np.any(radii < any_rank - DUPLICATE_TOLERANCE)
        ):
            raise ValueError("PVBP bottleneck ordering differs")
        any_rank = np.clip(any_rank, 0.0, radii)
        rank3 = np.clip(rank3, 0.0, any_rank)
        populations = {
            "isolation_any": np.clip(1.0 - any_rank / radii, 0.0, 1.0),
            "isolation_3d": np.clip(1.0 - rank3 / radii, 0.0, 1.0),
            "prominence_any": np.maximum(radii - any_rank, 0.0),
            "radius": radii,
        }
        features = aggregate_pvbp_features(populations)
        return PVBPFeatureResult(True, None, len(keys), len(edges), features)
    except Exception as exc:
        return _failure(exc)


def compute_pvbp_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pvbp_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row["pvbp_supported"] = bool(result.supported)
    row["pvbp_failure"] = result.failure_reason
    row["pvbp_node_count"] = int(result.node_count)
    row["pvbp_edge_count"] = int(result.edge_count)
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "pvbp_supported": False,
        "pvbp_failure": f"{type(exc).__name__}: {exc}",
        "pvbp_node_count": 0,
        "pvbp_edge_count": 0,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_pvbp_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_pvbp_row(AseAtomsAdaptor.get_atoms(structure))
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


def build_cross_source_pvbp_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT259 from physically isolated discovery geometry only."""

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
        raise ValueError("NEXT259 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT259 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT259 formal input identity differs: {differing}")
    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name)
        for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT259 frozen upstream source differs")
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
        raise ValueError("NEXT259 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT259 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT259 {source} discovery identity differs")
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
                raise RuntimeError(f"NEXT259 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery[source]):
                raise RuntimeError(f"NEXT259 {source} row accounting differs")
            supported = table["pvbp_supported"].fillna(False).astype(bool)
            finite_counts = {
                name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
                for name in FEATURE_NAMES
            }
            for count in finite_counts.values():
                if count != int(supported.sum()):
                    raise RuntimeError(f"NEXT259 {source} feature support differs")
            nodes = pd.to_numeric(table.loc[supported, "pvbp_node_count"], errors="coerce")
            edges = pd.to_numeric(table.loc[supported, "pvbp_edge_count"], errors="coerce")
            if (
                supported.any()
                and (
                    not np.isfinite(nodes).all()
                    or not np.isfinite(edges).all()
                    or (nodes <= 0).any()
                    or (edges <= 0).any()
                )
            ):
                raise RuntimeError(f"NEXT259 {source} graph accounting differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            failures = Counter(table.loc[~supported, "pvbp_failure"].astype(str))
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": dict(sorted(failures.items())),
                "finite_feature_counts": finite_counts,
                "node_count_min": int(nodes.min()) if len(nodes) else 0,
                "node_count_max": int(nodes.max()) if len(nodes) else 0,
                "node_count_sum": int(nodes.sum()) if len(nodes) else 0,
                "edge_count_min": int(edges.min()) if len(edges) else 0,
                "edge_count_max": int(edges.max()) if len(edges) else 0,
                "edge_count_sum": int(edges.sum()) if len(edges) else 0,
            }
        if counts["scigen"]["rows"] != 13_470 or counts["wyformer"]["rows"] != 5_232:  # type: ignore[index]
            raise RuntimeError("NEXT259 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "metric_names": list(METRIC_NAMES),
            "statistics": list(STATISTIC_NAMES),
            "hypothesis_direction": "protected_low",
            "voronoi": {
                "weight": "solid_angle",
                "tol": 0,
                "cutoff": 13,
                "compute_adj_neighbors": True,
            },
            "distance_field": "unweighted_nearest_atomic_center",
            "normalization": "cube_root_cell_volume_per_atom",
            "fractional_quantum": FRACTIONAL_QUANTUM,
            "round_decimals": ROUND_DECIMALS,
            "duplicate_tolerance": DUPLICATE_TOLERANCE,
            "quantile_method": "inverted_cdf",
            "quantiles": [0.75, 0.90],
            "upper_quartile_boundary_ties_included": True,
            "translation_rank_arithmetic": "exact_integer_cross_and_triple_products",
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_pvbp_feature_freeze",
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
                "src/next259_periodic_void_bottleneck_persistence.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT259 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT259 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build discovery-only periodic void-bottleneck persistence features."
    )
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_cross_source_pvbp_features(
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
