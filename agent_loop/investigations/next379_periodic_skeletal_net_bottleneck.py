#!/usr/bin/env python3
"""Periodic skeletal-net bottleneck from one raw x0 geometry only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping
import warnings

from ase import Atoms
import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor

import src.next267_periodic_radical_voronoi_packing as n267


PROTOCOL = "2026-08-13-next379-periodic-skeletal-net-bottleneck-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next379-next382-periodic-skeletal-net-bottleneck.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("psnb_skeletal_3d_bottleneck_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
REVERSE_ANGLE_TOLERANCE = 1.0e-8
NUMERICAL_TOLERANCE = 1.0e-12
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
class PSNBFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    directed_face_count: int
    undirected_edge_count: int
    rank3_site_count: int
    maximum_reverse_angle_error: float
    site_bottlenecks: tuple[float, ...]
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PSNBFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PSNBFeatureResult(False, reason, 0, 0, 0, 0, math.nan, (), {})


def _quantized(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("NEXT379 value is non-finite")
    return float(np.rint(value * OUTPUT_GRID) / OUTPUT_GRID)


def _reverse(key: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    left, right, x, y, z = key
    return right, left, -x, -y, -z


def _integer_rank(vectors: list[np.ndarray]) -> int:
    nonzero = [vector for vector in vectors if np.any(vector != 0)]
    if not nonzero:
        return 0
    first = nonzero[0]
    second = next(
        (vector for vector in nonzero[1:] if np.any(np.cross(first, vector))),
        None,
    )
    if second is None:
        return 1
    if any(
        int(np.dot(first, np.cross(second, vector))) != 0
        for vector in nonzero[1:]
    ):
        return 3
    return 2


def _reduced_basis(vectors: list[np.ndarray]) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    rank = 0
    for raw in vectors:
        vector = np.asarray(raw, dtype=np.int64)
        if vector.shape != (3,) or not np.any(vector):
            continue
        candidate = _integer_rank([*basis, vector])
        if candidate > rank:
            basis.append(vector.copy())
            rank = candidate
            if rank == 3:
                break
    return basis


class _TranslationDSU:
    """Disjoint-set forest with exact quotient-graph translation potentials."""

    def __init__(self, site_count: int) -> None:
        self.parent = np.arange(site_count, dtype=int)
        self.size = np.ones(site_count, dtype=int)
        self.offset = np.zeros((site_count, 3), dtype=np.int64)
        self.basis: list[list[np.ndarray]] = [[] for _ in range(site_count)]
        self.pending: list[list[int]] = [[index] for index in range(site_count)]

    def find(self, node: int) -> tuple[int, np.ndarray]:
        parent = int(self.parent[node])
        if parent == node:
            return node, np.zeros(3, dtype=np.int64)
        root, parent_offset = self.find(parent)
        total = self.offset[node] + parent_offset
        self.parent[node] = root
        self.offset[node] = total
        return root, total.copy()

    def add_edge(self, left: int, right: int, translation: np.ndarray) -> int:
        left_root, left_offset = self.find(left)
        right_root, right_offset = self.find(right)
        delta = np.asarray(translation, dtype=np.int64)
        if left_root == right_root:
            residual = left_offset + delta - right_offset
            self.basis[left_root] = _reduced_basis(
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
        self.basis[root] = _reduced_basis(
            [*self.basis[root], *self.basis[child]]
        )
        self.pending[root].extend(self.pending[child])
        self.pending[child] = []
        return root


def _inverted_cdf(values: np.ndarray, probability: float) -> float:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("NEXT379 site bottleneck population differs")
    ordered = np.sort(values, kind="stable")
    index = max(0, math.ceil(float(probability) * len(ordered)) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def skeletal_net_bottleneck(
    *, n_sites: int, endpoints: object, solid_angles: object
) -> PSNBFeatureResult:
    """Return the q10 site threshold for acquiring translation rank three."""

    try:
        if type(n_sites) is not int or n_sites < 1:
            raise ValueError("NEXT379 site population differs")
        raw_endpoints = np.asarray(endpoints)
        angles = np.asarray(solid_angles, dtype=float)
        if (
            raw_endpoints.ndim != 2
            or raw_endpoints.shape[1:] != (5,)
            or len(raw_endpoints) == 0
            or angles.shape != (len(raw_endpoints),)
            or np.any(~np.isfinite(angles))
            or np.any(angles <= 0.0)
        ):
            raise ValueError("NEXT379 directed facet population differs")
        endpoint_float = np.asarray(raw_endpoints, dtype=float)
        if (
            np.any(~np.isfinite(endpoint_float))
            or np.any(endpoint_float != np.floor(endpoint_float))
        ):
            raise ValueError("NEXT379 periodic contact key differs")
        endpoint_int = endpoint_float.astype(int)
        if (
            np.any(endpoint_int[:, :2] < 0)
            or np.any(endpoint_int[:, :2] >= n_sites)
        ):
            raise ValueError("NEXT379 periodic contact index differs")
        keys = [tuple(int(value) for value in row) for row in endpoint_int.tolist()]
        if len(keys) != len(set(keys)):
            raise ValueError("NEXT379 duplicate directed facet differs")
        for key in keys:
            if key[0] == key[1] and key[2:] == (0, 0, 0):
                raise ValueError("NEXT379 zero-image self contact differs")
        angle_by_key = dict(zip(keys, angles.tolist(), strict=True))
        shared: dict[tuple[int, int, int, int, int], float] = {}
        reverse_errors: list[float] = []
        for key in sorted(angle_by_key):
            reverse = _reverse(key)
            if reverse not in angle_by_key:
                raise ValueError("NEXT379 reverse incidence is incomplete")
            error = abs(float(angle_by_key[key]) - float(angle_by_key[reverse]))
            reverse_errors.append(error)
            if error > REVERSE_ANGLE_TOLERANCE:
                raise ValueError("NEXT379 reverse solid angle differs")
            shared[key] = 0.5 * (
                float(angle_by_key[key]) + float(angle_by_key[reverse])
            )

        local_maximum = np.zeros(n_sites, dtype=float)
        for key, weight in shared.items():
            local_maximum[key[0]] = max(local_maximum[key[0]], float(weight))
        if np.any(local_maximum <= 0.0) or not np.isfinite(local_maximum).all():
            raise ValueError("NEXT379 site has no periodic Voronoi contact")

        edges: list[tuple[float, tuple[int, int, int, int, int]]] = []
        for key in sorted(shared):
            reverse = _reverse(key)
            if key > reverse:
                continue
            left, right, *_ = key
            weight = float(shared[key])
            salience = _quantized(
                min(weight / local_maximum[left], weight / local_maximum[right])
            )
            if (
                salience < -NUMERICAL_TOLERANCE
                or salience > 1.0 + NUMERICAL_TOLERANCE
            ):
                raise ValueError("NEXT379 mutual salience is outside [0,1]")
            edges.append((float(np.clip(salience, 0.0, 1.0)), key))
        if not edges:
            raise ValueError("NEXT379 undirected facet population is empty")

        edges.sort(key=lambda item: (-item[0], item[1]))
        bottlenecks = np.zeros(n_sites, dtype=float)
        dsu = _TranslationDSU(n_sites)
        start = 0
        while start < len(edges):
            threshold = edges[start][0]
            stop = start + 1
            while stop < len(edges) and edges[stop][0] == threshold:
                stop += 1
            for _salience, key in edges[start:stop]:
                left, right, x, y, z = key
                dsu.add_edge(left, right, np.asarray((x, y, z), dtype=np.int64))
            roots = {dsu.find(index)[0] for index in range(n_sites)}
            for root in roots:
                root, _ = dsu.find(root)
                if len(dsu.basis[root]) == 3 and dsu.pending[root]:
                    bottlenecks[np.asarray(dsu.pending[root], dtype=int)] = threshold
                    dsu.pending[root] = []
            start = stop

        if (
            not np.isfinite(bottlenecks).all()
            or np.any(bottlenecks < 0.0)
            or np.any(bottlenecks > 1.0)
        ):
            raise RuntimeError("NEXT379 site bottlenecks differ")
        value = _quantized(_inverted_cdf(bottlenecks, 0.10))
        return PSNBFeatureResult(
            True,
            None,
            n_sites,
            len(keys),
            len(edges),
            int(np.sum(bottlenecks > 0.0)),
            max(reverse_errors, default=0.0),
            tuple(float(item) for item in bottlenecks.tolist()),
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def _strict_reduced_atoms(atoms: Atoms) -> Atoms:
    if (
        not isinstance(atoms, Atoms)
        or len(atoms) < 1
        or np.asarray(atoms.pbc, dtype=bool).shape != (3,)
        or not np.all(atoms.pbc)
        or atoms.calc is not None
        or bool(atoms.info)
        or set(atoms.arrays) != {"numbers", "positions"}
    ):
        raise ValueError("NEXT379 features require exact periodic geometry-only Atoms")
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    if (
        cell.shape != (3, 3)
        or positions.shape != (len(atoms), 3)
        or not np.isfinite(cell).all()
        or not np.isfinite(positions).all()
        or abs(float(np.linalg.det(cell))) <= 1.0e-12
    ):
        raise ValueError("NEXT379 features require exact periodic geometry-only Atoms")
    try:
        return n267._validated_reduced_atoms(atoms)
    except Exception as exc:
        raise ValueError(
            "NEXT379 features require exact periodic geometry-only Atoms"
        ) from exc


def compute_psnb_features(atoms: Atoms) -> PSNBFeatureResult:
    """Compute the frozen ordinary-Voronoi skeletal-net bottleneck."""

    try:
        work = _strict_reduced_atoms(atoms)
        structure = AseAtomsAdaptor.get_structure(work)
        finder = VoronoiNN(weight="solid_angle", tol=0, cutoff=13)
        angle_by_key: dict[tuple[int, int, int, int, int], float] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for center in range(len(structure)):
                for item in finder.get_nn_info(structure, center):
                    try:
                        neighbor = int(item["site_index"])
                        image_float = np.asarray(item["image"], dtype=float)
                        image = np.rint(image_float).astype(int)
                        angle = float(item["poly_info"]["solid_angle"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (
                        neighbor < 0
                        or neighbor >= len(structure)
                        or image_float.shape != (3,)
                        or not np.isfinite(image_float).all()
                        or not np.allclose(
                            image_float, image, rtol=0.0, atol=1.0e-10
                        )
                        or not math.isfinite(angle)
                        or angle <= 0.0
                    ):
                        continue
                    key = (center, neighbor, *(int(value) for value in image))
                    previous = angle_by_key.get(key)
                    if previous is not None and not math.isclose(
                        previous,
                        angle,
                        rel_tol=0.0,
                        abs_tol=REVERSE_ANGLE_TOLERANCE,
                    ):
                        raise ValueError("NEXT379 duplicate Voronoi facet differs")
                    angle_by_key[key] = (
                        angle if previous is None else 0.5 * (previous + angle)
                    )
        return skeletal_net_bottleneck(
            n_sites=len(structure),
            endpoints=np.asarray(sorted(angle_by_key), dtype=int),
            solid_angles=np.asarray(
                [angle_by_key[key] for key in sorted(angle_by_key)], dtype=float
            ),
        )
    except Exception as exc:
        return _failure(exc)


def compute_psnb_row(atoms: Atoms) -> dict[str, object]:
    result = compute_psnb_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "psnb_supported": bool(result.supported),
        "psnb_failure": result.failure_reason,
        "psnb_site_count": int(result.site_count),
        "psnb_directed_face_count": int(result.directed_face_count),
        "psnb_undirected_edge_count": int(result.undirected_edge_count),
        "psnb_rank3_site_count": int(result.rank3_site_count),
        "psnb_maximum_reverse_angle_error": result.maximum_reverse_angle_error,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "NUMERICAL_TOLERANCE",
    "OUTPUT_GRID",
    "PSNBFeatureResult",
    "PROTOCOL",
    "REVERSE_ANGLE_TOLERANCE",
    "compute_psnb_features",
    "compute_psnb_row",
    "skeletal_net_bottleneck",
]
