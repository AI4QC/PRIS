#!/usr/bin/env python3
"""Periodic skeletal path collision from one raw x0 geometry only."""

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

import src.next379_periodic_skeletal_net_bottleneck as n379


PROTOCOL = "2026-08-13-next383-periodic-skeletal-path-collision-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next383-next386-periodic-skeletal-path-collision.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("pspc_skeletal_nb3_collision_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
WALK_DEPTH = 3
BOUNDARY_FLAGS = dict(n379.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PSPCFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    directed_face_count: int
    undirected_edge_count: int
    rank3_site_count: int
    skeleton_edge_count: int
    skeleton_threshold: float
    total_walk_count: int
    total_endpoint_count: int
    maximum_reverse_angle_error: float
    site_collisions: tuple[float, ...]
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PSPCFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PSPCFeatureResult(
        False, reason, 0, 0, 0, 0, 0, math.nan, 0, 0, math.nan, (), {}
    )


def _validated_edges(
    *, n_sites: int, endpoints: object, solid_angles: object
) -> tuple[
    list[tuple[float, tuple[int, int, int, int, int]]], int, float
]:
    if type(n_sites) is not int or n_sites < 1:
        raise ValueError("NEXT383 site population differs")
    raw = np.asarray(endpoints)
    angles = np.asarray(solid_angles, dtype=float)
    if (
        raw.ndim != 2
        or raw.shape[1:] != (5,)
        or len(raw) == 0
        or angles.shape != (len(raw),)
        or np.any(~np.isfinite(angles))
        or np.any(angles <= 0.0)
    ):
        raise ValueError("NEXT383 directed facet population differs")
    numeric = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError("NEXT383 periodic contact key differs")
    integer = numeric.astype(int)
    if np.any(integer[:, :2] < 0) or np.any(integer[:, :2] >= n_sites):
        raise ValueError("NEXT383 periodic contact index differs")
    keys = [tuple(int(x) for x in row) for row in integer.tolist()]
    if len(keys) != len(set(keys)):
        raise ValueError("NEXT383 duplicate directed facet differs")
    angle_by_key = dict(zip(keys, angles.tolist(), strict=True))
    shared: dict[tuple[int, int, int, int, int], float] = {}
    errors: list[float] = []
    for key in sorted(angle_by_key):
        if key[0] == key[1] and key[2:] == (0, 0, 0):
            raise ValueError("NEXT383 zero-image self contact differs")
        reverse = n379._reverse(key)
        if reverse not in angle_by_key:
            raise ValueError("NEXT383 reverse incidence is incomplete")
        error = abs(float(angle_by_key[key]) - float(angle_by_key[reverse]))
        errors.append(error)
        if error > n379.REVERSE_ANGLE_TOLERANCE:
            raise ValueError("NEXT383 reverse solid angle differs")
        shared[key] = 0.5 * (angle_by_key[key] + angle_by_key[reverse])
    local_maximum = np.zeros(n_sites, dtype=float)
    for key, weight in shared.items():
        local_maximum[key[0]] = max(local_maximum[key[0]], float(weight))
    if np.any(local_maximum <= 0.0) or not np.isfinite(local_maximum).all():
        raise ValueError("NEXT383 site has no periodic Voronoi contact")
    edges = []
    for key in sorted(shared):
        reverse = n379._reverse(key)
        if key > reverse:
            continue
        left, right, *_ = key
        salience = n379._quantized(
            min(shared[key] / local_maximum[left], shared[key] / local_maximum[right])
        )
        if salience < -n379.NUMERICAL_TOLERANCE or salience > 1 + n379.NUMERICAL_TOLERANCE:
            raise ValueError("NEXT383 mutual salience is outside [0,1]")
        edges.append((float(np.clip(salience, 0.0, 1.0)), key))
    if not edges:
        raise ValueError("NEXT383 undirected facet population is empty")
    return sorted(edges, key=lambda item: (-item[0], item[1])), len(keys), max(errors)


def skeletal_path_collision(
    *, n_sites: int, endpoints: object, solid_angles: object
) -> PSPCFeatureResult:
    """Return q10 three-step non-backtracking endpoint collision."""

    try:
        edges, directed_count, maximum_error = _validated_edges(
            n_sites=n_sites, endpoints=endpoints, solid_angles=solid_angles
        )
        bottlenecks = np.zeros(n_sites, dtype=float)
        dsu = n379._TranslationDSU(n_sites)
        start = 0
        while start < len(edges):
            threshold = edges[start][0]
            stop = start + 1
            while stop < len(edges) and edges[stop][0] == threshold:
                stop += 1
            for _, key in edges[start:stop]:
                dsu.add_edge(key[0], key[1], np.asarray(key[2:], dtype=np.int64))
            for root in {dsu.find(index)[0] for index in range(n_sites)}:
                root, _ = dsu.find(root)
                if len(dsu.basis[root]) == 3 and dsu.pending[root]:
                    bottlenecks[np.asarray(dsu.pending[root], dtype=int)] = threshold
                    dsu.pending[root] = []
            start = stop
        threshold = n379._quantized(n379._inverted_cdf(bottlenecks, 0.10))
        skeleton = [key for salience, key in edges if salience >= threshold]
        adjacency: list[list[tuple[int, int, int, int, int]]] = [
            [] for _ in range(n_sites)
        ]
        for key in skeleton:
            adjacency[key[0]].append(key)
            adjacency[key[1]].append(n379._reverse(key))
        collisions: list[float] = []
        total_walks = 0
        total_endpoints = 0
        for origin in range(n_sites):
            states = {(origin, 0, 0, 0, None): 1}
            for _ in range(WALK_DEPTH):
                following: dict[tuple[object, ...], int] = {}
                for (site, x, y, z, prior), multiplicity in states.items():
                    for arc in adjacency[int(site)]:
                        if prior is not None and arc == n379._reverse(prior):
                            continue
                        state = (
                            arc[1], x + arc[2], y + arc[3], z + arc[4], arc
                        )
                        following[state] = following.get(state, 0) + multiplicity
                states = following
            endpoint_multiplicity: dict[tuple[int, int, int, int], int] = {}
            for (site, x, y, z, _), multiplicity in states.items():
                endpoint = (int(site), int(x), int(y), int(z))
                endpoint_multiplicity[endpoint] = (
                    endpoint_multiplicity.get(endpoint, 0) + multiplicity
                )
            walks = int(sum(endpoint_multiplicity.values()))
            distinct = int(len(endpoint_multiplicity))
            total_walks += walks
            total_endpoints += distinct
            collisions.append(0.0 if walks == 0 else 1.0 - distinct / walks)
        values = np.asarray(collisions, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise RuntimeError("NEXT383 site collision population differs")
        value = n379._quantized(n379._inverted_cdf(values, 0.10))
        return PSPCFeatureResult(
            True,
            None,
            n_sites,
            directed_count,
            len(edges),
            int(np.sum(bottlenecks > 0.0)),
            len(skeleton),
            threshold,
            total_walks,
            total_endpoints,
            maximum_error,
            tuple(float(x) for x in values.tolist()),
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pspc_features(atoms: Atoms) -> PSPCFeatureResult:
    try:
        work = n379._strict_reduced_atoms(atoms)
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
                        or not np.allclose(image_float, image, rtol=0, atol=1e-10)
                        or not math.isfinite(angle)
                        or angle <= 0
                    ):
                        continue
                    key = (center, neighbor, *(int(x) for x in image))
                    previous = angle_by_key.get(key)
                    if previous is not None and not math.isclose(
                        previous, angle, rel_tol=0, abs_tol=n379.REVERSE_ANGLE_TOLERANCE
                    ):
                        raise ValueError("NEXT383 duplicate Voronoi facet differs")
                    angle_by_key[key] = angle if previous is None else 0.5 * (previous + angle)
        return skeletal_path_collision(
            n_sites=len(structure),
            endpoints=np.asarray(sorted(angle_by_key), dtype=int),
            solid_angles=np.asarray([angle_by_key[key] for key in sorted(angle_by_key)]),
        )
    except Exception as exc:
        reason = str(exc)
        if "NEXT379" in reason:
            return _failure("NEXT383 features require exact periodic geometry-only Atoms")
        return _failure(exc)


def compute_pspc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pspc_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "pspc_supported": result.supported,
        "pspc_failure": result.failure_reason,
        "pspc_site_count": result.site_count,
        "pspc_directed_face_count": result.directed_face_count,
        "pspc_undirected_edge_count": result.undirected_edge_count,
        "pspc_rank3_site_count": result.rank3_site_count,
        "pspc_skeleton_edge_count": result.skeleton_edge_count,
        "pspc_skeleton_threshold": result.skeleton_threshold,
        "pspc_total_walk_count": result.total_walk_count,
        "pspc_total_endpoint_count": result.total_endpoint_count,
        "pspc_maximum_reverse_angle_error": result.maximum_reverse_angle_error,
    }


__all__ = [
    "BOUNDARY_FLAGS", "DESIGN_SHA256", "FEATURE_DIRECTIONS", "FEATURE_NAMES",
    "PROTOCOL", "PSPCFeatureResult", "WALK_DEPTH", "compute_pspc_features",
    "compute_pspc_row", "skeletal_path_collision",
]
