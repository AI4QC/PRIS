#!/usr/bin/env python3
"""Periodic skeletal vertex bypass from one raw x0 geometry only."""

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
import src.next383_periodic_skeletal_path_collision as n383


PROTOCOL = "2026-08-13-next387-periodic-skeletal-vertex-bypass-v1"
_DESIGN_PATH = Path(__file__).resolve().parents[1] / "docs/plans/2026-08-13-next387-next390-periodic-skeletal-vertex-bypass.md"
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("psvb_skeletal_vertex_bypass4_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
MAXIMUM_BYPASS_LENGTH = 4
BOUNDARY_FLAGS = dict(n379.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PSVBFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    directed_face_count: int
    undirected_edge_count: int
    rank3_site_count: int
    skeleton_edge_count: int
    skeleton_threshold: float
    total_neighbor_pair_count: int
    total_bypassed_pair_count: int
    maximum_reverse_angle_error: float
    site_bypass_fractions: tuple[float, ...]
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PSVBFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PSVBFeatureResult(False, reason, 0, 0, 0, 0, 0, math.nan, 0, 0, math.nan, (), {})


def _bypassed(
    start: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
    deleted: tuple[int, int, int, int],
    adjacency: list[list[tuple[int, int, int, int, int]]],
) -> bool:
    frontier = {start}
    visited = {start}
    for _ in range(MAXIMUM_BYPASS_LENGTH):
        following = set()
        for site, x, y, z in frontier:
            for arc in adjacency[site]:
                neighbor = (arc[1], x + arc[2], y + arc[3], z + arc[4])
                if neighbor == deleted or neighbor in visited:
                    continue
                if neighbor == target:
                    return True
                visited.add(neighbor)
                following.add(neighbor)
        frontier = following
        if not frontier:
            break
    return False


def skeletal_vertex_bypass(*, n_sites: int, endpoints: object, solid_angles: object) -> PSVBFeatureResult:
    try:
        edges, directed_count, maximum_error = n383._validated_edges(
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
        adjacency: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(n_sites)]
        for key in skeleton:
            adjacency[key[0]].append(key)
            adjacency[key[1]].append(n379._reverse(key))
        fractions = []
        total_pairs = 0
        total_bypassed = 0
        for origin in range(n_sites):
            neighbors = sorted({(arc[1], arc[2], arc[3], arc[4]) for arc in adjacency[origin]})
            pair_count = len(neighbors) * (len(neighbors) - 1) // 2
            bypassed = 0
            deleted = (origin, 0, 0, 0)
            for left_index, left in enumerate(neighbors):
                for right in neighbors[left_index + 1:]:
                    bypassed += int(_bypassed(left, right, deleted, adjacency))
            total_pairs += pair_count
            total_bypassed += bypassed
            fractions.append(0.0 if pair_count == 0 else bypassed / pair_count)
        values = np.asarray(fractions, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise RuntimeError("NEXT387 site bypass population differs")
        value = n379._quantized(n379._inverted_cdf(values, 0.10))
        return PSVBFeatureResult(
            True, None, n_sites, directed_count, len(edges), int(np.sum(bottlenecks > 0)),
            len(skeleton), threshold, total_pairs, total_bypassed, maximum_error,
            tuple(float(x) for x in values.tolist()), {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_psvb_features(atoms: Atoms) -> PSVBFeatureResult:
    try:
        work = n379._strict_reduced_atoms(atoms)
        structure = AseAtomsAdaptor.get_structure(work)
        finder = VoronoiNN(weight="solid_angle", tol=0, cutoff=13)
        angle_by_key = {}
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
                    if neighbor < 0 or neighbor >= len(structure) or image_float.shape != (3,) or not np.isfinite(image_float).all() or not np.allclose(image_float, image, rtol=0, atol=1e-10) or not math.isfinite(angle) or angle <= 0:
                        continue
                    key = (center, neighbor, *(int(x) for x in image))
                    previous = angle_by_key.get(key)
                    if previous is not None and not math.isclose(previous, angle, rel_tol=0, abs_tol=n379.REVERSE_ANGLE_TOLERANCE):
                        raise ValueError("NEXT387 duplicate Voronoi facet differs")
                    angle_by_key[key] = angle if previous is None else 0.5 * (previous + angle)
        return skeletal_vertex_bypass(
            n_sites=len(structure), endpoints=np.asarray(sorted(angle_by_key), dtype=int),
            solid_angles=np.asarray([angle_by_key[key] for key in sorted(angle_by_key)]),
        )
    except Exception as exc:
        if "NEXT379" in str(exc):
            return _failure("NEXT387 features require exact periodic geometry-only Atoms")
        return _failure(exc)


def compute_psvb_row(atoms: Atoms) -> dict[str, object]:
    result = compute_psvb_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "psvb_supported": result.supported, "psvb_failure": result.failure_reason,
        "psvb_site_count": result.site_count, "psvb_directed_face_count": result.directed_face_count,
        "psvb_undirected_edge_count": result.undirected_edge_count,
        "psvb_rank3_site_count": result.rank3_site_count,
        "psvb_skeleton_edge_count": result.skeleton_edge_count,
        "psvb_skeleton_threshold": result.skeleton_threshold,
        "psvb_total_neighbor_pair_count": result.total_neighbor_pair_count,
        "psvb_total_bypassed_pair_count": result.total_bypassed_pair_count,
        "psvb_maximum_reverse_angle_error": result.maximum_reverse_angle_error,
    }


__all__ = [
    "BOUNDARY_FLAGS", "DESIGN_SHA256", "FEATURE_DIRECTIONS", "FEATURE_NAMES",
    "MAXIMUM_BYPASS_LENGTH", "PROTOCOL", "PSVBFeatureResult",
    "compute_psvb_features", "compute_psvb_row", "skeletal_vertex_bypass",
]
