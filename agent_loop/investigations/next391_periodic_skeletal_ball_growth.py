#!/usr/bin/env python3
"""Periodic skeletal ball growth from one raw x0 geometry only."""

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


PROTOCOL = "2026-08-13-next391-periodic-skeletal-ball-growth-v1"
_DESIGN_PATH = Path(__file__).resolve().parents[1] / "docs/plans/2026-08-13-next391-next394-periodic-skeletal-ball-growth.md"
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("psbg_skeletal_ball4_growth_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
BALL_RADIUS = 4
BOUNDARY_FLAGS = dict(n379.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PSBGFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    directed_face_count: int
    undirected_edge_count: int
    rank3_site_count: int
    skeleton_edge_count: int
    skeleton_threshold: float
    total_ball_vertex_count: int
    maximum_reverse_angle_error: float
    site_ball_counts: tuple[int, ...]
    site_growth: tuple[float, ...]
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PSBGFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PSBGFeatureResult(False, reason, 0, 0, 0, 0, 0, math.nan, 0, math.nan, (), (), {})


def skeletal_ball_growth(*, n_sites: int, endpoints: object, solid_angles: object) -> PSBGFeatureResult:
    try:
        edges, directed_count, maximum_error = n383._validated_edges(n_sites=n_sites, endpoints=endpoints, solid_angles=solid_angles)
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
            for root in {dsu.find(i)[0] for i in range(n_sites)}:
                root, _ = dsu.find(root)
                if len(dsu.basis[root]) == 3 and dsu.pending[root]:
                    bottlenecks[np.asarray(dsu.pending[root], dtype=int)] = threshold
                    dsu.pending[root] = []
            start = stop
        threshold = n379._quantized(n379._inverted_cdf(bottlenecks, 0.10))
        skeleton = [key for salience, key in edges if salience >= threshold]
        adjacency = [[] for _ in range(n_sites)]
        for key in skeleton:
            adjacency[key[0]].append(key)
            adjacency[key[1]].append(n379._reverse(key))
        counts, growth = [], []
        for origin in range(n_sites):
            visited = {(origin, 0, 0, 0)}
            frontier = set(visited)
            for _ in range(BALL_RADIUS):
                following = set()
                for site, x, y, z in frontier:
                    for arc in adjacency[site]:
                        vertex = (arc[1], x + arc[2], y + arc[3], z + arc[4])
                        if vertex not in visited:
                            visited.add(vertex)
                            following.add(vertex)
                frontier = following
            count = len(visited)
            coefficient = count / float(BALL_RADIUS ** 3)
            counts.append(count)
            growth.append(coefficient / (1.0 + coefficient))
        values = np.asarray(growth)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= 1):
            raise RuntimeError("NEXT391 site growth population differs")
        value = n379._quantized(n379._inverted_cdf(values, 0.10))
        return PSBGFeatureResult(
            True, None, n_sites, directed_count, len(edges), int(np.sum(bottlenecks > 0)),
            len(skeleton), threshold, sum(counts), maximum_error, tuple(counts),
            tuple(float(x) for x in values), {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_psbg_features(atoms: Atoms) -> PSBGFeatureResult:
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
                        raise ValueError("NEXT391 duplicate Voronoi facet differs")
                    angle_by_key[key] = angle if previous is None else 0.5 * (previous + angle)
        return skeletal_ball_growth(n_sites=len(structure), endpoints=np.asarray(sorted(angle_by_key), dtype=int), solid_angles=np.asarray([angle_by_key[key] for key in sorted(angle_by_key)]))
    except Exception as exc:
        if "NEXT379" in str(exc):
            return _failure("NEXT391 features require exact periodic geometry-only Atoms")
        return _failure(exc)


def compute_psbg_row(atoms: Atoms) -> dict[str, object]:
    result = compute_psbg_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "psbg_supported": result.supported, "psbg_failure": result.failure_reason,
        "psbg_site_count": result.site_count, "psbg_directed_face_count": result.directed_face_count,
        "psbg_undirected_edge_count": result.undirected_edge_count,
        "psbg_rank3_site_count": result.rank3_site_count,
        "psbg_skeleton_edge_count": result.skeleton_edge_count,
        "psbg_skeleton_threshold": result.skeleton_threshold,
        "psbg_total_ball_vertex_count": result.total_ball_vertex_count,
        "psbg_maximum_reverse_angle_error": result.maximum_reverse_angle_error,
    }


__all__ = ["BALL_RADIUS", "BOUNDARY_FLAGS", "DESIGN_SHA256", "FEATURE_DIRECTIONS", "FEATURE_NAMES", "PROTOCOL", "PSBGFeatureResult", "compute_psbg_features", "compute_psbg_row", "skeletal_ball_growth"]
