#!/usr/bin/env python3
"""Worst-direction covering of raw periodic radical-cell facet normals."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ase import Atoms
import numpy as np

import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next327_radical_facet_positive_enclosure as n327
import src.next339_periodic_geometric_homogenized_transmissivity as n339


PROTOCOL = "2026-08-13-next359-radical-facet-normal-covering-v1"
DESIGN_SHA256 = "b7a278ccaacc81800938edb21a10096aaa70aac3af3aeb3c336f53e940e52dac"
FEATURE_NAMES = ("rfnc_directional_covering_floor_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
ORIGIN_INTERIOR_TOLERANCE = 1.0e-10
RECIPROCAL_AREA_RELATIVE_TOLERANCE = n339.RECIPROCAL_AREA_RELATIVE_TOLERANCE
VOLUME_TILING_RELATIVE_TOLERANCE = n267.VOLUME_TILING_RELATIVE_TOLERANCE
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class RFNCFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    minimum_unique_facet_count: int
    maximum_unique_facet_count: int
    minimum_facet_area: float
    maximum_reciprocal_area_relative_error: float
    volume_tiling_relative_error: float
    minimum_site_covering: float
    maximum_site_covering: float
    features: dict[str, float]


def _failure(exc: Exception | str) -> RFNCFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return RFNCFeatureResult(
        False, reason, 0, 0, 0, 0, math.nan, math.nan, math.nan, math.nan, math.nan, {}
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def normal_covering_radius(vectors: object) -> float:
    """Return the origin-centred inradius of the unit-direction convex hull."""

    directions = n327.unique_facet_directions(vectors)
    if len(directions) < 4 or np.linalg.matrix_rank(
        directions[1:] - directions[0], tol=ORIGIN_INTERIOR_TOLERANCE
    ) < 3:
        raise ValueError("NEXT359 facet-normal hull is not three-dimensional")
    try:
        hull = n267.ConvexHull(directions, qhull_options="Qx")
    except Exception as exc:
        raise ValueError("NEXT359 facet-normal hull is not three-dimensional") from exc
    equations = np.asarray(hull.equations, dtype=float)
    if equations.ndim != 2 or equations.shape[1:] != (4,) or len(equations) < 4:
        raise ValueError("NEXT359 facet-normal hull differs")
    normals = equations[:, :3]
    offsets = equations[:, 3]
    norms = np.linalg.norm(normals, axis=1)
    if (
        not np.isfinite(equations).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= ORIGIN_INTERIOR_TOLERANCE)
    ):
        raise ValueError("NEXT359 facet-normal hull differs")
    distances = -offsets / norms
    if np.any(distances <= ORIGIN_INTERIOR_TOLERANCE):
        raise ValueError("NEXT359 facet normals do not strictly enclose the origin")
    value = float(np.min(distances))
    if not math.isfinite(value) or value <= 0.0 or value > 1.0 + 1.0e-10:
        raise ValueError("NEXT359 normal covering radius differs")
    return float(np.clip(value, 0.0, 1.0))


def site_normal_coverings(
    *, n_sites: int, endpoints: object, displacements: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return site-aligned covering radii and unique-facet counts."""

    if type(n_sites) is not int or n_sites < 1:
        raise ValueError("NEXT359 site count differs")
    raw_pair = np.asarray(endpoints)
    vector = np.asarray(displacements, dtype=float)
    if raw_pair.ndim != 2 or raw_pair.shape[1:] != (2,) or len(raw_pair) < 1:
        raise ValueError("NEXT359 edge population differs")
    try:
        numeric_pair = np.asarray(raw_pair, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NEXT359 edge population differs") from exc
    if (
        not np.isfinite(numeric_pair).all()
        or not np.equal(numeric_pair, np.rint(numeric_pair)).all()
    ):
        raise ValueError("NEXT359 edge population differs")
    pair = numeric_pair.astype(int)
    if (
        vector.shape != (len(pair), 3)
        or not np.isfinite(vector).all()
        or np.any(pair < 0)
        or np.any(pair >= n_sites)
    ):
        raise ValueError("NEXT359 edge geometry differs")
    norms = np.linalg.norm(vector, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= n327.DIRECTION_NORM_TOLERANCE):
        raise ValueError("NEXT359 edge geometry differs")
    direction = vector / norms[:, None]
    by_site: list[list[np.ndarray]] = [[] for _ in range(n_sites)]
    for (left, right), normal in zip(pair, direction, strict=True):
        by_site[int(left)].append(normal)
        by_site[int(right)].append(-normal)
    coverings = np.empty(n_sites, dtype=float)
    counts = np.empty(n_sites, dtype=int)
    for site, population in enumerate(by_site):
        unique = n327.unique_facet_directions(np.asarray(population, dtype=float))
        counts[site] = len(unique)
        coverings[site] = normal_covering_radius(unique)
    return coverings, counts


def compute_rfnc_features(atoms: Atoms) -> RFNCFeatureResult:
    """Compute RFNC from elements and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        graph = n339.periodic_power_facet_graph(work, radii=radii)
        covering, counts = site_normal_coverings(
            n_sites=graph.site_count,
            endpoints=graph.endpoints,
            displacements=graph.displacements,
        )
        q10 = float(np.quantile(covering, 0.10, method="inverted_cdf"))
        value = _quantize(q10)
        if not 0.0 <= value <= 1.0:
            raise RuntimeError("NEXT359 feature domain differs")
        return RFNCFeatureResult(
            True,
            None,
            graph.site_count,
            len(graph.endpoints),
            int(counts.min()),
            int(counts.max()),
            graph.minimum_facet_area,
            graph.maximum_reciprocal_area_relative_error,
            graph.volume_tiling_relative_error,
            float(covering.min()),
            float(covering.max()),
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_rfnc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_rfnc_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "rfnc_supported": bool(result.supported),
        "rfnc_failure": result.failure_reason,
        "rfnc_site_count": int(result.site_count),
        "rfnc_edge_count": int(result.edge_count),
        "rfnc_minimum_unique_facet_count": int(result.minimum_unique_facet_count),
        "rfnc_maximum_unique_facet_count": int(result.maximum_unique_facet_count),
        "rfnc_minimum_facet_area": result.minimum_facet_area,
        "rfnc_maximum_reciprocal_area_relative_error": (
            result.maximum_reciprocal_area_relative_error
        ),
        "rfnc_volume_tiling_relative_error": result.volume_tiling_relative_error,
        "rfnc_minimum_site_covering": result.minimum_site_covering,
        "rfnc_maximum_site_covering": result.maximum_site_covering,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "ORIGIN_INTERIOR_TOLERANCE",
    "PROTOCOL",
    "RECIPROCAL_AREA_RELATIVE_TOLERANCE",
    "RFNCFeatureResult",
    "VOLUME_TILING_RELATIVE_TOLERANCE",
    "compute_rfnc_features",
    "compute_rfnc_row",
    "normal_covering_radius",
    "site_normal_coverings",
]
