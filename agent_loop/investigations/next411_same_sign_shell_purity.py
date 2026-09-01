#!/usr/bin/env python3
"""Same-sign intrusion purity from composition and one raw x0 geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next411-same-sign-shell-purity-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next411-next414-same-sign-shell-purity.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("sssp_same_sign_shell_purity_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
DISTANCE_TOLERANCE = 1.0e-8
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class SSSPResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    site_purities: tuple[float, ...]
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> SSSPResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return SSSPResult(False, reason, 0, 0, (), None, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def same_sign_shell_purity(
    *,
    opposite_shell_radii: Sequence[float] | np.ndarray,
    nearest_same_sign_distances: Sequence[float] | np.ndarray,
) -> SSSPResult:
    """Apply the frozen clipped distance ratio and lower-tail order statistic."""

    try:
        radii = np.asarray(opposite_shell_radii, dtype=float)
        distances = np.asarray(nearest_same_sign_distances, dtype=float)
        if (
            radii.ndim != 1
            or len(radii) < 1
            or distances.shape != radii.shape
            or not np.isfinite(radii).all()
            or not np.isfinite(distances).all()
            or np.any(radii <= 0.0)
            or np.any(distances <= 0.0)
        ):
            raise ValueError("SSSP distance populations differ")
        purities = np.minimum(1.0, distances / radii)
        if (
            not np.isfinite(purities).all()
            or np.any(purities <= 0.0)
            or np.any(purities > 1.0)
        ):
            raise RuntimeError("SSSP site purity domain differs")
        lower_tail = float(np.quantile(purities, 0.10, method="inverted_cdf"))
        value = _quantize(lower_tail)
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise RuntimeError("SSSP feature domain differs")
        return SSSPResult(
            True,
            None,
            len(radii),
            0,
            tuple(float(item) for item in purities),
            None,
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def _opposite_shell_radii(structure, charges: np.ndarray):
    geometry = n19.build_periodic_edge_geometry(
        structure, charges, graph_mode="voronoi"
    )
    if not geometry.supported:
        raise ValueError(geometry.failure_reason or "SSSP periodic graph failed")
    radii = np.zeros(len(structure), dtype=float)
    for edge in geometry.edges:
        radii[int(edge.cation)] = max(radii[int(edge.cation)], float(edge.distance))
        radii[int(edge.anion)] = max(radii[int(edge.anion)], float(edge.distance))
    if not np.isfinite(radii).all() or np.any(radii <= DISTANCE_TOLERANCE):
        raise ValueError("SSSP site has no incident opposite-sign contact")
    return radii, geometry


def _nearest_same_sign_distances_within_shell(
    structure, charges: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    nearest = np.asarray(radii, dtype=float).copy()
    for index, radius in enumerate(radii):
        search_radius = float(radius) * (1.0 + 1.0e-10) + DISTANCE_TOLERANCE
        sign = 1 if charges[index] > 0.0 else -1
        for neighbor in structure.get_neighbors(structure[index], search_radius):
            neighbor_index = int(neighbor.index)
            if (charges[neighbor_index] > 0.0) != (sign > 0):
                continue
            distance = float(neighbor.nn_distance)
            if distance <= DISTANCE_TOLERANCE or not math.isfinite(distance):
                continue
            nearest[index] = min(nearest[index], distance)
    return nearest


def compute_sssp_features(atoms: Atoms) -> SSSPResult:
    """Compute SSSP from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT411 valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        if (
            charges.shape != (len(structure),)
            or not np.isfinite(charges).all()
            or np.any(charges == 0.0)
        ):
            raise ValueError("NEXT411 charge signs differ")
        radii, geometry = _opposite_shell_radii(structure, charges)
        distances = _nearest_same_sign_distances_within_shell(
            structure, charges, radii
        )
        result = same_sign_shell_purity(
            opposite_shell_radii=radii,
            nearest_same_sign_distances=distances,
        )
        if not result.supported:
            return result
        return replace(
            result,
            edge_count=len(geometry.edges),
            valence_policy=str(assignment.policy),
        )
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason="NEXT411 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_sssp_row(atoms: Atoms) -> dict[str, object]:
    result = compute_sssp_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "sssp_supported": bool(result.supported),
        "sssp_failure": result.failure_reason,
        "sssp_site_count": int(result.site_count),
        "sssp_edge_count": int(result.edge_count),
        "sssp_min_site_purity": min(result.site_purities, default=math.nan),
        "sssp_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "DISTANCE_TOLERANCE",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "OUTPUT_GRID",
    "PROTOCOL",
    "SSSPResult",
    "compute_sssp_features",
    "compute_sssp_row",
    "same_sign_shell_purity",
]

