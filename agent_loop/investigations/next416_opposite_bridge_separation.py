#!/usr/bin/env python3
"""Opposite-ion bridge separation from composition and one raw x0 geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import math
from pathlib import Path
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next416-opposite-bridge-separation-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next416-next419-opposite-bridge-separation.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("obs_opposite_bridge_separation_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
DISTANCE_TOLERANCE = 1.0e-10
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class OBSResult:
    supported: bool
    failure_reason: str | None
    center_count: int
    edge_count: int
    center_separations: tuple[float, ...]
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> OBSResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return OBSResult(False, reason, 0, 0, (), None, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def opposite_bridge_separation(
    *, center_vectors: Sequence[Sequence[Sequence[float]]] | object
) -> OBSResult:
    """Apply the frozen triangle ratio, center minimum, and lower-tail q10."""

    try:
        if not isinstance(center_vectors, Sequence) or len(center_vectors) < 1:
            raise ValueError("OBS center-vector population differs")
        center_values: list[float] = []
        for raw in center_vectors:
            vectors = np.asarray(raw, dtype=float)
            if (
                vectors.ndim != 2
                or vectors.shape[0] < 2
                or vectors.shape[1] != 3
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("OBS incident-vector population differs")
            lengths = np.linalg.norm(vectors, axis=1)
            if not np.isfinite(lengths).all() or np.any(
                lengths <= DISTANCE_TOLERANCE
            ):
                raise ValueError("OBS incident-vector length differs")
            pairs = []
            for left, right in itertools.combinations(range(len(vectors)), 2):
                denominator = float(lengths[left] + lengths[right])
                value = float(np.linalg.norm(vectors[left] - vectors[right])) / denominator
                if not math.isfinite(value):
                    raise RuntimeError("OBS bridge separation is non-finite")
                pairs.append(min(1.0, max(0.0, value)))
            center_values.append(min(pairs))
        values = np.asarray(center_values, dtype=float)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise RuntimeError("OBS center separation domain differs")
        lower_tail = float(np.quantile(values, 0.10, method="inverted_cdf"))
        feature = _quantize(lower_tail)
        if not math.isfinite(feature) or feature < 0.0 or feature > 1.0:
            raise RuntimeError("OBS feature domain differs")
        return OBSResult(
            True,
            None,
            len(values),
            0,
            tuple(float(value) for value in values),
            None,
            {FEATURE_NAMES[0]: feature},
        )
    except Exception as exc:
        return _failure(exc)


def _center_vectors(structure, geometry) -> tuple[tuple[np.ndarray, ...], ...]:
    incident: list[list[np.ndarray]] = [[] for _ in range(len(structure))]
    for edge in geometry.edges:
        cation = int(edge.cation)
        anion = int(edge.anion)
        fractional = (
            np.asarray(structure[anion].frac_coords, dtype=float)
            + np.asarray(edge.image, dtype=float)
            - np.asarray(structure[cation].frac_coords, dtype=float)
        )
        vector = np.asarray(
            structure.lattice.get_cartesian_coords(fractional), dtype=float
        )
        distance = float(np.linalg.norm(vector))
        if (
            vector.shape != (3,)
            or not np.isfinite(vector).all()
            or distance <= DISTANCE_TOLERANCE
            or abs(distance - float(edge.distance))
            > 1.0e-7 * max(1.0, distance)
        ):
            raise ValueError("OBS periodic edge vector differs")
        incident[cation].append(vector)
        # Rebase the cation image by -edge.image around the reference-cell
        # anion. This is exactly the reverse of the same periodic edge.
        incident[anion].append(-vector)
    centers = tuple(
        tuple(vectors)
        for vectors in incident
        if len(vectors) >= 2
    )
    if not centers:
        raise ValueError("OBS graph has no center with two opposite-sign contacts")
    return centers


def compute_obs_features(atoms: Atoms) -> OBSResult:
    """Compute OBS from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT416 valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        if (
            charges.shape != (len(structure),)
            or not np.isfinite(charges).all()
            or np.any(charges == 0.0)
        ):
            raise ValueError("NEXT416 charge signs differ")
        geometry = n19.build_periodic_edge_geometry(
            structure, charges, graph_mode="voronoi"
        )
        if not geometry.supported:
            raise ValueError(geometry.failure_reason or "OBS periodic graph failed")
        centers = _center_vectors(structure, geometry)
        result = opposite_bridge_separation(center_vectors=centers)
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
                failure_reason="NEXT416 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_obs_row(atoms: Atoms) -> dict[str, object]:
    result = compute_obs_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "obs_supported": bool(result.supported),
        "obs_failure": result.failure_reason,
        "obs_center_count": int(result.center_count),
        "obs_edge_count": int(result.edge_count),
        "obs_min_center_separation": min(
            result.center_separations, default=math.nan
        ),
        "obs_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "DISTANCE_TOLERANCE",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "OBSResult",
    "OUTPUT_GRID",
    "PROTOCOL",
    "compute_obs_features",
    "compute_obs_row",
    "opposite_bridge_separation",
]
