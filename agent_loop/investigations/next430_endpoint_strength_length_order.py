#!/usr/bin/env python3
"""Endpoint characteristic bond strength versus raw local bond-length order."""

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


PROTOCOL = "2026-08-13-next430-endpoint-strength-length-order-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next430-next434-endpoint-strength-length-order.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("ecslo_endpoint_strength_length_order_protection",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
CHARGE_TOLERANCE = 1.0e-8
INFORMATIVE_TOLERANCE = 1.0e-15
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class ECSLOResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    pair_count: int
    informative_weight: float
    violation_weight: float
    edge_strengths: tuple[float, ...]
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> ECSLOResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return ECSLOResult(False, reason, 0, 0, 0, math.nan, math.nan, (), None, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def endpoint_strength_length_order_protection(
    *,
    charges: Sequence[float] | object,
    endpoints: Sequence[Sequence[int]] | object,
    distances: Sequence[float] | object,
) -> ECSLOResult:
    """Test local length order from symmetric endpoint charge/CN strengths."""

    try:
        charge = np.asarray(charges, dtype=float)
        raw_pair = np.asarray(endpoints)
        distance = np.asarray(distances, dtype=float)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("ECSLO charges differ")
        magnitude = float(np.abs(charge).sum())
        if (
            magnitude <= 0.0
            or abs(float(charge.sum())) > CHARGE_TOLERANCE * max(1.0, magnitude)
            or np.any(charge == 0.0)
            or not np.any(charge > 0.0)
            or not np.any(charge < 0.0)
        ):
            raise ValueError("ECSLO formal charges must be neutral and nonzero")
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
            or not np.isfinite(raw_pair.astype(float)).all()
            or not np.equal(raw_pair.astype(float), np.rint(raw_pair.astype(float))).all()
        ):
            raise ValueError("ECSLO endpoint population differs")
        pair = raw_pair.astype(int)
        if (
            np.any(pair < 0)
            or np.any(pair >= len(charge))
            or np.any(pair[:, 0] == pair[:, 1])
            or not np.all(charge[pair[:, 0]] > 0.0)
            or not np.all(charge[pair[:, 1]] < 0.0)
        ):
            raise ValueError("ECSLO cation-anion edge orientation differs")
        if (
            distance.shape != (len(pair),)
            or not np.isfinite(distance).all()
            or np.any(distance <= 0.0)
        ):
            raise ValueError("ECSLO edge lengths differ")
        degree = np.bincount(pair.ravel(), minlength=len(charge)).astype(float)
        if np.any(degree <= 0.0):
            raise ValueError("ECSLO periodic graph has an isolated charged site")
        endpoint_strength = np.abs(charge) / degree
        strengths = np.sqrt(
            endpoint_strength[pair[:, 0]] * endpoint_strength[pair[:, 1]]
        )
        if not np.isfinite(strengths).all() or np.any(strengths <= 0.0):
            raise RuntimeError("ECSLO endpoint characteristic strengths differ")

        products: list[float] = []
        for site in range(len(charge)):
            selected = np.flatnonzero((pair[:, 0] == site) | (pair[:, 1] == site))
            for left, right in itertools.combinations(selected.tolist(), 2):
                strength_contrast = float(
                    (strengths[left] - strengths[right])
                    / (strengths[left] + strengths[right])
                )
                length_contrast = float(
                    (distance[left] - distance[right])
                    / (distance[left] + distance[right])
                )
                products.append(strength_contrast * length_contrast)
        if not products:
            raise ValueError("ECSLO graph has no local edge pair")
        informative = math.fsum(abs(value) for value in products)
        violation = math.fsum(max(value, 0.0) for value in products)
        protection = (
            1.0 - violation / informative
            if informative > INFORMATIVE_TOLERANCE
            else 0.5
        )
        protection = _quantize(float(np.clip(protection, 0.0, 1.0)))
        if (
            not math.isfinite(informative)
            or not math.isfinite(violation)
            or violation < 0.0
            or violation > informative + 1.0e-12
            or not math.isfinite(protection)
            or protection < 0.0
            or protection > 1.0
        ):
            raise RuntimeError("ECSLO length-order statistic differs")
        return ECSLOResult(
            True, None, len(charge), len(pair), len(products),
            float(informative), float(violation),
            tuple(float(value) for value in strengths), None,
            {FEATURE_NAMES[0]: protection},
        )
    except Exception as exc:
        return _failure(exc)


def compute_ecslo_features(atoms: Atoms) -> ECSLOResult:
    """Compute ECSLO from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "NEXT430 valence assignment failed")
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT430 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            raise ValueError(geometry.failure_reason or "ECSLO periodic graph failed")
        edges = tuple(geometry.edges)
        result = endpoint_strength_length_order_protection(
            charges=charge,
            endpoints=tuple((int(edge.cation), int(edge.anion)) for edge in edges),
            distances=tuple(float(edge.distance) for edge in edges),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason="NEXT430 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_ecslo_row(atoms: Atoms) -> dict[str, object]:
    result = compute_ecslo_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "ecslo_supported": bool(result.supported),
        "ecslo_failure": result.failure_reason,
        "ecslo_site_count": int(result.site_count),
        "ecslo_edge_count": int(result.edge_count),
        "ecslo_pair_count": int(result.pair_count),
        "ecslo_informative_weight": float(result.informative_weight),
        "ecslo_violation_weight": float(result.violation_weight),
        "ecslo_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS", "CHARGE_TOLERANCE", "DESIGN_SHA256", "ECSLOResult",
    "FEATURE_DIRECTIONS", "FEATURE_NAMES", "INFORMATIVE_TOLERANCE",
    "OUTPUT_GRID", "PROTOCOL", "compute_ecslo_features", "compute_ecslo_row",
    "endpoint_strength_length_order_protection",
]
