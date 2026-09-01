#!/usr/bin/env python3
"""Strict-positive exact formal-valence transport margin on a periodic graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.optimize import linprog

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next435-positive-valence-transport-margin-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next435-next439-positive-valence-transport-margin.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("pvtm_positive_transport_margin",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
CHARGE_TOLERANCE = 1.0e-8
LP_RESIDUAL_TOLERANCE = 1.0e-8
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PVTMResult:
    supported: bool
    failure_reason: str | None
    feasible: bool
    site_count: int
    edge_count: int
    raw_margin: float
    maximum_equality_residual: float
    minimum_lower_bound_residual: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PVTMResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PVTMResult(False, reason, False, 0, 0, math.nan, math.nan, math.nan, None, {})


def _zero(*, site_count: int, edge_count: int, feasible: bool) -> PVTMResult:
    return PVTMResult(
        True, None, feasible, site_count, edge_count, 0.0,
        0.0 if feasible else math.nan, 0.0 if feasible else math.nan,
        None, {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def positive_valence_transport_margin(
    *, charges: Sequence[float] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> PVTMResult:
    """Maximize an all-edge characteristic-strength floor under Cx=|q|."""

    try:
        charge = np.asarray(charges, dtype=float)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("PVTM charges differ")
        magnitude = float(np.abs(charge).sum())
        if (
            magnitude <= 0.0
            or abs(float(charge.sum())) > CHARGE_TOLERANCE * max(1.0, magnitude)
            or np.any(charge == 0.0)
            or not np.any(charge > 0.0)
            or not np.any(charge < 0.0)
        ):
            raise ValueError("PVTM formal charges must be neutral and nonzero")
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
            or not np.isfinite(raw_pair.astype(float)).all()
            or not np.equal(raw_pair.astype(float), np.rint(raw_pair.astype(float))).all()
        ):
            raise ValueError("PVTM endpoint population differs")
        pair = raw_pair.astype(int)
        if (
            np.any(pair < 0)
            or np.any(pair >= len(charge))
            or np.any(pair[:, 0] == pair[:, 1])
            or not np.all(charge[pair[:, 0]] > 0.0)
            or not np.all(charge[pair[:, 1]] < 0.0)
        ):
            raise ValueError("PVTM cation-anion edge orientation differs")
        degree = np.bincount(pair.ravel(), minlength=len(charge)).astype(float)
        if np.any(degree <= 0.0):
            return _zero(site_count=len(charge), edge_count=len(pair), feasible=False)
        target = np.abs(charge)
        characteristic = target / degree
        reference = np.sqrt(
            characteristic[pair[:, 0]] * characteristic[pair[:, 1]]
        )
        if not np.isfinite(reference).all() or np.any(reference <= 0.0):
            raise RuntimeError("PVTM reference edge strengths differ")
        incidence = np.zeros((len(charge), len(pair)), dtype=float)
        columns = np.arange(len(pair))
        incidence[pair[:, 0], columns] = 1.0
        incidence[pair[:, 1], columns] = 1.0
        objective = np.zeros(len(pair) + 1, dtype=float)
        objective[-1] = -1.0
        equality = np.column_stack((incidence, np.zeros(len(charge))))
        lower_bound = np.column_stack((-np.eye(len(pair)), reference))
        result = linprog(
            objective,
            A_ub=lower_bound,
            b_ub=np.zeros(len(pair)),
            A_eq=equality,
            b_eq=target,
            bounds=[(0.0, None)] * (len(pair) + 1),
            method="highs",
        )
        if result.status == 2:
            return _zero(site_count=len(charge), edge_count=len(pair), feasible=False)
        if result.status != 0 or result.x is None:
            raise RuntimeError(
                f"PVTM linear program did not return optimal/infeasible: {result.status}"
            )
        flow = np.asarray(result.x[:-1], dtype=float)
        margin = float(result.x[-1])
        equality_residual = float(np.max(np.abs(incidence @ flow - target)))
        lower_residual = float(np.min(flow - margin * reference))
        scale = max(1.0, float(np.max(target)), float(np.max(flow)), abs(margin))
        if (
            not np.isfinite(flow).all()
            or not math.isfinite(margin)
            or margin < -LP_RESIDUAL_TOLERANCE
            or equality_residual > LP_RESIDUAL_TOLERANCE * scale
            or lower_residual < -LP_RESIDUAL_TOLERANCE * scale
        ):
            raise RuntimeError("PVTM independently verified LP residual differs")
        margin = max(0.0, margin)
        bounded = _quantize(margin / (1.0 + margin))
        if not math.isfinite(bounded) or bounded < 0.0 or bounded >= 1.0:
            raise RuntimeError("PVTM bounded feature differs")
        return PVTMResult(
            True, None, True, len(charge), len(pair), margin,
            equality_residual, lower_residual, None,
            {FEATURE_NAMES[0]: bounded},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pvtm_features(atoms: Atoms) -> PVTMResult:
    """Compute PVTM from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "NEXT435 valence assignment failed")
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT435 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "PVTM periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0, feasible=False),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        edges = tuple(geometry.edges)
        result = positive_valence_transport_margin(
            charges=charge,
            endpoints=tuple((int(edge.cation), int(edge.anion)) for edge in edges),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason="NEXT435 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_pvtm_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pvtm_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "pvtm_supported": bool(result.supported),
        "pvtm_failure": result.failure_reason,
        "pvtm_feasible": bool(result.feasible),
        "pvtm_site_count": int(result.site_count),
        "pvtm_edge_count": int(result.edge_count),
        "pvtm_raw_margin": float(result.raw_margin),
        "pvtm_maximum_equality_residual": float(result.maximum_equality_residual),
        "pvtm_minimum_lower_bound_residual": float(result.minimum_lower_bound_residual),
        "pvtm_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS", "CHARGE_TOLERANCE", "DESIGN_SHA256", "FEATURE_DIRECTIONS",
    "FEATURE_NAMES", "LP_RESIDUAL_TOLERANCE", "OUTPUT_GRID", "PROTOCOL",
    "PVTMResult", "compute_pvtm_features", "compute_pvtm_row",
    "positive_valence_transport_margin",
]
