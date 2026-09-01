"""Frozen NEXT445 path-constrained bond-strength matching (no DFT)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next440_path_constrained_apriori_bond_positivity as n440


PROTOCOL = "2026-08-13-next445-path-constrained-bond-strength-matching-v1"
DESIGN_SHA256 = "e7c91c51167a4c4653bfc8a0eb9ee7cfc25bacb7d7f1300c20f384f477da80b6"
FEATURE_NAMES = ("pcabsm_path_constrained_bond_strength_matching",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n440.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PCABSMResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    edge_strengths: tuple[float, ...]
    reference_strengths: tuple[float, ...]
    normalized_mismatch: float
    maximum_equality_residual: float
    maximum_path_residual: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> PCABSMResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return PCABSMResult(
        False,
        reason,
        None,
        0,
        0,
        (),
        (),
        math.nan,
        math.nan,
        math.nan,
        None,
        {},
    )


def _zero(*, site_count: int, edge_count: int) -> PCABSMResult:
    return PCABSMResult(
        True,
        None,
        False,
        int(site_count),
        int(edge_count),
        (),
        (),
        1.0,
        0.0,
        0.0,
        None,
        {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def path_constrained_bond_strength_matching(
    *,
    charges: Sequence[float] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> PCABSMResult:
    """Compare NEXT440's curl-free field with endpoint characteristic strength."""

    try:
        raw_charge = np.asarray(charges, dtype=float)
        raw_pair = np.asarray(endpoints)
        field = n440.path_constrained_apriori_bond_positivity(
            charges=charges, endpoints=endpoints
        )
        if not field.supported:
            raise ValueError(field.failure_reason or "PCABSM path field failed")
        if not field.feasible:
            return _zero(site_count=field.site_count, edge_count=field.edge_count)
        pair = raw_pair.astype(int)
        degree = np.bincount(pair.ravel(), minlength=len(raw_charge)).astype(float)
        if np.any(degree <= 0.0):
            return _zero(site_count=len(raw_charge), edge_count=len(pair))
        characteristic = np.abs(raw_charge) / degree
        reference = np.sqrt(
            characteristic[pair[:, 0]] * characteristic[pair[:, 1]]
        )
        strength = np.asarray(field.edge_strengths, dtype=float)
        if (
            reference.shape != strength.shape
            or not np.isfinite(reference).all()
            or np.any(reference <= 0.0)
        ):
            raise RuntimeError("PCABSM reference edge strengths differ")
        numerator = math.fsum(np.abs(strength - reference).tolist())
        denominator = math.fsum((np.abs(strength) + reference).tolist())
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or numerator < 0.0
            or denominator <= 0.0
            or numerator > denominator + 1.0e-10 * denominator
        ):
            raise RuntimeError("PCABSM normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        matching = _quantize(1.0 - mismatch)
        if not math.isfinite(matching) or matching < 0.0 or matching > 1.0:
            raise RuntimeError("PCABSM bounded feature differs")
        return PCABSMResult(
            True,
            None,
            True,
            field.site_count,
            field.edge_count,
            field.edge_strengths,
            tuple(float(value) for value in reference),
            mismatch,
            field.maximum_equality_residual,
            field.maximum_path_residual,
            None,
            {FEATURE_NAMES[0]: matching},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pcabsm_features(atoms: Atoms) -> PCABSMResult:
    """Compute PCABSM from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT445 valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT445 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "PCABSM periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        edges = tuple(geometry.edges)
        result = path_constrained_bond_strength_matching(
            charges=charge,
            endpoints=tuple(
                (int(edge.cation), int(edge.anion)) for edge in edges
            ),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason=str(exc).replace("NEXT295", "NEXT445"),
            )
        return result


def compute_pcabsm_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pcabsm_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "pcabsm_supported": bool(result.supported),
        "pcabsm_failure": result.failure_reason,
        "pcabsm_feasible": result.feasible,
        "pcabsm_site_count": int(result.site_count),
        "pcabsm_edge_count": int(result.edge_count),
        "pcabsm_normalized_mismatch": float(result.normalized_mismatch),
        "pcabsm_maximum_equality_residual": float(
            result.maximum_equality_residual
        ),
        "pcabsm_maximum_path_residual": float(result.maximum_path_residual),
        "pcabsm_valence_policy": result.valence_policy,
    }
