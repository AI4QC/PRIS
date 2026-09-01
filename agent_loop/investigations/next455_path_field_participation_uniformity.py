"""Frozen NEXT455 path-field participation uniformity (no DFT)."""

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


PROTOCOL = "2026-08-13-next455-path-field-participation-uniformity-v1"
DESIGN_SHA256 = "4ffde6f73aeb85d004383f0151d764c763fa8ca3dadbc34d9a0ae6b13eb76c7e"
FEATURE_NAMES = ("pfpu_path_field_participation_uniformity",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n440.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PFPUResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    edge_strengths: tuple[float, ...]
    effective_edge_count: float
    shannon_entropy: float
    maximum_equality_residual: float
    maximum_path_residual: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> PFPUResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return PFPUResult(
        False,
        reason,
        None,
        0,
        0,
        (),
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        None,
        {},
    )


def _zero(*, site_count: int, edge_count: int) -> PFPUResult:
    return PFPUResult(
        True,
        None,
        False,
        int(site_count),
        int(edge_count),
        (),
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def path_field_participation_uniformity(
    *,
    charges: Sequence[float] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> PFPUResult:
    """Return normalized Shannon effective participation of NEXT440's field."""

    try:
        field = n440.path_constrained_apriori_bond_positivity(
            charges=charges, endpoints=endpoints
        )
        if not field.supported:
            raise ValueError(field.failure_reason or "PFPU path field failed")
        if not field.feasible:
            return _zero(site_count=field.site_count, edge_count=field.edge_count)
        weights = np.abs(np.asarray(field.edge_strengths, dtype=float))
        total = math.fsum(weights.tolist())
        if (
            weights.shape != (field.edge_count,)
            or not np.isfinite(weights).all()
            or not math.isfinite(total)
            or total <= 0.0
        ):
            raise RuntimeError("PFPU absolute strength population differs")
        probability = weights / total
        positive = probability > 0.0
        entropy = -math.fsum(
            (probability[positive] * np.log(probability[positive])).tolist()
        )
        effective = math.exp(entropy)
        uniformity = _quantize(effective / field.edge_count)
        if (
            not math.isfinite(entropy)
            or not math.isfinite(effective)
            or not math.isfinite(uniformity)
            or uniformity <= 0.0
            or uniformity > 1.0
        ):
            raise RuntimeError("PFPU effective participation differs")
        return PFPUResult(
            True,
            None,
            True,
            field.site_count,
            field.edge_count,
            field.edge_strengths,
            float(effective),
            float(entropy),
            field.maximum_equality_residual,
            field.maximum_path_residual,
            None,
            {FEATURE_NAMES[0]: uniformity},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pfpu_features(atoms: Atoms) -> PFPUResult:
    """Compute PFPU from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT455 valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT455 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "PFPU periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        edges = tuple(geometry.edges)
        result = path_field_participation_uniformity(
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
                failure_reason=str(exc).replace("NEXT295", "NEXT455"),
            )
        return result


def compute_pfpu_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pfpu_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "pfpu_supported": bool(result.supported),
        "pfpu_failure": result.failure_reason,
        "pfpu_feasible": result.feasible,
        "pfpu_site_count": int(result.site_count),
        "pfpu_edge_count": int(result.edge_count),
        "pfpu_effective_edge_count": float(result.effective_edge_count),
        "pfpu_shannon_entropy": float(result.shannon_entropy),
        "pfpu_maximum_equality_residual": float(
            result.maximum_equality_residual
        ),
        "pfpu_maximum_path_residual": float(result.maximum_path_residual),
        "pfpu_valence_policy": result.valence_policy,
    }
