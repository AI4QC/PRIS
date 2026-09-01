"""Frozen NEXT440 path-constrained a-priori bond positivity (no DFT)."""

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


PROTOCOL = "2026-08-13-next440-path-constrained-apriori-bond-positivity-v1"
DESIGN_SHA256 = "b3a49a6a5f50c42e843b479551458b0a8a6c3e46252a20e64b241a5368f75153"
FEATURE_NAMES = ("pcabp_path_constrained_bond_positivity",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
CHARGE_TOLERANCE = 1.0e-9
SOLVE_RESIDUAL_TOLERANCE = 1.0e-9
BOUNDARY_FLAGS = {
    "dft_calculation_executed": False,
    "dft_values_used": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
    "trajectory_or_later_geometry_used": False,
    "same_composition_alternative_used": False,
}


@dataclass(frozen=True)
class PCABPResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    edge_strengths: tuple[float, ...]
    positive_strength_mass: float
    negative_strength_mass: float
    maximum_equality_residual: float
    maximum_path_residual: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> PCABPResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return PCABPResult(
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


def _zero(*, site_count: int, edge_count: int) -> PCABPResult:
    return PCABPResult(
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


def path_constrained_apriori_bond_positivity(
    *,
    charges: Sequence[float] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> PCABPResult:
    """Solve the unit-conductance curl-free field ``B s = q``.

    Edges must be oriented from positive to negative formal charge.  The
    minimum-norm solution is the unique edge-gradient field and consequently
    has zero directed sum on every graph loop.  Its negative mass measures
    the amount of path-constrained strength forced against the physical edge
    orientation.
    """

    try:
        charge = np.asarray(charges, dtype=float)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("PCABP charges differ")
        magnitude = float(np.abs(charge).sum())
        if (
            magnitude <= 0.0
            or abs(float(charge.sum())) > CHARGE_TOLERANCE * max(1.0, magnitude)
            or np.any(charge == 0.0)
            or not np.any(charge > 0.0)
            or not np.any(charge < 0.0)
        ):
            raise ValueError("PCABP formal charges must be neutral and nonzero")
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
            or not np.isfinite(raw_pair.astype(float)).all()
            or not np.equal(
                raw_pair.astype(float), np.rint(raw_pair.astype(float))
            ).all()
        ):
            raise ValueError("PCABP endpoint population differs")
        pair = raw_pair.astype(int)
        if (
            np.any(pair < 0)
            or np.any(pair >= len(charge))
            or np.any(pair[:, 0] == pair[:, 1])
            or not np.all(charge[pair[:, 0]] > 0.0)
            or not np.all(charge[pair[:, 1]] < 0.0)
        ):
            raise ValueError("PCABP cation-anion edge orientation differs")

        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree == 0):
            return _zero(site_count=len(charge), edge_count=len(pair))

        incidence = np.zeros((len(charge), len(pair)), dtype=float)
        columns = np.arange(len(pair))
        incidence[pair[:, 0], columns] = 1.0
        incidence[pair[:, 1], columns] = -1.0
        laplacian = incidence @ incidence.T
        potential = np.linalg.pinv(laplacian, rcond=1.0e-12, hermitian=True) @ charge
        strength = incidence.T @ potential

        equality_error = incidence @ strength - charge
        equality_scale = max(1.0, float(np.max(np.abs(charge))))
        equality_residual = float(np.max(np.abs(equality_error)) / equality_scale)
        # Project the field back through its divergence.  A curl-free field is
        # fixed by this operation; any cycle-space component would remain in
        # the difference.
        recovered_potential = np.linalg.pinv(
            laplacian, rcond=1.0e-12, hermitian=True
        ) @ (incidence @ strength)
        recovered = incidence.T @ recovered_potential
        path_scale = max(1.0, float(np.max(np.abs(strength))))
        path_residual = float(np.max(np.abs(strength - recovered)) / path_scale)
        if not np.isfinite(strength).all():
            raise RuntimeError("PCABP edge field is not finite")
        if equality_residual > SOLVE_RESIDUAL_TOLERANCE:
            # A disconnected component with nonzero net charge cannot satisfy
            # exact formal-charge conservation.  This is a physical topology
            # obstruction, not missing numeric data.
            return _zero(site_count=len(charge), edge_count=len(pair))
        if path_residual > SOLVE_RESIDUAL_TOLERANCE:
            raise RuntimeError("PCABP path-constrained field residual differs")

        positive = math.fsum(np.maximum(strength, 0.0).tolist())
        negative = math.fsum(np.maximum(-strength, 0.0).tolist())
        total = positive + negative
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("PCABP oriented strength mass differs")
        bounded = _quantize(positive / total)
        if not math.isfinite(bounded) or bounded < 0.0 or bounded > 1.0:
            raise RuntimeError("PCABP bounded feature differs")
        return PCABPResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            tuple(float(value) for value in strength),
            float(positive),
            float(negative),
            equality_residual,
            path_residual,
            None,
            {FEATURE_NAMES[0]: bounded},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pcabp_features(atoms: Atoms) -> PCABPResult:
    """Compute PCABP from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT440 valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT440 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "PCABP periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        edges = tuple(geometry.edges)
        result = path_constrained_apriori_bond_positivity(
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
                failure_reason=str(exc).replace("NEXT295", "NEXT440"),
            )
        return result


def compute_pcabp_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pcabp_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "pcabp_supported": bool(result.supported),
        "pcabp_failure": result.failure_reason,
        "pcabp_feasible": result.feasible,
        "pcabp_site_count": int(result.site_count),
        "pcabp_edge_count": int(result.edge_count),
        "pcabp_positive_strength_mass": float(result.positive_strength_mass),
        "pcabp_negative_strength_mass": float(result.negative_strength_mass),
        "pcabp_maximum_equality_residual": float(
            result.maximum_equality_residual
        ),
        "pcabp_maximum_path_residual": float(result.maximum_path_residual),
        "pcabp_valence_policy": result.valence_policy,
    }
