"""Frozen NEXT510 Madelung charge--potential antitonicity (zero DFT)."""

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


PROTOCOL = "2026-08-13-next510-madelung-charge-potential-antitonicity-v1"
DESIGN_SHA256 = "97c3d6220106b31d6dede1ed41c8e83eb3515f51d329d8a23d0076db2d7f5c21"
FEATURE_NAMES = ("mcpa_madelung_charge_potential_antitonicity",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
CHARGE_TOLERANCE = 1.0e-8
ROUNDING_GUARD_MULTIPLIER = 64.0
EWALD_IDENTITY_TOLERANCE = 1.0e-10
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class MCPAResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    pair_count: int
    minimum_exchange_margin: float
    mean_exchange_margin: float
    maximum_exchange_margin: float
    ewald_identity_relative_error: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> MCPAResult:
    reason = str(exc)
    if isinstance(exc, Exception) and not reason.startswith(type(exc).__name__):
        reason = f"{type(exc).__name__}: {reason}"
    return MCPAResult(
        False,
        reason,
        0,
        0,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        None,
        {},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def madelung_charge_potential_antitonicity(
    *,
    charges: Sequence[float] | object,
    site_potentials: Sequence[float] | object,
) -> MCPAResult:
    """Score antitone ordering of formal charge and Madelung site potential."""

    try:
        charge = np.asarray(charges, dtype=float)
        potential = np.asarray(site_potentials, dtype=float)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT510 formal-charge population differs")
        if potential.shape != charge.shape or not np.isfinite(potential).all():
            raise ValueError("NEXT510 site-potential population differs")
        magnitude = float(np.abs(charge).sum())
        if (
            magnitude <= 0.0
            or abs(float(charge.sum()))
            > CHARGE_TOLERANCE * max(1.0, magnitude)
            or np.any(charge == 0.0)
            or not np.any(charge > 0.0)
            or not np.any(charge < 0.0)
        ):
            raise ValueError("NEXT510 formal charges must be neutral and nonzero")

        margins: list[float] = []
        for left in range(len(charge)):
            for right in range(left + 1, len(charge)):
                difference = float(charge[left] - charge[right])
                if difference == 0.0:
                    continue
                denominator = float(abs(potential[left]) + abs(potential[right]))
                if denominator == 0.0:
                    margin = 0.0
                else:
                    margin = -math.copysign(1.0, difference) * float(
                        potential[left] - potential[right]
                    ) / denominator
                guard = ROUNDING_GUARD_MULTIPLIER * np.finfo(float).eps
                if not math.isfinite(margin) or margin < -1.0 - guard or margin > 1.0 + guard:
                    raise RuntimeError("NEXT510 exchange-margin bound differs")
                margins.append(float(np.clip(margin, -1.0, 1.0)))
        if not margins:
            raise ValueError("NEXT510 unequal-charge pair population is empty")

        mean_margin = float(math.fsum(margins) / len(margins))
        score = _quantize((1.0 + mean_margin) / 2.0)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("NEXT510 published score differs")
        return MCPAResult(
            True,
            None,
            len(charge),
            len(margins),
            float(min(margins)),
            mean_margin,
            float(max(margins)),
            math.nan,
            None,
            {FEATURE_NAMES[0]: score},
        )
    except Exception as exc:
        return _failure(exc)


def compute_mcpa_features(atoms: Atoms) -> MCPAResult:
    """Compute MCPA from composition and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT510 formal valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),) or not np.isfinite(charge).all():
            raise ValueError("NEXT510 formal-valence population differs")

        from pymatgen.analysis.ewald import EwaldSummation

        decorated = structure.copy()
        decorated.add_oxidation_state_by_site(charge.tolist())
        ewald = EwaldSummation(decorated, compute_forces=False)
        site_energy = np.asarray(
            [float(ewald.get_site_energy(index)) for index in range(len(decorated))],
            dtype=float,
        )
        if site_energy.shape != charge.shape or not np.isfinite(site_energy).all():
            raise RuntimeError("NEXT510 analytic Ewald site energies differ")
        potential = 2.0 * site_energy / charge
        total = float(ewald.total_energy)
        reconstructed = float(0.5 * np.dot(charge, potential))
        identity_error = abs(reconstructed - total) / max(
            1.0, abs(reconstructed), abs(total)
        )
        if (
            not np.isfinite(potential).all()
            or not math.isfinite(total)
            or not math.isfinite(identity_error)
            or identity_error > EWALD_IDENTITY_TOLERANCE
        ):
            raise RuntimeError("NEXT510 Ewald energy-potential identity differs")

        result = madelung_charge_potential_antitonicity(
            charges=charge,
            site_potentials=potential,
        )
        if not result.supported:
            return result
        return replace(
            result,
            ewald_identity_relative_error=identity_error,
            valence_policy=str(assignment.policy),
        )
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason=str(exc).replace("NEXT295", "NEXT510"),
            )
        return result


def compute_mcpa_row(atoms: Atoms) -> dict[str, object]:
    result = compute_mcpa_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]])
            if result.supported
            else math.nan
        ),
        "mcpa_supported": bool(result.supported),
        "mcpa_failure": result.failure_reason,
        "mcpa_site_count": int(result.site_count),
        "mcpa_pair_count": int(result.pair_count),
        "mcpa_minimum_exchange_margin": float(result.minimum_exchange_margin),
        "mcpa_mean_exchange_margin": float(result.mean_exchange_margin),
        "mcpa_maximum_exchange_margin": float(result.maximum_exchange_margin),
        "mcpa_ewald_identity_relative_error": float(
            result.ewald_identity_relative_error
        ),
        "mcpa_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "MCPAResult",
    "PROTOCOL",
    "compute_mcpa_features",
    "compute_mcpa_row",
    "madelung_charge_potential_antitonicity",
]
