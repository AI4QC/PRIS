"""Frozen NEXT505 Madelung valence-class homogeneity (zero DFT)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next21_normalized_madelung as n21
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next505-madelung-valence-class-homogeneity-v1"
DESIGN_SHA256 = "a440a4e4d345348b1e946b486d0b8a2f840db5bb44201a52f20ad1d341c832da"
FEATURE_NAMES = ("mvch_madelung_valence_class_homogeneity",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
CHARGE_TOLERANCE = 1.0e-8
ROUNDING_GUARD_MULTIPLIER = 64.0
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class MVCHResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    class_count: int
    repeated_class_site_fraction: float
    within_sum_squares: float
    total_sum_squares: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> MVCHResult:
    reason = str(exc)
    if isinstance(exc, Exception) and not reason.startswith(type(exc).__name__):
        reason = f"{type(exc).__name__}: {reason}"
    return MVCHResult(False, reason, 0, 0, math.nan, math.nan, math.nan, None, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def madelung_valence_class_homogeneity(
    *,
    site_energies: Sequence[float] | object,
    atomic_numbers: Sequence[int] | object,
    charges: Sequence[float] | object,
) -> MVCHResult:
    """Return the bounded ANOVA score for element/valence site classes."""

    try:
        energy = np.asarray(site_energies, dtype=float)
        raw_number = np.asarray(atomic_numbers)
        charge = np.asarray(charges, dtype=float)
        if energy.ndim != 1 or len(energy) < 2 or not np.isfinite(energy).all():
            raise ValueError("NEXT505 site-energy population differs")
        if (
            raw_number.shape != energy.shape
            or not np.isfinite(raw_number.astype(float)).all()
            or not np.equal(raw_number.astype(float), np.rint(raw_number.astype(float))).all()
        ):
            raise ValueError("NEXT505 atomic-number population differs")
        number = raw_number.astype(int)
        if np.any(number <= 0):
            raise ValueError("NEXT505 atomic numbers differ")
        if charge.shape != energy.shape or not np.isfinite(charge).all():
            raise ValueError("NEXT505 formal-valence population differs")
        magnitude = float(np.abs(charge).sum())
        if (
            magnitude <= 0.0
            or abs(float(charge.sum()))
            > CHARGE_TOLERANCE * max(1.0, magnitude)
            or np.any(charge == 0.0)
            or not np.any(charge > 0.0)
            or not np.any(charge < 0.0)
        ):
            raise ValueError("NEXT505 formal charges must be neutral and nonzero")

        groups: dict[tuple[int, float], list[int]] = {}
        for index, (atomic_number, formal_valence) in enumerate(
            zip(number.tolist(), charge.tolist(), strict=True)
        ):
            groups.setdefault((int(atomic_number), float(formal_valence)), []).append(index)
        if not groups:
            raise RuntimeError("NEXT505 valence-class partition differs")

        mean = float(np.mean(energy))
        total = float(math.fsum(((energy - mean) ** 2).tolist()))
        within_terms: list[float] = []
        for indices in groups.values():
            values = energy[np.asarray(indices, dtype=int)]
            class_mean = float(np.mean(values))
            within_terms.extend(((values - class_mean) ** 2).tolist())
        within = float(math.fsum(within_terms))
        scale = max(1.0, float(math.fsum((energy**2).tolist())))
        guard = ROUNDING_GUARD_MULTIPLIER * np.finfo(float).eps * scale
        if (
            not math.isfinite(total)
            or not math.isfinite(within)
            or total < -guard
            or within < -guard
            or within > total + guard
        ):
            raise RuntimeError("NEXT505 ANOVA identity differs")
        total = max(0.0, total)
        within = max(0.0, within)
        if total <= guard:
            score = 1.0
        else:
            score = 1.0 - within / total
            if score < -guard or score > 1.0 + guard:
                raise RuntimeError("NEXT505 bounded score differs")
            score = float(np.clip(score, 0.0, 1.0))
        score = _quantize(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("NEXT505 published score differs")
        repeated = sum(len(indices) for indices in groups.values() if len(indices) > 1)
        return MVCHResult(
            True,
            None,
            len(energy),
            len(groups),
            float(repeated / len(energy)),
            within,
            total,
            None,
            {FEATURE_NAMES[0]: score},
        )
    except Exception as exc:
        return _failure(exc)


def compute_mvch_features(atoms: Atoms) -> MVCHResult:
    """Compute MVCH from composition and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT505 formal valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),) or not np.isfinite(charge).all():
            raise ValueError("NEXT505 formal-valence population differs")
        q2 = float(charge @ charge)
        volume = float(structure.volume)
        if not math.isfinite(q2) or q2 <= 0.0 or not math.isfinite(volume) or volume <= 0.0:
            raise ValueError("NEXT505 Ewald normalization differs")

        from pymatgen.analysis.ewald import EwaldSummation

        decorated = structure.copy()
        decorated.add_oxidation_state_by_site(charge.tolist())
        ewald = EwaldSummation(decorated, compute_forces=False)
        site_energy = np.asarray(
            [float(ewald.get_site_energy(index)) for index in range(len(decorated))],
            dtype=float,
        )
        if site_energy.shape != charge.shape or not np.isfinite(site_energy).all():
            raise RuntimeError("NEXT505 analytic Ewald site energies differ")
        factor = volume ** (1.0 / 3.0) / (n21.COULOMB_EV_ANGSTROM * q2)
        reduced = site_energy * factor
        result = madelung_valence_class_homogeneity(
            site_energies=reduced,
            atomic_numbers=np.asarray(work.numbers, dtype=int),
            charges=charge,
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason=str(exc).replace("NEXT295", "NEXT505"),
            )
        return result


def compute_mvch_row(atoms: Atoms) -> dict[str, object]:
    result = compute_mvch_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]])
            if result.supported
            else math.nan
        ),
        "mvch_supported": bool(result.supported),
        "mvch_failure": result.failure_reason,
        "mvch_site_count": int(result.site_count),
        "mvch_class_count": int(result.class_count),
        "mvch_repeated_class_site_fraction": float(
            result.repeated_class_site_fraction
        ),
        "mvch_within_sum_squares": float(result.within_sum_squares),
        "mvch_total_sum_squares": float(result.total_sum_squares),
        "mvch_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "MVCHResult",
    "PROTOCOL",
    "compute_mvch_features",
    "compute_mvch_row",
    "madelung_valence_class_homogeneity",
]
