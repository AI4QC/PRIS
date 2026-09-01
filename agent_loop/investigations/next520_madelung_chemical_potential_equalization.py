"""Frozen NEXT520 Madelung chemical-potential equalization (zero DFT)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import reduce
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.core import Element
import pymatgen.core.periodic_table as periodic_table
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next520-madelung-chemical-potential-equalization-v1"
DESIGN_SHA256 = "9a3109ea4db49f0e4199eca651538fe561ea7df429137337a3d6291eddd8f660"
ATOMIC_TABLE_SHA256 = "b11669f8ccb0a9fe7647d9026ecbd30ee15ded7c464df828820a15768556d0aa"
FEATURE_NAMES = ("mcpe_madelung_chemical_potential_equalization",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
CHARGE_TOLERANCE = 1.0e-8
ROUNDING_GUARD_MULTIPLIER = 64.0
EWALD_IDENTITY_TOLERANCE = 1.0e-10
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class MCPEResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    pair_count: int
    electronegativities: tuple[float, ...]
    hardnesses: tuple[float, ...]
    chemical_potentials: tuple[float, ...]
    mean_normalized_discrepancy: float
    maximum_normalized_discrepancy: float
    chemical_potential_spread: float
    electronegativity_spread: float
    hardness_spread: float
    ewald_identity_relative_error: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> MCPEResult:
    reason = str(exc)
    if isinstance(exc, Exception) and not reason.startswith(type(exc).__name__):
        reason = f"{type(exc).__name__}: {reason}"
    return MCPEResult(
        False,
        reason,
        0,
        0,
        (),
        (),
        (),
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        None,
        {},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _atomic_table_path() -> Path:
    return Path(periodic_table.__file__).resolve().with_name("periodic_table.json.gz")


def _verify_atomic_table() -> None:
    path = _atomic_table_path()
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != ATOMIC_TABLE_SHA256
    ):
        raise ValueError("NEXT520 frozen atomic table identity differs")


def _intensive_formal_charges(
    charge: np.ndarray, elements: Sequence[Element], policy: str | None
) -> np.ndarray:
    """Undo NEXT19's cell normalization only for its partition fallback."""

    intensive = np.asarray(charge, dtype=float).copy()
    if policy != "electronegativity_partition":
        return intensive
    counts: dict[str, int] = {}
    for element in elements:
        symbol = str(element.symbol)
        counts[symbol] = counts.get(symbol, 0) + 1
    formula_units = reduce(math.gcd, counts.values())
    if formula_units < 1:
        raise RuntimeError("NEXT520 reduced-formula multiplicity differs")
    return intensive * float(formula_units)


def madelung_chemical_potential_equalization(
    *,
    charges: Sequence[float] | object,
    ionization_energies: Sequence[float] | object,
    electron_affinities: Sequence[float] | object,
    site_potentials: Sequence[float] | object,
) -> MCPEResult:
    """Score equality of QEq-style site chemical potentials at fixed charges."""

    try:
        charge = np.asarray(charges, dtype=float)
        ionization = np.asarray(ionization_energies, dtype=float)
        affinity = np.asarray(electron_affinities, dtype=float)
        potential = np.asarray(site_potentials, dtype=float)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT520 formal-charge population differs")
        if (
            ionization.shape != charge.shape
            or affinity.shape != charge.shape
            or potential.shape != charge.shape
            or not np.isfinite(ionization).all()
            or not np.isfinite(affinity).all()
            or not np.isfinite(potential).all()
        ):
            raise ValueError("NEXT520 atomic or potential population differs")
        magnitude = float(np.abs(charge).sum())
        if (
            magnitude <= 0.0
            or abs(float(charge.sum()))
            > CHARGE_TOLERANCE * max(1.0, magnitude)
            or np.any(charge == 0.0)
            or not np.any(charge > 0.0)
            or not np.any(charge < 0.0)
        ):
            raise ValueError("NEXT520 formal charges must be neutral and nonzero")
        electronegativity = 0.5 * (ionization + affinity)
        hardness = ionization - affinity
        if (
            not np.isfinite(electronegativity).all()
            or not np.isfinite(hardness).all()
            or np.any(electronegativity < 0.0)
            or np.any(hardness <= 0.0)
        ):
            raise ValueError("NEXT520 atomic electronegativity or hardness differs")
        chemical_potential = electronegativity + hardness * charge + potential
        scale = electronegativity + hardness * np.abs(charge) + np.abs(potential)
        if (
            not np.isfinite(chemical_potential).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise RuntimeError("NEXT520 chemical-potential scale differs")
        discrepancy = np.abs(
            chemical_potential[:, None] - chemical_potential[None, :]
        ) / (scale[:, None] + scale[None, :])
        guard = ROUNDING_GUARD_MULTIPLIER * np.finfo(float).eps
        if (
            not np.isfinite(discrepancy).all()
            or np.any(discrepancy < -guard)
            or np.any(discrepancy > 1.0 + guard)
        ):
            raise RuntimeError("NEXT520 normalized discrepancy bound differs")
        discrepancy = np.clip(discrepancy, 0.0, 1.0)
        mean_discrepancy = float(np.mean(discrepancy))
        score = _quantize(1.0 - mean_discrepancy)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("NEXT520 published score differs")
        return MCPEResult(
            True,
            None,
            len(charge),
            int(charge.size**2),
            tuple(float(value) for value in electronegativity),
            tuple(float(value) for value in hardness),
            tuple(float(value) for value in chemical_potential),
            mean_discrepancy,
            float(np.max(discrepancy)),
            float(np.max(chemical_potential) - np.min(chemical_potential)),
            float(np.max(electronegativity) - np.min(electronegativity)),
            float(np.max(hardness) - np.min(hardness)),
            math.nan,
            None,
            {FEATURE_NAMES[0]: score},
        )
    except Exception as exc:
        return _failure(exc)


def compute_mcpe_features(atoms: Atoms) -> MCPEResult:
    """Compute MCPE from composition and one raw unrelaxed periodic geometry."""

    try:
        _verify_atomic_table()
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT520 formal valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),) or not np.isfinite(charge).all():
            raise ValueError("NEXT520 formal-valence population differs")
        elements = tuple(Element(str(site.specie.symbol)) for site in structure)
        charge = _intensive_formal_charges(charge, elements, assignment.policy)
        ionization = np.asarray(
            [float(element.ionization_energy) for element in elements], dtype=float
        )
        affinity = np.asarray(
            [float(element.electron_affinity) for element in elements], dtype=float
        )
        if (
            ionization.shape != charge.shape
            or affinity.shape != charge.shape
            or not np.isfinite(ionization).all()
            or not np.isfinite(affinity).all()
        ):
            raise ValueError("NEXT520 frozen atomic data are incomplete")

        from pymatgen.analysis.ewald import EwaldSummation

        decorated = structure.copy()
        decorated.add_oxidation_state_by_site(charge.tolist())
        ewald = EwaldSummation(decorated, compute_forces=False)
        site_energy = np.asarray(
            [float(ewald.get_site_energy(index)) for index in range(len(decorated))],
            dtype=float,
        )
        if site_energy.shape != charge.shape or not np.isfinite(site_energy).all():
            raise RuntimeError("NEXT520 analytic Ewald site energies differ")
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
            raise RuntimeError("NEXT520 Ewald energy-potential identity differs")
        result = madelung_chemical_potential_equalization(
            charges=charge,
            ionization_energies=ionization,
            electron_affinities=affinity,
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
                failure_reason=str(exc).replace("NEXT295", "NEXT520"),
            )
        return result


def compute_mcpe_row(atoms: Atoms) -> dict[str, object]:
    result = compute_mcpe_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "mcpe_supported": bool(result.supported),
        "mcpe_failure": result.failure_reason,
        "mcpe_site_count": int(result.site_count),
        "mcpe_pair_count": int(result.pair_count),
        "mcpe_mean_normalized_discrepancy": float(result.mean_normalized_discrepancy),
        "mcpe_maximum_normalized_discrepancy": float(result.maximum_normalized_discrepancy),
        "mcpe_chemical_potential_spread": float(result.chemical_potential_spread),
        "mcpe_electronegativity_spread": float(result.electronegativity_spread),
        "mcpe_hardness_spread": float(result.hardness_spread),
        "mcpe_ewald_identity_relative_error": float(result.ewald_identity_relative_error),
        "mcpe_valence_policy": result.valence_policy,
        "mcpe_atomic_table_sha256": ATOMIC_TABLE_SHA256,
    }


__all__ = [
    "ATOMIC_TABLE_SHA256",
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "MCPEResult",
    "PROTOCOL",
    "compute_mcpe_features",
    "compute_mcpe_row",
    "madelung_chemical_potential_equalization",
]
