#!/usr/bin/env python3
"""First non-extinct formal-charge Bragg mode of one raw periodic crystal."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next36_charge_spectrum_features as n36
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next363-first-charge-bragg-extinction-v1"
DESIGN_SHA256 = "8184c6866d9f1f62aa61342b7d3ce39c87051e7b34393884c490fce6fa0568e9"
FEATURE_NAMES = ("fcbe_first_charge_bragg_wavenumber",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
INTENSITY_FLOOR = 1.0e-12
DIMENSIONLESS_CUTOFF = n36.DIMENSIONLESS_CUTOFF
MAX_ENUMERATED_RECIPROCAL_POINTS = n36.MAX_ENUMERATED_RECIPROCAL_POINTS
FIRST_SHELL_TOLERANCE = 1.0e-10
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class FirstChargeBraggResult:
    first_wavenumber: float
    first_intensity: float
    first_integer_vectors: np.ndarray
    reciprocal_vector_count: int
    nonextinct_vector_count: int


@dataclass(frozen=True)
class FCBEFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    reciprocal_vector_count: int
    nonextinct_vector_count: int
    first_peak_multiplicity: int
    first_intensity: float
    valence_policy: str | None
    features: dict[str, float]


def _failure(exc: Exception | str) -> FCBEFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return FCBEFeatureResult(False, reason, 0, 0, 0, 0, math.nan, None, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def first_charge_bragg_wavenumber(
    lattice_matrix: Sequence[Sequence[float]] | np.ndarray,
    fractional_coordinates: Sequence[Sequence[float]] | np.ndarray,
    charges: Sequence[float] | np.ndarray,
) -> FirstChargeBraggResult:
    """Return the first dimensionless reciprocal mode above the frozen floor."""

    lattice = np.asarray(lattice_matrix, dtype=float)
    fractional = np.asarray(fractional_coordinates, dtype=float)
    charge = np.asarray(charges, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("NEXT363 lattice must be a finite 3x3 matrix")
    if (
        fractional.ndim != 2
        or fractional.shape[1:] != (3,)
        or len(fractional) < 2
        or not np.isfinite(fractional).all()
    ):
        raise ValueError("NEXT363 fractional coordinates differ")
    if charge.shape != (len(fractional),) or not np.isfinite(charge).all():
        raise ValueError("NEXT363 charges differ")
    magnitude = float(np.abs(charge).sum())
    if magnitude <= 0.0 or abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
        raise ValueError("NEXT363 charges must be neutral")
    if not np.any(charge > 0.0) or not np.any(charge < 0.0):
        raise ValueError("NEXT363 charges must contain both signs")
    q2 = float(charge @ charge)
    volume = abs(float(np.linalg.det(lattice)))
    if not math.isfinite(q2) or q2 <= 0.0 or not math.isfinite(volume) or volume <= 1.0e-12:
        raise ValueError("NEXT363 normalization differs")
    length_per_site = float((volume / len(charge)) ** (1.0 / 3.0))
    integers, _, dimensionless = n36._reciprocal_integer_vectors(
        lattice, length_per_site=length_per_site
    )
    fractional = np.mod(fractional, 1.0)
    intensity = np.empty(len(integers), dtype=float)
    for start in range(0, len(integers), 16_384):
        stop = min(start + 16_384, len(integers))
        phase = 2.0 * np.pi * (integers[start:stop] @ fractional.T)
        amplitude = np.exp(-1j * phase) @ charge
        intensity[start:stop] = np.abs(amplitude) ** 2 / (len(charge) * q2)
    intensity = np.maximum(intensity, 0.0)
    if not np.isfinite(intensity).all():
        raise ValueError("NEXT363 reciprocal intensities differ")
    retained = intensity >= INTENSITY_FLOOR
    if not retained.any():
        raise ValueError("NEXT363 cutoff contains no non-extinct charge mode")
    first = float(np.min(dimensionless[retained]))
    shell = retained & np.isclose(
        dimensionless, first, rtol=0.0, atol=FIRST_SHELL_TOLERANCE
    )
    vectors = integers[shell]
    if len(vectors) < 1 or first <= 0.0 or first > DIMENSIONLESS_CUTOFF + 1.0e-12:
        raise RuntimeError("NEXT363 first charge Bragg shell differs")
    return FirstChargeBraggResult(
        first,
        float(np.max(intensity[shell])),
        np.asarray(vectors, dtype=int),
        int(len(integers)),
        int(retained.sum()),
    )


def compute_fcbe_features(atoms: Atoms) -> FCBEFeatureResult:
    """Compute FCBE from composition and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT363 valence assignment failed"
            )
        result = first_charge_bragg_wavenumber(
            np.asarray(work.cell.array, dtype=float),
            np.asarray(work.get_scaled_positions(wrap=True), dtype=float),
            np.asarray(assignment.values, dtype=float),
        )
        value = _quantize(result.first_wavenumber)
        if not 0.0 < value <= DIMENSIONLESS_CUTOFF:
            raise RuntimeError("NEXT363 feature domain differs")
        return FCBEFeatureResult(
            True,
            None,
            len(work),
            result.reciprocal_vector_count,
            result.nonextinct_vector_count,
            len(result.first_integer_vectors),
            result.first_intensity,
            str(assignment.policy),
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_fcbe_row(atoms: Atoms) -> dict[str, object]:
    result = compute_fcbe_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "fcbe_supported": bool(result.supported),
        "fcbe_failure": result.failure_reason,
        "fcbe_site_count": int(result.site_count),
        "fcbe_reciprocal_vector_count": int(result.reciprocal_vector_count),
        "fcbe_nonextinct_vector_count": int(result.nonextinct_vector_count),
        "fcbe_first_peak_multiplicity": int(result.first_peak_multiplicity),
        "fcbe_first_intensity": result.first_intensity,
        "fcbe_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS", "DESIGN_SHA256", "DIMENSIONLESS_CUTOFF",
    "FCBEFeatureResult", "FEATURE_DIRECTIONS", "FEATURE_NAMES",
    "FIRST_SHELL_TOLERANCE", "FirstChargeBraggResult", "INTENSITY_FLOOR",
    "MAX_ENUMERATED_RECIPROCAL_POINTS", "PROTOCOL", "compute_fcbe_features",
    "compute_fcbe_row", "first_charge_bragg_wavenumber",
]
