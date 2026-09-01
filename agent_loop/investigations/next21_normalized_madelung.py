#!/usr/bin/env python3
"""Scale- and charge-amplitude-invariant analytic Ewald descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PROTOCOL = "2026-08-02-next21-normalized-madelung-v1"
COULOMB_EV_ANGSTROM = 14.3996454784255
FEATURE_NAMES = (
    "nm_total_reduced",
    "nm_real_reduced",
    "nm_reciprocal_reduced",
    "nm_point_reduced",
    "nm_site_spread",
    "nm_site_max",
    "nm_site_min",
    "nm_site_positive_fraction",
    "nm_charge_concentration",
)


@dataclass(frozen=True)
class MadelungFeatureResult:
    """Fail-open normalized Madelung result for one structure."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> MadelungFeatureResult:
    return MadelungFeatureResult(False, reason, {})


def normalized_madelung_features(
    structure,
    charges: Sequence[float] | np.ndarray,
) -> MadelungFeatureResult:
    """Compute normalized analytic Ewald terms without changing the input."""

    values = np.asarray(charges, dtype=float)
    if values.shape != (len(structure),):
        return _failure("charges must match the structure sites")
    if not np.isfinite(values).all():
        return _failure("charges must be finite")
    q2 = float(np.sum(values**2))
    if not np.isfinite(q2) or q2 <= 0.0:
        return _failure("charges must have nonzero squared magnitude")
    if abs(float(values.sum())) > 1.0e-8 * max(1.0, float(np.abs(values).sum())):
        return _failure("charges must be neutral")
    volume = float(structure.volume)
    if not np.isfinite(volume) or volume <= 0.0:
        return _failure("structure volume must be finite and positive")

    decorated = structure.copy()
    try:
        from pymatgen.analysis.ewald import EwaldSummation

        decorated.add_oxidation_state_by_site(values.tolist())
        ewald = EwaldSummation(decorated, compute_forces=False)
        site_terms = np.asarray(
            [float(ewald.get_site_energy(index)) for index in range(len(decorated))],
            dtype=float,
        )
        raw = np.asarray(
            [
                float(ewald.total_energy),
                float(ewald.real_space_energy),
                float(ewald.reciprocal_space_energy),
                float(ewald.point_energy),
            ],
            dtype=float,
        )
    except Exception as exc:
        return _failure(f"analytic Ewald sum failed: {type(exc).__name__}")
    if not np.isfinite(raw).all() or not np.isfinite(site_terms).all():
        return _failure("analytic Ewald sum returned non-finite values")

    factor = volume ** (1.0 / 3.0) / (COULOMB_EV_ANGSTROM * q2)
    reduced = raw * factor
    site_reduced = site_terms * factor
    concentration = float(len(values) * np.sum(values**4) / q2**2)
    features = {
        "nm_total_reduced": float(reduced[0]),
        "nm_real_reduced": float(reduced[1]),
        "nm_reciprocal_reduced": float(reduced[2]),
        "nm_point_reduced": float(reduced[3]),
        "nm_site_spread": float(np.std(site_reduced)),
        "nm_site_max": float(np.max(site_reduced)),
        "nm_site_min": float(np.min(site_reduced)),
        "nm_site_positive_fraction": float(np.mean(site_reduced > 0.0)),
        "nm_charge_concentration": concentration,
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        return _failure("computed feature schema is invalid")
    return MadelungFeatureResult(True, None, features)


__all__ = [
    "FEATURE_NAMES",
    "PROTOCOL",
    "MadelungFeatureResult",
    "normalized_madelung_features",
]
