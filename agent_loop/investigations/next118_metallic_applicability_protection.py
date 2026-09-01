#!/usr/bin/env python3
"""Bounded no-DFT applicability protection for metallic packing regimes.

The optional operand uses only frozen elemental electronegativity and raw-x0
covalent-distance geometry.  It is fail-open: unavailable operands turn the
protection off, and composition never expands the support of the base law.
"""

from __future__ import annotations

import math

import numpy as np


COMBINATION_RULES = ("minimum", "product")


def metallic_packing_protection(
    *,
    electronegativity_mean: object,
    covalent_ratio_q05: object,
    electronegativity_upper: float,
    electronegativity_width: float,
    covalent_ratio_lower: float,
    covalent_ratio_width: float,
    combination: str = "minimum",
) -> np.ndarray:
    """Return a bounded conjunction of metallicity and noncompression legs."""

    electronegativity = np.asarray(electronegativity_mean, dtype=float)
    covalent_ratio = np.asarray(covalent_ratio_q05, dtype=float)
    parameters = (
        electronegativity_upper,
        electronegativity_width,
        covalent_ratio_lower,
        covalent_ratio_width,
    )
    if (
        electronegativity.ndim != 1
        or covalent_ratio.shape != electronegativity.shape
        or any(
            isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            for value in parameters
        )
        or float(electronegativity_width) <= 0.0
        or float(covalent_ratio_width) <= 0.0
        or combination not in COMBINATION_RULES
    ):
        raise ValueError("NEXT118 metallic-protection calibration differs")

    active = np.isfinite(electronegativity) & np.isfinite(covalent_ratio)
    electronegativity_leg = np.zeros(electronegativity.shape, dtype=float)
    covalent_ratio_leg = np.zeros(electronegativity.shape, dtype=float)
    electronegativity_leg[active] = np.clip(
        (float(electronegativity_upper) - electronegativity[active])
        / float(electronegativity_width),
        0.0,
        1.0,
    )
    covalent_ratio_leg[active] = np.clip(
        (covalent_ratio[active] - float(covalent_ratio_lower))
        / float(covalent_ratio_width),
        0.0,
        1.0,
    )
    if combination == "minimum":
        protection = np.minimum(electronegativity_leg, covalent_ratio_leg)
    else:
        protection = electronegativity_leg * covalent_ratio_leg
    protection[~active] = 0.0
    return protection


def compose_bounded_protection_score(
    *,
    base_score: object,
    base_supported: object,
    protection: object,
    protection_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract bounded protection without changing support or negative risk."""

    base = np.asarray(base_score, dtype=float)
    supported = np.asarray(base_supported, dtype=bool)
    operand = np.asarray(protection, dtype=float)
    weight = float(protection_weight)
    if (
        base.ndim != 1
        or supported.shape != base.shape
        or operand.shape != base.shape
        or not np.isfinite(base[supported]).all()
        or not np.isfinite(operand).all()
        or np.any(operand < 0.0)
        or np.any(operand > 1.0)
        or not math.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError("NEXT118 bounded-protection arrays differ")
    score = np.maximum(0.0, base - weight * operand)
    score[~supported] = np.nan
    return score, supported.copy()


__all__ = [
    "COMBINATION_RULES",
    "compose_bounded_protection_score",
    "metallic_packing_protection",
]
