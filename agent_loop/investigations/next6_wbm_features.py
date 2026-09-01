#!/usr/bin/env python3
"""Cheap, universal geometry features computed only from an unrelaxed structure."""

from __future__ import annotations

import math
import shlex
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from pymatgen.core import Element, Lattice


SCALE_GRID = (0.8, 0.9, 1.0, 1.1, 1.2)


@dataclass(frozen=True)
class ParsedExtXYZ:
    material_id: str
    lattice: np.ndarray
    species: tuple[str, ...]
    cart_coords: np.ndarray


def parse_extxyz(text: str) -> ParsedExtXYZ:
    """Parse one simple periodic extxyz frame used by the WBM archives."""

    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("extxyz frame is missing its atom count or comment")
    n_atoms = int(lines[0].strip())
    if len(lines) != n_atoms + 2:
        raise ValueError(f"expected {n_atoms} atom lines, found {len(lines) - 2}")

    fields: dict[str, str] = {}
    for token in shlex.split(lines[1]):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    if "Lattice" not in fields:
        raise ValueError("extxyz comment has no Lattice field")
    lattice_values = [float(value) for value in fields["Lattice"].split()]
    if len(lattice_values) != 9:
        raise ValueError("Lattice must contain exactly nine values")

    species: list[str] = []
    coordinates: list[list[float]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"malformed extxyz atom line: {line!r}")
        species.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return ParsedExtXYZ(
        material_id=fields.get("material_id", fields.get("material", "")),
        lattice=np.asarray(lattice_values, dtype=float).reshape(3, 3),
        species=tuple(species),
        cart_coords=np.asarray(coordinates, dtype=float),
    )


def _default_radius(symbol: str) -> float | None:
    element = Element(symbol)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No data available for atomic_radius_calculated.*",
            category=UserWarning,
        )
        value = element.atomic_radius_calculated
    if value is None:
        value = element.atomic_radius
    return None if value is None else float(value)


def _empty_features(error: str) -> dict[str, float | bool | str]:
    result: dict[str, float | bool | str] = {
        "feature_ok": False,
        "feature_error": error,
        "volume_per_atom": math.nan,
        "packing_fraction": math.nan,
        "min_pair_distance": math.nan,
        "min_pair_ratio": math.nan,
        "pair_ratio_q05": math.nan,
        "pair_ratio_median": math.nan,
        "repulsion_p2_per_atom": math.nan,
    }
    for scale in SCALE_GRID:
        suffix = f"l{int(round(scale * 100)):03d}"
        result[f"repulsion_p2_{suffix}"] = math.nan
        result[f"packing_{suffix}"] = math.nan
    return result


def geometry_features(
    frame: ParsedExtXYZ,
    *,
    radii: Mapping[str, float] | None = None,
) -> dict[str, float | bool | str]:
    """Compute packing and short-range Born-like overlap features.

    All quantities use only the supplied frame.  Larger repulsion means a more
    severe overlap.  Missing radii fail open via ``feature_ok=False`` so the
    caller can abstain.
    """

    radius_by_symbol: dict[str, float] = {}
    for symbol in sorted(set(frame.species)):
        value = radii.get(symbol) if radii is not None else _default_radius(symbol)
        if value is None or not np.isfinite(value) or value <= 0:
            return _empty_features(f"missing_radius:{symbol}")
        radius_by_symbol[symbol] = float(value)

    n_atoms = len(frame.species)
    if n_atoms < 2:
        return _empty_features("fewer_than_two_sites")
    lattice = Lattice(frame.lattice)
    volume = float(lattice.volume)
    if not np.isfinite(volume) or volume <= 0:
        return _empty_features("invalid_volume")

    fractional = lattice.get_fractional_coords(frame.cart_coords)
    distance_matrix = lattice.get_all_distances(fractional, fractional)
    left, right = np.triu_indices(n_atoms, k=1)
    distances = distance_matrix[left, right]
    site_radii = np.asarray([radius_by_symbol[s] for s in frame.species], dtype=float)
    radius_sums = site_radii[left] + site_radii[right]
    ratios = distances / radius_sums
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        return _empty_features("invalid_pair_distance")

    packing = float((4.0 * math.pi / 3.0) * np.sum(site_radii**3) / volume)
    result: dict[str, float | bool | str] = {
        "feature_ok": True,
        "feature_error": "",
        "volume_per_atom": volume / n_atoms,
        "packing_fraction": packing,
        "min_pair_distance": float(np.min(distances)),
        "min_pair_ratio": float(np.min(ratios)),
        "pair_ratio_q05": float(np.quantile(ratios, 0.05)),
        "pair_ratio_median": float(np.median(ratios)),
        "repulsion_p2_per_atom": float(np.sum(np.maximum(1.0 / ratios - 1.0, 0.0) ** 2) / n_atoms),
    }
    for scale in SCALE_GRID:
        suffix = f"l{int(round(scale * 100)):03d}"
        scaled_ratios = scale * ratios
        result[f"repulsion_p2_{suffix}"] = float(
            np.sum(np.maximum(1.0 / scaled_ratios - 1.0, 0.0) ** 2) / n_atoms
        )
        result[f"packing_{suffix}"] = packing / scale**3
    return result


__all__ = ["ParsedExtXYZ", "SCALE_GRID", "geometry_features", "parse_extxyz"]
