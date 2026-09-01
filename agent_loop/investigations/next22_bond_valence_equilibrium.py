#!/usr/bin/env python3
"""Scale-calibrated analytic bond-valence equilibrium descriptors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


PROTOCOL = "2026-08-02-next22-scale-calibrated-bond-valence-v2"
FEATURE_NAMES = (
    "scbv_mismatch_rms",
    "scbv_mismatch_q95",
    "scbv_mismatch_max",
    "scbv_cation_mismatch_rms",
    "scbv_anion_mismatch_rms",
    "scbv_vector_asymmetry_rms",
    "scbv_vector_asymmetry_max",
    "scbv_effective_cn_mean",
    "scbv_effective_cn_min",
    "scbv_isolated_site_fraction",
    "scbv_parameter_exact_fraction",
    "scbv_parameter_generic_fraction",
    "scbv_global_scale",
    "scbv_edge_count",
    "scbv_site_count",
)
PARAMETER_SOURCES = (
    "exact",
    "nearest_valence",
    "brown_generic",
    "radius_generic",
)


@dataclass(frozen=True)
class BondValenceFeatureResult:
    """Fail-open result for one independent bond-valence graph."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> BondValenceFeatureResult:
    return BondValenceFeatureResult(False, reason, {})


def scale_calibrated_bond_valence_features(
    *,
    charges: Sequence[float] | np.ndarray,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    strengths: Sequence[float] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
    parameter_sources: Sequence[str],
) -> BondValenceFeatureResult:
    """Aggregate a fixed bond graph after one closed-form amplitude scaling."""

    charge = np.asarray(charges, dtype=float)
    pair = np.asarray(endpoints, dtype=int)
    bond = np.asarray(strengths, dtype=float)
    displacement = np.asarray(vectors, dtype=float)
    if charge.ndim != 1 or len(charge) < 2:
        return _failure("charges must describe at least two sites")
    n_sites = len(charge)
    if not np.isfinite(charge).all():
        return _failure("charges must be finite")
    if not np.any(charge > 0) or not np.any(charge < 0):
        return _failure("charges need both signs")
    if abs(float(charge.sum())) > 1.0e-8 * max(1.0, float(np.abs(charge).sum())):
        return _failure("charges must be neutral")
    if pair.ndim != 2 or pair.shape[1:] != (2,) or len(pair) == 0:
        return _failure("endpoints must have nonempty shape (n_edges, 2)")
    n_edges = len(pair)
    if bond.shape != (n_edges,):
        return _failure("strengths must provide one value per edge")
    if displacement.shape != (n_edges, 3):
        return _failure("vectors must provide one Cartesian vector per edge")
    if len(parameter_sources) != n_edges:
        return _failure("parameter sources must provide one value per edge")
    if any(source not in PARAMETER_SOURCES for source in parameter_sources):
        return _failure("parameter source is unsupported")
    if np.any(pair < 0) or np.any(pair >= n_sites) or np.any(pair[:, 0] == pair[:, 1]):
        return _failure("endpoints contain invalid site indices")
    if not np.isfinite(bond).all() or np.any(bond <= 0):
        return _failure("bond strengths must be finite and positive")
    distance = np.linalg.norm(displacement, axis=1)
    if not np.isfinite(displacement).all() or np.any(distance <= 0):
        return _failure("bond vectors must be finite and nonzero")
    direction = displacement / distance[:, None]

    site_sum = np.zeros(n_sites, dtype=float)
    site_vector = np.zeros((n_sites, 3), dtype=float)
    site_strengths: list[list[float]] = [[] for _ in range(n_sites)]
    for index, (left, right) in enumerate(pair):
        value = float(bond[index])
        vector = value * direction[index]
        site_sum[left] += value
        site_sum[right] += value
        site_vector[left] += vector
        site_vector[right] -= vector
        site_strengths[left].append(value)
        site_strengths[right].append(value)

    target = np.abs(charge)
    denominator = float(np.dot(site_sum, site_sum))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return _failure("site bond-valence sums are degenerate")
    global_scale = float(np.dot(site_sum, target) / denominator)
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        return _failure("global bond-valence scale is invalid")
    charge_rms = float(np.sqrt(np.mean(target**2)))
    if not np.isfinite(charge_rms) or charge_rms <= 0.0:
        return _failure("charge magnitude is degenerate")
    mismatch = (global_scale * site_sum - target) / charge_rms
    absolute = np.abs(mismatch)

    isolated = site_sum <= 0.0
    asymmetry = np.ones(n_sites, dtype=float)
    active = ~isolated
    asymmetry[active] = np.linalg.norm(site_vector[active], axis=1) / site_sum[active]
    effective_cn = np.zeros(n_sites, dtype=float)
    for index, values in enumerate(site_strengths):
        if not values:
            continue
        probabilities = np.asarray(values, dtype=float)
        probabilities /= probabilities.sum()
        effective_cn[index] = float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        )

    cation = charge > 0
    anion = charge < 0
    sources = np.asarray(parameter_sources, dtype=object)
    features = {
        "scbv_mismatch_rms": float(np.sqrt(np.mean(mismatch**2))),
        "scbv_mismatch_q95": float(np.quantile(absolute, 0.95)),
        "scbv_mismatch_max": float(absolute.max()),
        "scbv_cation_mismatch_rms": float(np.sqrt(np.mean(mismatch[cation] ** 2))),
        "scbv_anion_mismatch_rms": float(np.sqrt(np.mean(mismatch[anion] ** 2))),
        "scbv_vector_asymmetry_rms": float(np.sqrt(np.mean(asymmetry**2))),
        "scbv_vector_asymmetry_max": float(asymmetry.max()),
        "scbv_effective_cn_mean": float(np.mean(effective_cn)),
        "scbv_effective_cn_min": float(np.min(effective_cn)),
        "scbv_isolated_site_fraction": float(np.mean(isolated)),
        "scbv_parameter_exact_fraction": float(np.mean(sources == "exact")),
        "scbv_parameter_generic_fraction": float(
            np.mean(np.isin(sources, ("brown_generic", "radius_generic")))
        ),
        "scbv_global_scale": global_scale,
        "scbv_edge_count": float(n_edges),
        "scbv_site_count": float(n_sites),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        return _failure("computed feature schema is invalid")
    return BondValenceFeatureResult(True, None, features)


def bond_valence_features_from_periodic_geometry(
    structure,
    charges: Sequence[float] | np.ndarray,
    geometry,
    *,
    parameters=None,
) -> BondValenceFeatureResult:
    """Evaluate one cached periodic graph with frozen bond-valence parameters."""

    charge = np.asarray(charges, dtype=float)
    if charge.shape != (len(structure),) or not np.isfinite(charge).all():
        return _failure("charges must match the finite structure sites")
    if geometry is None or not getattr(geometry, "supported", False):
        return _failure(
            getattr(geometry, "failure_reason", None)
            or "periodic graph is unsupported"
        )
    try:
        from src.advanced_local_features import resolve_bond_valence_parameter
        from src.elec_feat import bv_table

        table = bv_table() if parameters is None else parameters
    except Exception as exc:
        return _failure(f"bond-valence parameter table failed: {type(exc).__name__}")

    endpoints: list[tuple[int, int]] = []
    strengths: list[float] = []
    vectors: list[np.ndarray] = []
    sources: list[str] = []
    for edge in geometry.edges:
        left = int(edge.cation)
        right = int(edge.anion)
        key = (
            structure[left].specie.symbol,
            int(round(float(charge[left]))),
            structure[right].specie.symbol,
            int(round(float(charge[right]))),
        )
        resolved = resolve_bond_valence_parameter(
            key,
            table,
            policy="frozen-fallback",
        )
        if resolved is None:
            try:
                from src.next20_valence_rigidity import _tabulated_radius

                left_radius = _tabulated_radius(structure[left].specie.symbol)
                right_radius = _tabulated_radius(structure[right].specie.symbol)
            except Exception as exc:
                return _failure(f"radius-generic parameter failed: {type(exc).__name__}")
            if left_radius is None or right_radius is None:
                return _failure(
                    "bond-valence and radius-generic parameters are missing for "
                    f"{structure[left].specie.symbol}-{structure[right].specie.symbol}"
                )
            resolved = (left_radius + right_radius, 0.37, "radius_generic")
        r0, decay, source = resolved
        if not np.isfinite(r0) or not np.isfinite(decay) or decay <= 0:
            return _failure("bond-valence parameter is invalid")
        exponent = (float(r0) - float(edge.distance)) / float(decay)
        try:
            strength = math.exp(exponent)
        except OverflowError:
            return _failure("bond strength overflowed")
        if not np.isfinite(strength) or strength <= 0:
            return _failure("bond strength is invalid")
        image = np.asarray(edge.image, dtype=float)
        fractional = (
            np.asarray(structure[right].frac_coords, dtype=float)
            + image
            - np.asarray(structure[left].frac_coords, dtype=float)
        )
        vector = np.asarray(
            structure.lattice.get_cartesian_coords(fractional), dtype=float
        )
        endpoints.append((left, right))
        strengths.append(strength)
        vectors.append(vector)
        sources.append(source)
    return scale_calibrated_bond_valence_features(
        charges=charge,
        endpoints=np.asarray(endpoints, dtype=int),
        strengths=np.asarray(strengths, dtype=float),
        vectors=np.asarray(vectors, dtype=float),
        parameter_sources=tuple(sources),
    )


def compute_scale_calibrated_bond_valence_features(
    structure,
    *,
    graph_mode: str,
) -> BondValenceFeatureResult:
    """Compute NEXT22 for one raw structure without modifying it."""

    if graph_mode not in {"crystalnn", "voronoi"}:
        return _failure("unsupported graph mode")
    try:
        from src.next19_valence_transport import (
            build_periodic_edge_geometry,
            infer_valence_assignment,
        )

        assignment = infer_valence_assignment(structure)
    except Exception as exc:
        return _failure(f"valence assignment failed: {type(exc).__name__}")
    if not assignment.supported or assignment.values is None:
        return _failure(assignment.failure_reason or "valence assignment is unsupported")
    try:
        geometry = build_periodic_edge_geometry(
            structure,
            assignment.values,
            graph_mode=graph_mode,
        )
    except Exception as exc:
        return _failure(f"periodic graph failed: {type(exc).__name__}")
    return bond_valence_features_from_periodic_geometry(
        structure,
        assignment.values,
        geometry,
    )


__all__ = [
    "FEATURE_NAMES",
    "PARAMETER_SOURCES",
    "PROTOCOL",
    "BondValenceFeatureResult",
    "bond_valence_features_from_periodic_geometry",
    "compute_scale_calibrated_bond_valence_features",
    "scale_calibrated_bond_valence_features",
]
