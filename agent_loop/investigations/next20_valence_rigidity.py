#!/usr/bin/env python3
"""Pure-analytic scale-invariant valence-rigidity descriptors.

The public kernel consumes one edge system at a time.  It has no dataset,
label, endpoint, relaxed-structure, or learned-potential input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import warnings

import numpy as np


PROTOCOL = "2026-08-02-next20-scale-invariant-valence-rigidity-v1"
FEATURE_NAMES = (
    "sivr_scale_log_median",
    "sivr_edge_mismatch_rms",
    "sivr_edge_mismatch_q95",
    "sivr_edge_mismatch_max",
    "sivr_site_imbalance_rms",
    "sivr_site_imbalance_max",
    "sivr_cell_hydro_abs",
    "sivr_cell_anisotropy",
    "sivr_stiffness_min",
    "sivr_negative_mode_fraction",
    "sivr_soft_mode_fraction",
    "sivr_edge_count",
    "sivr_site_count",
)


@dataclass(frozen=True)
class RigidityFeatureResult:
    """Fail-open result for one independent analytic edge system."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> RigidityFeatureResult:
    return RigidityFeatureResult(False, reason, {})


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Return a deterministic midpoint-CDF weighted quantile."""

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / np.sum(
        sorted_weights
    )
    return float(
        np.interp(
            float(quantile),
            positions,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    )


def rigidity_features_from_edges(
    *,
    n_sites: int,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
    radius_sums: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    soft_tolerance: float = 1.0e-6,
    negative_tolerance: float = 1.0e-8,
) -> RigidityFeatureResult:
    """Compute SIVR descriptors from one fixed periodic contact graph.

    The single weighted-median centering is a dimensionless normalization.  It
    neither changes coordinates nor searches for an alternative structure.
    """

    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        return _failure("edge system needs at least two sites")
    n_sites = int(n_sites)
    pair = np.asarray(endpoints, dtype=int)
    displacement = np.asarray(vectors, dtype=float)
    radii = np.asarray(radius_sums, dtype=float)
    edge_weight = np.asarray(weights, dtype=float)
    if pair.ndim != 2 or pair.shape[1:] != (2,):
        return _failure("endpoints must have shape (n_edges, 2)")
    n_edges = len(pair)
    if n_edges == 0:
        return _failure("edge system has no contacts")
    if displacement.shape != (n_edges, 3):
        return _failure("vectors must have shape (n_edges, 3)")
    if radii.shape != (n_edges,):
        return _failure("radius sums must have one value per edge")
    if edge_weight.shape != (n_edges,):
        return _failure("weights must have one value per edge")
    if np.any(pair < 0) or np.any(pair >= n_sites) or np.any(pair[:, 0] == pair[:, 1]):
        return _failure("endpoints contain invalid site indices")
    if not np.isfinite(displacement).all():
        return _failure("vectors contain non-finite values")
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        return _failure("radius sums must be finite and positive")
    if not np.isfinite(edge_weight).all() or np.any(edge_weight <= 0):
        return _failure("weights must be finite and positive")
    if not np.isfinite(soft_tolerance) or soft_tolerance <= 0:
        return _failure("soft tolerance must be finite and positive")
    if not np.isfinite(negative_tolerance) or negative_tolerance <= 0:
        return _failure("negative tolerance must be finite and positive")

    distance = np.linalg.norm(displacement, axis=1)
    if not np.isfinite(distance).all() or np.any(distance <= 0):
        return _failure("edge distances must be finite and positive")
    direction = displacement / distance[:, None]
    log_ratio = np.log(distance / radii)
    scale = _weighted_quantile(log_ratio, edge_weight, 0.5)
    residual = log_ratio - scale
    absolute = np.abs(residual)
    weight_sum = float(edge_weight.sum())

    mismatch_rms = float(np.sqrt(np.sum(edge_weight * residual**2) / weight_sum))
    mismatch_q95 = _weighted_quantile(absolute, edge_weight, 0.95)
    mismatch_max = float(absolute.max())

    imbalance = np.zeros((n_sites, 3), dtype=float)
    site_weight = np.zeros(n_sites, dtype=float)
    edge_vector = edge_weight[:, None] * residual[:, None] * direction
    for index, (left, right) in enumerate(pair):
        imbalance[left] += edge_vector[index]
        imbalance[right] -= edge_vector[index]
        site_weight[left] += edge_weight[index]
        site_weight[right] += edge_weight[index]
    active_sites = site_weight > 0
    normalized_imbalance = np.zeros_like(imbalance)
    normalized_imbalance[active_sites] = (
        imbalance[active_sites] / site_weight[active_sites, None]
    )
    imbalance_norm = np.linalg.norm(normalized_imbalance, axis=1)
    imbalance_rms = float(np.sqrt(np.mean(imbalance_norm**2)))
    imbalance_max = float(imbalance_norm.max())

    cell_tensor = np.einsum(
        "e,ei,ej->ij", edge_weight * residual, direction, direction
    ) / weight_sum
    hydro = float(np.trace(cell_tensor) / 3.0)
    deviator = cell_tensor - hydro * np.eye(3)
    anisotropy = float(np.linalg.norm(deviator, ord="fro"))

    hessian = np.zeros((3 * n_sites, 3 * n_sites), dtype=float)
    identity = np.eye(3)
    for index, (left, right) in enumerate(pair):
        e_value = float(residual[index])
        unit = direction[index]
        block = (
            edge_weight[index]
            * np.exp(-2.0 * e_value)
            * (e_value * identity + (1.0 - 2.0 * e_value) * np.outer(unit, unit))
        )
        left_slice = slice(3 * left, 3 * left + 3)
        right_slice = slice(3 * right, 3 * right + 3)
        hessian[left_slice, left_slice] += block
        hessian[right_slice, right_slice] += block
        hessian[left_slice, right_slice] -= block
        hessian[right_slice, left_slice] -= block

    degree_coordinates = np.repeat(site_weight, 3)
    inverse_root = np.zeros_like(degree_coordinates)
    positive_degree = degree_coordinates > 0
    inverse_root[positive_degree] = 1.0 / np.sqrt(degree_coordinates[positive_degree])
    normalized_hessian = hessian * np.outer(inverse_root, inverse_root)
    eigenvalues = np.linalg.eigvalsh(normalized_hessian)
    if not np.isfinite(eigenvalues).all():
        return _failure("stiffness spectrum is non-finite")
    # A central-force network has three exact translational modes.  Remove the
    # three eigenvalues closest to zero without assuming their sorted position
    # when prestress makes other modes negative.
    keep = np.ones(len(eigenvalues), dtype=bool)
    translation = np.argsort(np.abs(eigenvalues), kind="stable")[:3]
    keep[translation] = False
    active_eigenvalues = eigenvalues[keep]
    if not len(active_eigenvalues):
        return _failure("stiffness spectrum has no non-translational modes")
    stiffness_min = float(active_eigenvalues.min())
    negative_fraction = float(np.mean(active_eigenvalues < -negative_tolerance))
    soft_fraction = float(np.mean(np.abs(active_eigenvalues) <= soft_tolerance))

    features = {
        "sivr_scale_log_median": float(scale),
        "sivr_edge_mismatch_rms": mismatch_rms,
        "sivr_edge_mismatch_q95": mismatch_q95,
        "sivr_edge_mismatch_max": mismatch_max,
        "sivr_site_imbalance_rms": imbalance_rms,
        "sivr_site_imbalance_max": imbalance_max,
        "sivr_cell_hydro_abs": abs(hydro),
        "sivr_cell_anisotropy": anisotropy,
        "sivr_stiffness_min": stiffness_min,
        "sivr_negative_mode_fraction": negative_fraction,
        "sivr_soft_mode_fraction": soft_fraction,
        "sivr_edge_count": float(n_edges),
        "sivr_site_count": float(n_sites),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        return _failure("computed feature schema is invalid")
    return RigidityFeatureResult(True, None, features)


def _tabulated_radius(symbol: str) -> float | None:
    """Return the same frozen elemental-radius policy used by NEXT6."""

    from pymatgen.core.periodic_table import Element

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
    if value is None:
        return None
    radius = float(value)
    return radius if np.isfinite(radius) and radius > 0 else None


def compute_valence_rigidity_features(
    structure,
    *,
    graph_mode: str,
    charge_weight_exponent: float,
) -> RigidityFeatureResult:
    """Compute SIVR for one raw structure without modifying the input."""

    if graph_mode not in {"crystalnn", "voronoi"}:
        return _failure("unsupported graph mode")
    if charge_weight_exponent not in {0.0, 0.5}:
        return _failure("unsupported charge-weight exponent")
    if len(structure) < 2:
        return _failure("structure needs at least two sites")
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
    charges = np.asarray(assignment.values, dtype=float)
    try:
        geometry = build_periodic_edge_geometry(
            structure,
            charges,
            graph_mode=graph_mode,
        )
    except Exception as exc:
        return _failure(f"periodic graph failed: {type(exc).__name__}")
    if not geometry.supported:
        return _failure(geometry.failure_reason or "periodic graph is unsupported")

    return rigidity_features_from_periodic_geometry(
        structure,
        charges,
        geometry,
        charge_weight_exponent=charge_weight_exponent,
    )


def rigidity_features_from_periodic_geometry(
    structure,
    charges: Sequence[float] | np.ndarray,
    geometry,
    *,
    charge_weight_exponent: float,
) -> RigidityFeatureResult:
    """Reweight one cached NEXT19 graph and evaluate the SIVR kernel."""

    if charge_weight_exponent not in {0.0, 0.5}:
        return _failure("unsupported charge-weight exponent")
    charge_array = np.asarray(charges, dtype=float)
    if charge_array.shape != (len(structure),) or not np.isfinite(charge_array).all():
        return _failure("charges must match the finite structure sites")
    if geometry is None or not getattr(geometry, "supported", False):
        return _failure(
            getattr(geometry, "failure_reason", None)
            or "periodic graph is unsupported"
        )

    radii_by_site: list[float] = []
    for site in structure:
        try:
            radius = _tabulated_radius(site.specie.symbol)
        except Exception as exc:
            return _failure(f"tabulated radius failed: {type(exc).__name__}")
        if radius is None:
            return _failure(f"tabulated radius is missing for {site.specie.symbol}")
        radii_by_site.append(radius)

    endpoints: list[tuple[int, int]] = []
    vectors: list[np.ndarray] = []
    radius_sums: list[float] = []
    weights: list[float] = []
    for edge in geometry.edges:
        left = int(edge.cation)
        right = int(edge.anion)
        image = np.asarray(edge.image, dtype=float)
        fractional = (
            np.asarray(structure[right].frac_coords, dtype=float)
            + image
            - np.asarray(structure[left].frac_coords, dtype=float)
        )
        vector = np.asarray(
            structure.lattice.get_cartesian_coords(fractional), dtype=float
        )
        charge_product = abs(float(charge_array[left] * charge_array[right]))
        weight = float(edge.neighbor_weight) * charge_product**charge_weight_exponent
        endpoints.append((left, right))
        vectors.append(vector)
        radius_sums.append(radii_by_site[left] + radii_by_site[right])
        weights.append(weight)
    return rigidity_features_from_edges(
        n_sites=len(structure),
        endpoints=np.asarray(endpoints, dtype=int),
        vectors=np.asarray(vectors, dtype=float),
        radius_sums=np.asarray(radius_sums, dtype=float),
        weights=np.asarray(weights, dtype=float),
    )


__all__ = [
    "FEATURE_NAMES",
    "PROTOCOL",
    "RigidityFeatureResult",
    "compute_valence_rigidity_features",
    "rigidity_features_from_periodic_geometry",
    "rigidity_features_from_edges",
]
