#!/usr/bin/env python3
"""Closed-form finite-budget bond-valence transport path on one raw x0.

This module solves only dimensionless linear algebra.  It never changes an
atomic coordinate or cell and does not consume an endpoint, relaxed geometry,
DFT quantity, or learned potential.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np


BUDGETS = (0.01, 0.03, 0.10)
MAX_SITES = 64
CERTIFICATE_METHOD = "closed_form_radial_minimum_norm_path"
FEATURE_NAMES = (
    "bvtbd_unbounded_residual_fraction",
    "bvtbd_required_linf_budget",
    "bvtbd_minimum_motion_rms",
    "bvtbd_atomic_motion_max",
    "bvtbd_cell_strain_frobenius",
    "bvtbd_residual_fraction_tau01",
    "bvtbd_residual_fraction_tau03",
    "bvtbd_residual_fraction_tau10",
    "bvtbd_deformation_debt_tau01",
    "bvtbd_deformation_debt_tau03",
    "bvtbd_deformation_debt_tau10",
)


@dataclass(frozen=True)
class BoundedTransportResult:
    """Fail-open result for one bounded analytic transport certificate."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> BoundedTransportResult:
    return BoundedTransportResult(False, reason, {})


def _budget_suffix(budget: float) -> str:
    return f"tau{int(round(100.0 * budget)):02d}"


def bounded_transport_budget_certificate(
    *,
    correction: Sequence[float] | np.ndarray,
    jacobian: Sequence[Sequence[float]] | np.ndarray,
    n_sites: int,
    characteristic_length: float,
) -> BoundedTransportResult:
    """Measure residual along a frozen minimum-norm correction path.

    Atomic generalized coordinates are normalized by ``characteristic_length``;
    the final six coordinates are the dimensionless symmetric strain tensor in
    ``xx, yy, zz, yz, xz, xy`` order. Each finite budget uniformly truncates
    the unbounded minimum-norm solution so its largest absolute generalized
    coordinate equals the budget. This is not a box-constrained global solve.
    """

    try:
        if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
            return _failure("n_sites must be an integer of at least two")
        n_sites = int(n_sites)
        if n_sites > MAX_SITES:
            return _failure(f"site cap exceeded: {n_sites} > {MAX_SITES}")
        target = np.asarray(correction, dtype=float)
        response = np.asarray(jacobian, dtype=float)
        length = float(characteristic_length)
        if target.ndim != 1 or len(target) < 1 or not np.isfinite(target).all():
            return _failure("correction must be a nonempty finite vector")
        if response.shape != (len(target), 3 * n_sites + 6):
            return _failure("Jacobian shape must match edges and generalized coordinates")
        if not np.isfinite(response).all():
            return _failure("Jacobian must be finite")
        if not math.isfinite(length) or length <= 0.0:
            return _failure("characteristic length must be finite and positive")

        target_norm = float(np.linalg.norm(target))
        if target_norm == 0.0:
            return BoundedTransportResult(
                True,
                None,
                {name: 0.0 for name in FEATURE_NAMES},
            )

        dimensionless = response.copy()
        dimensionless[:, : 3 * n_sites] *= length
        # Normalize the complete least-squares system so solver termination is
        # relative to the correction rather than its arbitrary valence scale.
        normalized_response = dimensionless / target_norm
        normalized_target = target / target_norm
        minimum_motion, _residuals, _rank, _singular = np.linalg.lstsq(
            normalized_response,
            normalized_target,
            rcond=None,
        )
        compatible_response = normalized_response @ minimum_motion
        unbounded_residual = normalized_target - compatible_response
        floor = float(np.linalg.norm(unbounded_residual))
        floor = float(np.clip(floor, 0.0, 1.0))
        required_linf_budget = float(np.max(np.abs(minimum_motion)))

        atomic = minimum_motion[: 3 * n_sites].reshape(n_sites, 3)
        cell = minimum_motion[3 * n_sites :]
        atomic_max = float(np.max(np.linalg.norm(atomic, axis=1)))
        cell_frobenius = float(
            math.sqrt(
                float(np.sum(cell[:3] ** 2))
                + 2.0 * float(np.sum(cell[3:] ** 2))
            )
        )
        features: dict[str, float] = {
            "bvtbd_unbounded_residual_fraction": floor,
            "bvtbd_required_linf_budget": required_linf_budget,
            "bvtbd_minimum_motion_rms": float(
                np.linalg.norm(minimum_motion) / math.sqrt(len(minimum_motion))
            ),
            "bvtbd_atomic_motion_max": atomic_max,
            "bvtbd_cell_strain_frobenius": cell_frobenius,
        }

        previous = math.inf
        radial_residuals: dict[str, float] = {}
        debts: dict[str, float] = {}
        for budget in BUDGETS:
            alpha = (
                1.0
                if required_linf_budget <= float(budget)
                else float(budget) / required_linf_budget
            )
            residual = float(
                np.linalg.norm(
                    unbounded_residual + (1.0 - alpha) * compatible_response
                )
            )
            residual = float(np.clip(residual, 0.0, 1.0))
            if residual > previous + 1.0e-9:
                return _failure("bounded residual is not monotone with budget")
            previous = residual
            suffix = _budget_suffix(float(budget))
            radial_residuals[f"bvtbd_residual_fraction_{suffix}"] = residual
            debts[f"bvtbd_deformation_debt_{suffix}"] = float(
                math.sqrt(max(0.0, residual * residual - floor * floor))
            )
        features.update(radial_residuals)
        features.update(debts)
        if tuple(features) != FEATURE_NAMES or not np.isfinite(
            list(features.values())
        ).all():
            return _failure("bounded transport feature schema or values differ")
        return BoundedTransportResult(True, None, features)
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def bounded_bond_valence_transport_features(
    *,
    charges: Sequence[float] | np.ndarray,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
    strengths: Sequence[float] | np.ndarray,
    decays: Sequence[float] | np.ndarray,
) -> BoundedTransportResult:
    """Build the NEXT38 correction/Jacobian and apply finite motion budgets."""

    try:
        charge = np.asarray(charges, dtype=float)
        pair = np.asarray(endpoints, dtype=int)
        vector = np.asarray(vectors, dtype=float)
        strength = np.asarray(strengths, dtype=float)
        decay = np.asarray(decays, dtype=float)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            return _failure("charges must be a finite site vector")
        n_sites = len(charge)
        if n_sites > MAX_SITES:
            return _failure(f"site cap exceeded: {n_sites} > {MAX_SITES}")
        magnitude = float(np.abs(charge).sum())
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            return _failure("charges need both signs")
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            return _failure("charges must be neutral")
        if pair.ndim != 2 or pair.shape[1:] != (2,) or len(pair) < 1:
            return _failure("endpoints must have nonempty shape (E,2)")
        n_edges = len(pair)
        if vector.shape != (n_edges, 3):
            return _failure("vectors must have shape (E,3)")
        if strength.shape != (n_edges,) or decay.shape != (n_edges,):
            return _failure("strengths and decays must match all edges")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            return _failure("endpoints contain invalid site indices")
        if not np.all(charge[pair[:, 0]] > 0.0) or not np.all(
            charge[pair[:, 1]] < 0.0
        ):
            return _failure("edges must be ordered cation to anion")
        if not np.isfinite(vector).all():
            return _failure("edge vectors must be finite")
        if not np.isfinite(strength).all() or np.any(strength <= 0.0):
            return _failure("bond strengths must be finite and positive")
        if not np.isfinite(decay).all() or np.any(decay <= 0.0):
            return _failure("bond decays must be finite and positive")
        distance = np.linalg.norm(vector, axis=1)
        if not np.isfinite(distance).all() or np.any(distance <= 0.0):
            return _failure("edge distances must be finite and positive")
        direction = vector / distance[:, None]

        prior = np.zeros(n_edges, dtype=float)
        for cation in np.flatnonzero(charge > 0.0):
            selected = pair[:, 0] == cation
            if not selected.any():
                return _failure("cation has no bond-valence edge")
            normalizer = float(strength[selected].sum())
            if not math.isfinite(normalizer) or normalizer <= 0.0:
                return _failure("cation bond-valence normalizer is invalid")
            prior[selected] = float(charge[cation]) * strength[selected] / normalizer

        distance_jacobian = np.zeros((n_edges, 3 * n_sites + 6), dtype=float)
        rows = np.arange(n_edges)
        for axis in range(3):
            distance_jacobian[rows, 3 * pair[:, 0] + axis] -= direction[:, axis]
            distance_jacobian[rows, 3 * pair[:, 1] + axis] += direction[:, axis]
        nx, ny, nz = direction.T
        distance_jacobian[:, -6:] = distance[:, None] * np.column_stack(
            (nx**2, ny**2, nz**2, 2.0 * ny * nz, 2.0 * nx * nz, 2.0 * ny * nx)
        )
        log_strength_jacobian = -distance_jacobian / decay[:, None]
        normalized_jacobian = np.zeros_like(log_strength_jacobian)
        for cation in np.flatnonzero(charge > 0.0):
            selected = pair[:, 0] == cation
            weights = prior[selected] / float(charge[cation])
            star_mean = weights @ log_strength_jacobian[selected]
            normalized_jacobian[selected] = prior[selected, None] * (
                log_strength_jacobian[selected] - star_mean[None, :]
            )

        incidence = np.zeros((n_sites, n_edges), dtype=float)
        columns = np.arange(n_edges)
        incidence[pair[:, 0], columns] = 1.0
        incidence[pair[:, 1], columns] = 1.0
        site_deficit = np.abs(charge) - incidence @ prior
        correction, _residuals, _rank, _singular = np.linalg.lstsq(
            incidence,
            site_deficit,
            rcond=None,
        )
        conservation_residual = site_deficit - incidence @ correction
        conservation_scale = max(1.0, float(np.linalg.norm(site_deficit)))
        if float(np.linalg.norm(conservation_residual)) > 1.0e-8 * conservation_scale:
            return _failure("periodic graph cannot carry the site-valence correction")
        exact_zero_scale = max(
            1.0,
            float(np.linalg.norm(prior)),
            float(np.linalg.norm(np.abs(charge))),
        )
        if float(np.linalg.norm(site_deficit)) <= 1.0e-12 * exact_zero_scale:
            correction = np.zeros_like(correction)
        return bounded_transport_budget_certificate(
            correction=correction,
            jacobian=normalized_jacobian,
            n_sites=n_sites,
            characteristic_length=float(np.median(distance)),
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def compute_bounded_bond_valence_transport_features(
    structure,
    charges: Sequence[float] | np.ndarray,
) -> BoundedTransportResult:
    """Resolve the frozen NEXT38 graph/parameters and evaluate NEXT119."""

    try:
        from src.advanced_local_features import resolve_bond_valence_parameter
        from src.elec_feat import bv_table
        from src.next19_valence_transport import build_periodic_edge_geometry
        from src.next20_valence_rigidity import _tabulated_radius
        from src.next22_bond_valence_equilibrium import PARAMETER_SOURCES

        if len(structure) > MAX_SITES:
            return _failure(f"site cap exceeded: {len(structure)} > {MAX_SITES}")
        charge = np.asarray(charges, dtype=float)
        if charge.shape != (len(structure),) or not np.isfinite(charge).all():
            return _failure("charges must be finite and match all sites")
        magnitude = float(np.abs(charge).sum())
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            return _failure("charges must be neutral")
        geometry = build_periodic_edge_geometry(structure, charge, graph_mode="voronoi")
        if not geometry.supported:
            return _failure(geometry.failure_reason or "periodic graph is unsupported")
        parameters = bv_table()
        endpoints: list[tuple[int, int]] = []
        vectors: list[np.ndarray] = []
        strengths: list[float] = []
        decays: list[float] = []
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
                parameters,
                policy="frozen-fallback",
            )
            if resolved is None:
                left_radius = _tabulated_radius(structure[left].specie.symbol)
                right_radius = _tabulated_radius(structure[right].specie.symbol)
                if left_radius is None or right_radius is None:
                    return _failure("bond-valence and radius-generic parameters are missing")
                resolved = (left_radius + right_radius, 0.37, "radius_generic")
            r0, decay, source = resolved
            if (
                not np.isfinite(r0)
                or not np.isfinite(decay)
                or float(decay) <= 0.0
                or source not in PARAMETER_SOURCES
            ):
                return _failure("bond-valence parameter is invalid")
            fractional = (
                np.asarray(structure[right].frac_coords, dtype=float)
                + np.asarray(edge.image, dtype=float)
                - np.asarray(structure[left].frac_coords, dtype=float)
            )
            displacement = np.asarray(
                structure.lattice.get_cartesian_coords(fractional),
                dtype=float,
            )
            distance = float(np.linalg.norm(displacement))
            try:
                strength = math.exp((float(r0) - distance) / float(decay))
            except OverflowError:
                return _failure("bond strength overflowed")
            if not np.isfinite(strength) or strength <= 0.0:
                return _failure("bond strength is invalid")
            endpoints.append((left, right))
            vectors.append(displacement)
            strengths.append(float(strength))
            decays.append(float(decay))
        return bounded_bond_valence_transport_features(
            charges=charge,
            endpoints=np.asarray(endpoints, dtype=int),
            vectors=np.asarray(vectors, dtype=float),
            strengths=np.asarray(strengths, dtype=float),
            decays=np.asarray(decays, dtype=float),
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


__all__ = [
    "BUDGETS",
    "CERTIFICATE_METHOD",
    "FEATURE_NAMES",
    "MAX_SITES",
    "BoundedTransportResult",
    "bounded_bond_valence_transport_features",
    "bounded_transport_budget_certificate",
    "compute_bounded_bond_valence_transport_features",
]
