#!/usr/bin/env python3
"""Topology-only a-priori bond strength versus raw local bond-length order."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import math
from pathlib import Path
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next425-apriori-bond-strength-length-order-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next425-next429-apriori-bond-strength-length-order.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("aprbs_length_order_protection",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
CHARGE_TOLERANCE = 1.0e-8
MARGINAL_TOLERANCE = 1.0e-10
INFORMATIVE_TOLERANCE = 1.0e-15
MAXIMUM_ITERATIONS = 20_000
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class APRBSResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    pair_count: int
    informative_weight: float
    violation_weight: float
    maximum_marginal_residual: float
    iterations: int
    edge_strengths: tuple[float, ...]
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> APRBSResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return APRBSResult(
        False, reason, 0, 0, 0, math.nan, math.nan, math.nan, 0, (), None, {}
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _validated_problem(
    *, charges: object, endpoints: object, distances: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    charge = np.asarray(charges, dtype=float)
    raw_pair = np.asarray(endpoints)
    distance = np.asarray(distances, dtype=float)
    if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
        raise ValueError("APRBS charges differ")
    magnitude = float(np.abs(charge).sum())
    if (
        magnitude <= 0.0
        or abs(float(charge.sum())) > CHARGE_TOLERANCE * max(1.0, magnitude)
        or np.any(charge == 0.0)
        or not np.any(charge > 0.0)
        or not np.any(charge < 0.0)
    ):
        raise ValueError("APRBS formal charges must be neutral and nonzero")
    if (
        raw_pair.ndim != 2
        or raw_pair.shape[1:] != (2,)
        or len(raw_pair) < 1
        or not np.isfinite(raw_pair.astype(float)).all()
        or not np.equal(raw_pair.astype(float), np.rint(raw_pair.astype(float))).all()
    ):
        raise ValueError("APRBS endpoint population differs")
    pair = raw_pair.astype(int)
    if (
        np.any(pair < 0)
        or np.any(pair >= len(charge))
        or np.any(pair[:, 0] == pair[:, 1])
        or not np.all(charge[pair[:, 0]] > 0.0)
        or not np.all(charge[pair[:, 1]] < 0.0)
    ):
        raise ValueError("APRBS cation-anion edge orientation differs")
    if (
        distance.shape != (len(pair),)
        or not np.isfinite(distance).all()
        or np.any(distance <= 0.0)
    ):
        raise ValueError("APRBS edge lengths differ")
    groups = tuple(np.flatnonzero((pair[:, 0] == site) | (pair[:, 1] == site)) for site in range(len(charge)))
    if any(len(group) < 1 for group in groups):
        raise ValueError("APRBS periodic graph has an isolated charged site")

    adjacency: list[set[int]] = [set() for _ in range(len(charge))]
    for left, right in pair:
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    unseen = set(range(len(charge)))
    while unseen:
        stack = [unseen.pop()]
        component: list[int] = []
        while stack:
            site = stack.pop()
            component.append(site)
            fresh = adjacency[site] & unseen
            unseen.difference_update(fresh)
            stack.extend(sorted(fresh, reverse=True))
        component_charge = charge[np.asarray(component, dtype=int)]
        supply = float(component_charge[component_charge > 0.0].sum())
        demand = float(-component_charge[component_charge < 0.0].sum())
        if not math.isclose(
            supply,
            demand,
            rel_tol=MARGINAL_TOLERANCE,
            abs_tol=MARGINAL_TOLERANCE * max(1.0, supply, demand),
        ):
            raise ValueError("APRBS connected charge marginals are infeasible")
    return charge, pair, distance, groups


def _maximum_entropy_field(
    charge: np.ndarray, pair: np.ndarray, groups: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, int, float]:
    """Scale a uniform multigraph measure to exact site-charge marginals."""

    log_strength = np.zeros(len(pair), dtype=float)
    targets = np.abs(charge)
    positive_sites = tuple(int(site) for site in np.flatnonzero(charge > 0.0))
    negative_sites = tuple(int(site) for site in np.flatnonzero(charge < 0.0))
    residual = math.inf
    strengths = np.ones(len(pair), dtype=float)
    for iteration in range(1, MAXIMUM_ITERATIONS + 1):
        for sites in (positive_sites, negative_sites):
            for site in sites:
                selected = groups[site]
                values = log_strength[selected]
                maximum = float(np.max(values))
                log_total = maximum + math.log(
                    math.fsum(np.exp(values - maximum).tolist())
                )
                log_strength[selected] += math.log(float(targets[site])) - log_total
        strengths = np.exp(log_strength)
        if not np.isfinite(strengths).all() or np.any(strengths <= 0.0):
            raise RuntimeError("APRBS maximum-entropy field lost positivity")
        residual = 0.0
        for site, selected in enumerate(groups):
            observed = math.fsum(strengths[selected].tolist())
            residual = max(
                residual, abs(observed - float(targets[site])) / float(targets[site])
            )
        if residual <= MARGINAL_TOLERANCE:
            return strengths, iteration, float(residual)
    raise ValueError(
        f"APRBS maximum-entropy field did not converge: residual={residual:.12g}"
    )


def apriori_length_order_protection(
    *,
    charges: Sequence[float] | object,
    endpoints: Sequence[Sequence[int]] | object,
    distances: Sequence[float] | object,
) -> APRBSResult:
    """Compare topology-only conserved edge-strength order with local lengths."""

    try:
        charge, pair, distance, groups = _validated_problem(
            charges=charges, endpoints=endpoints, distances=distances
        )
        strengths, iterations, residual = _maximum_entropy_field(
            charge, pair, groups
        )
        products: list[float] = []
        for selected in groups:
            for left, right in itertools.combinations(selected.tolist(), 2):
                strength_contrast = float(
                    (strengths[left] - strengths[right])
                    / (strengths[left] + strengths[right])
                )
                length_contrast = float(
                    (distance[left] - distance[right])
                    / (distance[left] + distance[right])
                )
                products.append(strength_contrast * length_contrast)
        if not products:
            raise ValueError("APRBS graph has no local edge pair")
        informative = math.fsum(abs(value) for value in products)
        violation = math.fsum(max(value, 0.0) for value in products)
        protection = (
            1.0 - violation / informative
            if informative > INFORMATIVE_TOLERANCE
            else 0.5
        )
        protection = _quantize(float(np.clip(protection, 0.0, 1.0)))
        if (
            not math.isfinite(informative)
            or not math.isfinite(violation)
            or violation < 0.0
            or violation > informative + 1.0e-12
            or not math.isfinite(protection)
            or protection < 0.0
            or protection > 1.0
        ):
            raise RuntimeError("APRBS length-order statistic differs")
        return APRBSResult(
            True,
            None,
            len(charge),
            len(pair),
            len(products),
            float(informative),
            float(violation),
            float(residual),
            int(iterations),
            tuple(float(value) for value in strengths),
            None,
            {FEATURE_NAMES[0]: protection},
        )
    except Exception as exc:
        return _failure(exc)


def compute_aprbs_features(atoms: Atoms) -> APRBSResult:
    """Compute APRBS from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "NEXT425 valence assignment failed")
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT425 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            raise ValueError(geometry.failure_reason or "APRBS periodic graph failed")
        edges = tuple(geometry.edges)
        result = apriori_length_order_protection(
            charges=charge,
            endpoints=tuple((int(edge.cation), int(edge.anion)) for edge in edges),
            distances=tuple(float(edge.distance) for edge in edges),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason="NEXT425 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_aprbs_row(atoms: Atoms) -> dict[str, object]:
    result = compute_aprbs_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "aprbs_supported": bool(result.supported),
        "aprbs_failure": result.failure_reason,
        "aprbs_site_count": int(result.site_count),
        "aprbs_edge_count": int(result.edge_count),
        "aprbs_pair_count": int(result.pair_count),
        "aprbs_informative_weight": float(result.informative_weight),
        "aprbs_violation_weight": float(result.violation_weight),
        "aprbs_maximum_marginal_residual": float(result.maximum_marginal_residual),
        "aprbs_iterations": int(result.iterations),
        "aprbs_valence_policy": result.valence_policy,
    }


__all__ = [
    "APRBSResult",
    "BOUNDARY_FLAGS",
    "CHARGE_TOLERANCE",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "INFORMATIVE_TOLERANCE",
    "MARGINAL_TOLERANCE",
    "MAXIMUM_ITERATIONS",
    "OUTPUT_GRID",
    "PROTOCOL",
    "apriori_length_order_protection",
    "compute_aprbs_features",
    "compute_aprbs_row",
]
