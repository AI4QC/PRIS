#!/usr/bin/env python3
"""Canonical morphology of Brown-free convex mixed-valence obstruction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from importlib.metadata import version
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from src.next104_convex_mixed_valence_flow import (
    MAX_SIGN_PATTERNS,
    _catalogue_digest,
    _oriented_sign_patterns,
    _state_catalogue,
)
from src.next109_convex_mixed_valence_obstruction import (
    _normalized_interval_gap,
    _opposite_sign_endpoints,
)


PROTOCOL = "2026-08-08-next112-obstruction-morphology-v1"
FEATURE_NAMES = (
    "cmvom_component_gap_site_mean",
    "cmvom_component_gap_site_rms",
    "cmvom_obstructed_site_fraction",
    "cmvom_localized_slack_severity",
    "cmvom_side_slack_asymmetry",
    "cmvom_side_slack_flexibility",
)


@dataclass(frozen=True)
class MixedValenceObstructionMorphologyResult:
    """Primary obstruction and canonical morphology for one signed graph."""

    supported: bool
    failure_reason: str | None
    min_interval_slack: float | None
    global_balance_gap: float | None
    component_balance_gap: float | None
    unserved_site_fraction: float | None
    morphology: Mapping[str, float]


@dataclass(frozen=True)
class ConvexMixedValenceObstructionMorphologyFeatureResult:
    """Auditable structure-level morphology for one frozen catalogue mode."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]
    catalogue_sha256: str
    pymatgen_version: str
    scipy_version: str


def _failure(reason: str) -> MixedValenceObstructionMorphologyResult:
    return MixedValenceObstructionMorphologyResult(
        False, reason, None, None, None, None, {}
    )


def _feature_failure(
    reason: str,
    *,
    catalogue_sha256: str,
    pymatgen_version: str,
    scipy_version: str,
) -> ConvexMixedValenceObstructionMorphologyFeatureResult:
    return ConvexMixedValenceObstructionMorphologyFeatureResult(
        False,
        reason,
        {},
        catalogue_sha256,
        pymatgen_version,
        scipy_version,
    )


def _component_morphology(
    *,
    n_sites: int,
    endpoints: np.ndarray,
    magnitude_low: np.ndarray,
    magnitude_high: np.ndarray,
    positive: np.ndarray,
    tolerance: float,
) -> tuple[float, float, float, float]:
    """Return max, site mean, site RMS and support of component gaps."""

    parent = np.arange(n_sites, dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in endpoints:
        union(int(first), int(second))
    groups: dict[int, list[int]] = {}
    for site in range(n_sites):
        groups.setdefault(find(site), []).append(site)

    weighted_sum = 0.0
    weighted_square_sum = 0.0
    obstructed_sites = 0
    maximum = 0.0
    for sites in groups.values():
        indices = np.asarray(sites, dtype=int)
        gap = _normalized_interval_gap(
            magnitude_low,
            magnitude_high,
            positive,
            indices,
        )
        weight = len(indices) / n_sites
        weighted_sum += weight * gap
        weighted_square_sum += weight * gap * gap
        if gap > tolerance:
            obstructed_sites += len(indices)
        maximum = max(maximum, gap)
    return (
        float(maximum),
        float(weighted_sum),
        float(math.sqrt(weighted_square_sum)),
        float(obstructed_sites / n_sites),
    )


def _secondary_morphology(
    *,
    inequalities: np.ndarray,
    equality: np.ndarray,
    n_edges: int,
    n_sites: int,
    positive: np.ndarray,
    primary_slack: float,
    tolerance: float,
) -> tuple[float, float, float] | str:
    """Optimize morphology scalars on the exact primary optimum face."""

    slack_start = n_edges + 1
    n_variables = slack_start + n_sites
    slack_sum = 2.0 * primary_slack
    optimum_face = np.vstack(
        [
            equality,
            np.concatenate(
                [np.zeros(slack_start, dtype=float), np.ones(n_sites, dtype=float)]
            )[None, :],
        ]
    )
    optimum_rhs = np.asarray([1.0, slack_sum], dtype=float)
    inequality_rhs = np.zeros(len(inequalities), dtype=float)

    minimax_inequalities = np.pad(inequalities, ((0, 0), (0, 1)))
    peak_constraints = np.zeros((n_sites, n_variables + 1), dtype=float)
    for site in range(n_sites):
        peak_constraints[site, slack_start + site] = 1.0
        peak_constraints[site, -1] = -1.0
    minimax_objective = np.zeros(n_variables + 1, dtype=float)
    minimax_objective[-1] = 1.0
    minimax = linprog(
        minimax_objective,
        A_ub=np.vstack([minimax_inequalities, peak_constraints]),
        b_ub=np.zeros(len(inequalities) + n_sites, dtype=float),
        A_eq=np.pad(optimum_face, ((0, 0), (0, 1))),
        b_eq=optimum_rhs,
        bounds=[(0.0, None)] * (n_variables + 1),
        method="highs",
    )
    if not minimax.success or minimax.fun is None:
        return f"minimax optimum-face solve failed: {minimax.message}"
    peak = float(minimax.fun)
    if not math.isfinite(peak) or peak <= 0.0:
        return "minimax optimum-face peak is invalid"

    positive_slack_objective = np.zeros(n_variables, dtype=float)
    positive_slack_objective[
        slack_start + np.flatnonzero(positive)
    ] = 1.0
    side_minimum = linprog(
        positive_slack_objective,
        A_ub=inequalities,
        b_ub=inequality_rhs,
        A_eq=optimum_face,
        b_eq=optimum_rhs,
        bounds=[(0.0, None)] * n_variables,
        method="highs",
    )
    side_maximum = linprog(
        -positive_slack_objective,
        A_ub=inequalities,
        b_ub=inequality_rhs,
        A_eq=optimum_face,
        b_eq=optimum_rhs,
        bounds=[(0.0, None)] * n_variables,
        method="highs",
    )
    if (
        not side_minimum.success
        or side_minimum.fun is None
        or not side_maximum.success
        or side_maximum.fun is None
    ):
        detail = side_minimum.message if not side_minimum.success else side_maximum.message
        return f"side-range optimum-face solve failed: {detail}"

    effective_support = float(np.clip(slack_sum / (n_sites * peak), 0.0, 1.0))
    positive_minimum = float(np.clip(side_minimum.fun / slack_sum, 0.0, 1.0))
    positive_maximum = float(np.clip(-side_maximum.fun / slack_sum, 0.0, 1.0))
    if positive_minimum > positive_maximum + 100.0 * tolerance:
        return "side-range optimum-face bounds are inconsistent"
    positive_maximum = max(positive_minimum, positive_maximum)
    forced_side = float(
        np.clip(
            2.0
            * max(
                0.0,
                positive_minimum - 0.5,
                0.5 - positive_maximum,
            ),
            0.0,
            1.0,
        )
    )
    flexibility = float(np.clip(positive_maximum - positive_minimum, 0.0, 1.0))
    return (
        float(primary_slack * (1.0 - effective_support)),
        float(primary_slack * forced_side),
        float(primary_slack * flexibility),
    )


def solve_obstruction_morphology(
    *,
    signed_charge_bounds: Sequence[Sequence[float]],
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    tolerance: float = 1.0e-9,
) -> MixedValenceObstructionMorphologyResult:
    """Return canonical component and optimum-face obstruction morphology."""

    interval = np.asarray(signed_charge_bounds, dtype=float)
    pair = np.asarray(endpoints, dtype=int)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if (
        interval.ndim != 2
        or interval.shape[1:] != (2,)
        or len(interval) < 2
        or not np.isfinite(interval).all()
        or np.any(interval[:, 0] > interval[:, 1])
        or np.any(interval[:, 0] * interval[:, 1] <= 0.0)
    ):
        raise ValueError("signed charge bounds must be finite same-sign intervals")
    n_sites = len(interval)
    if pair.size == 0:
        pair = np.empty((0, 2), dtype=int)
    if pair.ndim != 2 or pair.shape[1:] != (2,):
        raise ValueError("endpoints must have shape (n_edges, 2)")
    if len(pair) and (
        np.any(pair < 0)
        or np.any(pair >= n_sites)
        or np.any(pair[:, 0] == pair[:, 1])
    ):
        raise ValueError("endpoints contain invalid site indices")

    positive = interval[:, 0] > 0.0
    negative = interval[:, 1] < 0.0
    if not positive.any() or not negative.any():
        return _failure("charge intervals need both signs")
    if len(pair) and (
        not np.all(positive[pair[:, 0]]) or not np.all(negative[pair[:, 1]])
    ):
        raise ValueError("edges must be ordered from positive to negative sites")
    pair = np.unique(pair, axis=0)

    magnitude_low = np.min(np.abs(interval), axis=1)
    magnitude_high = np.max(np.abs(interval), axis=1)
    all_sites = np.arange(n_sites, dtype=int)
    global_gap = _normalized_interval_gap(
        magnitude_low, magnitude_high, positive, all_sites
    )
    component_gap, component_mean, component_rms, obstructed_fraction = (
        _component_morphology(
            n_sites=n_sites,
            endpoints=pair,
            magnitude_low=magnitude_low,
            magnitude_high=magnitude_high,
            positive=positive,
            tolerance=tolerance,
        )
    )

    incidence = np.zeros((n_sites, len(pair)), dtype=float)
    if len(pair):
        columns = np.arange(len(pair))
        incidence[pair[:, 0], columns] = 1.0
        incidence[pair[:, 1], columns] = 1.0
    unserved = float(np.mean(np.sum(incidence, axis=1) == 0.0))
    if not len(pair):
        morphology = {
            "cmvom_component_gap_site_mean": component_mean,
            "cmvom_component_gap_site_rms": component_rms,
            "cmvom_obstructed_site_fraction": obstructed_fraction,
            "cmvom_localized_slack_severity": 0.0,
            "cmvom_side_slack_asymmetry": 0.0,
            "cmvom_side_slack_flexibility": 0.0,
        }
        return MixedValenceObstructionMorphologyResult(
            True, None, 1.0, global_gap, component_gap, unserved, morphology
        )

    n_edges = len(pair)
    r_index = n_edges
    slack_start = n_edges + 1
    n_variables = slack_start + n_sites
    inequalities = np.zeros((2 * n_sites, n_variables), dtype=float)
    for site in range(n_sites):
        inequalities[2 * site, :n_edges] = incidence[site]
        inequalities[2 * site, r_index] = -magnitude_high[site]
        inequalities[2 * site, slack_start + site] = -1.0
        inequalities[2 * site + 1, :n_edges] = -incidence[site]
        inequalities[2 * site + 1, r_index] = magnitude_low[site]
        inequalities[2 * site + 1, slack_start + site] = -1.0
    equality = np.zeros((1, n_variables), dtype=float)
    equality[0, :n_edges] = 1.0
    objective = np.zeros(n_variables, dtype=float)
    objective[slack_start:] = 0.5
    primary = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(2 * n_sites, dtype=float),
        A_eq=equality,
        b_eq=np.ones(1, dtype=float),
        bounds=[(0.0, None)] * n_variables,
        method="highs",
    )
    if not primary.success or primary.fun is None:
        return _failure(f"obstruction morphology primary solve failed: {primary.message}")
    raw_slack = float(primary.fun)
    if (
        not math.isfinite(raw_slack)
        or raw_slack < -tolerance
        or raw_slack > 1.0 + tolerance
    ):
        return _failure("obstruction morphology primary optimum is invalid")
    primary_slack = float(np.clip(raw_slack, 0.0, 1.0))

    localized = forced_side = flexibility = 0.0
    if primary_slack > tolerance:
        secondary = _secondary_morphology(
            inequalities=inequalities,
            equality=equality,
            n_edges=n_edges,
            n_sites=n_sites,
            positive=positive,
            primary_slack=primary_slack,
            tolerance=tolerance,
        )
        if isinstance(secondary, str):
            return _failure(secondary)
        localized, forced_side, flexibility = secondary

    morphology = {
        "cmvom_component_gap_site_mean": component_mean,
        "cmvom_component_gap_site_rms": component_rms,
        "cmvom_obstructed_site_fraction": obstructed_fraction,
        "cmvom_localized_slack_severity": localized,
        "cmvom_side_slack_asymmetry": forced_side,
        "cmvom_side_slack_flexibility": flexibility,
    }
    if tuple(morphology) != FEATURE_NAMES or not np.isfinite(
        list(morphology.values())
    ).all():
        return _failure("obstruction morphology schema is invalid")
    morphology = {
        name: float(np.clip(value, 0.0, 1.0))
        for name, value in morphology.items()
    }
    return MixedValenceObstructionMorphologyResult(
        True,
        None,
        primary_slack,
        global_gap,
        component_gap,
        unserved,
        morphology,
    )


def compute_obstruction_morphology(
    structure,
    *,
    graph_mode: str,
    catalogue_mode: str,
    max_sign_patterns: int = MAX_SIGN_PATTERNS,
) -> ConvexMixedValenceObstructionMorphologyFeatureResult:
    """Evaluate morphology on the sign pattern selected exactly as in NEXT109."""

    pymatgen_version = version("pymatgen")
    scipy_version = version("scipy")
    if graph_mode != "voronoi":
        raise ValueError("NEXT112 freezes graph_mode=voronoi")
    if not isinstance(max_sign_patterns, int) or max_sign_patterns < 1:
        raise ValueError("max_sign_patterns must be a positive integer")
    symbols = tuple(str(site.specie.symbol) for site in structure)
    catalogue = _state_catalogue(symbols, catalogue_mode)
    digest = _catalogue_digest(catalogue, catalogue_mode=catalogue_mode)
    if not symbols or len(set(symbols)) < 2:
        return _feature_failure(
            "obstruction morphology needs at least two elements",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    patterns = _oriented_sign_patterns(
        symbols=symbols,
        catalogue=catalogue,
        max_sign_patterns=max_sign_patterns,
    )
    if isinstance(patterns, str):
        return _feature_failure(
            patterns,
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    if not patterns:
        return _feature_failure(
            "no electronegativity-oriented sign pattern exists",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )

    candidates: list[
        tuple[
            tuple[float, ...],
            tuple[tuple[str, int], ...],
            dict[str, float],
        ]
    ] = []
    failures: list[str] = []
    for pattern in patterns:
        sign_by_element = dict(pattern)
        signed_bounds: list[tuple[float, float]] = []
        for symbol in symbols:
            sign = sign_by_element[symbol]
            states = tuple(
                int(value) for value in catalogue[symbol] if sign * int(value) > 0
            )
            signed_bounds.append((float(min(states)), float(max(states))))
        graph = _opposite_sign_endpoints(
            structure,
            symbols=symbols,
            sign_by_element=sign_by_element,
            graph_mode=graph_mode,
        )
        if isinstance(graph, str):
            failures.append(graph)
            continue
        solution = solve_obstruction_morphology(
            signed_charge_bounds=signed_bounds,
            endpoints=graph,
        )
        if not solution.supported:
            failures.append(solution.failure_reason or "obstruction morphology failed")
            continue
        rank = (
            float(solution.min_interval_slack),
            float(solution.global_balance_gap),
            float(solution.component_balance_gap),
            float(solution.unserved_site_fraction),
        )
        features = {name: float(solution.morphology[name]) for name in FEATURE_NAMES}
        candidates.append((rank, pattern, features))
    if not candidates:
        detail = sorted(set(failures))[0] if failures else "unknown failure"
        return _feature_failure(
            f"all sign patterns are unsupported: {detail}",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    _, _, best = min(candidates, key=lambda item: (item[0], item[1]))
    if tuple(best) != FEATURE_NAMES or not np.isfinite(list(best.values())).all():
        return _feature_failure(
            "computed obstruction morphology schema is invalid",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    return ConvexMixedValenceObstructionMorphologyFeatureResult(
        True,
        None,
        best,
        digest,
        pymatgen_version,
        scipy_version,
    )
