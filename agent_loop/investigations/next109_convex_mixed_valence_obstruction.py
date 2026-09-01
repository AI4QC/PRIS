#!/usr/bin/env python3
"""Brown-free obstruction magnitude for convex mixed-valence graph flow."""

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


PROTOCOL = "2026-08-08-next109-convex-mixed-valence-obstruction-v1"
FEATURE_NAMES = (
    "cmvo_min_interval_slack",
    "cmvo_global_balance_gap",
    "cmvo_component_balance_gap",
    "cmvo_unserved_site_fraction",
)


@dataclass(frozen=True)
class MixedValenceObstructionResult:
    """Normalized distance from one fixed-sign graph to interval feasibility."""

    supported: bool
    failure_reason: str | None
    min_interval_slack: float | None
    global_balance_gap: float | None
    component_balance_gap: float | None
    unserved_site_fraction: float | None


@dataclass(frozen=True)
class ConvexMixedValenceObstructionFeatureResult:
    """Auditable structure-level obstruction for one frozen catalogue mode."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]
    catalogue_sha256: str
    pymatgen_version: str
    scipy_version: str


def _failure(reason: str) -> MixedValenceObstructionResult:
    return MixedValenceObstructionResult(False, reason, None, None, None, None)


def _feature_failure(
    reason: str,
    *,
    catalogue_sha256: str,
    pymatgen_version: str,
    scipy_version: str,
) -> ConvexMixedValenceObstructionFeatureResult:
    return ConvexMixedValenceObstructionFeatureResult(
        False,
        reason,
        {},
        catalogue_sha256,
        pymatgen_version,
        scipy_version,
    )


def _normalized_interval_gap(
    magnitude_low: np.ndarray,
    magnitude_high: np.ndarray,
    positive: np.ndarray,
    indices: np.ndarray,
) -> float:
    positive_indices = indices[positive[indices]]
    negative_indices = indices[~positive[indices]]
    if not len(positive_indices) or not len(negative_indices):
        return 1.0
    positive_low = float(np.sum(magnitude_low[positive_indices]))
    positive_high = float(np.sum(magnitude_high[positive_indices]))
    negative_low = float(np.sum(magnitude_low[negative_indices]))
    negative_high = float(np.sum(magnitude_high[negative_indices]))
    gap = max(0.0, positive_low - negative_high, negative_low - positive_high)
    denominator = max(positive_low, negative_low)
    return float(min(1.0, gap / denominator))


def _component_balance_gap(
    *,
    n_sites: int,
    endpoints: np.ndarray,
    magnitude_low: np.ndarray,
    magnitude_high: np.ndarray,
    positive: np.ndarray,
) -> float:
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
    return max(
        _normalized_interval_gap(
            magnitude_low,
            magnitude_high,
            positive,
            np.asarray(sites, dtype=int),
        )
        for sites in groups.values()
    )


def solve_mixed_valence_obstruction(
    *,
    signed_charge_bounds: Sequence[Sequence[float]],
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    tolerance: float = 1.0e-9,
) -> MixedValenceObstructionResult:
    """Return the normalized L1 interval relaxation for one bipartite graph."""

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
    component_gap = _component_balance_gap(
        n_sites=n_sites,
        endpoints=pair,
        magnitude_low=magnitude_low,
        magnitude_high=magnitude_high,
        positive=positive,
    )
    incidence = np.zeros((n_sites, len(pair)), dtype=float)
    if len(pair):
        columns = np.arange(len(pair))
        incidence[pair[:, 0], columns] = 1.0
        incidence[pair[:, 1], columns] = 1.0
    unserved = float(np.mean(np.sum(incidence, axis=1) == 0.0))
    if not len(pair):
        return MixedValenceObstructionResult(
            True, None, 1.0, global_gap, component_gap, unserved
        )

    n_edges = len(pair)
    r_index = n_edges
    slack_start = n_edges + 1
    n_variables = n_edges + 1 + n_sites
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
    solved = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(2 * n_sites, dtype=float),
        A_eq=equality,
        b_eq=np.ones(1, dtype=float),
        bounds=[(0.0, None)] * n_variables,
        method="highs",
    )
    if not solved.success or solved.fun is None:
        return _failure(f"mixed-valence obstruction solve failed: {solved.message}")
    slack = float(solved.fun)
    if not math.isfinite(slack) or slack < -tolerance or slack > 1.0 + tolerance:
        return _failure("mixed-valence obstruction optimum is invalid")
    return MixedValenceObstructionResult(
        True,
        None,
        float(min(1.0, max(0.0, slack))),
        global_gap,
        component_gap,
        unserved,
    )


def _opposite_sign_endpoints(
    structure,
    *,
    symbols: Sequence[str],
    sign_by_element: Mapping[str, int],
    graph_mode: str,
) -> np.ndarray | str:
    """Build all valid projected opposite-sign endpoints without early abort."""

    from src.next19_valence_transport import _neighbor_finder

    positive_sites = tuple(
        index
        for index, symbol in enumerate(symbols)
        if sign_by_element[symbol] > 0
    )
    negative_sites = {
        index
        for index, symbol in enumerate(symbols)
        if sign_by_element[symbol] < 0
    }
    if not positive_sites or not negative_sites:
        return "sign pattern needs both signs"
    dummy_charges = tuple(
        1.0 / len(positive_sites)
        if index in positive_sites
        else -1.0 / len(negative_sites)
        for index in range(len(symbols))
    )
    working_structure = structure.copy()
    try:
        working_structure.add_oxidation_state_by_site(dummy_charges)
    except Exception as exc:
        return f"valence decoration failed: {type(exc).__name__}"
    finder = _neighbor_finder(graph_mode)
    endpoints: set[tuple[int, int]] = set()
    for cation in positive_sites:
        try:
            neighbor_info = finder.get_nn_info(working_structure, cation)
        except Exception as exc:
            return f"{graph_mode} neighbor construction failed: {type(exc).__name__}"
        origin = np.asarray(working_structure[cation].coords, dtype=float)
        for info in neighbor_info:
            try:
                anion = int(info["site_index"])
                if anion not in negative_sites:
                    continue
                image = tuple(int(round(float(value))) for value in info["image"])
                if len(image) != 3:
                    continue
                neighbor_coords = np.asarray(info["site"].coords, dtype=float)
                distance = float(np.linalg.norm(neighbor_coords - origin))
                weight = float(info.get("weight", 1.0))
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(distance) and distance > 1.0e-8):
                continue
            if not (math.isfinite(weight) and weight > 0.0):
                continue
            endpoints.add((int(cation), int(anion)))
    if not endpoints:
        return np.empty((0, 2), dtype=int)
    return np.asarray(sorted(endpoints), dtype=int)


def compute_convex_mixed_valence_obstruction(
    structure,
    *,
    graph_mode: str,
    catalogue_mode: str,
    max_sign_patterns: int = MAX_SIGN_PATTERNS,
) -> ConvexMixedValenceObstructionFeatureResult:
    """Evaluate Brown-free obstruction over every frozen oriented sign pattern."""

    pymatgen_version = version("pymatgen")
    scipy_version = version("scipy")
    if graph_mode != "voronoi":
        raise ValueError("NEXT109 freezes graph_mode=voronoi")
    if not isinstance(max_sign_patterns, int) or max_sign_patterns < 1:
        raise ValueError("max_sign_patterns must be a positive integer")
    symbols = tuple(str(site.specie.symbol) for site in structure)
    catalogue = _state_catalogue(symbols, catalogue_mode)
    digest = _catalogue_digest(catalogue, catalogue_mode=catalogue_mode)
    if not symbols or len(set(symbols)) < 2:
        return _feature_failure(
            "mixed-valence obstruction needs at least two elements",
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
        solution = solve_mixed_valence_obstruction(
            signed_charge_bounds=signed_bounds,
            endpoints=graph,
        )
        if not solution.supported:
            failures.append(solution.failure_reason or "obstruction solve failed")
            continue
        features = {
            "cmvo_min_interval_slack": float(solution.min_interval_slack),
            "cmvo_global_balance_gap": float(solution.global_balance_gap),
            "cmvo_component_balance_gap": float(solution.component_balance_gap),
            "cmvo_unserved_site_fraction": float(solution.unserved_site_fraction),
        }
        rank = tuple(features[name] for name in FEATURE_NAMES)
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
            "computed feature schema is invalid",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    return ConvexMixedValenceObstructionFeatureResult(
        True,
        None,
        best,
        digest,
        pymatgen_version,
        scipy_version,
    )
