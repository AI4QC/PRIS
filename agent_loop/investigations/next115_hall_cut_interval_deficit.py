#!/usr/bin/env python3
"""Exact generalized-Hall interval deficits on a signed contact graph."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import math
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
    _opposite_sign_endpoints,
    solve_mixed_valence_obstruction,
)


PROTOCOL = "2026-08-08-next115-hall-cut-interval-deficit-v1"
_SIDES = ("positive", "negative")
_SUFFIXES = (
    "global_deficit",
    "local_density",
    "origin_site_fraction_min",
    "origin_site_fraction_max",
    "neighbor_site_fraction_min",
)
FEATURE_NAMES = tuple(
    f"hcid_{side}_{suffix}" for side in _SIDES for suffix in _SUFFIXES
)


@dataclass(frozen=True)
class HallCutIntervalDeficitResult:
    supported: bool
    failure_reason: str | None
    features: dict[str, float]
    positive_max_deficit: float
    negative_max_deficit: float


@dataclass(frozen=True)
class HallCutIntervalDeficitFeatureResult:
    """Auditable structure-level certificate for one frozen catalogue mode."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]
    catalogue_sha256: str
    pymatgen_version: str
    scipy_version: str


def _failure(message: str) -> HallCutIntervalDeficitResult:
    return HallCutIntervalDeficitResult(False, message, {}, math.nan, math.nan)


def _feature_failure(
    message: str,
    *,
    catalogue_sha256: str,
    pymatgen_version: str,
    scipy_version: str,
) -> HallCutIntervalDeficitFeatureResult:
    return HallCutIntervalDeficitFeatureResult(
        False,
        message,
        {},
        catalogue_sha256,
        pymatgen_version,
        scipy_version,
    )


def _unit(value: float, *, tolerance: float, name: str) -> float:
    if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
        raise RuntimeError(f"{name} is outside the normalized unit interval")
    return float(min(1.0, max(0.0, value)))


def _secondary_optimum(
    objective: np.ndarray,
    *,
    closure_inequalities: np.ndarray | None,
    primary_vector: np.ndarray,
    primary_optimum: float,
) -> float:
    solved = linprog(
        objective,
        A_ub=closure_inequalities,
        b_ub=(
            np.zeros(len(closure_inequalities), dtype=float)
            if closure_inequalities is not None
            else None
        ),
        A_eq=primary_vector.reshape(1, -1),
        b_eq=np.asarray([primary_optimum], dtype=float),
        bounds=[(0.0, 1.0)] * len(objective),
        method="highs",
    )
    if not solved.success or solved.fun is None or not math.isfinite(float(solved.fun)):
        raise RuntimeError(f"Hall-cut secondary solve failed: {solved.message}")
    return float(solved.fun)


def _directional_certificate(
    *,
    magnitude_low: np.ndarray,
    magnitude_high: np.ndarray,
    origin: np.ndarray,
    neighbor: np.ndarray,
    endpoints: np.ndarray,
    tolerance: float,
) -> tuple[float, tuple[float, ...]]:
    n_origin = len(origin)
    n_neighbor = len(neighbor)
    origin_position = {int(site): index for index, site in enumerate(origin)}
    neighbor_position = {
        int(site): n_origin + index for index, site in enumerate(neighbor)
    }
    closure = np.zeros((len(endpoints), n_origin + n_neighbor), dtype=float)
    for row, (left, right) in enumerate(endpoints):
        closure[row, origin_position[int(left)]] = 1.0
        closure[row, neighbor_position[int(right)]] = -1.0
    closure_inequalities = closure if len(closure) else None
    origin_low = magnitude_low[origin]
    neighbor_high = magnitude_high[neighbor]
    primary_vector = np.concatenate([origin_low, -neighbor_high])
    solved = linprog(
        -primary_vector,
        A_ub=closure_inequalities,
        b_ub=(
            np.zeros(len(closure), dtype=float) if closure_inequalities is not None else None
        ),
        bounds=[(0.0, 1.0)] * len(primary_vector),
        method="highs",
    )
    if not solved.success or solved.fun is None:
        raise RuntimeError(f"Hall-cut primary solve failed: {solved.message}")
    deficit = -float(solved.fun)
    if not math.isfinite(deficit) or deficit < -tolerance:
        raise RuntimeError("Hall-cut primary optimum is invalid")
    if deficit <= tolerance:
        return 0.0, (0.0,) * len(_SUFFIXES)

    origin_charge_objective = np.concatenate(
        [origin_low, np.zeros(n_neighbor, dtype=float)]
    )
    origin_count_objective = np.concatenate(
        [np.ones(n_origin, dtype=float), np.zeros(n_neighbor, dtype=float)]
    )
    neighbor_count_objective = np.concatenate(
        [np.zeros(n_origin, dtype=float), np.ones(n_neighbor, dtype=float)]
    )
    secondary = {
        "origin_charge_min": _secondary_optimum(
            origin_charge_objective,
            closure_inequalities=closure_inequalities,
            primary_vector=primary_vector,
            primary_optimum=deficit,
        ),
        "origin_count_min": _secondary_optimum(
            origin_count_objective,
            closure_inequalities=closure_inequalities,
            primary_vector=primary_vector,
            primary_optimum=deficit,
        ),
        "origin_count_max": -_secondary_optimum(
            -origin_count_objective,
            closure_inequalities=closure_inequalities,
            primary_vector=primary_vector,
            primary_optimum=deficit,
        ),
        "neighbor_count_min": _secondary_optimum(
            neighbor_count_objective,
            closure_inequalities=closure_inequalities,
            primary_vector=primary_vector,
            primary_optimum=deficit,
        ),
    }
    if secondary["origin_charge_min"] <= tolerance:
        raise RuntimeError("Hall-cut optimum has invalid origin charge support")
    values = (
        deficit / float(np.sum(origin_low)),
        deficit / secondary["origin_charge_min"],
        secondary["origin_count_min"] / n_origin,
        secondary["origin_count_max"] / n_origin,
        secondary["neighbor_count_min"] / n_neighbor,
    )
    normalized = tuple(
        _unit(value, tolerance=10.0 * tolerance, name=name)
        for name, value in zip(_SUFFIXES, values, strict=True)
    )
    return float(deficit), normalized


def solve_hall_cut_interval_deficit(
    *,
    signed_charge_bounds: Sequence[Sequence[float]],
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    tolerance: float = 1.0e-9,
) -> HallCutIntervalDeficitResult:
    """Return exact directional Hall-cut violations and canonical morphology."""

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
    positive_sites = np.flatnonzero(positive)
    negative_sites = np.flatnonzero(negative)
    try:
        positive_deficit, positive_values = _directional_certificate(
            magnitude_low=magnitude_low,
            magnitude_high=magnitude_high,
            origin=positive_sites,
            neighbor=negative_sites,
            endpoints=pair,
            tolerance=tolerance,
        )
        negative_deficit, negative_values = _directional_certificate(
            magnitude_low=magnitude_low,
            magnitude_high=magnitude_high,
            origin=negative_sites,
            neighbor=positive_sites,
            endpoints=pair[:, ::-1],
            tolerance=tolerance,
        )
    except RuntimeError as exc:
        return _failure(str(exc))
    features = {
        f"hcid_{side}_{suffix}": float(value)
        for side, values in zip(_SIDES, (positive_values, negative_values), strict=True)
        for suffix, value in zip(_SUFFIXES, values, strict=True)
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        return _failure("computed Hall-cut feature schema is invalid")
    return HallCutIntervalDeficitResult(
        True,
        None,
        features,
        float(positive_deficit),
        float(negative_deficit),
    )


def compute_hall_cut_interval_deficit(
    structure,
    *,
    graph_mode: str,
    catalogue_mode: str,
    max_sign_patterns: int = MAX_SIGN_PATTERNS,
) -> HallCutIntervalDeficitFeatureResult:
    """Evaluate HCID on the sign pattern selected exactly as in NEXT109."""

    pymatgen_version = version("pymatgen")
    scipy_version = version("scipy")
    if graph_mode != "voronoi":
        raise ValueError("NEXT115 freezes graph_mode=voronoi")
    if not isinstance(max_sign_patterns, int) or max_sign_patterns < 1:
        raise ValueError("max_sign_patterns must be a positive integer")
    symbols = tuple(str(site.specie.symbol) for site in structure)
    catalogue = _state_catalogue(symbols, catalogue_mode)
    digest = _catalogue_digest(catalogue, catalogue_mode=catalogue_mode)
    if not symbols or len(set(symbols)) < 2:
        return _feature_failure(
            "Hall-cut certificate needs at least two elements",
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
        legacy = solve_mixed_valence_obstruction(
            signed_charge_bounds=signed_bounds,
            endpoints=graph,
        )
        if not legacy.supported:
            failures.append(legacy.failure_reason or "NEXT109 obstruction failed")
            continue
        certificate = solve_hall_cut_interval_deficit(
            signed_charge_bounds=signed_bounds,
            endpoints=graph,
        )
        if not certificate.supported:
            failures.append(certificate.failure_reason or "Hall-cut certificate failed")
            continue
        rank = (
            float(legacy.min_interval_slack),
            float(legacy.global_balance_gap),
            float(legacy.component_balance_gap),
            float(legacy.unserved_site_fraction),
        )
        features = {
            name: float(certificate.features[name]) for name in FEATURE_NAMES
        }
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
            "computed Hall-cut feature schema is invalid",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    return HallCutIntervalDeficitFeatureResult(
        True,
        None,
        best,
        digest,
        pymatgen_version,
        scipy_version,
    )
