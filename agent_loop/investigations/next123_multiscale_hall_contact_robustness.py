#!/usr/bin/env python3
"""Multiscale generalized-Hall robustness on a weighted contact graph."""

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


PROTOCOL = "2026-08-08-next123-multiscale-hall-contact-robustness-v1"
STRENGTH_THRESHOLDS = (0.05, 0.10, 0.25, 0.50)
_SIDES = ("positive", "negative")


def _threshold_code(value: float) -> str:
    return f"tau{int(round(100.0 * value)):02d}"


FEATURE_NAMES = tuple(
    f"mhcr_{side}_deficit_gain_{_threshold_code(threshold)}"
    for side in _SIDES
    for threshold in STRENGTH_THRESHOLDS
)


@dataclass(frozen=True)
class MultiscaleHallContactRobustnessResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]
    positive_deficits: tuple[float, ...]
    negative_deficits: tuple[float, ...]


@dataclass(frozen=True)
class MultiscaleHallContactRobustnessFeatureResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]
    catalogue_sha256: str
    pymatgen_version: str
    scipy_version: str


def _failure(message: str) -> MultiscaleHallContactRobustnessResult:
    return MultiscaleHallContactRobustnessResult(False, message, {}, (), ())


def _feature_failure(
    message: str,
    *,
    catalogue_sha256: str,
    pymatgen_version: str,
    scipy_version: str,
) -> MultiscaleHallContactRobustnessFeatureResult:
    return MultiscaleHallContactRobustnessFeatureResult(
        False,
        message,
        {},
        catalogue_sha256,
        pymatgen_version,
        scipy_version,
    )


def _validate_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in thresholds)
    if (
        not values
        or any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in values)
        or any(right <= left for left, right in zip(values, values[1:]))
    ):
        raise ValueError("strength thresholds must be finite, increasing, and in (0, 1]")
    return values


def _validated_graph(
    signed_charge_bounds: Sequence[Sequence[float]],
    weighted_endpoints: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    interval = np.asarray(signed_charge_bounds, dtype=float)
    if (
        interval.ndim != 2
        or interval.shape[1:] != (2,)
        or len(interval) < 2
        or not np.isfinite(interval).all()
        or np.any(interval[:, 0] > interval[:, 1])
        or np.any(interval[:, 0] * interval[:, 1] <= 0.0)
    ):
        raise ValueError("signed charge bounds must be finite same-sign intervals")
    positive = interval[:, 0] > 0.0
    negative = interval[:, 1] < 0.0
    if not positive.any() or not negative.any():
        raise ValueError("signed charge bounds need both signs")

    raw = np.asarray(weighted_endpoints, dtype=float)
    if raw.size == 0:
        raw = np.empty((0, 3), dtype=float)
    if raw.ndim != 2 or raw.shape[1:] != (3,) or not np.isfinite(raw).all():
        raise ValueError("weighted endpoints must have shape (n_edges, 3) and be finite")
    if len(raw):
        integer_endpoints = np.rint(raw[:, :2])
        if (
            not np.allclose(raw[:, :2], integer_endpoints, rtol=0.0, atol=0.0)
            or np.any(integer_endpoints < 0)
            or np.any(integer_endpoints >= len(interval))
            or np.any(integer_endpoints[:, 0] == integer_endpoints[:, 1])
            or not np.all(positive[integer_endpoints[:, 0].astype(int)])
            or not np.all(negative[integer_endpoints[:, 1].astype(int)])
            or np.any(raw[:, 2] <= 0.0)
            or np.any(raw[:, 2] > 1.0)
        ):
            raise ValueError("weighted endpoints have invalid orientation, index, or strength")

    strongest: dict[tuple[int, int], float] = {}
    for left, right, strength in raw:
        key = (int(round(left)), int(round(right)))
        strongest[key] = max(strongest.get(key, 0.0), float(strength))
    graph = np.asarray(
        [(left, right, strongest[(left, right)]) for left, right in sorted(strongest)],
        dtype=float,
    ).reshape(-1, 3)
    return interval, graph


def _directional_max_deficit(
    *,
    magnitude_low: np.ndarray,
    magnitude_high: np.ndarray,
    origin: np.ndarray,
    neighbor: np.ndarray,
    endpoints: np.ndarray,
    tolerance: float,
) -> float:
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
    primary = np.concatenate(
        [magnitude_low[origin], -magnitude_high[neighbor]]
    )
    solved = linprog(
        -primary,
        A_ub=closure if len(closure) else None,
        b_ub=np.zeros(len(closure), dtype=float) if len(closure) else None,
        bounds=[(0.0, 1.0)] * len(primary),
        method="highs",
    )
    if not solved.success or solved.fun is None:
        raise RuntimeError(f"Hall-contact closure solve failed: {solved.message}")
    deficit = -float(solved.fun)
    maximum = float(magnitude_low[origin].sum())
    if (
        not math.isfinite(deficit)
        or deficit < -tolerance
        or deficit > maximum + 10.0 * tolerance
    ):
        raise RuntimeError("Hall-contact closure optimum is invalid")
    return float(min(maximum, max(0.0, deficit)))


def solve_multiscale_hall_contact_robustness(
    *,
    signed_charge_bounds: Sequence[Sequence[float]],
    weighted_endpoints: Sequence[Sequence[float]],
    thresholds: Sequence[float] = STRENGTH_THRESHOLDS,
    tolerance: float = 1.0e-9,
) -> MultiscaleHallContactRobustnessResult:
    """Return normalized deficit gains after fixed weak-contact deletion."""

    threshold_values = _validate_thresholds(thresholds)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    interval, graph = _validated_graph(signed_charge_bounds, weighted_endpoints)
    magnitude_low = np.min(np.abs(interval), axis=1)
    magnitude_high = np.max(np.abs(interval), axis=1)
    positive = np.flatnonzero(interval[:, 0] > 0.0)
    negative = np.flatnonzero(interval[:, 1] < 0.0)
    endpoints = graph[:, :2].astype(int)

    features: dict[str, float] = {}
    deficits_by_side: dict[str, tuple[float, ...]] = {}
    try:
        for side, origin, neighbor, reverse in (
            ("positive", positive, negative, False),
            ("negative", negative, positive, True),
        ):
            oriented = endpoints[:, ::-1] if reverse else endpoints
            base = _directional_max_deficit(
                magnitude_low=magnitude_low,
                magnitude_high=magnitude_high,
                origin=origin,
                neighbor=neighbor,
                endpoints=oriented,
                tolerance=tolerance,
            )
            denominator = float(magnitude_low[origin].sum())
            values: list[float] = []
            raw_deficits: list[float] = [base]
            for threshold in threshold_values:
                retained = graph[:, 2] >= threshold
                threshold_endpoints = endpoints[retained]
                if reverse:
                    threshold_endpoints = threshold_endpoints[:, ::-1]
                deficit = _directional_max_deficit(
                    magnitude_low=magnitude_low,
                    magnitude_high=magnitude_high,
                    origin=origin,
                    neighbor=neighbor,
                    endpoints=threshold_endpoints,
                    tolerance=tolerance,
                )
                gain = (deficit - base) / denominator
                if not math.isfinite(gain) or gain < -10.0 * tolerance or gain > 1.0 + 10.0 * tolerance:
                    raise RuntimeError("Hall-contact robustness gain is invalid")
                gain = float(min(1.0, max(0.0, gain)))
                if values and gain + 10.0 * tolerance < values[-1]:
                    raise RuntimeError("Hall-contact robustness is not threshold-monotone")
                values.append(gain)
                raw_deficits.append(deficit)
                features[f"mhcr_{side}_deficit_gain_{_threshold_code(threshold)}"] = gain
            deficits_by_side[side] = tuple(raw_deficits)
    except RuntimeError as exc:
        return _failure(str(exc))

    expected_names = tuple(
        f"mhcr_{side}_deficit_gain_{_threshold_code(threshold)}"
        for side in _SIDES
        for threshold in threshold_values
    )
    if tuple(features) != expected_names or not np.isfinite(list(features.values())).all():
        return _failure("computed multiscale Hall-contact schema is invalid")
    return MultiscaleHallContactRobustnessResult(
        True,
        None,
        features,
        deficits_by_side["positive"],
        deficits_by_side["negative"],
    )


def _weighted_opposite_sign_endpoints(
    structure,
    *,
    symbols: Sequence[str],
    sign_by_element: Mapping[str, int],
    graph_mode: str,
) -> np.ndarray | str:
    """Return the NEXT109 endpoints with deterministic per-origin strengths."""

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
    strongest: dict[tuple[int, int], float] = {}
    for cation in positive_sites:
        try:
            neighbor_info = finder.get_nn_info(working_structure, cation)
        except Exception as exc:
            return f"{graph_mode} neighbor construction failed: {type(exc).__name__}"
        origin = np.asarray(working_structure[cation].coords, dtype=float)
        valid: list[tuple[int, float]] = []
        for info in neighbor_info:
            try:
                neighbor = int(info["site_index"])
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
            valid.append((neighbor, weight))
        if not valid:
            continue
        normalizer = max(weight for _, weight in valid)
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            return "Voronoi contact normalization failed"
        for anion, weight in valid:
            if anion not in negative_sites:
                continue
            strength = float(weight / normalizer)
            if not math.isfinite(strength) or strength <= 0.0 or strength > 1.0 + 1.0e-12:
                return "Voronoi contact strength is invalid"
            key = (int(cation), int(anion))
            strongest[key] = max(strongest.get(key, 0.0), min(1.0, strength))
    return np.asarray(
        [(left, right, strongest[(left, right)]) for left, right in sorted(strongest)],
        dtype=float,
    ).reshape(-1, 3)


def compute_multiscale_hall_contact_robustness(
    structure,
    *,
    graph_mode: str,
    catalogue_mode: str,
    max_sign_patterns: int = MAX_SIGN_PATTERNS,
) -> MultiscaleHallContactRobustnessFeatureResult:
    """Evaluate MHCR on the exact NEXT109-selected analytic sign pattern."""

    pymatgen_version = version("pymatgen")
    scipy_version = version("scipy")
    if graph_mode != "voronoi":
        raise ValueError("NEXT123 freezes graph_mode=voronoi")
    if not isinstance(max_sign_patterns, int) or max_sign_patterns < 1:
        raise ValueError("max_sign_patterns must be a positive integer")
    symbols = tuple(str(site.specie.symbol) for site in structure)
    catalogue = _state_catalogue(symbols, catalogue_mode)
    digest = _catalogue_digest(catalogue, catalogue_mode=catalogue_mode)
    if not symbols or len(set(symbols)) < 2:
        return _feature_failure(
            "multiscale Hall-contact robustness needs at least two elements",
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
            list[tuple[float, float]],
            np.ndarray,
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
            failures.append(legacy.failure_reason or "legacy obstruction failed")
            continue
        rank = (
            float(legacy.min_interval_slack),
            float(legacy.global_balance_gap),
            float(legacy.component_balance_gap),
            float(legacy.unserved_site_fraction),
        )
        candidates.append((rank, pattern, signed_bounds, graph))
    if not candidates:
        detail = sorted(set(failures))[0] if failures else "unknown failure"
        return _feature_failure(
            f"all sign patterns are unsupported: {detail}",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )

    _, pattern, signed_bounds, legacy_graph = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    weighted = _weighted_opposite_sign_endpoints(
        structure,
        symbols=symbols,
        sign_by_element=dict(pattern),
        graph_mode=graph_mode,
    )
    if isinstance(weighted, str):
        return _feature_failure(
            weighted,
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    if not np.array_equal(weighted[:, :2].astype(int), legacy_graph):
        return _feature_failure(
            "weighted endpoints differ from the frozen legacy graph",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    result = solve_multiscale_hall_contact_robustness(
        signed_charge_bounds=signed_bounds,
        weighted_endpoints=weighted,
    )
    if not result.supported:
        return _feature_failure(
            result.failure_reason or "multiscale Hall-contact solve failed",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    if tuple(result.features) != FEATURE_NAMES:
        return _feature_failure(
            "computed multiscale Hall-contact feature schema is invalid",
            catalogue_sha256=digest,
            pymatgen_version=pymatgen_version,
            scipy_version=scipy_version,
        )
    return MultiscaleHallContactRobustnessFeatureResult(
        True,
        None,
        result.features,
        digest,
        pymatgen_version,
        scipy_version,
    )


__all__ = [
    "FEATURE_NAMES",
    "PROTOCOL",
    "STRENGTH_THRESHOLDS",
    "MultiscaleHallContactRobustnessFeatureResult",
    "MultiscaleHallContactRobustnessResult",
    "compute_multiscale_hall_contact_robustness",
    "solve_multiscale_hall_contact_robustness",
]
