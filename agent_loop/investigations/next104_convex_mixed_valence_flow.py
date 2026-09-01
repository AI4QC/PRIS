"""Representation-invariant convex mixed-valence bond-flow certificate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog


PROTOCOL = "2026-08-04-next104-convex-mixed-valence-flow-v1"
MAX_SIGN_PATTERNS = 128
FEATURE_NAMES = (
    "cmvf_reallocation",
    "cmvf_overload",
    "cmvf_log_scale_mismatch",
    "cmvf_domain_width_mean",
    "cmvf_domain_width_max",
    "cmvf_sign_pattern_log_count",
)


@dataclass(frozen=True)
class ConvexMixedValenceFlowResult:
    """Optimum values of one fixed-sign convex mixed-valence network."""

    supported: bool
    failure_reason: str | None
    reallocation: float | None
    overload: float | None
    log_scale_mismatch: float | None


@dataclass(frozen=True)
class ConvexMixedValenceFeatureResult:
    """Auditable structure-level certificate for one frozen catalogue mode."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]
    catalogue_sha256: str
    pymatgen_version: str
    scipy_version: str


def _failure(reason: str) -> ConvexMixedValenceFlowResult:
    return ConvexMixedValenceFlowResult(False, reason, None, None, None)


def _feature_failure(
    reason: str,
    *,
    catalogue_sha256: str,
    pymatgen_version: str,
    scipy_version: str,
) -> ConvexMixedValenceFeatureResult:
    return ConvexMixedValenceFeatureResult(
        False,
        reason,
        {},
        catalogue_sha256,
        pymatgen_version,
        scipy_version,
    )


def solve_convex_mixed_valence_flow(
    *,
    signed_charge_bounds: Sequence[Sequence[float]],
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    raw_priors: Sequence[float] | np.ndarray,
    tolerance: float = 1.0e-9,
) -> ConvexMixedValenceFlowResult:
    """Jointly reconcile geometric edge priors with mixed-valence intervals.

    Edge flows are normalized to unit total charge.  ``r`` is the inverse
    physical charge scale, so every site constraint remains linear:
    ``lower_i * r <= incident_flow_i <= upper_i * r``.
    """

    interval = np.asarray(signed_charge_bounds, dtype=float)
    pair = np.asarray(endpoints, dtype=int)
    raw = np.asarray(raw_priors, dtype=float)
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
    if pair.ndim != 2 or pair.shape[1:] != (2,) or len(pair) < 1:
        raise ValueError("endpoints must have nonempty shape (n_edges, 2)")
    n_edges = len(pair)
    if raw.shape != (n_edges,) or not np.isfinite(raw).all() or np.any(raw <= 0.0):
        raise ValueError("raw priors must be finite and positive for every edge")
    if np.any(pair < 0) or np.any(pair >= n_sites) or np.any(pair[:, 0] == pair[:, 1]):
        raise ValueError("endpoints contain invalid site indices")

    positive = interval[:, 0] > 0.0
    negative = interval[:, 1] < 0.0
    if not positive.any() or not negative.any():
        return _failure("charge intervals need both signs")
    if not np.all(positive[pair[:, 0]]) or not np.all(negative[pair[:, 1]]):
        raise ValueError("edges must be ordered from positive to negative sites")
    incidence = np.zeros((n_sites, n_edges), dtype=float)
    columns = np.arange(n_edges)
    incidence[pair[:, 0], columns] = 1.0
    incidence[pair[:, 1], columns] = 1.0
    if np.any(np.sum(incidence, axis=1) == 0.0):
        return _failure("opposite-sign graph contains an isolated site")

    magnitude_low = np.min(np.abs(interval), axis=1)
    magnitude_high = np.max(np.abs(interval), axis=1)
    positive_low = float(np.sum(magnitude_low[positive]))
    positive_high = float(np.sum(magnitude_high[positive]))
    negative_low = float(np.sum(magnitude_low[negative]))
    negative_high = float(np.sum(magnitude_high[negative]))
    total_low = max(positive_low, negative_low)
    total_high = min(positive_high, negative_high)
    if total_low > total_high + tolerance * max(1.0, total_high):
        return _failure("positive and negative charge intervals cannot balance")
    if total_low <= 0.0 or not math.isfinite(total_high):
        raise ValueError("charge-scale interval is invalid")

    raw_sum = float(np.sum(raw))
    prior = raw / raw_sum
    # Variables are [normalized edge flow y, inverse charge r, abs deviation u].
    y_slice = slice(0, n_edges)
    r_index = n_edges
    u_slice = slice(n_edges + 1, 2 * n_edges + 1)
    n_base = 2 * n_edges + 1

    equality = np.zeros((1, n_base), dtype=float)
    equality[0, y_slice] = 1.0
    equality_rhs = np.asarray([1.0])
    inequalities: list[np.ndarray] = []
    inequality_rhs: list[float] = []
    for site in range(n_sites):
        upper = np.zeros(n_base, dtype=float)
        upper[y_slice] = incidence[site]
        upper[r_index] = -magnitude_high[site]
        inequalities.append(upper)
        inequality_rhs.append(0.0)

        lower = np.zeros(n_base, dtype=float)
        lower[y_slice] = -incidence[site]
        lower[r_index] = magnitude_low[site]
        inequalities.append(lower)
        inequality_rhs.append(0.0)
    for edge in range(n_edges):
        above = np.zeros(n_base, dtype=float)
        above[edge] = 1.0
        above[n_edges + 1 + edge] = -1.0
        inequalities.append(above)
        inequality_rhs.append(float(prior[edge]))

        below = np.zeros(n_base, dtype=float)
        below[edge] = -1.0
        below[n_edges + 1 + edge] = -1.0
        inequalities.append(below)
        inequality_rhs.append(float(-prior[edge]))
    base_upper = np.vstack(inequalities)
    base_rhs = np.asarray(inequality_rhs, dtype=float)
    base_bounds = (
        [(0.0, None)] * n_edges
        + [(1.0 / total_high, 1.0 / total_low)]
        + [(0.0, None)] * n_edges
    )
    first_objective = np.zeros(n_base, dtype=float)
    first_objective[u_slice] = 0.5
    first = linprog(
        first_objective,
        A_ub=base_upper,
        b_ub=base_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=base_bounds,
        method="highs",
    )
    if not first.success or first.x is None or first.fun is None:
        return _failure(f"mixed-valence reallocation solve failed: {first.message}")
    reallocation = max(0.0, float(first.fun))
    face_tolerance = max(1.0e-9, 100.0 * tolerance)

    # On the total-variation optimum face, minimize max(y_e / p_e).
    second_base = np.pad(base_upper, ((0, 0), (0, 1)))
    tv_face = np.zeros((1, n_base + 1), dtype=float)
    tv_face[0, u_slice] = 1.0
    overload_rows = np.zeros((n_edges, n_base + 1), dtype=float)
    overload_rows[np.arange(n_edges), np.arange(n_edges)] = 1.0
    overload_rows[:, -1] = -prior
    second_upper = np.vstack([second_base, tv_face, overload_rows])
    second_rhs = np.concatenate(
        [base_rhs, [2.0 * reallocation + face_tolerance], np.zeros(n_edges)]
    )
    second_objective = np.zeros(n_base + 1, dtype=float)
    second_objective[-1] = 1.0
    second = linprog(
        second_objective,
        A_ub=second_upper,
        b_ub=second_rhs,
        A_eq=np.pad(equality, ((0, 0), (0, 1))),
        b_eq=equality_rhs,
        bounds=base_bounds + [(0.0, None)],
        method="highs",
    )
    if not second.success or second.x is None or second.fun is None:
        return _failure(f"mixed-valence overload solve failed: {second.message}")
    kappa = max(1.0, float(second.fun))
    overload = max(0.0, kappa - 1.0)

    # Resolve the last scalar tie by choosing the raw-scale-compatible charge.
    third_base = np.pad(base_upper, ((0, 0), (0, 1)))
    third_tv = np.zeros((1, n_base + 1), dtype=float)
    third_tv[0, u_slice] = 1.0
    fixed_overload = np.zeros((n_edges, n_base + 1), dtype=float)
    fixed_overload[np.arange(n_edges), np.arange(n_edges)] = 1.0
    fixed_overload_rhs = (kappa + face_tolerance) * prior
    target_inverse_charge = 1.0 / raw_sum
    scale_rows = np.zeros((2, n_base + 1), dtype=float)
    scale_rows[0, r_index] = 1.0
    scale_rows[0, -1] = -1.0
    scale_rows[1, r_index] = -1.0
    scale_rows[1, -1] = -1.0
    scale_rhs = np.asarray([target_inverse_charge, -target_inverse_charge])
    third_upper = np.vstack(
        [third_base, third_tv, fixed_overload, scale_rows]
    )
    third_rhs = np.concatenate(
        [
            base_rhs,
            [2.0 * reallocation + face_tolerance],
            fixed_overload_rhs,
            scale_rhs,
        ]
    )
    third_objective = np.zeros(n_base + 1, dtype=float)
    third_objective[-1] = 1.0
    third = linprog(
        third_objective,
        A_ub=third_upper,
        b_ub=third_rhs,
        A_eq=np.pad(equality, ((0, 0), (0, 1))),
        b_eq=equality_rhs,
        bounds=base_bounds + [(0.0, None)],
        method="highs",
        # HiGHS presolve can falsely mark duplicated symmetric components
        # infeasible after both preceding optimum faces have been fixed.
        options={"presolve": False},
    )
    if not third.success or third.x is None:
        return _failure(f"mixed-valence scale solve failed: {third.message}")
    inverse_charge = float(third.x[r_index])
    if not math.isfinite(inverse_charge) or inverse_charge <= 0.0:
        return _failure("mixed-valence scale solve returned invalid charge")
    charge_scale = 1.0 / inverse_charge
    log_scale_mismatch = abs(math.log(charge_scale / raw_sum))
    values = (reallocation, overload, log_scale_mismatch)
    if not np.isfinite(values).all():
        return _failure("mixed-valence optimum is non-finite")
    return ConvexMixedValenceFlowResult(
        True,
        None,
        float(reallocation),
        float(overload),
        float(log_scale_mismatch),
    )


def _state_catalogue(
    elements: Sequence[str], catalogue_mode: str
) -> dict[str, tuple[int, ...]]:
    from pymatgen.core import Element

    if catalogue_mode not in {"core", "expanded"}:
        raise ValueError("catalogue_mode must be core or expanded")
    catalogue: dict[str, tuple[int, ...]] = {}
    for symbol in sorted(set(elements)):
        element = Element(symbol)
        raw = (
            element.oxidation_states
            if catalogue_mode == "expanded"
            else tuple(element.common_oxidation_states)
            + tuple(element.icsd_oxidation_states)
        )
        states = tuple(
            sorted(
                {
                    int(value)
                    for value in raw
                    if float(value).is_integer() and int(value) != 0
                }
            )
        )
        catalogue[symbol] = states
    return catalogue


def _catalogue_digest(
    catalogue: Mapping[str, Sequence[int]], *, catalogue_mode: str
) -> str:
    payload = json.dumps(
        {
            "catalogue_mode": catalogue_mode,
            "states": {
                symbol: [int(value) for value in catalogue[symbol]]
                for symbol in sorted(catalogue)
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _oriented_sign_patterns(
    *,
    symbols: Sequence[str],
    catalogue: Mapping[str, Sequence[int]],
    max_sign_patterns: int,
) -> tuple[tuple[tuple[str, int], ...], ...] | str:
    from pymatgen.core import Element

    elements = tuple(sorted(set(symbols)))
    options: list[tuple[int, ...]] = []
    for symbol in elements:
        states = tuple(int(value) for value in catalogue.get(symbol, ()))
        signs = tuple(sign for sign in (-1, 1) if any(sign * value > 0 for value in states))
        if not signs:
            return f"oxidation-state catalogue is empty for {symbol}"
        options.append(signs)
    try:
        electronegativity = {symbol: float(Element(symbol).X) for symbol in elements}
    except (TypeError, ValueError) as exc:
        return f"electronegativity catalogue failed: {type(exc).__name__}"
    if not np.isfinite(list(electronegativity.values())).all():
        return "electronegativity catalogue is incomplete"

    patterns: list[tuple[tuple[str, int], ...]] = []
    for signs in itertools.product(*options):
        if len(set(signs)) < 2:
            continue
        sign_by_element = dict(zip(elements, signs, strict=True))
        positive_x = [
            electronegativity[symbol]
            for symbol in symbols
            if sign_by_element[symbol] > 0
        ]
        negative_x = [
            electronegativity[symbol]
            for symbol in symbols
            if sign_by_element[symbol] < 0
        ]
        if not positive_x or not negative_x:
            continue
        if float(np.mean(negative_x)) <= float(np.mean(positive_x)):
            continue
        patterns.append(tuple(zip(elements, signs, strict=True)))
        if len(patterns) > max_sign_patterns:
            return f"sign pattern count exceeds {max_sign_patterns}"
    return tuple(sorted(patterns))


def _brown_generic_strengths(structure, geometry) -> np.ndarray:
    from pymatgen.analysis.bond_valence import BV_PARAMS
    from pymatgen.core import Element

    strengths: list[float] = []
    for edge in geometry.edges:
        first = Element(structure[int(edge.cation)].specie.symbol)
        second = Element(structure[int(edge.anion)].specie.symbol)
        try:
            r1, c1 = float(BV_PARAMS[first]["r"]), float(BV_PARAMS[first]["c"])
            r2, c2 = float(BV_PARAMS[second]["r"]), float(BV_PARAMS[second]["c"])
            denominator = c1 * r1 + c2 * r2
            r0 = r1 + r2 - (
                r1
                * r2
                * (math.sqrt(c1) - math.sqrt(c2)) ** 2
                / denominator
            )
            strength = math.exp((r0 - float(edge.distance)) / 0.37)
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(
                f"Brown generic parameter failed for {first.symbol}-{second.symbol}: "
                f"{type(exc).__name__}"
            ) from exc
        if not math.isfinite(strength) or strength <= 0.0:
            raise ValueError(
                f"Brown generic strength is invalid for {first.symbol}-{second.symbol}"
            )
        strengths.append(float(strength))
    return np.asarray(strengths, dtype=float)


def compute_convex_mixed_valence_flow(
    structure,
    *,
    graph_mode: str,
    catalogue_mode: str,
    max_sign_patterns: int = MAX_SIGN_PATTERNS,
) -> ConvexMixedValenceFeatureResult:
    """Evaluate every frozen sign pattern on one untouched raw structure."""

    pymatgen_version = version("pymatgen")
    scipy_version = version("scipy")
    if graph_mode != "voronoi":
        raise ValueError("NEXT104 freezes graph_mode=voronoi")
    if not isinstance(max_sign_patterns, int) or max_sign_patterns < 1:
        raise ValueError("max_sign_patterns must be a positive integer")
    symbols = tuple(str(site.specie.symbol) for site in structure)
    catalogue = _state_catalogue(symbols, catalogue_mode)
    digest = _catalogue_digest(catalogue, catalogue_mode=catalogue_mode)
    if not symbols or len(set(symbols)) < 2:
        return _feature_failure(
            "mixed-valence flow needs at least two elements",
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

    from src.next19_valence_transport import build_periodic_edge_geometry

    candidates: list[tuple[tuple[float, ...], tuple[tuple[str, int], ...], dict[str, float]]] = []
    failures: list[str] = []
    for pattern in patterns:
        sign_by_element = dict(pattern)
        positive_sites = sum(sign_by_element[symbol] > 0 for symbol in symbols)
        negative_sites = len(symbols) - positive_sites
        dummy_charges = tuple(
            1.0 / positive_sites
            if sign_by_element[symbol] > 0
            else -1.0 / negative_sites
            for symbol in symbols
        )
        geometry = build_periodic_edge_geometry(
            structure,
            dummy_charges,
            graph_mode=graph_mode,
        )
        if not geometry.supported:
            failures.append(geometry.failure_reason or "periodic graph failed")
            continue
        signed_bounds: list[tuple[float, float]] = []
        normalized_width: list[float] = []
        for symbol in symbols:
            sign = sign_by_element[symbol]
            states = tuple(value for value in catalogue[symbol] if sign * value > 0)
            lower, upper = min(states), max(states)
            signed_bounds.append((float(lower), float(upper)))
            low_magnitude = min(abs(lower), abs(upper))
            high_magnitude = max(abs(lower), abs(upper))
            normalized_width.append(
                float((high_magnitude - low_magnitude) / (high_magnitude + low_magnitude))
            )
        try:
            priors = _brown_generic_strengths(structure, geometry)
            endpoints = np.asarray(
                [(int(edge.cation), int(edge.anion)) for edge in geometry.edges],
                dtype=int,
            )
            solution = solve_convex_mixed_valence_flow(
                signed_charge_bounds=signed_bounds,
                endpoints=endpoints,
                raw_priors=priors,
            )
        except (TypeError, ValueError) as exc:
            failures.append(f"certificate construction failed: {type(exc).__name__}: {exc}")
            continue
        if not solution.supported:
            failures.append(solution.failure_reason or "certificate solve failed")
            continue
        features = {
            "cmvf_reallocation": float(solution.reallocation),
            "cmvf_overload": float(solution.overload),
            "cmvf_log_scale_mismatch": float(solution.log_scale_mismatch),
            "cmvf_domain_width_mean": float(np.mean(normalized_width)),
            "cmvf_domain_width_max": float(np.max(normalized_width)),
            "cmvf_sign_pattern_log_count": float(math.log1p(len(patterns))),
        }
        rank = (
            features["cmvf_reallocation"],
            features["cmvf_overload"],
            features["cmvf_log_scale_mismatch"],
        )
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
    return ConvexMixedValenceFeatureResult(
        True,
        None,
        best,
        digest,
        pymatgen_version,
        scipy_version,
    )
