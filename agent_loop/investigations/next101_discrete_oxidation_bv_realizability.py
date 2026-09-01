#!/usr/bin/env python3
"""Discrete oxidation-state bond-valence realizability on one raw structure."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from typing import Mapping, Sequence

import numpy as np


PROTOCOL = "2026-08-04-next101-uniform-dobvr-v1"
FEATURE_NAMES = (
    "dobvr_neutral_assignment_count",
    "dobvr_supported_assignment_fraction",
    "dobvr_best_mismatch_rms",
    "dobvr_best_mismatch_q95",
    "dobvr_best_mismatch_max",
    "dobvr_median_mismatch_rms",
    "dobvr_runner_up_gap_rms",
    "dobvr_best_parameter_exact_fraction",
    "dobvr_best_parameter_generic_fraction",
    "dobvr_best_mean_abs_oxidation",
    "dobvr_best_max_abs_oxidation",
    "dobvr_assignment_log_count",
)
MAX_CARTESIAN_PRODUCT = 65_536
MAX_NEUTRAL_ASSIGNMENTS = 512


@dataclass(frozen=True, order=True)
class OxidationAssignment:
    """One canonical element-uniform integer oxidation-state assignment."""

    element_states: tuple[tuple[str, int], ...]
    site_charges: tuple[int, ...]


@dataclass(frozen=True)
class OxidationEnumerationResult:
    """Auditable result of bounded charge-neutral assignment enumeration."""

    supported: bool
    failure_reason: str | None
    assignments: tuple[OxidationAssignment, ...]
    catalogue_sha256: str
    pymatgen_version: str


@dataclass(frozen=True)
class DOBVRFeatureResult:
    """Fail-open ensemble evaluation for one unchanged raw structure."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]
    best_assignment: OxidationAssignment | None
    assignments: tuple[OxidationAssignment, ...]
    catalogue_sha256: str
    pymatgen_version: str


def _state_options(symbol: str) -> tuple[int, ...]:
    from pymatgen.core import Element

    element = Element(symbol)
    raw = tuple(element.common_oxidation_states) + tuple(
        element.icsd_oxidation_states
    )
    states: set[int] = set()
    for value in raw:
        scalar = float(value)
        integer = int(round(scalar))
        if math.isfinite(scalar) and scalar != 0.0 and scalar == integer:
            states.add(integer)
    return tuple(sorted(states))


def _normalize_catalogue(
    elements: Sequence[str],
    state_catalogue: Mapping[str, Sequence[int]] | None,
) -> tuple[dict[str, tuple[int, ...]], str | None]:
    catalogue: dict[str, tuple[int, ...]] = {}
    for symbol in elements:
        raw = _state_options(symbol) if state_catalogue is None else state_catalogue.get(symbol, ())
        states: set[int] = set()
        for value in raw:
            try:
                scalar = float(value)
                integer = int(round(scalar))
            except (TypeError, ValueError, OverflowError):
                return {}, f"oxidation-state catalogue is invalid for {symbol}"
            if not math.isfinite(scalar) or scalar == 0.0 or scalar != integer:
                return {}, f"oxidation-state catalogue is invalid for {symbol}"
            states.add(integer)
        if not states:
            return {}, f"oxidation-state catalogue is empty for {symbol}"
        catalogue[symbol] = tuple(sorted(states))
    return catalogue, None


def _catalogue_digest(catalogue: Mapping[str, Sequence[int]]) -> str:
    payload = json.dumps(
        {symbol: list(catalogue[symbol]) for symbol in sorted(catalogue)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def enumerate_uniform_neutral_assignments(
    structure,
    *,
    state_catalogue: Mapping[str, Sequence[int]] | None = None,
    max_cartesian_product: int = MAX_CARTESIAN_PRODUCT,
    max_neutral_assignments: int = MAX_NEUTRAL_ASSIGNMENTS,
) -> OxidationEnumerationResult:
    """Enumerate every bounded element-uniform neutral integer assignment."""

    pymatgen_version = version("pymatgen")
    symbols = tuple(str(site.specie.symbol) for site in structure)
    elements = tuple(sorted(set(symbols)))
    if not symbols or len(elements) < 2:
        return OxidationEnumerationResult(
            False,
            "no neutral assignment exists without at least two elements",
            (),
            _catalogue_digest({}),
            pymatgen_version,
        )
    catalogue, error = _normalize_catalogue(elements, state_catalogue)
    digest = _catalogue_digest(catalogue)
    if error is not None:
        return OxidationEnumerationResult(
            False, error, (), digest, pymatgen_version
        )
    if max_cartesian_product < 1 or max_neutral_assignments < 1:
        return OxidationEnumerationResult(
            False,
            "enumeration bounds must be positive",
            (),
            digest,
            pymatgen_version,
        )
    product_size = math.prod(len(catalogue[symbol]) for symbol in elements)
    if product_size > max_cartesian_product:
        return OxidationEnumerationResult(
            False,
            f"oxidation-state Cartesian product exceeds {max_cartesian_product}",
            (),
            digest,
            pymatgen_version,
        )
    counts = {symbol: symbols.count(symbol) for symbol in elements}
    assignments: list[OxidationAssignment] = []
    option_rows = (catalogue[symbol] for symbol in elements)
    for states in itertools.product(*option_rows):
        state_by_element = dict(zip(elements, states, strict=True))
        if sum(counts[symbol] * state_by_element[symbol] for symbol in elements) != 0:
            continue
        site_charges = tuple(state_by_element[symbol] for symbol in symbols)
        if not any(value > 0 for value in site_charges) or not any(
            value < 0 for value in site_charges
        ):
            continue
        assignments.append(
            OxidationAssignment(
                tuple(zip(elements, states, strict=True)),
                site_charges,
            )
        )
        if len(assignments) > max_neutral_assignments:
            return OxidationEnumerationResult(
                False,
                f"neutral assignment count exceeds {max_neutral_assignments}",
                (),
                digest,
                pymatgen_version,
            )
    canonical = tuple(sorted(assignments))
    if not canonical:
        return OxidationEnumerationResult(
            False,
            "no charge-neutral oxidation-state assignment exists",
            (),
            digest,
            pymatgen_version,
        )
    return OxidationEnumerationResult(
        True, None, canonical, digest, pymatgen_version
    )


def _feature_failure(
    reason: str,
    enumeration: OxidationEnumerationResult,
) -> DOBVRFeatureResult:
    return DOBVRFeatureResult(
        False,
        reason,
        {},
        None,
        enumeration.assignments,
        enumeration.catalogue_sha256,
        enumeration.pymatgen_version,
    )


def compute_discrete_oxidation_bv_realizability(
    structure,
    *,
    graph_mode: str,
    state_catalogue: Mapping[str, Sequence[int]] | None = None,
    max_cartesian_product: int = MAX_CARTESIAN_PRODUCT,
    max_neutral_assignments: int = MAX_NEUTRAL_ASSIGNMENTS,
) -> DOBVRFeatureResult:
    """Evaluate all frozen neutral assignments on one raw periodic structure."""

    enumeration = enumerate_uniform_neutral_assignments(
        structure,
        state_catalogue=state_catalogue,
        max_cartesian_product=max_cartesian_product,
        max_neutral_assignments=max_neutral_assignments,
    )
    if not enumeration.supported:
        return _feature_failure(
            enumeration.failure_reason or "oxidation-state enumeration failed",
            enumeration,
        )
    if graph_mode not in {"crystalnn", "voronoi"}:
        return _feature_failure("unsupported graph mode", enumeration)

    from src.next19_valence_transport import build_periodic_edge_geometry
    from src.next22_bond_valence_equilibrium import (
        bond_valence_features_from_periodic_geometry,
    )

    geometry_cache: dict[tuple[int, ...], object] = {}
    evaluated: list[tuple[OxidationAssignment, Mapping[str, float]]] = []
    failure_reasons: list[str] = []
    for assignment in enumeration.assignments:
        sign_pattern = tuple(
            1 if charge > 0 else -1 if charge < 0 else 0
            for charge in assignment.site_charges
        )
        geometry = geometry_cache.get(sign_pattern)
        if geometry is None:
            try:
                geometry = build_periodic_edge_geometry(
                    structure,
                    assignment.site_charges,
                    graph_mode=graph_mode,
                )
            except Exception as exc:
                failure_reasons.append(
                    f"periodic graph failed: {type(exc).__name__}"
                )
                continue
            geometry_cache[sign_pattern] = geometry
        result = bond_valence_features_from_periodic_geometry(
            structure,
            assignment.site_charges,
            geometry,
        )
        if result.supported:
            evaluated.append((assignment, result.features))
        else:
            failure_reasons.append(
                result.failure_reason or "bond-valence evaluation failed"
            )
    if not evaluated:
        detail = failure_reasons[0] if failure_reasons else "no assignment was evaluable"
        return _feature_failure(
            f"all neutral assignments are unsupported: {detail}",
            enumeration,
        )

    ranked = sorted(
        evaluated,
        key=lambda item: (
            float(item[1]["scbv_mismatch_rms"]),
            float(item[1]["scbv_mismatch_q95"]),
            float(item[1]["scbv_mismatch_max"]),
            item[0].element_states,
        ),
    )
    best_assignment, best = ranked[0]
    rms_values = [float(item[1]["scbv_mismatch_rms"]) for item in ranked]
    runner_up_gap = rms_values[1] - rms_values[0] if len(rms_values) > 1 else 0.0
    absolute_charges = [abs(value) for value in best_assignment.site_charges]
    assignment_count = len(enumeration.assignments)
    features = {
        "dobvr_neutral_assignment_count": float(assignment_count),
        "dobvr_supported_assignment_fraction": float(len(evaluated) / assignment_count),
        "dobvr_best_mismatch_rms": float(best["scbv_mismatch_rms"]),
        "dobvr_best_mismatch_q95": float(best["scbv_mismatch_q95"]),
        "dobvr_best_mismatch_max": float(best["scbv_mismatch_max"]),
        "dobvr_median_mismatch_rms": float(np.median(rms_values)),
        "dobvr_runner_up_gap_rms": float(max(0.0, runner_up_gap)),
        "dobvr_best_parameter_exact_fraction": float(
            best["scbv_parameter_exact_fraction"]
        ),
        "dobvr_best_parameter_generic_fraction": float(
            best["scbv_parameter_generic_fraction"]
        ),
        "dobvr_best_mean_abs_oxidation": float(np.mean(absolute_charges)),
        "dobvr_best_max_abs_oxidation": float(max(absolute_charges)),
        "dobvr_assignment_log_count": float(math.log(assignment_count)),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(
        list(features.values())
    ).all():
        return _feature_failure("computed feature schema is invalid", enumeration)
    return DOBVRFeatureResult(
        True,
        None,
        features,
        best_assignment,
        enumeration.assignments,
        enumeration.catalogue_sha256,
        enumeration.pymatgen_version,
    )


__all__ = [
    "FEATURE_NAMES",
    "MAX_CARTESIAN_PRODUCT",
    "MAX_NEUTRAL_ASSIGNMENTS",
    "PROTOCOL",
    "DOBVRFeatureResult",
    "OxidationAssignment",
    "OxidationEnumerationResult",
    "compute_discrete_oxidation_bv_realizability",
    "enumerate_uniform_neutral_assignments",
]
