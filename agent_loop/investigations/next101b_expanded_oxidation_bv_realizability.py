#!/usr/bin/env python3
"""Expanded-table uniform DOBVR with electronegativity orientation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from src.next101_discrete_oxidation_bv_realizability import (
    MAX_CARTESIAN_PRODUCT,
    MAX_NEUTRAL_ASSIGNMENTS,
    OxidationEnumerationResult,
    enumerate_uniform_neutral_assignments,
)


PROTOCOL = "2026-08-04-next101b-expanded-uniform-dobvr-v1"
FEATURE_NAMES = (
    "dobvrb_neutral_assignment_count",
    "dobvrb_supported_assignment_fraction",
    "dobvrb_best_mismatch_rms",
    "dobvrb_best_mismatch_q95",
    "dobvrb_best_mismatch_max",
    "dobvrb_median_mismatch_rms",
    "dobvrb_runner_up_gap_rms",
    "dobvrb_best_parameter_exact_fraction",
    "dobvrb_best_parameter_generic_fraction",
    "dobvrb_best_mean_abs_oxidation",
    "dobvrb_best_max_abs_oxidation",
    "dobvrb_assignment_log_count",
    "dobvrb_best_catalogue_tier",
    "dobvrb_core_assignment_fraction",
    "dobvrb_best_eneg_margin",
)


@dataclass(frozen=True)
class ExpandedOxidationAssignment:
    """One electronegativity-oriented assignment with catalogue provenance."""

    element_states: tuple[tuple[str, int], ...]
    site_charges: tuple[int, ...]
    catalogue_tier: int
    electronegativity_margin: float


@dataclass(frozen=True)
class ExpandedOxidationEnumerationResult:
    """Bounded all-table enumeration after the chemical orientation gate."""

    supported: bool
    failure_reason: str | None
    assignments: tuple[ExpandedOxidationAssignment, ...]
    catalogue_sha256: str
    pymatgen_version: str


@dataclass(frozen=True)
class ExpandedDOBVRFeatureResult:
    """Fail-open expanded DOBVR result for one unchanged raw structure."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]
    best_assignment: ExpandedOxidationAssignment | None
    assignments: tuple[ExpandedOxidationAssignment, ...]
    catalogue_sha256: str
    pymatgen_version: str


def _catalogues(structure) -> tuple[dict[str, tuple[int, ...]], dict[str, set[int]]]:
    from pymatgen.core import Element

    symbols = sorted({str(site.specie.symbol) for site in structure})
    expanded: dict[str, tuple[int, ...]] = {}
    core: dict[str, set[int]] = {}
    for symbol in symbols:
        element = Element(symbol)
        expanded[symbol] = tuple(
            sorted(
                {
                    int(value)
                    for value in element.oxidation_states
                    if float(value).is_integer() and int(value) != 0
                }
            )
        )
        core[symbol] = {
            int(value)
            for value in (
                *element.common_oxidation_states,
                *element.icsd_oxidation_states,
            )
            if float(value).is_integer() and int(value) != 0
        }
    return expanded, core


def _expanded_failure(
    reason: str,
    base: OxidationEnumerationResult,
) -> ExpandedOxidationEnumerationResult:
    return ExpandedOxidationEnumerationResult(
        False,
        reason,
        (),
        base.catalogue_sha256,
        base.pymatgen_version,
    )


def enumerate_expanded_uniform_neutral_assignments(
    structure,
    *,
    max_cartesian_product: int = MAX_CARTESIAN_PRODUCT,
    max_neutral_assignments: int = MAX_NEUTRAL_ASSIGNMENTS,
) -> ExpandedOxidationEnumerationResult:
    """Enumerate all-table assignments and reject reversed charge chemistry."""

    from pymatgen.core import Element

    expanded, core = _catalogues(structure)
    base = enumerate_uniform_neutral_assignments(
        structure,
        state_catalogue=expanded,
        max_cartesian_product=max_cartesian_product,
        max_neutral_assignments=max_neutral_assignments,
    )
    if not base.supported:
        return _expanded_failure(
            base.failure_reason or "expanded oxidation-state enumeration failed",
            base,
        )
    electronegativity: dict[str, float] = {}
    for symbol in expanded:
        try:
            value = float(Element(symbol).X)
        except (TypeError, ValueError):
            value = math.nan
        electronegativity[symbol] = value

    symbols = tuple(str(site.specie.symbol) for site in structure)
    assignments: list[ExpandedOxidationAssignment] = []
    for item in base.assignments:
        state_by_element = dict(item.element_states)
        positive_x = [
            electronegativity[symbol]
            for symbol, charge in zip(symbols, item.site_charges, strict=True)
            if charge > 0
        ]
        negative_x = [
            electronegativity[symbol]
            for symbol, charge in zip(symbols, item.site_charges, strict=True)
            if charge < 0
        ]
        if (
            not positive_x
            or not negative_x
            or not np.isfinite(positive_x).all()
            or not np.isfinite(negative_x).all()
        ):
            continue
        margin = float(np.mean(negative_x) - np.mean(positive_x))
        if not math.isfinite(margin) or margin <= 0.0:
            continue
        tier = max(
            int(state_by_element[symbol] not in core[symbol])
            for symbol in expanded
        )
        assignments.append(
            ExpandedOxidationAssignment(
                item.element_states,
                item.site_charges,
                tier,
                margin,
            )
        )
    canonical = tuple(sorted(assignments, key=lambda item: item.element_states))
    if not canonical:
        return _expanded_failure(
            "no neutral assignment passes electronegativity orientation",
            base,
        )
    return ExpandedOxidationEnumerationResult(
        True,
        None,
        canonical,
        base.catalogue_sha256,
        base.pymatgen_version,
    )


def _feature_failure(
    reason: str,
    enumeration: ExpandedOxidationEnumerationResult,
) -> ExpandedDOBVRFeatureResult:
    return ExpandedDOBVRFeatureResult(
        False,
        reason,
        {},
        None,
        enumeration.assignments,
        enumeration.catalogue_sha256,
        enumeration.pymatgen_version,
    )


def compute_expanded_discrete_oxidation_bv_realizability(
    structure,
    *,
    graph_mode: str,
    max_cartesian_product: int = MAX_CARTESIAN_PRODUCT,
    max_neutral_assignments: int = MAX_NEUTRAL_ASSIGNMENTS,
) -> ExpandedDOBVRFeatureResult:
    """Evaluate the tiered expanded ensemble on one raw periodic structure."""

    enumeration = enumerate_expanded_uniform_neutral_assignments(
        structure,
        max_cartesian_product=max_cartesian_product,
        max_neutral_assignments=max_neutral_assignments,
    )
    if not enumeration.supported:
        return _feature_failure(
            enumeration.failure_reason or "expanded enumeration failed",
            enumeration,
        )
    if graph_mode not in {"crystalnn", "voronoi"}:
        return _feature_failure("unsupported graph mode", enumeration)

    from src.next19_valence_transport import build_periodic_edge_geometry
    from src.next22_bond_valence_equilibrium import (
        bond_valence_features_from_periodic_geometry,
    )

    geometry_cache: dict[tuple[int, ...], object] = {}
    evaluated: list[tuple[ExpandedOxidationAssignment, Mapping[str, float]]] = []
    failures: list[str] = []
    for assignment in enumeration.assignments:
        sign_pattern = tuple(1 if value > 0 else -1 for value in assignment.site_charges)
        geometry = geometry_cache.get(sign_pattern)
        if geometry is None:
            try:
                geometry = build_periodic_edge_geometry(
                    structure,
                    assignment.site_charges,
                    graph_mode=graph_mode,
                )
            except Exception as exc:
                failures.append(f"periodic graph failed: {type(exc).__name__}")
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
            failures.append(result.failure_reason or "bond-valence evaluation failed")
    if not evaluated:
        detail = failures[0] if failures else "no assignment was evaluable"
        return _feature_failure(
            f"all expanded assignments are unsupported: {detail}", enumeration
        )

    ranked = sorted(
        evaluated,
        key=lambda item: (
            item[0].catalogue_tier,
            float(item[1]["scbv_mismatch_rms"]),
            float(item[1]["scbv_mismatch_q95"]),
            float(item[1]["scbv_mismatch_max"]),
            item[0].element_states,
        ),
    )
    best_assignment, best = ranked[0]
    rms_values = [float(item[1]["scbv_mismatch_rms"]) for item in ranked]
    same_tier_rms = [
        float(values["scbv_mismatch_rms"])
        for assignment, values in ranked
        if assignment.catalogue_tier == best_assignment.catalogue_tier
    ]
    runner_up_gap = (
        same_tier_rms[1] - same_tier_rms[0] if len(same_tier_rms) > 1 else 0.0
    )
    absolute = [abs(value) for value in best_assignment.site_charges]
    assignment_count = len(enumeration.assignments)
    features = {
        "dobvrb_neutral_assignment_count": float(assignment_count),
        "dobvrb_supported_assignment_fraction": float(len(evaluated) / assignment_count),
        "dobvrb_best_mismatch_rms": float(best["scbv_mismatch_rms"]),
        "dobvrb_best_mismatch_q95": float(best["scbv_mismatch_q95"]),
        "dobvrb_best_mismatch_max": float(best["scbv_mismatch_max"]),
        "dobvrb_median_mismatch_rms": float(np.median(rms_values)),
        "dobvrb_runner_up_gap_rms": float(max(0.0, runner_up_gap)),
        "dobvrb_best_parameter_exact_fraction": float(
            best["scbv_parameter_exact_fraction"]
        ),
        "dobvrb_best_parameter_generic_fraction": float(
            best["scbv_parameter_generic_fraction"]
        ),
        "dobvrb_best_mean_abs_oxidation": float(np.mean(absolute)),
        "dobvrb_best_max_abs_oxidation": float(max(absolute)),
        "dobvrb_assignment_log_count": float(math.log(assignment_count)),
        "dobvrb_best_catalogue_tier": float(best_assignment.catalogue_tier),
        "dobvrb_core_assignment_fraction": float(
            sum(item.catalogue_tier == 0 for item in enumeration.assignments)
            / assignment_count
        ),
        "dobvrb_best_eneg_margin": float(best_assignment.electronegativity_margin),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(
        list(features.values())
    ).all():
        return _feature_failure("computed expanded feature schema is invalid", enumeration)
    return ExpandedDOBVRFeatureResult(
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
    "PROTOCOL",
    "ExpandedDOBVRFeatureResult",
    "ExpandedOxidationAssignment",
    "ExpandedOxidationEnumerationResult",
    "compute_expanded_discrete_oxidation_bv_realizability",
    "enumerate_expanded_uniform_neutral_assignments",
]
