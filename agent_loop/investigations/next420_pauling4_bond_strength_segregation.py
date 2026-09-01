#!/usr/bin/env python3
"""Continuous Pauling-4 bond-strength segregation from one raw x0 geometry."""

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


PROTOCOL = "2026-08-13-next420-pauling4-bond-strength-segregation-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next420-next424-pauling4-bond-strength-segregation.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("p4bss_bond_strength_pair_avoidance",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
CHARGE_TOLERANCE = 1.0e-8
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class P4BSSResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    pair_count: int
    expected_product: float
    observed_product: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> P4BSSResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return P4BSSResult(False, reason, 0, 0, 0, math.nan, math.nan, None, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def pauling4_bond_strength_pair_avoidance(
    *,
    stub_strengths: Sequence[float] | object,
    anion_stub_indices: Sequence[Sequence[int]] | object,
) -> P4BSSResult:
    """Compare observed co-anion strength products with independent stubs."""

    try:
        strengths = np.asarray(stub_strengths, dtype=float)
        if (
            strengths.ndim != 1
            or len(strengths) < 2
            or not np.isfinite(strengths).all()
            or np.any(strengths <= 0.0)
        ):
            raise ValueError("P4BSS contact-stub strengths differ")
        if not isinstance(anion_stub_indices, Sequence) or not anion_stub_indices:
            raise ValueError("P4BSS anion contact groups differ")

        seen: list[int] = []
        products: list[float] = []
        for raw_group in anion_stub_indices:
            if not isinstance(raw_group, Sequence) or len(raw_group) < 1:
                raise ValueError("P4BSS anion contact group differs")
            group: list[int] = []
            for raw_index in raw_group:
                if isinstance(raw_index, (bool, np.bool_)):
                    raise ValueError("P4BSS contact-stub index differs")
                index = int(raw_index)
                if float(raw_index) != float(index) or index < 0 or index >= len(strengths):
                    raise ValueError("P4BSS contact-stub index differs")
                group.append(index)
            if len(set(group)) != len(group):
                raise ValueError("P4BSS anion group repeats a contact stub")
            seen.extend(group)
            products.extend(
                float(strengths[left] * strengths[right])
                for left, right in itertools.combinations(group, 2)
            )
        if sorted(seen) != list(range(len(strengths))):
            raise ValueError("P4BSS anion groups must partition contact stubs")
        if not products:
            raise ValueError("P4BSS graph has no co-anion contact pair")

        expected = float(np.mean(strengths)) ** 2
        observed = float(np.mean(np.asarray(products, dtype=float)))
        denominator = expected + observed
        if (
            not math.isfinite(expected)
            or not math.isfinite(observed)
            or expected <= 0.0
            or observed <= 0.0
            or not math.isfinite(denominator)
            or denominator <= 0.0
        ):
            raise RuntimeError("P4BSS strength-product moments differ")
        avoidance = _quantize(expected / denominator)
        if not math.isfinite(avoidance) or avoidance < 0.0 or avoidance > 1.0:
            raise RuntimeError("P4BSS feature domain differs")
        return P4BSSResult(
            True,
            None,
            0,
            len(strengths),
            len(products),
            expected,
            observed,
            None,
            {FEATURE_NAMES[0]: avoidance},
        )
    except Exception as exc:
        return _failure(exc)


def _periodic_stub_population(
    *, charges: np.ndarray, geometry: object
) -> tuple[tuple[float, ...], tuple[tuple[int, ...], ...]]:
    if charges.ndim != 1 or len(charges) < 2 or not np.isfinite(charges).all():
        raise ValueError("P4BSS charges must be a finite vector")
    magnitude = float(np.abs(charges).sum())
    if (
        magnitude <= 0.0
        or abs(float(charges.sum())) > CHARGE_TOLERANCE * max(1.0, magnitude)
        or not np.any(charges > 0.0)
        or not np.any(charges < 0.0)
        or np.any(charges == 0.0)
    ):
        raise ValueError("P4BSS formal charges must be neutral and nonzero")
    edges = tuple(geometry.edges)
    if not edges:
        raise ValueError("P4BSS periodic graph has no contacts")

    cation_counts = np.zeros(len(charges), dtype=np.int64)
    anion_groups: list[list[int]] = [[] for _ in range(len(charges))]
    for edge in edges:
        cation = int(edge.cation)
        anion = int(edge.anion)
        if (
            cation < 0
            or cation >= len(charges)
            or anion < 0
            or anion >= len(charges)
            or charges[cation] <= 0.0
            or charges[anion] >= 0.0
        ):
            raise ValueError("P4BSS cation-anion edge orientation differs")
        cation_counts[cation] += 1
    positive = charges > 0.0
    if np.any(cation_counts[positive] <= 0):
        raise ValueError("P4BSS periodic graph has a zero-degree cation")

    strengths: list[float] = []
    for edge in edges:
        cation = int(edge.cation)
        anion = int(edge.anion)
        strength = abs(float(charges[cation])) / int(cation_counts[cation])
        if not math.isfinite(strength) or strength <= 0.0:
            raise RuntimeError("P4BSS Pauling bond strength differs")
        stub = len(strengths)
        strengths.append(strength)
        anion_groups[anion].append(stub)
    if any(not anion_groups[index] for index in np.flatnonzero(charges < 0.0)):
        raise ValueError("P4BSS periodic graph has a zero-degree anion")
    return tuple(strengths), tuple(
        tuple(group) for group in anion_groups if group
    )


def compute_p4bss_features(atoms: Atoms) -> P4BSSResult:
    """Compute P4BSS from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "NEXT420 valence assignment failed")
        charges = np.asarray(assignment.values, dtype=float)
        if charges.shape != (len(structure),):
            raise ValueError("NEXT420 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charges, graph_mode="voronoi"
        )
        if not geometry.supported:
            raise ValueError(geometry.failure_reason or "P4BSS periodic graph failed")
        strengths, groups = _periodic_stub_population(
            charges=charges, geometry=geometry
        )
        result = pauling4_bond_strength_pair_avoidance(
            stub_strengths=strengths, anion_stub_indices=groups
        )
        if not result.supported:
            return result
        return replace(
            result,
            site_count=len(structure),
            valence_policy=str(assignment.policy),
        )
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason="NEXT420 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_p4bss_row(atoms: Atoms) -> dict[str, object]:
    result = compute_p4bss_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "p4bss_supported": bool(result.supported),
        "p4bss_failure": result.failure_reason,
        "p4bss_site_count": int(result.site_count),
        "p4bss_edge_count": int(result.edge_count),
        "p4bss_pair_count": int(result.pair_count),
        "p4bss_expected_product": float(result.expected_product),
        "p4bss_observed_product": float(result.observed_product),
        "p4bss_valence_policy": result.valence_policy,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "CHARGE_TOLERANCE",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "OUTPUT_GRID",
    "P4BSSResult",
    "PROTOCOL",
    "compute_p4bss_features",
    "compute_p4bss_row",
    "pauling4_bond_strength_pair_avoidance",
]
