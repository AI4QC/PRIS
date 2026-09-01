#!/usr/bin/env python3
"""Periodic equal-valence uniformity from one raw unrelaxed geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next307_periodic_bond_valence_hodge_loop as n307


PROTOCOL = "2026-08-13-next367-periodic-bond-valence-equal-uniformity-v1"
DESIGN_SHA256 = "c63b1042315a6df72a7368de31921f2f8e10cce67aa1e408a581bb5bd197132c"
FEATURE_NAMES = ("pbveu_equal_valence_uniformity_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PBVEUFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    minimum_degree: int
    maximum_degree: int
    valence_policy: str | None
    parameter_exact_fraction: float
    parameter_generic_fraction: float
    site_uniformities: np.ndarray
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PBVEUFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PBVEUFeatureResult(
        False, reason, 0, 0, 0, 0, None, math.nan, math.nan,
        np.empty(0, dtype=float), {},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _inverted_cdf(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    index = max(0, int(math.ceil(float(quantile) * len(ordered))) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def equal_valence_uniformity_features(
    *,
    n_sites: int,
    endpoints: object,
    bond_valences: object,
) -> PBVEUFeatureResult:
    """Evaluate exp(-KL(p || uniform)) on every periodic site star."""

    try:
        if type(n_sites) is not int or n_sites < 2:
            raise ValueError("PBVEU site count differs")
        raw_pair = np.asarray(endpoints)
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
            or not np.isfinite(raw_pair.astype(float)).all()
            or not np.equal(raw_pair.astype(float), np.rint(raw_pair.astype(float))).all()
        ):
            raise ValueError("PBVEU endpoint population differs")
        pair = raw_pair.astype(int)
        values = np.asarray(bond_valences, dtype=float)
        if (
            values.shape != (len(pair),)
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
        ):
            raise ValueError("PBVEU bond-valence population differs")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            raise ValueError("PBVEU endpoint indices differ")

        degrees = np.zeros(n_sites, dtype=int)
        incident: list[list[float]] = [[] for _ in range(n_sites)]
        for (left, right), value in zip(pair, values, strict=True):
            incident[int(left)].append(float(value))
            incident[int(right)].append(float(value))
            degrees[int(left)] += 1
            degrees[int(right)] += 1
        if np.any(degrees < 1):
            raise ValueError("PBVEU graph contains an isolated site")

        uniformities = np.empty(n_sites, dtype=float)
        for site, raw in enumerate(incident):
            weights = np.asarray(raw, dtype=float)
            probabilities = weights / math.fsum(float(value) for value in weights)
            entropy = -math.fsum(
                float(value) * math.log(float(value)) for value in probabilities
            )
            uniformities[site] = math.exp(entropy) / len(weights)
        if (
            not np.isfinite(uniformities).all()
            or np.any(uniformities <= 0.0)
            or np.any(uniformities > 1.0 + 1.0e-12)
        ):
            raise RuntimeError("PBVEU site uniformity domain differs")
        uniformities = np.clip(uniformities, 0.0, 1.0)
        feature = _quantize(_inverted_cdf(uniformities, 0.10))
        if not 0.0 < feature <= 1.0:
            raise RuntimeError("PBVEU aggregate domain differs")
        return PBVEUFeatureResult(
            True,
            None,
            n_sites,
            len(pair),
            int(degrees.min()),
            int(degrees.max()),
            None,
            math.nan,
            math.nan,
            uniformities,
            {FEATURE_NAMES[0]: feature},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pbveu_features(atoms: Atoms) -> PBVEUFeatureResult:
    """Compute PBVEU from composition and one raw periodic geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT367 valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        endpoints, bond_valences, sources = n307._resolved_bond_valence_field(
            structure, charges
        )
        result = equal_valence_uniformity_features(
            n_sites=len(structure),
            endpoints=endpoints,
            bond_valences=bond_valences,
        )
        if not result.supported:
            return result
        source_array = np.asarray(sources, dtype=object)
        return replace(
            result,
            valence_policy=str(assignment.policy),
            parameter_exact_fraction=float(np.mean(source_array == "exact")),
            parameter_generic_fraction=float(
                np.mean(np.isin(source_array, ("brown_generic", "radius_generic")))
            ),
        )
    except Exception as exc:
        return _failure(exc)


def compute_pbveu_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pbveu_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "pbveu_supported": bool(result.supported),
        "pbveu_failure": result.failure_reason,
        "pbveu_site_count": int(result.site_count),
        "pbveu_edge_count": int(result.edge_count),
        "pbveu_minimum_degree": int(result.minimum_degree),
        "pbveu_maximum_degree": int(result.maximum_degree),
        "pbveu_valence_policy": result.valence_policy,
        "pbveu_parameter_exact_fraction": result.parameter_exact_fraction,
        "pbveu_parameter_generic_fraction": result.parameter_generic_fraction,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "OUTPUT_GRID",
    "PBVEUFeatureResult",
    "PROTOCOL",
    "compute_pbveu_features",
    "compute_pbveu_row",
    "equal_valence_uniformity_features",
]
