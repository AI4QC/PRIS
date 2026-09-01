#!/usr/bin/env python3
"""Bond-valence-weighted local direction isotropy from one raw x0 geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Mapping

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next38_bond_valence_transport_compatibility_features as n38
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next395-periodic-bond-valence-tensor-isotropy-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next395-next398-periodic-bond-valence-tensor-isotropy.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("pbvti_bond_valence_tensor_isotropy_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
ROUND_OFF_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PBVTIResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    minimum_degree: int
    maximum_degree: int
    valence_policy: str | None
    parameter_exact_fraction: float
    parameter_generic_fraction: float
    site_isotropy: np.ndarray
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PBVTIResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PBVTIResult(
        False,
        reason,
        0,
        0,
        0,
        0,
        None,
        math.nan,
        math.nan,
        np.empty(0, dtype=float),
        {},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _inverted_cdf(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    index = max(0, int(math.ceil(float(quantile) * len(ordered))) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def bond_valence_tensor_isotropy(
    *,
    n_sites: int,
    endpoints: object,
    vectors: object,
    bond_valences: object,
) -> PBVTIResult:
    """Evaluate the weakest normalized bond-valence direction tensor."""

    try:
        if type(n_sites) is not int or n_sites < 2:
            raise ValueError("PBVTI site count differs")
        raw_pair = np.asarray(endpoints)
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
            or not np.isfinite(raw_pair.astype(float)).all()
            or not np.equal(
                raw_pair.astype(float), np.rint(raw_pair.astype(float))
            ).all()
        ):
            raise ValueError("PBVTI endpoint population differs")
        pair = raw_pair.astype(int)
        displacement = np.asarray(vectors, dtype=float)
        values = np.asarray(bond_valences, dtype=float)
        if displacement.shape != (len(pair), 3) or not np.isfinite(displacement).all():
            raise ValueError("PBVTI vector population differs")
        if (
            values.shape != (len(pair),)
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
        ):
            raise ValueError("PBVTI bond-valence population differs")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            raise ValueError("PBVTI endpoint indices differ")
        lengths = np.linalg.norm(displacement, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("PBVTI vector lengths differ")
        directions = displacement / lengths[:, None]

        tensors = np.zeros((n_sites, 3, 3), dtype=float)
        weight_sums = np.zeros(n_sites, dtype=float)
        degrees = np.zeros(n_sites, dtype=int)
        for edge, (left, right) in enumerate(pair):
            weighted_outer = float(values[edge]) * np.outer(
                directions[edge], directions[edge]
            )
            tensors[left] += weighted_outer
            tensors[right] += weighted_outer
            weight_sums[left] += float(values[edge])
            weight_sums[right] += float(values[edge])
            degrees[left] += 1
            degrees[right] += 1
        if np.any(degrees < 1) or np.any(weight_sums <= 0.0):
            raise ValueError("PBVTI graph contains an isolated site")

        isotropy = np.empty(n_sites, dtype=float)
        for site in range(n_sites):
            tensor = tensors[site] / weight_sums[site]
            eigenvalues = np.linalg.eigvalsh(tensor)
            if (
                not np.isfinite(eigenvalues).all()
                or float(eigenvalues[0]) < -ROUND_OFF_TOLERANCE
                or float(eigenvalues[-1]) > 1.0 + ROUND_OFF_TOLERANCE
                or not math.isclose(
                    float(eigenvalues.sum()), 1.0, abs_tol=ROUND_OFF_TOLERANCE
                )
            ):
                raise RuntimeError("PBVTI local tensor spectrum differs")
            isotropy[site] = float(
                np.clip(3.0 * float(eigenvalues[0]), 0.0, 1.0)
            )
        feature = _quantize(_inverted_cdf(isotropy, 0.10))
        if not math.isfinite(feature) or not 0.0 <= feature <= 1.0:
            raise RuntimeError("PBVTI aggregate domain differs")
        return PBVTIResult(
            True,
            None,
            n_sites,
            len(pair),
            int(degrees.min()),
            int(degrees.max()),
            None,
            math.nan,
            math.nan,
            isotropy,
            {FEATURE_NAMES[0]: feature},
        )
    except Exception as exc:
        return _failure(exc)


def _resolved_bond_valence_vectors(
    structure, charges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    geometry = n19.build_periodic_edge_geometry(
        structure, charges, graph_mode="voronoi"
    )
    if not geometry.supported:
        raise ValueError(geometry.failure_reason or "PBVTI periodic graph failed")
    parameters = n38.bv_table()
    endpoints: list[tuple[int, int]] = []
    vectors: list[np.ndarray] = []
    values: list[float] = []
    sources: list[str] = []
    for edge in geometry.edges:
        cation = int(edge.cation)
        anion = int(edge.anion)
        key = (
            structure[cation].specie.symbol,
            int(round(float(charges[cation]))),
            structure[anion].specie.symbol,
            int(round(float(charges[anion]))),
        )
        resolved = n38.resolve_bond_valence_parameter(
            key, parameters, policy="frozen-fallback"
        )
        if resolved is None:
            cation_radius = n38._tabulated_radius(structure[cation].specie.symbol)
            anion_radius = n38._tabulated_radius(structure[anion].specie.symbol)
            if cation_radius is None or anion_radius is None:
                raise ValueError("PBVTI bond-valence and radius parameters are missing")
            resolved = (cation_radius + anion_radius, 0.37, "radius_generic")
        r0, decay, source = resolved
        if (
            not math.isfinite(float(r0))
            or not math.isfinite(float(decay))
            or float(decay) <= 0.0
            or str(source) not in n38.PARAMETER_SOURCES
        ):
            raise ValueError("PBVTI bond-valence parameter differs")
        fractional = (
            np.asarray(structure[anion].frac_coords, dtype=float)
            + np.asarray(edge.image, dtype=float)
            - np.asarray(structure[cation].frac_coords, dtype=float)
        )
        displacement = np.asarray(
            structure.lattice.get_cartesian_coords(fractional), dtype=float
        )
        distance = float(np.linalg.norm(displacement))
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("PBVTI edge distance differs")
        try:
            value = math.exp((float(r0) - distance) / float(decay))
        except OverflowError as exc:
            raise ValueError("PBVTI bond valence overflowed") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("PBVTI bond valence differs")
        endpoints.append((cation, anion))
        vectors.append(displacement)
        values.append(value)
        sources.append(str(source))
    return (
        np.asarray(endpoints, dtype=int),
        np.asarray(vectors, dtype=float),
        np.asarray(values, dtype=float),
        tuple(sources),
    )


def compute_pbvti_features(atoms: Atoms) -> PBVTIResult:
    """Compute PBVTI from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT395 valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        endpoints, vectors, values, sources = _resolved_bond_valence_vectors(
            structure, charges
        )
        result = bond_valence_tensor_isotropy(
            n_sites=len(structure),
            endpoints=endpoints,
            vectors=vectors,
            bond_valences=values,
        )
        if not result.supported:
            return result
        source_array = np.asarray(sources, dtype=object)
        return replace(
            result,
            valence_policy=str(assignment.policy),
            parameter_exact_fraction=float(np.mean(source_array == "exact")),
            parameter_generic_fraction=float(
                np.mean(
                    np.isin(source_array, ("brown_generic", "radius_generic"))
                )
            ),
        )
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason="NEXT395 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_pbvti_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pbvti_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "pbvti_supported": bool(result.supported),
        "pbvti_failure": result.failure_reason,
        "pbvti_site_count": int(result.site_count),
        "pbvti_edge_count": int(result.edge_count),
        "pbvti_minimum_degree": int(result.minimum_degree),
        "pbvti_maximum_degree": int(result.maximum_degree),
        "pbvti_valence_policy": result.valence_policy,
        "pbvti_parameter_exact_fraction": result.parameter_exact_fraction,
        "pbvti_parameter_generic_fraction": result.parameter_generic_fraction,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "OUTPUT_GRID",
    "PBVTIResult",
    "PROTOCOL",
    "bond_valence_tensor_isotropy",
    "compute_pbvti_features",
    "compute_pbvti_row",
]
