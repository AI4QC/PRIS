#!/usr/bin/env python3
"""Zachara nonlinear bond-valence-vector closure from one raw x0 geometry."""

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
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next395_periodic_bond_valence_tensor_isotropy as n395


PROTOCOL = "2026-08-13-next399-zachara-bond-valence-vector-closure-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next399-next402-zachara-bond-valence-vector-closure.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("zbvvc_zachara_vector_closure_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
ROUND_OFF_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class ZBVVCResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    minimum_degree: int
    maximum_degree: int
    valence_policy: str | None
    parameter_exact_fraction: float
    parameter_generic_fraction: float
    site_closure: np.ndarray
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> ZBVVCResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return ZBVVCResult(
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


def zachara_vector_closure(
    *,
    site_valences: object,
    endpoints: object,
    vectors: object,
    bond_valences: object,
) -> ZBVVCResult:
    """Evaluate normalized ``v=s(1-s/V)`` vector closure at every site."""

    try:
        valences = np.asarray(site_valences, dtype=float)
        if (
            valences.ndim != 1
            or len(valences) < 2
            or not np.isfinite(valences).all()
            or np.any(valences <= 0.0)
        ):
            raise ValueError("ZBVVC site-valence population differs")
        n_sites = len(valences)
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
            raise ValueError("ZBVVC endpoint population differs")
        pair = raw_pair.astype(int)
        displacement = np.asarray(vectors, dtype=float)
        strengths = np.asarray(bond_valences, dtype=float)
        if displacement.shape != (len(pair), 3) or not np.isfinite(displacement).all():
            raise ValueError("ZBVVC vector population differs")
        if (
            strengths.shape != (len(pair),)
            or not np.isfinite(strengths).all()
            or np.any(strengths <= 0.0)
        ):
            raise ValueError("ZBVVC bond-valence population differs")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            raise ValueError("ZBVVC endpoint indices differ")
        lengths = np.linalg.norm(displacement, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("ZBVVC vector lengths differ")
        directions = displacement / lengths[:, None]

        incident_strength = np.zeros(n_sites, dtype=float)
        degrees = np.zeros(n_sites, dtype=int)
        for edge, (left, right) in enumerate(pair):
            incident_strength[left] += float(strengths[edge])
            incident_strength[right] += float(strengths[edge])
            degrees[left] += 1
            degrees[right] += 1
        if np.any(degrees < 1) or np.any(incident_strength <= 0.0):
            raise ValueError("ZBVVC graph contains an isolated site")

        resultant = np.zeros((n_sites, 3), dtype=float)
        vector_sum = np.zeros(n_sites, dtype=float)
        for edge, (left, right) in enumerate(pair):
            for site, sign in ((int(left), 1.0), (int(right), -1.0)):
                share = float(strengths[edge] / incident_strength[site])
                if not 0.0 < share <= 1.0 + ROUND_OFF_TOLERANCE:
                    raise RuntimeError("ZBVVC normalized bond valence differs")
                share = float(np.clip(share, 0.0, 1.0))
                zachara_magnitude = float(
                    valences[site] * share * (1.0 - share)
                )
                resultant[site] += sign * zachara_magnitude * directions[edge]
                vector_sum[site] += zachara_magnitude

        closure = np.zeros(n_sites, dtype=float)
        active = vector_sum > ROUND_OFF_TOLERANCE
        closure[active] = 1.0 - (
            np.linalg.norm(resultant[active], axis=1) / vector_sum[active]
        )
        if (
            not np.isfinite(closure).all()
            or np.any(closure < -ROUND_OFF_TOLERANCE)
            or np.any(closure > 1.0 + ROUND_OFF_TOLERANCE)
        ):
            raise RuntimeError("ZBVVC site closure domain differs")
        closure = np.clip(closure, 0.0, 1.0)
        feature = _quantize(_inverted_cdf(closure, 0.10))
        if not 0.0 <= feature <= 1.0:
            raise RuntimeError("ZBVVC aggregate domain differs")
        return ZBVVCResult(
            True,
            None,
            n_sites,
            len(pair),
            int(degrees.min()),
            int(degrees.max()),
            None,
            math.nan,
            math.nan,
            closure,
            {FEATURE_NAMES[0]: feature},
        )
    except Exception as exc:
        return _failure(exc)


def compute_zbvvc_features(atoms: Atoms) -> ZBVVCResult:
    """Compute ZBVVC from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT399 valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        endpoints, vectors, strengths, sources = n395._resolved_bond_valence_vectors(
            structure, charges
        )
        result = zachara_vector_closure(
            site_valences=np.abs(charges),
            endpoints=endpoints,
            vectors=vectors,
            bond_valences=strengths,
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
                failure_reason="NEXT399 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_zbvvc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_zbvvc_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "zbvvc_supported": bool(result.supported),
        "zbvvc_failure": result.failure_reason,
        "zbvvc_site_count": int(result.site_count),
        "zbvvc_edge_count": int(result.edge_count),
        "zbvvc_minimum_degree": int(result.minimum_degree),
        "zbvvc_maximum_degree": int(result.maximum_degree),
        "zbvvc_valence_policy": result.valence_policy,
        "zbvvc_parameter_exact_fraction": result.parameter_exact_fraction,
        "zbvvc_parameter_generic_fraction": result.parameter_generic_fraction,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "OUTPUT_GRID",
    "PROTOCOL",
    "ZBVVCResult",
    "compute_zbvvc_features",
    "compute_zbvvc_row",
    "zachara_vector_closure",
]
