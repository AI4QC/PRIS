#!/usr/bin/env python3
"""Zachara nonlinear-over-linear vector gain from one raw x0 geometry."""

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


PROTOCOL = "2026-08-13-next403-zachara-over-linear-vector-gain-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next403-next406-zachara-over-linear-vector-gain.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("zbvvg_zachara_over_linear_gain_q10",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
ROUND_OFF_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class ZBVVGResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    minimum_degree: int
    maximum_degree: int
    valence_policy: str | None
    parameter_exact_fraction: float
    parameter_generic_fraction: float
    site_linear_closure: np.ndarray
    site_zachara_closure: np.ndarray
    site_gain: np.ndarray
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> ZBVVGResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    empty = np.empty(0, dtype=float)
    return ZBVVGResult(
        False,
        reason,
        0,
        0,
        0,
        0,
        None,
        math.nan,
        math.nan,
        empty,
        empty,
        empty,
        {},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _inverted_cdf(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    index = max(0, int(math.ceil(float(quantile) * len(ordered))) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def zachara_over_linear_vector_gain(
    *,
    site_valences: object,
    endpoints: object,
    vectors: object,
    bond_valences: object,
) -> ZBVVGResult:
    """Contrast site-normalized Zachara and traditional vector closure."""

    try:
        valences = np.asarray(site_valences, dtype=float)
        if (
            valences.ndim != 1
            or len(valences) < 2
            or not np.isfinite(valences).all()
            or np.any(valences <= 0.0)
        ):
            raise ValueError("ZBVVG site-valence population differs")
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
            raise ValueError("ZBVVG endpoint population differs")
        pair = raw_pair.astype(int)
        displacement = np.asarray(vectors, dtype=float)
        strengths = np.asarray(bond_valences, dtype=float)
        if displacement.shape != (len(pair), 3) or not np.isfinite(displacement).all():
            raise ValueError("ZBVVG vector population differs")
        if (
            strengths.shape != (len(pair),)
            or not np.isfinite(strengths).all()
            or np.any(strengths <= 0.0)
        ):
            raise ValueError("ZBVVG bond-valence population differs")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            raise ValueError("ZBVVG endpoint indices differ")
        lengths = np.linalg.norm(displacement, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("ZBVVG vector lengths differ")
        directions = displacement / lengths[:, None]

        incident_strength = np.zeros(n_sites, dtype=float)
        degrees = np.zeros(n_sites, dtype=int)
        for edge, (left, right) in enumerate(pair):
            incident_strength[left] += float(strengths[edge])
            incident_strength[right] += float(strengths[edge])
            degrees[left] += 1
            degrees[right] += 1
        if np.any(degrees < 1) or np.any(incident_strength <= 0.0):
            raise ValueError("ZBVVG graph contains an isolated site")

        linear_resultant = np.zeros((n_sites, 3), dtype=float)
        zachara_resultant = np.zeros((n_sites, 3), dtype=float)
        zachara_sum = np.zeros(n_sites, dtype=float)
        for edge, (left, right) in enumerate(pair):
            for site, sign in ((int(left), 1.0), (int(right), -1.0)):
                share = float(strengths[edge] / incident_strength[site])
                if not 0.0 < share <= 1.0 + ROUND_OFF_TOLERANCE:
                    raise RuntimeError("ZBVVG normalized bond valence differs")
                share = float(np.clip(share, 0.0, 1.0))
                nonlinear = share * (1.0 - share)
                linear_resultant[site] += sign * share * directions[edge]
                zachara_resultant[site] += sign * nonlinear * directions[edge]
                zachara_sum[site] += nonlinear
        linear = 1.0 - np.linalg.norm(linear_resultant, axis=1)
        zachara = np.zeros(n_sites, dtype=float)
        active = zachara_sum > ROUND_OFF_TOLERANCE
        zachara[active] = 1.0 - (
            np.linalg.norm(zachara_resultant[active], axis=1) / zachara_sum[active]
        )
        gain = (1.0 + zachara - linear) / 2.0
        for name, values in (("linear", linear), ("Zachara", zachara), ("gain", gain)):
            if (
                not np.isfinite(values).all()
                or np.any(values < -ROUND_OFF_TOLERANCE)
                or np.any(values > 1.0 + ROUND_OFF_TOLERANCE)
            ):
                raise RuntimeError(f"ZBVVG {name} domain differs")
        linear = np.clip(linear, 0.0, 1.0)
        zachara = np.clip(zachara, 0.0, 1.0)
        gain = np.clip(gain, 0.0, 1.0)
        feature = _quantize(_inverted_cdf(gain, 0.10))
        return ZBVVGResult(
            True,
            None,
            n_sites,
            len(pair),
            int(degrees.min()),
            int(degrees.max()),
            None,
            math.nan,
            math.nan,
            linear,
            zachara,
            gain,
            {FEATURE_NAMES[0]: feature},
        )
    except Exception as exc:
        return _failure(exc)


def compute_zbvvg_features(atoms: Atoms) -> ZBVVGResult:
    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT403 valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        endpoints, vectors, strengths, sources = n395._resolved_bond_valence_vectors(
            structure, charges
        )
        result = zachara_over_linear_vector_gain(
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
                failure_reason="NEXT403 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_zbvvg_row(atoms: Atoms) -> dict[str, object]:
    result = compute_zbvvg_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "zbvvg_supported": bool(result.supported),
        "zbvvg_failure": result.failure_reason,
        "zbvvg_site_count": int(result.site_count),
        "zbvvg_edge_count": int(result.edge_count),
        "zbvvg_minimum_degree": int(result.minimum_degree),
        "zbvvg_maximum_degree": int(result.maximum_degree),
        "zbvvg_valence_policy": result.valence_policy,
        "zbvvg_parameter_exact_fraction": result.parameter_exact_fraction,
        "zbvvg_parameter_generic_fraction": result.parameter_generic_fraction,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "OUTPUT_GRID",
    "PROTOCOL",
    "ZBVVGResult",
    "compute_zbvvg_features",
    "compute_zbvvg_row",
    "zachara_over_linear_vector_gain",
]
