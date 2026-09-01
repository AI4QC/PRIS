#!/usr/bin/env python3
"""Periodic CHARDI backward charge-return consistency from raw geometry."""

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


PROTOCOL = "2026-08-13-next371-periodic-chardi-return-consistency-v1"
DESIGN_SHA256 = "4517c5a01f65293665c7029322cafa1878767248628a72ec33d9b035187014fa"
FEATURE_NAMES = ("pchardi_cation_return_mapd",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_low"}
MEAN_DISTANCE_TOLERANCE = 1.0e-12
MAXIMUM_ITERATIONS = 10_000
OUTPUT_GRID = 10_000_000_000
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class PCHARDIFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    cation_count: int
    anion_count: int
    iterations: int
    maximum_mean_distance_residual: float
    valence_policy: str | None
    anion_species: str | None
    anion_received_charges: np.ndarray
    cation_returned_charges: np.ndarray
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PCHARDIFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PCHARDIFeatureResult(
        False,
        reason,
        0,
        0,
        0,
        0,
        0,
        math.nan,
        None,
        None,
        np.empty(0, dtype=float),
        np.empty(0, dtype=float),
        {},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def chardi_return_consistency(
    *, charges: object, endpoints: object, distances: object
) -> PCHARDIFeatureResult:
    """Apply published homoligand CHARDI forward/backward charge equations."""

    try:
        charge = np.asarray(charges, dtype=float)
        raw_pair = np.asarray(endpoints)
        length = np.asarray(distances, dtype=float)
        if (
            charge.ndim != 1
            or len(charge) < 2
            or not np.isfinite(charge).all()
            or np.any(charge == 0.0)
            or not np.any(charge > 0.0)
            or not np.any(charge < 0.0)
        ):
            raise ValueError("PCHARDI formal charge population differs")
        if abs(math.fsum(charge.tolist())) > 1.0e-8 * max(
            1.0, math.fsum(np.abs(charge).tolist())
        ):
            raise ValueError("PCHARDI formal charges are not neutral")
        if (
            raw_pair.ndim != 2
            or raw_pair.shape[1:] != (2,)
            or len(raw_pair) < 1
        ):
            raise ValueError("PCHARDI endpoint population differs")
        try:
            numeric_pair = np.asarray(raw_pair, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PCHARDI endpoint population differs") from exc
        if (
            not np.isfinite(numeric_pair).all()
            or not np.equal(numeric_pair, np.rint(numeric_pair)).all()
        ):
            raise ValueError("PCHARDI endpoint population differs")
        pair = numeric_pair.astype(int)
        if (
            length.shape != (len(pair),)
            or not np.isfinite(length).all()
            or np.any(length <= 0.0)
        ):
            raise ValueError("PCHARDI distance population differs")
        n_sites = len(charge)
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
            or not np.all(charge[pair[:, 0]] > 0.0)
            or not np.all(charge[pair[:, 1]] < 0.0)
        ):
            raise ValueError("PCHARDI endpoints are not cation-to-anion edges")

        cations = np.flatnonzero(charge > 0.0)
        anions = np.flatnonzero(charge < 0.0)
        cation_row = {int(site): row for row, site in enumerate(cations)}
        anion_row = {int(site): row for row, site in enumerate(anions)}
        cation_edge_rows: list[np.ndarray] = []
        for site in cations:
            selected = np.flatnonzero(pair[:, 0] == site)
            if not len(selected):
                raise ValueError("PCHARDI cation has no periodic edge")
            cation_edge_rows.append(selected)
        for site in anions:
            if not np.any(pair[:, 1] == site):
                raise ValueError("PCHARDI anion has no periodic edge")

        means = np.asarray(
            [float(np.min(length[selected])) for selected in cation_edge_rows],
            dtype=float,
        )
        residual = math.inf
        iterations = 0
        converged = False
        for step in range(MAXIMUM_ITERATIONS):
            updated = np.empty_like(means)
            for row, selected in enumerate(cation_edge_rows):
                weights = np.exp(1.0 - (length[selected] / means[row]) ** 6)
                normalizer = math.fsum(weights.tolist())
                if not math.isfinite(normalizer) or normalizer <= 0.0:
                    raise ValueError("PCHARDI bond weights have zero mass")
                updated[row] = math.fsum(
                    (length[selected] * weights).tolist()
                ) / normalizer
            residual = float(np.max(np.abs(updated - means)))
            means = updated
            iterations = step + 1
            if residual <= MEAN_DISTANCE_TOLERANCE:
                converged = True
                break
        if not converged:
            raise ValueError("PCHARDI weighted mean distance did not converge")

        fractions = np.empty(len(pair), dtype=float)
        for row, selected in enumerate(cation_edge_rows):
            weights = np.exp(1.0 - (length[selected] / means[row]) ** 6)
            fractions[selected] = weights / math.fsum(weights.tolist())
        distributed = np.asarray(
            [abs(float(charge[cation])) * fraction for cation, fraction in zip(pair[:, 0], fractions, strict=True)],
            dtype=float,
        )
        received = np.zeros(len(anions), dtype=float)
        for edge, amount in enumerate(distributed):
            received[anion_row[int(pair[edge, 1])]] += float(amount)
        if not np.isfinite(received).all() or np.any(received <= 0.0):
            raise ValueError("PCHARDI anion received charge differs")

        returned = np.zeros(len(cations), dtype=float)
        for edge, amount in enumerate(distributed):
            cation = int(pair[edge, 0])
            anion = int(pair[edge, 1])
            returned[cation_row[cation]] += (
                float(amount)
                * abs(float(charge[anion]))
                / received[anion_row[anion]]
            )
        expected = np.abs(charge[cations])
        if not np.isfinite(returned).all() or np.any(returned <= 0.0):
            raise ValueError("PCHARDI returned cation charge differs")
        mapd = _quantize(float(np.mean(np.abs(expected - returned) / expected)))
        if not math.isfinite(mapd) or mapd < 0.0:
            raise RuntimeError("PCHARDI aggregate domain differs")
        return PCHARDIFeatureResult(
            True,
            None,
            n_sites,
            len(pair),
            len(cations),
            len(anions),
            iterations,
            residual,
            None,
            None,
            received,
            returned,
            {FEATURE_NAMES[0]: mapd},
        )
    except Exception as exc:
        return _failure(exc)


def compute_pchardi_features(atoms: Atoms) -> PCHARDIFeatureResult:
    """Compute the frozen CHARDI return MAPD from one raw geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT371 valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        negative_species = {
            str(structure[index].specie.symbol)
            for index in np.flatnonzero(charges < 0.0)
        }
        if len(negative_species) != 1:
            raise ValueError(
                "PCHARDI requires exactly one negative-valence chemical species"
            )
        geometry = n19.build_periodic_edge_geometry(
            structure, charges, graph_mode="voronoi"
        )
        if not geometry.supported:
            raise ValueError(
                geometry.failure_reason or "NEXT371 periodic graph failed"
            )
        endpoints = np.asarray(
            [(edge.cation, edge.anion) for edge in geometry.edges], dtype=int
        )
        distances = np.asarray(
            [edge.distance for edge in geometry.edges], dtype=float
        )
        result = chardi_return_consistency(
            charges=charges, endpoints=endpoints, distances=distances
        )
        if not result.supported:
            return result
        return replace(
            result,
            valence_policy=str(assignment.policy),
            anion_species=next(iter(negative_species)),
        )
    except Exception as exc:
        return _failure(exc)


def compute_pchardi_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pchardi_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "pchardi_supported": bool(result.supported),
        "pchardi_failure": result.failure_reason,
        "pchardi_site_count": int(result.site_count),
        "pchardi_edge_count": int(result.edge_count),
        "pchardi_cation_count": int(result.cation_count),
        "pchardi_anion_count": int(result.anion_count),
        "pchardi_iterations": int(result.iterations),
        "pchardi_maximum_mean_distance_residual": (
            result.maximum_mean_distance_residual
        ),
        "pchardi_valence_policy": result.valence_policy,
        "pchardi_anion_species": result.anion_species,
    }


__all__ = [
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "MAXIMUM_ITERATIONS",
    "MEAN_DISTANCE_TOLERANCE",
    "OUTPUT_GRID",
    "PCHARDIFeatureResult",
    "PROTOCOL",
    "chardi_return_consistency",
    "compute_pchardi_features",
    "compute_pchardi_row",
]
