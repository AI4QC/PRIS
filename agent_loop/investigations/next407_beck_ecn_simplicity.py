#!/usr/bin/env python3
"""Beck ECN integer-distribution simplicity from one raw x0 geometry."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295


PROTOCOL = "2026-08-13-next407-beck-ecn-simplicity-v1"
_DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/plans/2026-08-13-next407-next410-beck-ecn-simplicity.md"
)
DESIGN_SHA256 = hashlib.sha256(_DESIGN_PATH.read_bytes()).hexdigest()
FEATURE_NAMES = ("becns_beck_ecn_simplicity",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 10_000_000_000
ROUND_OFF_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class BECNSResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    cation_ecn_class_count: int
    anion_type_count: int
    beck_excess: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> BECNSResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return BECNSResult(False, reason, 0, 0, 0, 0, math.nan, None, {})


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def beck_ecn_simplicity(
    *,
    symbols: Sequence[str] | np.ndarray,
    formal_valences: Sequence[float] | np.ndarray,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
) -> BECNSResult:
    """Measure excess over Beck's adjacent-integer contact distribution."""

    try:
        symbol = np.asarray(symbols, dtype=object)
        charge = np.asarray(formal_valences, dtype=float)
        if (
            symbol.ndim != 1
            or len(symbol) < 2
            or charge.shape != symbol.shape
            or not np.isfinite(charge).all()
            or np.any(charge == 0.0)
            or any(not isinstance(value, str) or not value for value in symbol)
        ):
            raise ValueError("BECNS site population differs")
        magnitude = float(np.abs(charge).sum())
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            raise ValueError("BECNS formal valences are not neutral")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            raise ValueError("BECNS formal valences require both signs")

        raw_pair = np.asarray(endpoints)
        if raw_pair.ndim != 2 or raw_pair.shape[1:] != (2,) or len(raw_pair) < 1:
            raise ValueError("BECNS contact population differs")
        try:
            pair = raw_pair.astype(np.int64)
            numeric = raw_pair.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("BECNS contact population differs") from exc
        if (
            not np.isfinite(numeric).all()
            or not np.equal(numeric, pair).all()
            or np.any(pair < 0)
            or np.any(pair >= len(symbol))
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            raise ValueError("BECNS contact population differs")
        if np.any(charge[pair[:, 0]] <= 0.0) or np.any(charge[pair[:, 1]] >= 0.0):
            raise ValueError("BECNS contacts must be oriented cation to anion")

        degrees = np.zeros(len(symbol), dtype=int)
        np.add.at(degrees, pair[:, 0], 1)
        np.add.at(degrees, pair[:, 1], 1)
        if np.any(degrees < 1):
            raise ValueError("BECNS graph contains an isolated site")

        cation_class: dict[int, tuple[str, int, int]] = {}
        for site in np.flatnonzero(charge > 0.0):
            rounded = int(round(float(charge[site])))
            if not math.isclose(
                float(charge[site]), rounded, rel_tol=0.0, abs_tol=1.0e-8
            ):
                raise ValueError("BECNS formal cation valence is not integral")
            cation_class[int(site)] = (
                str(symbol[site]),
                rounded,
                int(degrees[site]),
            )

        anion_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for site in np.flatnonzero(charge < 0.0):
            rounded = int(round(float(charge[site])))
            if not math.isclose(
                float(charge[site]), rounded, rel_tol=0.0, abs_tol=1.0e-8
            ):
                raise ValueError("BECNS formal anion valence is not integral")
            anion_groups[(str(symbol[site]), rounded)].append(int(site))

        contact_counts: Counter[tuple[int, tuple[str, int, int]]] = Counter()
        for cation, anion in pair:
            contact_counts[(int(anion), cation_class[int(cation)])] += 1

        excess = 0.0
        classes = tuple(sorted(set(cation_class.values())))
        for sites in anion_groups.values():
            population = len(sites)
            if population < 1:
                raise RuntimeError("BECNS anion population differs")
            for group in classes:
                counts = np.fromiter(
                    (contact_counts[(site, group)] for site in sites),
                    dtype=float,
                    count=population,
                )
                total = int(round(float(counts.sum())))
                if total == 0:
                    continue
                mean = total / population
                observed = float(np.sum((counts - mean) ** 2))
                remainder = total % population
                minimum = remainder * (population - remainder) / population
                column_excess = observed - minimum
                if column_excess < -ROUND_OFF_TOLERANCE:
                    raise RuntimeError("BECNS integer minimum exceeds observation")
                excess += max(0.0, column_excess)

        if not math.isfinite(excess) or excess < 0.0:
            raise RuntimeError("BECNS segregation excess differs")
        simplicity = 1.0 / (1.0 + excess / len(pair))
        if (
            not math.isfinite(simplicity)
            or simplicity <= 0.0
            or simplicity > 1.0 + ROUND_OFF_TOLERANCE
        ):
            raise RuntimeError("BECNS simplicity domain differs")
        simplicity = _quantize(min(1.0, simplicity))
        return BECNSResult(
            True,
            None,
            len(symbol),
            len(pair),
            len(classes),
            len(anion_groups),
            float(excess),
            None,
            {FEATURE_NAMES[0]: simplicity},
        )
    except Exception as exc:
        return _failure(exc)


def _opposite_sign_endpoints(structure, charges: np.ndarray) -> np.ndarray:
    geometry = n19.build_periodic_edge_geometry(
        structure, charges, graph_mode="voronoi"
    )
    if not geometry.supported:
        raise ValueError(geometry.failure_reason or "BECNS periodic graph failed")
    endpoints = np.asarray(
        [(int(edge.cation), int(edge.anion)) for edge in geometry.edges],
        dtype=int,
    )
    if endpoints.ndim != 2 or endpoints.shape[1:] != (2,) or not len(endpoints):
        raise ValueError("BECNS periodic graph contains no contact")
    return endpoints


def compute_becns_features(atoms: Atoms) -> BECNSResult:
    """Compute BECNS from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT407 valence assignment failed"
            )
        charges = np.asarray(assignment.values, dtype=float)
        result = beck_ecn_simplicity(
            symbols=tuple(site.specie.symbol for site in structure),
            formal_valences=charges,
            endpoints=_opposite_sign_endpoints(structure, charges),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason="NEXT407 features require exact periodic geometry-only Atoms",
            )
        return result


def compute_becns_row(atoms: Atoms) -> dict[str, object]:
    result = compute_becns_features(atoms)
    return {
        FEATURE_NAMES[0]: result.features.get(FEATURE_NAMES[0], math.nan),
        "becns_supported": bool(result.supported),
        "becns_failure": result.failure_reason,
        "becns_site_count": int(result.site_count),
        "becns_edge_count": int(result.edge_count),
        "becns_cation_ecn_class_count": int(result.cation_ecn_class_count),
        "becns_anion_type_count": int(result.anion_type_count),
        "becns_beck_excess": result.beck_excess,
        "becns_valence_policy": result.valence_policy,
    }


__all__ = [
    "BECNSResult",
    "BOUNDARY_FLAGS",
    "DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "OUTPUT_GRID",
    "PROTOCOL",
    "beck_ecn_simplicity",
    "compute_becns_features",
    "compute_becns_row",
]
