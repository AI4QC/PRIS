"""Frozen NEXT490 coordination-conditioned Lewis-acidity balance (no DFT)."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next460_characteristic_lewis_anion_matching as n460


PROTOCOL = "2026-08-13-next490-coordination-conditioned-lewis-acidity-balance-v1"
DESIGN_SHA256 = "1237ec348ea7b830935f831b7e5c0877497b3afa5e3c52815989b16e1b462ae6"
ASSET_SHA256 = n460.ASSET_SHA256
FEATURE_NAMES = ("cclab_coordination_conditioned_lewis_acidity_balance",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n460.BOUNDARY_FLAGS)


def _load_characteristic_states() -> Mapping[str, tuple[tuple[int, float, float], ...]]:
    values: dict[str, list[tuple[int, float, float]]] = {}
    with n460.ASSET_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        element = str(row["element"])
        state = (
            int(row["oxidation"]),
            float(row["characteristic_cn"]),
            float(row["acidity_e"]),
        )
        values.setdefault(element, []).append(state)
    return MappingProxyType(
        {element: tuple(sorted(states)) for element, states in values.items()}
    )


CHARACTERISTIC_STATES_BY_ELEMENT = _load_characteristic_states()


@dataclass(frozen=True)
class CCLABResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    cation_indices: tuple[int, ...]
    anion_indices: tuple[int, ...]
    cation_coordination: tuple[int, ...]
    nearest_characteristic_cn: tuple[float, ...]
    cation_acidity_lower: tuple[float, ...]
    cation_acidity_upper: tuple[float, ...]
    received_lower: tuple[float, ...]
    received_upper: tuple[float, ...]
    anion_demand: tuple[float, ...]
    normalized_mismatch: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> CCLABResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return CCLABResult(
        False, reason, None, 0, 0, (), (), (), (), (), (), (), (), (),
        math.nan, None, {},
    )


def _zero(*, site_count: int, edge_count: int) -> CCLABResult:
    return CCLABResult(
        True, None, False, int(site_count), int(edge_count), (), (), (), (),
        (), (), (), (), (), 1.0, None, {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _representation_invariant_charges(
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    policy: str,
) -> np.ndarray:
    """Undo only the whole-cell normalization of the fallback assignment."""

    charge = np.asarray(charges, dtype=float)
    symbol = tuple(str(item) for item in symbols)
    if charge.shape != (len(symbol),) or not np.isfinite(charge).all():
        raise ValueError("NEXT490 valence population differs")
    if str(policy) != "electronegativity_partition":
        return charge.copy()
    counts = tuple(Counter(symbol).values())
    formula_factor = math.gcd(*counts)
    return charge * float(formula_factor)


def coordination_conditioned_lewis_acidity_balance(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> CCLABResult:
    """Balance anion demand against CN-selected characteristic acidity."""

    try:
        charge = np.asarray(charges, dtype=float)
        symbol = tuple(str(item) for item in symbols)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT490 charged-site population differs")
        if len(symbol) != len(charge) or any(not item for item in symbol):
            raise ValueError("NEXT490 element-symbol population differs")
        if np.any(charge == 0.0):
            raise ValueError("NEXT490 every site must be formally charged")
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, float(np.abs(charge).sum())):
            raise ValueError("NEXT490 formal charges must be neutral")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            raise ValueError("NEXT490 charged-site signs differ")
        if raw_pair.size == 0:
            raise ValueError("NEXT490 requires an opposite-sign contact population")
        if raw_pair.ndim != 2 or raw_pair.shape[1] != 2:
            raise ValueError("NEXT490 contact endpoint population differs")
        if not np.issubdtype(raw_pair.dtype, np.integer):
            numeric = raw_pair.astype(float)
            if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.rint(numeric)):
                raise ValueError("NEXT490 contact endpoints must be integer indices")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= len(charge)):
            raise ValueError("NEXT490 contact endpoint index differs")
        if np.any(charge[pair[:, 0]] <= 0) or np.any(charge[pair[:, 1]] >= 0):
            raise ValueError("NEXT490 contacts must be ordered cation-to-anion")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree <= 0):
            raise ValueError("NEXT490 isolated charged site is unsupported")

        cations = tuple(int(index) for index in np.flatnonzero(charge > 0))
        anions = tuple(int(index) for index in np.flatnonzero(charge < 0))
        coordination = tuple(int(degree[index]) for index in cations)
        nearest_cn: list[float] = []
        lower_by_cation: dict[int, float] = {}
        upper_by_cation: dict[int, float] = {}
        for index, cn in zip(cations, coordination):
            element = symbol[index]
            states = CHARACTERISTIC_STATES_BY_ELEMENT.get(element)
            if states is None:
                raise ValueError(
                    f"NEXT490 characteristic Lewis acidity lookup missing for {element}"
                )
            distances = tuple(abs(float(cn) - state[1]) for state in states)
            minimum = min(distances)
            selected = tuple(
                state for state, distance in zip(states, distances) if distance == minimum
            )
            nearest_cn.append(min(state[1] for state in selected))
            lower_by_cation[index] = min(state[2] for state in selected)
            upper_by_cation[index] = max(state[2] for state in selected)

        received_lower_map = {index: [] for index in anions}
        received_upper_map = {index: [] for index in anions}
        for cation, anion in pair:
            received_lower_map[int(anion)].append(lower_by_cation[int(cation)])
            received_upper_map[int(anion)].append(upper_by_cation[int(cation)])
        received_lower = tuple(math.fsum(received_lower_map[index]) for index in anions)
        received_upper = tuple(math.fsum(received_upper_map[index]) for index in anions)
        demand = tuple(float(-charge[index]) for index in anions)
        projected = tuple(
            min(max(target, lower), upper)
            for lower, upper, target in zip(received_lower, received_upper, demand)
        )
        if any(
            lower <= 0.0 or upper < lower or not math.isfinite(lower + upper)
            for lower, upper in zip(received_lower, received_upper)
        ):
            raise RuntimeError("NEXT490 received acidity interval differs")
        numerator = math.fsum(
            abs(value - target) for value, target in zip(projected, demand)
        )
        denominator = math.fsum(
            value + target for value, target in zip(projected, demand)
        )
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or numerator < 0.0
            or denominator <= 0.0
            or numerator > denominator + 1.0e-10 * denominator
        ):
            raise RuntimeError("NEXT490 normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        matching = _quantize(1.0 - mismatch)
        if not math.isfinite(matching) or matching < 0.0 or matching > 1.0:
            raise RuntimeError("NEXT490 bounded feature differs")
        return CCLABResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            cations,
            anions,
            coordination,
            tuple(float(value) for value in nearest_cn),
            tuple(lower_by_cation[index] for index in cations),
            tuple(upper_by_cation[index] for index in cations),
            tuple(float(value) for value in received_lower),
            tuple(float(value) for value in received_upper),
            demand,
            mismatch,
            None,
            {FEATURE_NAMES[0]: matching},
        )
    except Exception as exc:
        return _failure(exc)


def compute_cclab_features(atoms: Atoms) -> CCLABResult:
    """Compute CCLAB from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT490 valence assignment failed"
            )
        symbols = tuple(str(site.specie.symbol) for site in structure)
        charge = _representation_invariant_charges(
            assignment.values, symbols, str(assignment.policy)
        )
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "CCLAB periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        result = coordination_conditioned_lewis_acidity_balance(
            charges=charge,
            symbols=symbols,
            endpoints=tuple(
                (int(edge.cation), int(edge.anion)) for edge in geometry.edges
            ),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(result, failure_reason=str(exc).replace("NEXT295", "NEXT490"))
        return result


def compute_cclab_row(atoms: Atoms) -> dict[str, object]:
    result = compute_cclab_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "cclab_supported": bool(result.supported),
        "cclab_failure": result.failure_reason,
        "cclab_feasible": result.feasible,
        "cclab_site_count": int(result.site_count),
        "cclab_edge_count": int(result.edge_count),
        "cclab_cation_count": int(len(result.cation_indices)),
        "cclab_anion_count": int(len(result.anion_indices)),
        "cclab_normalized_mismatch": float(result.normalized_mismatch),
        "cclab_valence_policy": result.valence_policy,
        "cclab_asset_sha256": ASSET_SHA256,
    }
