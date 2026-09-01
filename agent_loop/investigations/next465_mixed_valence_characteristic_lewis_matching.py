"""Frozen NEXT465 mixed-valence characteristic-Lewis matching (no DFT)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next460_characteristic_lewis_anion_matching as n460


PROTOCOL = "2026-08-13-next465-mixed-valence-characteristic-lewis-matching-v1"
DESIGN_SHA256 = "86c99667ece5f91517ee4fe8de452e6f3ac3d88910785781059e4d1c2d6b6623"
ASSET_SHA256 = n460.ASSET_SHA256
FEATURE_NAMES = ("mvclam_mixed_valence_characteristic_lewis_matching",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n460.BOUNDARY_FLAGS)


_STATES_BY_ELEMENT: dict[str, tuple[int, ...]] = {}
for _element, _oxidation in n460.CHARACTERISTIC_LEWIS_ACIDITY:
    _STATES_BY_ELEMENT.setdefault(_element, ())
    _STATES_BY_ELEMENT[_element] = tuple(
        sorted((*_STATES_BY_ELEMENT[_element], _oxidation))
    )


def mixed_valence_characteristic_acidity(element: str, charge: float) -> float:
    """Return a printed value or the unique adjacent-state lever interpolation."""

    symbol = str(element)
    value = float(charge)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("NEXT465 positive cation charge differs")
    rounded = int(np.rint(value))
    if abs(value - rounded) <= 1.0e-8:
        key = (symbol, rounded)
        if key not in n460.CHARACTERISTIC_LEWIS_ACIDITY:
            raise ValueError(f"NEXT465 characteristic acidity lookup missing for {symbol}{rounded}+")
        return float(n460.CHARACTERISTIC_LEWIS_ACIDITY[key])
    states = _STATES_BY_ELEMENT.get(symbol, ())
    lower = tuple(state for state in states if state < value)
    upper = tuple(state for state in states if state > value)
    if not lower or not upper:
        raise ValueError(f"NEXT465 characteristic acidity bracket missing for {symbol}{value:g}+")
    low = max(lower)
    high = min(upper)
    low_value = float(n460.CHARACTERISTIC_LEWIS_ACIDITY[(symbol, low)])
    high_value = float(n460.CHARACTERISTIC_LEWIS_ACIDITY[(symbol, high)])
    weight = (value - low) / (high - low)
    acidity = (1.0 - weight) * low_value + weight * high_value
    if not math.isfinite(acidity) or acidity <= 0.0:
        raise RuntimeError("NEXT465 lever-rule acidity differs")
    return float(acidity)


@dataclass(frozen=True)
class MVCLAMResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    cation_indices: tuple[int, ...]
    anion_indices: tuple[int, ...]
    cation_acidity: tuple[float, ...]
    received_acidity: tuple[float, ...]
    anion_demand: tuple[float, ...]
    normalized_mismatch: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> MVCLAMResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return MVCLAMResult(
        False, reason, None, 0, 0, (), (), (), (), (), math.nan, None, {}
    )


def _zero(*, site_count: int, edge_count: int) -> MVCLAMResult:
    return MVCLAMResult(
        True,
        None,
        False,
        int(site_count),
        int(edge_count),
        (),
        (),
        (),
        (),
        (),
        1.0,
        None,
        {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def mixed_valence_characteristic_lewis_matching(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> MVCLAMResult:
    """Match lever-interpolated mixed-valence acidities to anion demand."""

    try:
        charge = np.asarray(charges, dtype=float)
        symbol = tuple(str(item) for item in symbols)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT465 charged-site population differs")
        if len(symbol) != len(charge) or any(not item for item in symbol):
            raise ValueError("NEXT465 element-symbol population differs")
        if np.any(charge == 0.0):
            raise ValueError("NEXT465 every site must be formally charged")
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, float(np.abs(charge).sum())):
            raise ValueError("NEXT465 formal charges must be neutral")
        if raw_pair.size == 0:
            raise ValueError("NEXT465 requires an opposite-sign contact population")
        if raw_pair.ndim != 2 or raw_pair.shape[1] != 2:
            raise ValueError("NEXT465 contact endpoint population differs")
        if not np.issubdtype(raw_pair.dtype, np.integer):
            numeric = raw_pair.astype(float)
            if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.rint(numeric)):
                raise ValueError("NEXT465 contact endpoints must be integer indices")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= len(charge)):
            raise ValueError("NEXT465 contact endpoint index differs")
        if np.any(charge[pair[:, 0]] <= 0) or np.any(charge[pair[:, 1]] >= 0):
            raise ValueError("NEXT465 contacts must be ordered cation-to-anion")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree <= 0):
            raise ValueError("NEXT465 isolated charged site is unsupported")

        cations = tuple(int(index) for index in np.flatnonzero(charge > 0))
        anions = tuple(int(index) for index in np.flatnonzero(charge < 0))
        acidity_by_cation = {
            index: mixed_valence_characteristic_acidity(symbol[index], charge[index])
            for index in cations
        }
        received_by_anion = {index: [] for index in anions}
        for cation, anion in pair:
            received_by_anion[int(anion)].append(acidity_by_cation[int(cation)])
        received = tuple(math.fsum(received_by_anion[index]) for index in anions)
        demand = tuple(float(-charge[index]) for index in anions)
        if any(value <= 0.0 or not math.isfinite(value) for value in received):
            raise RuntimeError("NEXT465 received acidity population differs")
        numerator = math.fsum(abs(value - target) for value, target in zip(received, demand))
        denominator = math.fsum(value + target for value, target in zip(received, demand))
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or numerator < 0.0
            or denominator <= 0.0
            or numerator > denominator + 1.0e-10 * denominator
        ):
            raise RuntimeError("NEXT465 normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        matching = _quantize(1.0 - mismatch)
        if not math.isfinite(matching) or matching < 0.0 or matching > 1.0:
            raise RuntimeError("NEXT465 bounded feature differs")
        return MVCLAMResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            cations,
            anions,
            tuple(acidity_by_cation[index] for index in cations),
            tuple(float(value) for value in received),
            demand,
            mismatch,
            None,
            {FEATURE_NAMES[0]: matching},
        )
    except Exception as exc:
        return _failure(exc)


def compute_mvclam_features(atoms: Atoms) -> MVCLAMResult:
    """Compute MV-CLAM from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT465 valence assignment failed"
            )
        if assignment.policy == "electronegativity_partition":
            raise ValueError("NEXT465 electronegativity partition is unsupported")
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT465 valence population differs")
        geometry = n19.build_periodic_edge_geometry(structure, charge, graph_mode="voronoi")
        if not geometry.supported:
            reason = str(geometry.failure_reason or "MVCLAM periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        result = mixed_valence_characteristic_lewis_matching(
            charges=charge,
            symbols=tuple(str(site.specie.symbol) for site in structure),
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
            return replace(result, failure_reason=str(exc).replace("NEXT295", "NEXT465"))
        return result


def compute_mvclam_row(atoms: Atoms) -> dict[str, object]:
    result = compute_mvclam_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "mvclam_supported": bool(result.supported),
        "mvclam_failure": result.failure_reason,
        "mvclam_feasible": result.feasible,
        "mvclam_site_count": int(result.site_count),
        "mvclam_edge_count": int(result.edge_count),
        "mvclam_cation_count": int(len(result.cation_indices)),
        "mvclam_anion_count": int(len(result.anion_indices)),
        "mvclam_normalized_mismatch": float(result.normalized_mismatch),
        "mvclam_valence_policy": result.valence_policy,
        "mvclam_asset_sha256": ASSET_SHA256,
    }
