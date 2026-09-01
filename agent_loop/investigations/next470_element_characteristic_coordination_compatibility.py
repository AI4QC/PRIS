"""Frozen NEXT470 element characteristic-CN compatibility (no DFT)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next295_positive_contact_force_closure as n295
import src.next460_characteristic_lewis_anion_matching as n460


PROTOCOL = "2026-08-13-next470-element-characteristic-coordination-compatibility-v1"
DESIGN_SHA256 = "2bc60aa284b61d650f9602168827d29bba9b80028bdd7de73f91ea169b710063"
ASSET_SHA256 = n460.ASSET_SHA256
FEATURE_NAMES = ("eccc_element_characteristic_coordination_compatibility",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n460.BOUNDARY_FLAGS)


def _load_characteristic_cn_sets() -> Mapping[str, tuple[float, ...]]:
    with n460.ASSET_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[str, set[float]] = {}
    for row in rows:
        values.setdefault(str(row["element"]), set()).add(
            float(row["characteristic_cn"])
        )
    return MappingProxyType(
        {element: tuple(sorted(numbers)) for element, numbers in values.items()}
    )


CHARACTERISTIC_CN_BY_ELEMENT = _load_characteristic_cn_sets()


@dataclass(frozen=True)
class ECCCResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    cation_indices: tuple[int, ...]
    coordination: tuple[int, ...]
    nearest_characteristic_cn: tuple[float, ...]
    normalized_mismatch: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> ECCCResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return ECCCResult(False, reason, None, 0, 0, (), (), (), math.nan, None, {})


def _zero(*, site_count: int, edge_count: int) -> ECCCResult:
    return ECCCResult(
        True,
        None,
        False,
        int(site_count),
        int(edge_count),
        (),
        (),
        (),
        1.0,
        None,
        {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def element_characteristic_coordination_compatibility(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> ECCCResult:
    """Compare cation contact counts with element-specific characteristic CNs."""

    try:
        charge = np.asarray(charges, dtype=float)
        symbol = tuple(str(item) for item in symbols)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT470 charged-site population differs")
        if len(symbol) != len(charge) or any(not item for item in symbol):
            raise ValueError("NEXT470 element-symbol population differs")
        if np.any(charge == 0.0):
            raise ValueError("NEXT470 every site must be formally charged")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            raise ValueError("NEXT470 charged-site signs differ")
        if raw_pair.size == 0:
            raise ValueError("NEXT470 requires an opposite-sign contact population")
        if raw_pair.ndim != 2 or raw_pair.shape[1] != 2:
            raise ValueError("NEXT470 contact endpoint population differs")
        if not np.issubdtype(raw_pair.dtype, np.integer):
            numeric = raw_pair.astype(float)
            if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.rint(numeric)):
                raise ValueError("NEXT470 contact endpoints must be integer indices")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= len(charge)):
            raise ValueError("NEXT470 contact endpoint index differs")
        if np.any(charge[pair[:, 0]] <= 0) or np.any(charge[pair[:, 1]] >= 0):
            raise ValueError("NEXT470 contacts must be ordered cation-to-anion")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree <= 0):
            raise ValueError("NEXT470 isolated charged site is unsupported")

        cations = tuple(int(index) for index in np.flatnonzero(charge > 0))
        coordination = tuple(int(degree[index]) for index in cations)
        nearest: list[float] = []
        for index, cn in zip(cations, coordination):
            element = symbol[index]
            if element not in CHARACTERISTIC_CN_BY_ELEMENT:
                raise ValueError(
                    f"NEXT470 characteristic coordination lookup missing for {element}"
                )
            nearest.append(
                min(
                    CHARACTERISTIC_CN_BY_ELEMENT[element],
                    key=lambda value: (abs(cn - value), value),
                )
            )
        numerator = math.fsum(
            abs(cn - reference) for cn, reference in zip(coordination, nearest)
        )
        denominator = math.fsum(
            cn + reference for cn, reference in zip(coordination, nearest)
        )
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or numerator < 0.0
            or denominator <= 0.0
            or numerator > denominator + 1.0e-10 * denominator
        ):
            raise RuntimeError("NEXT470 normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        matching = _quantize(1.0 - mismatch)
        if not math.isfinite(matching) or matching < 0.0 or matching > 1.0:
            raise RuntimeError("NEXT470 bounded feature differs")
        return ECCCResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            cations,
            coordination,
            tuple(float(value) for value in nearest),
            mismatch,
            None,
            {FEATURE_NAMES[0]: matching},
        )
    except Exception as exc:
        return _failure(exc)


def compute_eccc_features(atoms: Atoms) -> ECCCResult:
    """Compute ECCC from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT470 valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT470 valence population differs")
        geometry = n19.build_periodic_edge_geometry(structure, charge, graph_mode="voronoi")
        if not geometry.supported:
            reason = str(geometry.failure_reason or "ECCC periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        result = element_characteristic_coordination_compatibility(
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
            return replace(result, failure_reason=str(exc).replace("NEXT295", "NEXT470"))
        return result


def compute_eccc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_eccc_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "eccc_supported": bool(result.supported),
        "eccc_failure": result.failure_reason,
        "eccc_feasible": result.feasible,
        "eccc_site_count": int(result.site_count),
        "eccc_edge_count": int(result.edge_count),
        "eccc_cation_count": int(len(result.cation_indices)),
        "eccc_normalized_mismatch": float(result.normalized_mismatch),
        "eccc_valence_policy": result.valence_policy,
        "eccc_asset_sha256": ASSET_SHA256,
    }
