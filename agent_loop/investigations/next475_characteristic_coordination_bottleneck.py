"""Frozen NEXT475 characteristic-coordination bottleneck (no DFT)."""

from __future__ import annotations

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
import src.next470_element_characteristic_coordination_compatibility as n470


PROTOCOL = "2026-08-13-next475-characteristic-coordination-bottleneck-v1"
DESIGN_SHA256 = "2040d819d398de30d58fc73426ff7b1f8b77bc0fbfa6590981c49b7439602dbe"
ASSET_SHA256 = n470.ASSET_SHA256
FEATURE_NAMES = ("cccb_characteristic_coordination_bottleneck",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n470.BOUNDARY_FLAGS)
CHARACTERISTIC_CN_BY_ELEMENT = MappingProxyType(
    {**dict(n470.CHARACTERISTIC_CN_BY_ELEMENT), "H": (2.03,)}
)


@dataclass(frozen=True)
class CCCBResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    cation_indices: tuple[int, ...]
    coordination: tuple[int, ...]
    nearest_characteristic_cn: tuple[float, ...]
    site_compatibility: tuple[float, ...]
    missing_element_count: int
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> CCCBResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return CCCBResult(False, reason, None, 0, 0, (), (), (), (), 0, None, {})


def _zero(*, site_count: int, edge_count: int) -> CCCBResult:
    return CCCBResult(
        True, None, False, int(site_count), int(edge_count), (), (), (), (), 0,
        None, {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def characteristic_coordination_bottleneck(
    *, charges: Sequence[float] | object, symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> CCCBResult:
    """Return the worst cation compatibility with published characteristic CN."""

    try:
        charge = np.asarray(charges, dtype=float)
        symbol = tuple(str(item) for item in symbols)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT475 charged-site population differs")
        if len(symbol) != len(charge) or any(not item for item in symbol):
            raise ValueError("NEXT475 element-symbol population differs")
        if np.any(charge == 0.0):
            raise ValueError("NEXT475 every site must be formally charged")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            raise ValueError("NEXT475 charged-site signs differ")
        if raw_pair.size == 0:
            raise ValueError("NEXT475 requires an opposite-sign contact population")
        if raw_pair.ndim != 2 or raw_pair.shape[1] != 2:
            raise ValueError("NEXT475 contact endpoint population differs")
        if not np.issubdtype(raw_pair.dtype, np.integer):
            numeric = raw_pair.astype(float)
            if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.rint(numeric)):
                raise ValueError("NEXT475 contact endpoints must be integer indices")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= len(charge)):
            raise ValueError("NEXT475 contact endpoint index differs")
        if np.any(charge[pair[:, 0]] <= 0) or np.any(charge[pair[:, 1]] >= 0):
            raise ValueError("NEXT475 contacts must be ordered cation-to-anion")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree <= 0):
            raise ValueError("NEXT475 isolated charged site is unsupported")
        cations = tuple(int(index) for index in np.flatnonzero(charge > 0))
        coordination = tuple(int(degree[index]) for index in cations)
        nearest, compatibility = [], []
        missing = 0
        for index, cn in zip(cations, coordination):
            options = CHARACTERISTIC_CN_BY_ELEMENT.get(symbol[index])
            if options is None:
                nearest.append(math.nan)
                compatibility.append(0.0)
                missing += 1
                continue
            reference = min(options, key=lambda value: (abs(cn - value), value))
            score = 1.0 - abs(cn - reference) / (cn + reference)
            nearest.append(float(reference))
            compatibility.append(float(score))
        bottleneck = _quantize(min(compatibility))
        if not math.isfinite(bottleneck) or bottleneck < 0.0 or bottleneck > 1.0:
            raise RuntimeError("NEXT475 bounded feature differs")
        return CCCBResult(
            True, None, True, len(charge), len(pair), cations, coordination,
            tuple(nearest), tuple(compatibility), missing, None,
            {FEATURE_NAMES[0]: bottleneck},
        )
    except Exception as exc:
        return _failure(exc)


def compute_cccb_features(atoms: Atoms) -> CCCBResult:
    """Compute CCCB from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "NEXT475 valence assignment failed")
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT475 valence population differs")
        geometry = n19.build_periodic_edge_geometry(structure, charge, graph_mode="voronoi")
        if not geometry.supported:
            reason = str(geometry.failure_reason or "CCCB periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(_zero(site_count=len(structure), edge_count=0), valence_policy=str(assignment.policy))
            raise ValueError(reason)
        result = characteristic_coordination_bottleneck(
            charges=charge,
            symbols=tuple(str(site.specie.symbol) for site in structure),
            endpoints=tuple((int(edge.cation), int(edge.anion)) for edge in geometry.edges),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(result, failure_reason=str(exc).replace("NEXT295", "NEXT475"))
        return result


def compute_cccb_row(atoms: Atoms) -> dict[str, object]:
    result = compute_cccb_features(atoms)
    return {
        FEATURE_NAMES[0]: float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan,
        "cccb_supported": bool(result.supported), "cccb_failure": result.failure_reason,
        "cccb_feasible": result.feasible, "cccb_site_count": int(result.site_count),
        "cccb_edge_count": int(result.edge_count), "cccb_cation_count": int(len(result.cation_indices)),
        "cccb_missing_element_count": int(result.missing_element_count),
        "cccb_valence_policy": result.valence_policy, "cccb_asset_sha256": ASSET_SHA256,
    }
