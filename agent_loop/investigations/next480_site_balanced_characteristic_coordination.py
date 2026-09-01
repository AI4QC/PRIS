"""Frozen NEXT480 site-balanced characteristic coordination (no DFT)."""

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
import src.next475_characteristic_coordination_bottleneck as n475


PROTOCOL = "2026-08-13-next480-site-balanced-characteristic-coordination-v1"
DESIGN_SHA256 = "17ab08acda98c2c06ef9410e9cb34d0c89a4abcb068a609deeb08322a5274c50"
ASSET_SHA256 = n475.ASSET_SHA256
FEATURE_NAMES = ("sbcc_site_balanced_characteristic_coordination",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n475.BOUNDARY_FLAGS)
CHARACTERISTIC_CN_BY_ELEMENT = n475.CHARACTERISTIC_CN_BY_ELEMENT


@dataclass(frozen=True)
class SBCCResult:
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


def _failure(exc: object) -> SBCCResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return SBCCResult(False, reason, None, 0, 0, (), (), (), (), 0, None, {})


def _zero(*, site_count: int, edge_count: int) -> SBCCResult:
    return SBCCResult(
        True, None, False, int(site_count), int(edge_count), (), (), (), (), 0,
        None, {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def site_balanced_characteristic_coordination(
    *, charges: Sequence[float] | object, symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> SBCCResult:
    """Return the equal-site mean compatibility with characteristic CN."""

    local = n475.characteristic_coordination_bottleneck(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    if not local.supported:
        return _failure(ValueError(local.failure_reason or "NEXT480 local population failed"))
    if not local.feasible:
        return _zero(site_count=local.site_count, edge_count=local.edge_count)
    try:
        value = _quantize(math.fsum(local.site_compatibility) / len(local.site_compatibility))
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise RuntimeError("NEXT480 bounded feature differs")
        return SBCCResult(
            True, None, True, local.site_count, local.edge_count,
            local.cation_indices, local.coordination, local.nearest_characteristic_cn,
            local.site_compatibility, local.missing_element_count, None,
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_sbcc_features(atoms: Atoms) -> SBCCResult:
    """Compute SBCC from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "NEXT480 valence assignment failed")
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT480 valence population differs")
        geometry = n19.build_periodic_edge_geometry(structure, charge, graph_mode="voronoi")
        if not geometry.supported:
            reason = str(geometry.failure_reason or "SBCC periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(_zero(site_count=len(structure), edge_count=0), valence_policy=str(assignment.policy))
            raise ValueError(reason)
        result = site_balanced_characteristic_coordination(
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
            return replace(result, failure_reason=str(exc).replace("NEXT295", "NEXT480"))
        return result


def compute_sbcc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_sbcc_features(atoms)
    return {
        FEATURE_NAMES[0]: float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan,
        "sbcc_supported": bool(result.supported), "sbcc_failure": result.failure_reason,
        "sbcc_feasible": result.feasible, "sbcc_site_count": int(result.site_count),
        "sbcc_edge_count": int(result.edge_count), "sbcc_cation_count": int(len(result.cation_indices)),
        "sbcc_missing_element_count": int(result.missing_element_count),
        "sbcc_valence_policy": result.valence_policy, "sbcc_asset_sha256": ASSET_SHA256,
    }
