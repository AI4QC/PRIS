"""Frozen NEXT515 conservative applicability-domain extension of ECCC."""

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
import src.next470_element_characteristic_coordination_compatibility as n470


PROTOCOL = "2026-08-13-next515-eccc-conservative-domain-extension-v1"
DESIGN_SHA256 = "f6d6a35f44379471a9aff9e75b09a5e299c135fb56c0dd806bbc4d92b6bb10c9"
ASSET_SHA256 = n470.ASSET_SHA256
CHARACTERISTIC_CN_BY_ELEMENT = n470.CHARACTERISTIC_CN_BY_ELEMENT
FEATURE_NAMES = ("eccc_cde_conservative_domain_extension",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = n470.OUTPUT_GRID
BOUNDARY_FLAGS = dict(n470.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class ECCCDEResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    cation_indices: tuple[int, ...]
    coordination: tuple[int, ...]
    nearest_characteristic_cn: tuple[float, ...]
    unknown_cation_count: int
    isolated_site_count: int
    normalized_mismatch: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> ECCCDEResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return ECCCDEResult(
        False, reason, None, 0, 0, (), (), (), 0, 0, math.nan, None, {}
    )


def _zero(
    *,
    site_count: int,
    edge_count: int,
    cation_indices: tuple[int, ...] = (),
    coordination: tuple[int, ...] = (),
    unknown_cation_count: int = 0,
    isolated_site_count: int = 0,
) -> ECCCDEResult:
    return ECCCDEResult(
        True,
        None,
        False,
        int(site_count),
        int(edge_count),
        cation_indices,
        coordination,
        (),
        int(unknown_cation_count),
        int(isolated_site_count),
        1.0,
        None,
        {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _eccc_domain_extension(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
    ignore_unknown_cations: bool,
) -> ECCCDEResult:
    try:
        charge = np.asarray(charges, dtype=float)
        symbol = tuple(str(item) for item in symbols)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT515 charged-site population differs")
        if len(symbol) != len(charge) or any(not item for item in symbol):
            raise ValueError("NEXT515 element-symbol population differs")
        if np.any(charge == 0.0):
            raise ValueError("NEXT515 every site must be formally charged")
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, float(np.abs(charge).sum())):
            raise ValueError("NEXT515 formal charges must be neutral")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            raise ValueError("NEXT515 charged-site signs differ")
        if raw_pair.size == 0:
            return _zero(site_count=len(charge), edge_count=0, isolated_site_count=len(charge))
        if raw_pair.ndim != 2 or raw_pair.shape[1] != 2:
            raise ValueError("NEXT515 contact endpoint population differs")
        if not np.issubdtype(raw_pair.dtype, np.integer):
            numeric = raw_pair.astype(float)
            if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.rint(numeric)):
                raise ValueError("NEXT515 contact endpoints must be integer indices")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= len(charge)):
            raise ValueError("NEXT515 contact endpoint index differs")
        if np.any(charge[pair[:, 0]] <= 0) or np.any(charge[pair[:, 1]] >= 0):
            raise ValueError("NEXT515 contacts must be ordered cation-to-anion")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        isolated = int(np.sum(degree <= 0))
        cations = tuple(int(index) for index in np.flatnonzero(charge > 0))
        coordination = tuple(int(degree[index]) for index in cations)
        unknown = tuple(
            index for index in cations if symbol[index] not in CHARACTERISTIC_CN_BY_ELEMENT
        )
        if isolated:
            return _zero(
                site_count=len(charge),
                edge_count=len(pair),
                cation_indices=cations,
                coordination=coordination,
                unknown_cation_count=len(unknown),
                isolated_site_count=isolated,
            )
        if unknown and not ignore_unknown_cations:
            return _zero(
                site_count=len(charge),
                edge_count=len(pair),
                cation_indices=cations,
                coordination=coordination,
                unknown_cation_count=len(unknown),
            )

        retained = tuple(index for index in cations if index not in unknown)
        if not retained:
            return _zero(
                site_count=len(charge),
                edge_count=len(pair),
                cation_indices=cations,
                coordination=coordination,
                unknown_cation_count=len(unknown),
            )
        nearest: list[float] = []
        retained_coordination: list[int] = []
        for index in retained:
            cn = int(degree[index])
            nearest.append(
                min(
                    CHARACTERISTIC_CN_BY_ELEMENT[symbol[index]],
                    key=lambda value: (abs(cn - value), value),
                )
            )
            retained_coordination.append(cn)
        numerator = math.fsum(
            abs(cn - reference)
            for cn, reference in zip(retained_coordination, nearest)
        )
        denominator = math.fsum(
            cn + reference for cn, reference in zip(retained_coordination, nearest)
        )
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or numerator < 0.0
            or denominator <= 0.0
            or numerator > denominator + 1.0e-10 * denominator
        ):
            raise RuntimeError("NEXT515 normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        matching = _quantize(1.0 - mismatch)
        if not math.isfinite(matching) or matching < 0.0 or matching > 1.0:
            raise RuntimeError("NEXT515 bounded feature differs")
        return ECCCDEResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            cations,
            coordination,
            tuple(float(value) for value in nearest),
            len(unknown),
            0,
            mismatch,
            None,
            {FEATURE_NAMES[0]: matching},
        )
    except Exception as exc:
        return _failure(exc)


def eccc_conservative_domain_extension(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> ECCCDEResult:
    """Return exact ECCC in-domain and zero outside its empirical domain."""

    return _eccc_domain_extension(
        charges=charges,
        symbols=symbols,
        endpoints=endpoints,
        ignore_unknown_cations=False,
    )


def eccc_ignore_unknown_cations_diagnostic(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> ECCCDEResult:
    """Optimistic label-blind comparator; never exposed as the candidate."""

    return _eccc_domain_extension(
        charges=charges,
        symbols=symbols,
        endpoints=endpoints,
        ignore_unknown_cations=True,
    )


def _compute_eccc_cde_features(
    atoms: Atoms, *, optimistic_diagnostic: bool
) -> ECCCDEResult:
    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT515 valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT515 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "ECCC-CDE periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0, isolated_site_count=len(structure)),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        function = (
            eccc_ignore_unknown_cations_diagnostic
            if optimistic_diagnostic
            else eccc_conservative_domain_extension
        )
        result = function(
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
            return replace(result, failure_reason=str(exc).replace("NEXT295", "NEXT515"))
        return result


def compute_eccc_cde_features(atoms: Atoms) -> ECCCDEResult:
    return _compute_eccc_cde_features(atoms, optimistic_diagnostic=False)


def compute_eccc_ignore_unknown_diagnostic(atoms: Atoms) -> ECCCDEResult:
    return _compute_eccc_cde_features(atoms, optimistic_diagnostic=True)


def compute_eccc_cde_row(atoms: Atoms) -> dict[str, object]:
    result = compute_eccc_cde_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "eccc_cde_supported": bool(result.supported),
        "eccc_cde_failure": result.failure_reason,
        "eccc_cde_feasible": result.feasible,
        "eccc_cde_site_count": int(result.site_count),
        "eccc_cde_edge_count": int(result.edge_count),
        "eccc_cde_cation_count": int(len(result.cation_indices)),
        "eccc_cde_unknown_cation_count": int(result.unknown_cation_count),
        "eccc_cde_isolated_site_count": int(result.isolated_site_count),
        "eccc_cde_normalized_mismatch": float(result.normalized_mismatch),
        "eccc_cde_valence_policy": result.valence_policy,
        "eccc_cde_asset_sha256": ASSET_SHA256,
    }


__all__ = [
    "ASSET_SHA256",
    "BOUNDARY_FLAGS",
    "CHARACTERISTIC_CN_BY_ELEMENT",
    "DESIGN_SHA256",
    "ECCCDEResult",
    "FEATURE_DIRECTIONS",
    "FEATURE_NAMES",
    "PROTOCOL",
    "compute_eccc_cde_features",
    "compute_eccc_cde_row",
    "compute_eccc_ignore_unknown_diagnostic",
    "eccc_conservative_domain_extension",
    "eccc_ignore_unknown_cations_diagnostic",
]
