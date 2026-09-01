"""Frozen NEXT495 conservative applicability-domain extension of CCLAB."""

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
import src.next490_coordination_conditioned_lewis_acidity_balance as n490


PROTOCOL = "2026-08-13-next495-cclab-conservative-domain-extension-v1"
DESIGN_SHA256 = "26cefbfee766bd408064738a076e6b29730b44cd2a4b3a035a1426ae761ead04"
ASSET_SHA256 = n490.ASSET_SHA256
CHARACTERISTIC_STATES_BY_ELEMENT = n490.CHARACTERISTIC_STATES_BY_ELEMENT
FEATURE_NAMES = ("cclab_cde_conservative_domain_extension",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = n490.OUTPUT_GRID
BOUNDARY_FLAGS = dict(n490.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class CCLABCODEResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    cation_indices: tuple[int, ...]
    anion_indices: tuple[int, ...]
    cation_coordination: tuple[int, ...]
    unknown_cation_count: int
    unknown_anion_neighborhood_count: int
    projected_received: tuple[float, ...]
    anion_demand: tuple[float, ...]
    normalized_mismatch: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> CCLABCODEResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return CCLABCODEResult(
        False, reason, None, 0, 0, (), (), (), 0, 0, (), (), math.nan, None, {}
    )


def _zero(*, site_count: int, edge_count: int) -> CCLABCODEResult:
    return CCLABCODEResult(
        True,
        None,
        False,
        int(site_count),
        int(edge_count),
        (),
        (),
        (),
        0,
        0,
        (),
        (),
        1.0,
        None,
        {FEATURE_NAMES[0]: 0.0},
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _cclab_domain_extension(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
    penalize_unknown_neighborhood: bool,
) -> CCLABCODEResult:
    """Evaluate the frozen extension or its label-blind optimistic diagnostic."""

    try:
        charge = np.asarray(charges, dtype=float)
        symbol = tuple(str(item) for item in symbols)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT495 charged-site population differs")
        if len(symbol) != len(charge) or any(not item for item in symbol):
            raise ValueError("NEXT495 element-symbol population differs")
        if np.any(charge == 0.0):
            raise ValueError("NEXT495 every site must be formally charged")
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, float(np.abs(charge).sum())):
            raise ValueError("NEXT495 formal charges must be neutral")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            raise ValueError("NEXT495 charged-site signs differ")
        if raw_pair.size == 0:
            raise ValueError("NEXT495 requires an opposite-sign contact population")
        if raw_pair.ndim != 2 or raw_pair.shape[1] != 2:
            raise ValueError("NEXT495 contact endpoint population differs")
        if not np.issubdtype(raw_pair.dtype, np.integer):
            numeric = raw_pair.astype(float)
            if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.rint(numeric)):
                raise ValueError("NEXT495 contact endpoints must be integer indices")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= len(charge)):
            raise ValueError("NEXT495 contact endpoint index differs")
        if np.any(charge[pair[:, 0]] <= 0) or np.any(charge[pair[:, 1]] >= 0):
            raise ValueError("NEXT495 contacts must be ordered cation-to-anion")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree <= 0):
            raise ValueError("NEXT495 isolated charged site is unsupported")

        cations = tuple(int(index) for index in np.flatnonzero(charge > 0))
        anions = tuple(int(index) for index in np.flatnonzero(charge < 0))
        coordination = tuple(int(degree[index]) for index in cations)
        acidity: dict[int, tuple[float, float] | None] = {}
        for index, cn in zip(cations, coordination):
            states = CHARACTERISTIC_STATES_BY_ELEMENT.get(symbol[index])
            if states is None:
                acidity[index] = None
                continue
            distances = tuple(abs(float(cn) - state[1]) for state in states)
            minimum = min(distances)
            selected = tuple(
                state for state, distance in zip(states, distances) if distance == minimum
            )
            acidity[index] = (
                min(state[2] for state in selected),
                max(state[2] for state in selected),
            )

        incident = {index: [] for index in anions}
        for cation, anion in pair:
            incident[int(anion)].append(int(cation))
        demand = tuple(float(-charge[index]) for index in anions)
        projected: list[float] = []
        unknown_neighborhoods = 0
        for index, target in zip(anions, demand):
            donors = incident[index]
            has_unknown = any(acidity[donor] is None for donor in donors)
            if has_unknown:
                unknown_neighborhoods += 1
            if has_unknown and penalize_unknown_neighborhood:
                projected.append(0.0)
                continue
            known_donors = tuple(donor for donor in donors if acidity[donor] is not None)
            if not known_donors:
                projected.append(0.0)
                continue
            lower = math.fsum(acidity[donor][0] for donor in known_donors)  # type: ignore[index]
            upper = math.fsum(acidity[donor][1] for donor in known_donors)  # type: ignore[index]
            if lower <= 0.0 or upper < lower or not math.isfinite(lower + upper):
                raise RuntimeError("NEXT495 received acidity interval differs")
            projected.append(min(max(target, lower), upper))
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
            raise RuntimeError("NEXT495 normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        matching = _quantize(1.0 - mismatch)
        if not math.isfinite(matching) or matching < 0.0 or matching > 1.0:
            raise RuntimeError("NEXT495 bounded feature differs")
        return CCLABCODEResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            cations,
            anions,
            coordination,
            sum(acidity[index] is None for index in cations),
            unknown_neighborhoods,
            tuple(float(value) for value in projected),
            demand,
            mismatch,
            None,
            {FEATURE_NAMES[0]: matching},
        )
    except Exception as exc:
        return _failure(exc)


def cclab_conservative_domain_extension(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> CCLABCODEResult:
    """Apply maximum local mismatch where Appendix 3 lacks a donor element."""

    return _cclab_domain_extension(
        charges=charges,
        symbols=symbols,
        endpoints=endpoints,
        penalize_unknown_neighborhood=True,
    )


def cclab_ignore_unknown_contacts_diagnostic(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> CCLABCODEResult:
    """Optimistic label-blind comparator; never exposed as the candidate."""

    return _cclab_domain_extension(
        charges=charges,
        symbols=symbols,
        endpoints=endpoints,
        penalize_unknown_neighborhood=False,
    )


def _compute_cclab_cde_features(
    atoms: Atoms, *, optimistic_diagnostic: bool
) -> CCLABCODEResult:
    """Compute the extension or audit-only comparator from raw initial geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT495 valence assignment failed"
            )
        symbols = tuple(str(site.specie.symbol) for site in structure)
        charge = n490._representation_invariant_charges(
            assignment.values, symbols, str(assignment.policy)
        )
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "CCLAB-CDE periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        function = (
            cclab_ignore_unknown_contacts_diagnostic
            if optimistic_diagnostic
            else cclab_conservative_domain_extension
        )
        result = function(
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
            return replace(result, failure_reason=str(exc).replace("NEXT295", "NEXT495"))
        return result


def compute_cclab_cde_features(atoms: Atoms) -> CCLABCODEResult:
    """Compute conservative CCLAB extension from one raw initial geometry."""

    return _compute_cclab_cde_features(atoms, optimistic_diagnostic=False)


def compute_cclab_ignore_unknown_diagnostic(atoms: Atoms) -> CCLABCODEResult:
    """Compute the audit-only optimistic unknown-contact comparator."""

    return _compute_cclab_cde_features(atoms, optimistic_diagnostic=True)


def compute_cclab_cde_row(atoms: Atoms) -> dict[str, object]:
    result = compute_cclab_cde_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "cclab_cde_supported": bool(result.supported),
        "cclab_cde_failure": result.failure_reason,
        "cclab_cde_feasible": result.feasible,
        "cclab_cde_site_count": int(result.site_count),
        "cclab_cde_edge_count": int(result.edge_count),
        "cclab_cde_cation_count": int(len(result.cation_indices)),
        "cclab_cde_anion_count": int(len(result.anion_indices)),
        "cclab_cde_unknown_cation_count": int(result.unknown_cation_count),
        "cclab_cde_unknown_anion_neighborhood_count": int(
            result.unknown_anion_neighborhood_count
        ),
        "cclab_cde_normalized_mismatch": float(result.normalized_mismatch),
        "cclab_cde_valence_policy": result.valence_policy,
        "cclab_cde_asset_sha256": ASSET_SHA256,
    }
