"""Frozen NEXT460 characteristic-Lewis anion matching (no DFT)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import hashlib
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
import src.next440_path_constrained_apriori_bond_positivity as n440


PROTOCOL = "2026-08-13-next460-characteristic-lewis-anion-matching-v1"
DESIGN_SHA256 = "ae0da92aa5abfcd3a116a9899f5eaa04b2fd5119d74a60aa3d1087ac6f16d090"
ASSET_SHA256 = "1f8cceb8eaade9368f96aefbf8da5e5665627c02271641cf079e199da70e4c9c"
ASSET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "characteristic_lewis_acidity_hawthorne_2026.csv"
)
FEATURE_NAMES = ("clam_characteristic_lewis_anion_matching",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n440.BOUNDARY_FLAGS)


def _load_frozen_asset() -> Mapping[tuple[str, int], float]:
    payload = ASSET_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ASSET_SHA256:
        raise RuntimeError("NEXT460 characteristic-acidity asset SHA-256 differs")
    with ASSET_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 134:
        raise RuntimeError("NEXT460 characteristic-acidity asset row count differs")
    table: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (str(row["element"]), int(row["oxidation"]))
        value = float(row["acidity_e"])
        cn = float(row["characteristic_cn"])
        if (
            key in table
            or key[0] == "H"
            or key[1] <= 0
            or not math.isfinite(value)
            or value <= 0.0
            or not math.isfinite(cn)
            or cn <= 0.0
            or abs(key[1] / cn - value) > 0.06
            or row["source_doi"] != "10.1180/mgm.2026.10215"
            or row["license"] != "CC-BY-4.0"
        ):
            raise RuntimeError("NEXT460 characteristic-acidity asset row differs")
        table[key] = value
    return MappingProxyType(table)


CHARACTERISTIC_LEWIS_ACIDITY = _load_frozen_asset()


@dataclass(frozen=True)
class CLAMResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    anion_indices: tuple[int, ...]
    received_acidity: tuple[float, ...]
    anion_demand: tuple[float, ...]
    normalized_mismatch: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> CLAMResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return CLAMResult(
        False, reason, None, 0, 0, (), (), (), math.nan, None, {}
    )


def _zero(*, site_count: int, edge_count: int) -> CLAMResult:
    return CLAMResult(
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


def characteristic_lewis_anion_matching(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> CLAMResult:
    """Match fixed experimental cation acidities to local anion demand."""

    try:
        charge = np.asarray(charges, dtype=float)
        symbol = tuple(str(item) for item in symbols)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT460 charged-site population differs")
        if len(symbol) != len(charge) or any(not item for item in symbol):
            raise ValueError("NEXT460 element-symbol population differs")
        if np.any(charge == 0.0):
            raise ValueError("NEXT460 every site must be formally charged")
        integral = np.rint(charge)
        if not np.allclose(charge, integral, rtol=0.0, atol=1.0e-8):
            raise ValueError("NEXT460 formal charges must be integer oxidation states")
        charge = integral.astype(int)
        if abs(int(charge.sum())) != 0:
            raise ValueError("NEXT460 formal charges must be neutral")
        if raw_pair.size == 0:
            raise ValueError("NEXT460 requires an opposite-sign contact population")
        if raw_pair.ndim != 2 or raw_pair.shape[1] != 2:
            raise ValueError("NEXT460 contact endpoint population differs")
        if not np.issubdtype(raw_pair.dtype, np.integer):
            numeric = raw_pair.astype(float)
            if not np.isfinite(numeric).all() or not np.array_equal(
                numeric, np.rint(numeric)
            ):
                raise ValueError("NEXT460 contact endpoints must be integer indices")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= len(charge)):
            raise ValueError("NEXT460 contact endpoint index differs")
        if np.any(charge[pair[:, 0]] <= 0) or np.any(charge[pair[:, 1]] >= 0):
            raise ValueError("NEXT460 contacts must be ordered cation-to-anion")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree <= 0):
            raise ValueError("NEXT460 isolated charged site is unsupported")

        acidity_by_cation: dict[int, float] = {}
        for index in np.flatnonzero(charge > 0):
            key = (symbol[int(index)], int(charge[int(index)]))
            if key not in CHARACTERISTIC_LEWIS_ACIDITY:
                raise ValueError(
                    f"NEXT460 characteristic Lewis acidity lookup missing for {key[0]}{key[1]}+"
                )
            acidity_by_cation[int(index)] = CHARACTERISTIC_LEWIS_ACIDITY[key]

        anions = tuple(int(index) for index in np.flatnonzero(charge < 0))
        received_by_anion = {index: [] for index in anions}
        for cation, anion in pair:
            received_by_anion[int(anion)].append(acidity_by_cation[int(cation)])
        received = tuple(
            math.fsum(received_by_anion[index]) for index in anions
        )
        demand = tuple(float(-charge[index]) for index in anions)
        if any(value <= 0.0 or not math.isfinite(value) for value in received):
            raise RuntimeError("NEXT460 received acidity population differs")
        numerator = math.fsum(abs(value - target) for value, target in zip(received, demand))
        denominator = math.fsum(value + target for value, target in zip(received, demand))
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or numerator < 0.0
            or denominator <= 0.0
            or numerator > denominator + 1.0e-10 * denominator
        ):
            raise RuntimeError("NEXT460 normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        matching = _quantize(1.0 - mismatch)
        if not math.isfinite(matching) or matching < 0.0 or matching > 1.0:
            raise RuntimeError("NEXT460 bounded feature differs")
        return CLAMResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            anions,
            tuple(float(value) for value in received),
            demand,
            mismatch,
            None,
            {FEATURE_NAMES[0]: matching},
        )
    except Exception as exc:
        return _failure(exc)


def compute_clam_features(atoms: Atoms) -> CLAMResult:
    """Compute CLAM from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT460 valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT460 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "CLAM periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        edges = tuple(geometry.edges)
        result = characteristic_lewis_anion_matching(
            charges=charge,
            symbols=tuple(str(site.specie.symbol) for site in structure),
            endpoints=tuple(
                (int(edge.cation), int(edge.anion)) for edge in edges
            ),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        result = _failure(exc)
        if "NEXT295" in str(exc):
            return replace(
                result,
                failure_reason=str(exc).replace("NEXT295", "NEXT460"),
            )
        return result


def compute_clam_row(atoms: Atoms) -> dict[str, object]:
    result = compute_clam_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "clam_supported": bool(result.supported),
        "clam_failure": result.failure_reason,
        "clam_feasible": result.feasible,
        "clam_site_count": int(result.site_count),
        "clam_edge_count": int(result.edge_count),
        "clam_anion_count": int(len(result.anion_indices)),
        "clam_normalized_mismatch": float(result.normalized_mismatch),
        "clam_valence_policy": result.valence_policy,
        "clam_asset_sha256": ASSET_SHA256,
    }
