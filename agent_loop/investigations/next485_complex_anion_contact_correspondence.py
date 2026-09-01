"""Frozen NEXT485 complex-anion contact correspondence (no DFT)."""

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


PROTOCOL = "2026-08-13-next485-complex-anion-contact-correspondence-v1"
DESIGN_SHA256 = "abaf274b66f2a54b14838ee00a3b18bca37bc304ef9835db890bef78582eaeb1"
ASSET_SHA256 = "4d1474e8ee9dc5819dd64d45d44e224dfb68226d2e8b9658535dad9bbdb50868"
ASSET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "complex_anion_lewis_basicity_hawthorne_2026.csv"
)
FEATURE_NAMES = ("cacc_complex_anion_contact_correspondence",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
OUTPUT_GRID = 1.0e10
BOUNDARY_FLAGS = dict(n295.BOUNDARY_FLAGS)


@dataclass(frozen=True)
class ComplexAnionSpec:
    group: str
    center_element: str
    center_cn: int
    group_charge: int
    lewis_basicity: float


def _load_appendix4_asset() -> Mapping[tuple[str, int], ComplexAnionSpec]:
    payload = ASSET_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ASSET_SHA256:
        raise RuntimeError("NEXT485 Appendix 4 asset SHA-256 differs")
    with ASSET_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 11:
        raise RuntimeError("NEXT485 Appendix 4 asset row count differs")
    table: dict[tuple[str, int], ComplexAnionSpec] = {}
    for row in rows:
        spec = ComplexAnionSpec(
            str(row["group"]),
            str(row["center_element"]),
            int(row["center_cn"]),
            int(row["group_charge"]),
            float(row["lewis_basicity_e"]),
        )
        key = (spec.center_element, spec.center_cn)
        if (
            key in table
            or spec.center_cn not in (3, 4)
            or spec.group_charge >= 0
            or not math.isfinite(spec.lewis_basicity)
            or spec.lewis_basicity <= 0.0
            or row["source_doi"] != "10.1180/mgm.2026.10215"
            or row["license"] != "CC-BY-4.0"
        ):
            raise RuntimeError("NEXT485 Appendix 4 asset row differs")
        table[key] = spec
    return MappingProxyType(table)


APPENDIX4_GROUPS = _load_appendix4_asset()


@dataclass(frozen=True)
class CACCResult:
    supported: bool
    failure_reason: str | None
    feasible: bool | None
    site_count: int
    edge_count: int
    recognized_group_count: int
    group_names: tuple[str, ...]
    group_centers: tuple[int, ...]
    external_contact_counts: tuple[int, ...]
    expected_external_contact_counts: tuple[float, ...]
    normalized_mismatch: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: object) -> CACCResult:
    reason = str(exc)
    if not reason.startswith(type(exc).__name__) and isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {reason}"
    return CACCResult(
        False, reason, None, 0, 0, 0, (), (), (), (), math.nan, None, {}
    )


def _zero(*, site_count: int, edge_count: int) -> CACCResult:
    return CACCResult(
        True,
        None,
        False,
        int(site_count),
        int(edge_count),
        0,
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


def complex_anion_contact_correspondence(
    *,
    charges: Sequence[float] | object,
    symbols: Sequence[str] | object,
    endpoints: Sequence[Sequence[int]] | object,
) -> CACCResult:
    """Compare isolated Appendix 4 oxyanion external contacts with basicity."""

    try:
        charge = np.asarray(charges, dtype=float)
        symbol = tuple(str(item) for item in symbols)
        raw_pair = np.asarray(endpoints)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("NEXT485 charged-site population differs")
        if len(symbol) != len(charge) or any(not item for item in symbol):
            raise ValueError("NEXT485 element-symbol population differs")
        if np.any(charge == 0.0):
            raise ValueError("NEXT485 every site must be formally charged")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            raise ValueError("NEXT485 charged-site signs differ")
        if raw_pair.size == 0:
            raise ValueError("NEXT485 requires an opposite-sign contact population")
        if raw_pair.ndim != 2 or raw_pair.shape[1] != 2:
            raise ValueError("NEXT485 contact endpoint population differs")
        if not np.issubdtype(raw_pair.dtype, np.integer):
            numeric = raw_pair.astype(float)
            if not np.isfinite(numeric).all() or not np.array_equal(
                numeric, np.rint(numeric)
            ):
                raise ValueError("NEXT485 contact endpoints must be integer indices")
        pair = raw_pair.astype(int)
        if np.any(pair < 0) or np.any(pair >= len(charge)):
            raise ValueError("NEXT485 contact endpoint index differs")
        if np.any(charge[pair[:, 0]] <= 0) or np.any(charge[pair[:, 1]] >= 0):
            raise ValueError("NEXT485 contacts must be ordered cation-to-anion")
        degree = np.bincount(pair.ravel(), minlength=len(charge))
        if np.any(degree <= 0):
            raise ValueError("NEXT485 isolated charged site is unsupported")
        if "H" in symbol:
            return _zero(site_count=len(charge), edge_count=len(pair))

        distinct_anions: dict[int, set[int]] = {}
        for cation, anion in pair:
            distinct_anions.setdefault(int(cation), set()).add(int(anion))
        candidates: dict[int, tuple[ComplexAnionSpec, tuple[int, ...]]] = {}
        for center in np.flatnonzero(charge > 0.0):
            ligands = tuple(sorted(distinct_anions.get(int(center), set())))
            spec = APPENDIX4_GROUPS.get((symbol[int(center)], len(ligands)))
            if spec is None or any(symbol[index] != "O" for index in ligands):
                continue
            candidates[int(center)] = (spec, ligands)

        ligand_center_count: dict[int, int] = {}
        for _, ligands in candidates.values():
            for ligand in ligands:
                ligand_center_count[ligand] = ligand_center_count.get(ligand, 0) + 1
        accepted = tuple(
            (center, spec, ligands)
            for center, (spec, ligands) in sorted(candidates.items())
            if all(ligand_center_count[ligand] == 1 for ligand in ligands)
        )
        if not accepted:
            return _zero(site_count=len(charge), edge_count=len(pair))

        names: list[str] = []
        centers: list[int] = []
        observed: list[int] = []
        expected: list[float] = []
        for center, spec, ligands in accepted:
            ligand_set = set(ligands)
            external = sum(
                1
                for cation, anion in pair
                if int(anion) in ligand_set and int(cation) != center
            )
            names.append(spec.group)
            centers.append(center)
            observed.append(int(external))
            expected.append(abs(spec.group_charge) / spec.lewis_basicity)
        numerator = math.fsum(
            abs(count - target) for count, target in zip(observed, expected)
        )
        denominator = math.fsum(
            count + target for count, target in zip(observed, expected)
        )
        if (
            not math.isfinite(numerator)
            or not math.isfinite(denominator)
            or numerator < 0.0
            or denominator <= 0.0
            or numerator > denominator + 1.0e-10 * denominator
        ):
            raise RuntimeError("NEXT485 normalized mismatch differs")
        mismatch = float(np.clip(numerator / denominator, 0.0, 1.0))
        matching = _quantize(1.0 - mismatch)
        if not math.isfinite(matching) or matching < 0.0 or matching > 1.0:
            raise RuntimeError("NEXT485 bounded feature differs")
        return CACCResult(
            True,
            None,
            True,
            len(charge),
            len(pair),
            len(accepted),
            tuple(names),
            tuple(centers),
            tuple(observed),
            tuple(expected),
            mismatch,
            None,
            {FEATURE_NAMES[0]: matching},
        )
    except Exception as exc:
        return _failure(exc)


def compute_cacc_features(atoms: Atoms) -> CACCResult:
    """Compute CACC from composition and one raw unrelaxed geometry only."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(
                assignment.failure_reason or "NEXT485 valence assignment failed"
            )
        charge = np.asarray(assignment.values, dtype=float)
        if charge.shape != (len(structure),):
            raise ValueError("NEXT485 valence population differs")
        geometry = n19.build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            reason = str(geometry.failure_reason or "CACC periodic graph failed")
            if "no opposite-sign periodic neighbor" in reason:
                return replace(
                    _zero(site_count=len(structure), edge_count=0),
                    valence_policy=str(assignment.policy),
                )
            raise ValueError(reason)
        result = complex_anion_contact_correspondence(
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
            return replace(
                result,
                failure_reason=str(exc).replace("NEXT295", "NEXT485"),
            )
        return result


def compute_cacc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_cacc_features(atoms)
    return {
        FEATURE_NAMES[0]: (
            float(result.features[FEATURE_NAMES[0]]) if result.supported else math.nan
        ),
        "cacc_supported": bool(result.supported),
        "cacc_failure": result.failure_reason,
        "cacc_feasible": result.feasible,
        "cacc_site_count": int(result.site_count),
        "cacc_edge_count": int(result.edge_count),
        "cacc_recognized_group_count": int(result.recognized_group_count),
        "cacc_valence_policy": result.valence_policy,
        "cacc_asset_sha256": ASSET_SHA256,
    }
