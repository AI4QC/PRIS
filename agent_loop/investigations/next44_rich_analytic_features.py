#!/usr/bin/env python3
"""Additive rich analytic descriptors from the sealed NEXT42 raw-x0 cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
from ase.data import atomic_masses, vdw_radii
import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next19_valence_transport import build_periodic_edge_geometry, infer_valence_assignment
from src.next20_valence_rigidity import (
    FEATURE_NAMES as SIVR_ALL_FEATURE_NAMES,
    rigidity_features_from_periodic_geometry,
)
from src.next21_normalized_madelung import (
    FEATURE_NAMES as MADELUNG_ALL_FEATURE_NAMES,
    normalized_madelung_features,
)
from src.next22_bond_valence_equilibrium import (
    FEATURE_NAMES as SCBVE_ALL_FEATURE_NAMES,
    bond_valence_features_from_periodic_geometry,
)
from src.next26_packing import FEATURE_COLUMNS as PACKING_FEATURE_NAMES, compute_packing_features
from src.next27_periodic_packing import (
    NEXT27_FEATURE_COLUMNS as PERIODIC_NONBOND_FEATURE_NAMES,
    compute_periodic_features,
)
from src.next32_inorganic_response_features import (
    INORGANIC_FEATURE_NAMES,
    _canonical_periodic_ratios,
    _resolve_radii,
)
from src.next43_analytic_feature_bank import _validated_inputs


PROTOCOL = "2026-08-03-next44-rich-analytic-feature-bank-v1"
FEATURE_NAME = "next44_rich_analytic_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
SIVR_EXTRA_FEATURE_NAMES = tuple(
    name for name in SIVR_ALL_FEATURE_NAMES if name not in INORGANIC_FEATURE_NAMES
)
MADELUNG_EXTRA_FEATURE_NAMES = tuple(
    name for name in MADELUNG_ALL_FEATURE_NAMES if name not in INORGANIC_FEATURE_NAMES
)
SCBVE_EXTRA_FEATURE_NAMES = tuple(
    name for name in SCBVE_ALL_FEATURE_NAMES if name not in INORGANIC_FEATURE_NAMES
)
CELL_COMPOSITION_FEATURE_NAMES = (
    "geom_volume_pa",
    "geom_mass_density_proxy",
    "geom_covalent_packing",
    "geom_vdw_packing",
    "geom_cell_length_ratio",
    "geom_cell_angle_dev",
    "geom_species_count",
    "geom_atomic_number_mean",
    "geom_atomic_number_std",
    "geom_atomic_number_range",
    "geom_covalent_radius_mean",
    "geom_covalent_radius_cv",
    "geom_covalent_radius_range_rel",
)
ELECTRONEGATIVITY_FEATURE_NAMES = (
    "geom_electronegativity_mean",
    "geom_electronegativity_std",
    "geom_electronegativity_range",
)
EXTENDED_CONTACT_FEATURE_NAMES = (
    "cov_ratio_q50",
    "cov_ratio_q95",
    "cov_coord100_mean",
    "cov_coord100_std",
    "cov_coord100_q95",
    "cov_coord100_max",
    "cov_coord100_zero_fraction",
    "cov_coord110_mean",
    "cov_coord110_std",
    "cov_coord110_q95",
    "cov_coord110_max",
    "cov_coord110_zero_fraction",
    "cov_site_nearest_mean",
    "cov_site_nearest_q95",
    "cov_site_nearest_max",
)
CANDIDATE_FEATURE_NAMES = (
    SIVR_EXTRA_FEATURE_NAMES
    + MADELUNG_EXTRA_FEATURE_NAMES
    + SCBVE_EXTRA_FEATURE_NAMES
    + tuple(PERIODIC_NONBOND_FEATURE_NAMES)
    + tuple(PACKING_FEATURE_NAMES)
    + CELL_COMPOSITION_FEATURE_NAMES
    + ELECTRONEGATIVITY_FEATURE_NAMES
    + EXTENDED_CONTACT_FEATURE_NAMES
)
FAMILY_NAMES = (
    "full_sivr",
    "full_madelung",
    "full_scbve",
    "periodic_nonbonded",
    "legacy_packing",
    "cell_composition",
    "electronegativity",
    "extended_contact",
)


if len(CANDIDATE_FEATURE_NAMES) != len(set(CANDIDATE_FEATURE_NAMES)):
    raise RuntimeError("NEXT44 rich feature names are duplicated")


@dataclass(frozen=True)
class _Result:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> _Result:
    if isinstance(exc, Exception):
        reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = str(exc)
    return _Result(False, reason, {})


def _cell_composition_features(atoms: Atoms) -> _Result:
    try:
        numbers = np.asarray(atoms.numbers, dtype=int)
        cell = np.asarray(atoms.cell.array, dtype=float)
        if (
            len(numbers) < 1
            or cell.shape != (3, 3)
            or not np.all(atoms.pbc)
            or not np.isfinite(cell).all()
            or np.any(numbers <= 0)
        ):
            raise ValueError("invalid periodic geometry")
        volume = abs(float(np.linalg.det(cell)))
        if volume <= 1.0e-10:
            raise ValueError("periodic cell volume is zero")
        radii = _resolve_radii(numbers, None)
        vdw = np.asarray(
            [
                float(vdw_radii[number])
                if number < len(vdw_radii) and np.isfinite(vdw_radii[number])
                else 1.70
                for number in numbers
            ],
            dtype=float,
        )
        lengths = np.linalg.norm(cell, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("periodic cell vector is zero")
        cosines = np.asarray(
            [
                np.dot(cell[1], cell[2]) / (lengths[1] * lengths[2]),
                np.dot(cell[0], cell[2]) / (lengths[0] * lengths[2]),
                np.dot(cell[0], cell[1]) / (lengths[0] * lengths[1]),
            ],
            dtype=float,
        )
        angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
        radius_mean = float(np.mean(radii))
        values = {
            "geom_volume_pa": volume / len(numbers),
            "geom_mass_density_proxy": float(np.sum(atomic_masses[numbers]) / volume),
            "geom_covalent_packing": float(4.0 * np.pi * np.sum(radii**3) / (3.0 * volume)),
            "geom_vdw_packing": float(4.0 * np.pi * np.sum(vdw**3) / (3.0 * volume)),
            "geom_cell_length_ratio": float(np.max(lengths) / np.min(lengths)),
            "geom_cell_angle_dev": float(np.max(np.abs(angles - 90.0))),
            "geom_species_count": float(len(set(numbers.tolist()))),
            "geom_atomic_number_mean": float(np.mean(numbers)),
            "geom_atomic_number_std": float(np.std(numbers)),
            "geom_atomic_number_range": float(np.max(numbers) - np.min(numbers)),
            "geom_covalent_radius_mean": radius_mean,
            "geom_covalent_radius_cv": float(np.std(radii) / radius_mean),
            "geom_covalent_radius_range_rel": float(
                (np.max(radii) - np.min(radii)) / radius_mean
            ),
        }
        if tuple(values) != CELL_COMPOSITION_FEATURE_NAMES or not np.isfinite(list(values.values())).all():
            raise ValueError("cell/composition feature schema differs")
        return _Result(True, None, values)
    except Exception as exc:
        return _failure(exc)


def _electronegativity_features(atoms: Atoms) -> _Result:
    try:
        values = np.asarray(
            [float(Element(symbol).X) for symbol in atoms.get_chemical_symbols()],
            dtype=float,
        )
        if values.shape != (len(atoms),) or not np.isfinite(values).all():
            raise ValueError("Pauling electronegativity is unavailable")
        features = {
            "geom_electronegativity_mean": float(np.mean(values)),
            "geom_electronegativity_std": float(np.std(values)),
            "geom_electronegativity_range": float(np.max(values) - np.min(values)),
        }
        return _Result(True, None, features)
    except Exception as exc:
        return _failure(exc)


def _extended_contact_features(atoms: Atoms) -> _Result:
    try:
        numbers = np.asarray(atoms.numbers, dtype=int)
        radii = _resolve_radii(numbers, None)
        pairs = _canonical_periodic_ratios(atoms, radii)
        if not pairs:
            raise ValueError("periodic geometry has no radius-scaled contacts")
        ratios = np.asarray([row[2] for row in pairs], dtype=float)
        coord100 = np.zeros(len(atoms), dtype=float)
        coord110 = np.zeros(len(atoms), dtype=float)
        nearest = np.full(len(atoms), 1.6, dtype=float)
        for left, right, ratio in pairs:
            nearest[left] = min(nearest[left], ratio)
            nearest[right] = min(nearest[right], ratio)
            if ratio <= 1.0:
                coord100[left] += 1.0
                coord100[right] += 1.0
            if ratio <= 1.1:
                coord110[left] += 1.0
                coord110[right] += 1.0
        features = {
            "cov_ratio_q50": float(np.quantile(ratios, 0.50)),
            "cov_ratio_q95": float(np.quantile(ratios, 0.95)),
            "cov_coord100_mean": float(np.mean(coord100)),
            "cov_coord100_std": float(np.std(coord100)),
            "cov_coord100_q95": float(np.quantile(coord100, 0.95)),
            "cov_coord100_max": float(np.max(coord100)),
            "cov_coord100_zero_fraction": float(np.mean(coord100 == 0.0)),
            "cov_coord110_mean": float(np.mean(coord110)),
            "cov_coord110_std": float(np.std(coord110)),
            "cov_coord110_q95": float(np.quantile(coord110, 0.95)),
            "cov_coord110_max": float(np.max(coord110)),
            "cov_coord110_zero_fraction": float(np.mean(coord110 == 0.0)),
            "cov_site_nearest_mean": float(np.mean(nearest)),
            "cov_site_nearest_q95": float(np.quantile(nearest, 0.95)),
            "cov_site_nearest_max": float(np.max(nearest)),
        }
        if tuple(features) != EXTENDED_CONTACT_FEATURE_NAMES or not np.isfinite(list(features.values())).all():
            raise ValueError("extended contact feature schema differs")
        return _Result(True, None, features)
    except Exception as exc:
        return _failure(exc)


def _mapping_result(function, atoms: Atoms) -> _Result:
    try:
        values = function(atoms)
        if not isinstance(values, Mapping) or not np.isfinite(list(values.values())).all():
            raise ValueError("analytic mapping result differs")
        return _Result(True, None, {str(key): float(value) for key, value in values.items()})
    except Exception as exc:
        return _failure(exc)


def _set_result(
    row: dict[str, object], *, family: str, result, names: Sequence[str]
) -> None:
    supported = bool(result.supported)
    row[f"{family}_supported"] = supported
    row[f"{family}_failure"] = None if supported else (
        result.failure_reason or f"{family} unsupported"
    )
    if supported:
        for name in names:
            if name not in result.features:
                raise RuntimeError(f"{family} omitted required feature {name}")
            value = float(result.features[name])
            if not np.isfinite(value):
                raise RuntimeError(f"{family} emitted non-finite feature {name}")
            row[name] = value


def compute_rich_feature_row(atoms: Atoms) -> dict[str, object]:
    """Compute the additive NEXT44 families for one unmodified periodic x0."""

    row: dict[str, object] = {name: math.nan for name in CANDIDATE_FEATURE_NAMES}
    for family in FAMILY_NAMES:
        row[f"{family}_supported"] = False
        row[f"{family}_failure"] = None
    _set_result(
        row,
        family="cell_composition",
        result=_cell_composition_features(atoms),
        names=CELL_COMPOSITION_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="electronegativity",
        result=_electronegativity_features(atoms),
        names=ELECTRONEGATIVITY_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="extended_contact",
        result=_extended_contact_features(atoms),
        names=EXTENDED_CONTACT_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="periodic_nonbonded",
        result=_mapping_result(compute_periodic_features, atoms),
        names=PERIODIC_NONBOND_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="legacy_packing",
        result=_mapping_result(compute_packing_features, atoms),
        names=PACKING_FEATURE_NAMES,
    )

    try:
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
    except Exception as exc:
        assignment = None
        assignment_reason = f"valence assignment failed: {type(exc).__name__}: {exc}"
    else:
        assignment_reason = None if assignment.supported else (
            assignment.failure_reason or "valence assignment unsupported"
        )
    if assignment is None or not assignment.supported or assignment.values is None:
        for family in ("full_sivr", "full_madelung", "full_scbve"):
            row[f"{family}_failure"] = assignment_reason
        return row
    charges = np.asarray(assignment.values, dtype=float)
    try:
        geometry = build_periodic_edge_geometry(structure, charges, graph_mode="voronoi")
    except Exception as exc:
        geometry = None
        geometry_reason = f"periodic graph failed: {type(exc).__name__}: {exc}"
    else:
        geometry_reason = None if geometry.supported else (
            geometry.failure_reason or "periodic graph unsupported"
        )
    if geometry is None or not geometry.supported:
        for family in ("full_sivr", "full_scbve"):
            row[f"{family}_failure"] = geometry_reason
    else:
        _set_result(
            row,
            family="full_sivr",
            result=rigidity_features_from_periodic_geometry(
                structure, charges, geometry, charge_weight_exponent=0.0
            ),
            names=SIVR_EXTRA_FEATURE_NAMES,
        )
        _set_result(
            row,
            family="full_scbve",
            result=bond_valence_features_from_periodic_geometry(
                structure, charges, geometry
            ),
            names=SCBVE_EXTRA_FEATURE_NAMES,
        )
    _set_result(
        row,
        family="full_madelung",
        result=normalized_madelung_features(structure, charges),
        names=MADELUNG_EXTRA_FEATURE_NAMES,
    )
    return row


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_rich_feature_bank(
    *,
    metadata_path: Path,
    geometry_path: Path,
    cohort_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build and atomically publish the label-free NEXT44 table."""

    metadata_path = Path(metadata_path).resolve()
    geometry_path = Path(geometry_path).resolve()
    cohort_manifest_path = Path(cohort_manifest_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    metadata, structures = _validated_inputs(
        metadata_path=metadata_path,
        geometry_path=geometry_path,
        cohort_manifest_path=cohort_manifest_path,
    )
    rows: list[dict[str, object]] = []
    for index, atoms in enumerate(structures, start=1):
        rows.append(compute_rich_feature_row(atoms))
        if index % 50 == 0 or index == len(structures):
            print(f"NEXT44 rich analytic features: {index}/{len(structures)}", flush=True)
    features = pd.DataFrame(rows)
    expected = set(CANDIDATE_FEATURE_NAMES) | {
        f"{family}_{suffix}"
        for family in FAMILY_NAMES
        for suffix in ("supported", "failure")
    }
    if set(features.columns) != expected:
        raise RuntimeError("NEXT44 computed feature schema differs")
    table = pd.concat([metadata.reset_index(drop=True), features], axis=1)
    for family in FAMILY_NAMES:
        if not table[f"{family}_supported"].map(lambda value: type(value) is bool).all():
            raise RuntimeError(f"NEXT44 {family} support flags differ")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURE_NAME
        table.to_parquet(feature_path, index=False)
        repository = Path(__file__).resolve().parents[1]
        source_names = (
            "src/next19_valence_transport.py",
            "src/next20_valence_rigidity.py",
            "src/next21_normalized_madelung.py",
            "src/next22_bond_valence_equilibrium.py",
            "src/next26_packing.py",
            "src/next27_periodic_packing.py",
            "src/next32_inorganic_response_features.py",
            "src/next43_analytic_feature_bank.py",
            "src/next44_rich_analytic_features.py",
        )
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "additive rich analytic descriptors from one raw x0",
            "input_role": "one_raw_pre_dft_pre_mlip_x0_only",
            "rows": len(table),
            "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
            "candidate_features": list(CANDIDATE_FEATURE_NAMES),
            "family_support_counts": {
                family: int(table[f"{family}_supported"].sum())
                for family in FAMILY_NAMES
            },
            "family_failure_counts": {
                family: dict(
                    sorted(
                        Counter(
                            str(value)
                            for value in table.loc[
                                ~table[f"{family}_supported"], f"{family}_failure"
                            ]
                            if value is not None
                        ).items()
                    )
                )
                for family in FAMILY_NAMES
            },
            "labels_opened": False,
            "endpoint_fields_read": False,
            "dft_values_used": False,
            "mlip_or_model_potential_used": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "same_composition_alternatives_used": False,
            "inputs_sha256": {
                "metadata": _sha256(metadata_path),
                "geometry": _sha256(geometry_path),
                "cohort_manifest": _sha256(cohort_manifest_path),
            },
            "executed_source_sha256": {
                name: _sha256(repository / name) for name in source_names
            },
            "scientific_improvement_claim": False,
        }
        manifest["outputs_sha256"] = {FEATURE_NAME: _sha256(feature_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for path, digest in (
            (metadata_path, manifest["inputs_sha256"]["metadata"]),
            (geometry_path, manifest["inputs_sha256"]["geometry"]),
            (cohort_manifest_path, manifest["inputs_sha256"]["cohort_manifest"]),
        ):
            if _sha256(path) != digest:
                raise RuntimeError(f"NEXT44 input changed during build: {path}")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_rich_feature_bank(
        metadata_path=args.metadata,
        geometry_path=args.geometry,
        cohort_manifest_path=args.cohort_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({"rows": manifest["rows"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
