#!/usr/bin/env python3
"""Compose the frozen pure-analytic x0 kernels for NEXT43 development.

This program has no endpoint, calculator, relaxation, or model-potential input.
Every descriptor is computed from one sanitized periodic x0 and failures are
explicitly fail-open.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256, _strict_json
from src.next19_valence_transport import (
    compute_valence_transport_features,
    infer_valence_assignment,
)
from src.next32_inorganic_response_features import (
    INORGANIC_FEATURE_NAMES,
    compute_inorganic_response_features,
)
from src.next33_symmetry_steric_features import (
    STERIC_FEATURE_NAMES,
    SYMMETRY_FEATURE_NAMES,
    compute_directional_steric_features,
    compute_symmetry_recovery_features,
)
from src.next34_analytic_field_features import (
    CANDIDATE_FEATURE_NAMES as ANALYTIC_FIELD_FEATURE_NAMES,
    compute_analytic_field_features,
)
from src.next35_coulomb_steric_balance_features import (
    CANDIDATE_FEATURE_NAMES as COULOMB_STERIC_FEATURE_NAMES,
    compute_coulomb_steric_balance_features,
)
from src.next36_charge_spectrum_features import (
    CANDIDATE_FEATURE_NAMES as CHARGE_SPECTRUM_FEATURE_NAMES,
    compute_charge_spectrum_features,
)
from src.next37_self_stress_compatibility_features import (
    CANDIDATE_FEATURE_NAMES as SELF_STRESS_FEATURE_NAMES,
    compute_self_stress_compatibility_features,
)
from src.next38_bond_valence_transport_compatibility_features import (
    CANDIDATE_FEATURE_NAMES as BOND_VALENCE_TRANSPORT_FEATURE_NAMES,
    compute_bond_valence_transport_compatibility_features,
)
from src.next42_alexandria_cohort import (
    COHORT_NAME,
    GEOMETRY_NAME,
    INPUT_ROLE as COHORT_INPUT_ROLE,
    PROTOCOL as COHORT_PROTOCOL,
)


PROTOCOL = "2026-08-03-next43-analytic-feature-bank-v1"
FEATURE_NAME = "next43_analytic_feature_bank.parquet"
MANIFEST_NAME = "MANIFEST.json"
VALENCE_TRANSPORT_ALPHAS = (0.0, 2.0, 4.0, 6.0)
VALENCE_TRANSPORT_METRICS = (
    "overload",
    "reallocation",
    "anion_mismatch_max",
)


def _alpha_tag(alpha: float) -> str:
    return f"a{int(alpha)}"


VALENCE_TRANSPORT_FEATURE_NAMES = tuple(
    f"vt_{_alpha_tag(alpha)}_{metric}"
    for alpha in VALENCE_TRANSPORT_ALPHAS
    for metric in VALENCE_TRANSPORT_METRICS
)
CANDIDATE_FEATURE_NAMES = (
    tuple(INORGANIC_FEATURE_NAMES)
    + tuple(SYMMETRY_FEATURE_NAMES)
    + tuple(STERIC_FEATURE_NAMES)
    + VALENCE_TRANSPORT_FEATURE_NAMES
    + tuple(ANALYTIC_FIELD_FEATURE_NAMES)
    + tuple(COULOMB_STERIC_FEATURE_NAMES)
    + tuple(CHARGE_SPECTRUM_FEATURE_NAMES)
    + tuple(SELF_STRESS_FEATURE_NAMES)
    + tuple(BOND_VALENCE_TRANSPORT_FEATURE_NAMES)
)
FAMILY_NAMES = (
    "contact",
    "sivr",
    "madelung",
    "scbve",
    "symmetry",
    "steric",
    "valence_transport_a0",
    "valence_transport_a2",
    "valence_transport_a4",
    "valence_transport_a6",
    "analytic_field",
    "coulomb_steric_balance",
    "charge_spectrum",
    "self_stress",
    "bond_valence_transport_compatibility",
)
METADATA_COLUMNS = (
    "material_id",
    "source_family",
    "source_shard",
    "formula",
    "reduced_formula",
    "natoms",
    "input_role",
)
_CHARGE_FAMILIES = FAMILY_NAMES[6:]


if len(CANDIDATE_FEATURE_NAMES) != len(set(CANDIDATE_FEATURE_NAMES)):
    raise RuntimeError("NEXT43 analytic feature names are duplicated")


def _set_result(
    row: dict[str, object],
    *,
    family: str,
    result,
    names: Sequence[str],
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


def compute_analytic_feature_row(atoms: Atoms) -> dict[str, object]:
    """Compute every finite single-x0 candidate family with fail-open support."""

    row: dict[str, object] = {name: math.nan for name in CANDIDATE_FEATURE_NAMES}
    for family in FAMILY_NAMES:
        row[f"{family}_supported"] = False
        row[f"{family}_failure"] = None
    row["valence_assignment_policy"] = None

    inorganic = compute_inorganic_response_features(atoms)
    for family in ("contact", "sivr", "madelung", "scbve"):
        row[f"{family}_supported"] = bool(inorganic.family_supported[family])
        row[f"{family}_failure"] = inorganic.family_failures[family]
    for name, value in inorganic.features.items():
        if name in row and np.isfinite(float(value)):
            row[name] = float(value)

    _set_result(
        row,
        family="symmetry",
        result=compute_symmetry_recovery_features(atoms),
        names=SYMMETRY_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="steric",
        result=compute_directional_steric_features(atoms),
        names=STERIC_FEATURE_NAMES,
    )

    try:
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
    except Exception as exc:
        assignment = None
        reason = f"valence assignment failed: {type(exc).__name__}: {exc}"
    else:
        reason = None if assignment.supported else (
            assignment.failure_reason or "valence assignment unsupported"
        )
    if assignment is None or not assignment.supported or assignment.values is None:
        for family in _CHARGE_FAMILIES:
            row[f"{family}_failure"] = reason
        return row

    charges = np.asarray(assignment.values, dtype=float)
    row["valence_assignment_policy"] = assignment.policy
    for alpha in VALENCE_TRANSPORT_ALPHAS:
        family = f"valence_transport_{_alpha_tag(alpha)}"
        result = compute_valence_transport_features(
            structure,
            charges,
            graph_mode="voronoi",
            alpha=alpha,
        )
        names = tuple(
            f"vt_{_alpha_tag(alpha)}_{metric}"
            for metric in VALENCE_TRANSPORT_METRICS
        )
        if result.supported:
            translated = {
                names[0]: result.features["vt_overload"],
                names[1]: result.features["vt_reallocation"],
                names[2]: result.features["vt_anion_mismatch_max"],
            }
            result = type(result)(True, None, translated)
        _set_result(row, family=family, result=result, names=names)

    _set_result(
        row,
        family="analytic_field",
        result=compute_analytic_field_features(structure, charges),
        names=ANALYTIC_FIELD_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="coulomb_steric_balance",
        result=compute_coulomb_steric_balance_features(structure, charges),
        names=COULOMB_STERIC_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="charge_spectrum",
        result=compute_charge_spectrum_features(structure, charges),
        names=CHARGE_SPECTRUM_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="self_stress",
        result=compute_self_stress_compatibility_features(structure, charges),
        names=SELF_STRESS_FEATURE_NAMES,
    )
    _set_result(
        row,
        family="bond_valence_transport_compatibility",
        result=compute_bond_valence_transport_compatibility_features(
            structure, charges
        ),
        names=BOND_VALENCE_TRANSPORT_FEATURE_NAMES,
    )
    return row


def _validated_inputs(
    *, metadata_path: Path, geometry_path: Path, cohort_manifest_path: Path
) -> tuple[pd.DataFrame, list[Atoms]]:
    paths = (Path(metadata_path), Path(geometry_path), Path(cohort_manifest_path))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("NEXT43 feature-bank input is missing")
    if metadata_path.name != COHORT_NAME or geometry_path.name != GEOMETRY_NAME:
        raise ValueError("NEXT43 cohort filenames differ")
    manifest = _strict_json(cohort_manifest_path, role="NEXT43 cohort manifest")
    selection = manifest.get("selection")
    if (
        manifest.get("protocol") != COHORT_PROTOCOL
        or manifest.get("input_role") != COHORT_INPUT_ROLE
        or manifest.get("later_geometry_accessed") is not False
        or manifest.get("dft_values_read") is not False
        or manifest.get("mlip_prerelaxation_used") is not False
        or manifest.get("physical_relaxation_executed") is not False
        or not isinstance(selection, Mapping)
        or selection.get("endpoint_fields_used") is not False
    ):
        raise ValueError("NEXT43 cohort crossed the geometry-only boundary")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or any(
        outputs.get(path.name) != _sha256(path)
        for path in (metadata_path, geometry_path)
    ):
        raise ValueError("NEXT43 cohort output hash differs")
    metadata = pd.read_parquet(metadata_path)
    if set(metadata.columns) != set(METADATA_COLUMNS):
        raise ValueError("NEXT43 cohort metadata schema differs")
    metadata = metadata.loc[:, METADATA_COLUMNS].sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    if (
        metadata.empty
        or metadata.material_id.isna().any()
        or metadata.material_id.astype(str).duplicated().any()
        or not metadata.input_role.eq(COHORT_INPUT_ROLE).all()
    ):
        raise ValueError("NEXT43 cohort metadata identities differ")
    identifiers = tuple(metadata.material_id.astype(str))
    loaded_ids, structures = _load_archive_only(geometry_path, identifiers)
    if loaded_ids != list(identifiers) or len(structures) != len(metadata):
        raise ValueError("NEXT43 geometry coverage differs")
    if [len(atoms) for atoms in structures] != metadata.natoms.astype(int).tolist():
        raise ValueError("NEXT43 geometry atom counts differ")
    return metadata, structures


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_feature_bank(
    *,
    metadata_path: Path,
    geometry_path: Path,
    cohort_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build and atomically publish the complete label-free NEXT43 bank."""

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
        rows.append(compute_analytic_feature_row(atoms))
        if index % 50 == 0 or index == len(structures):
            print(f"NEXT43 analytic features: {index}/{len(structures)}", flush=True)
    features = pd.DataFrame(rows)
    expected_columns = set(CANDIDATE_FEATURE_NAMES) | {
        f"{family}_{suffix}"
        for family in FAMILY_NAMES
        for suffix in ("supported", "failure")
    } | {"valence_assignment_policy"}
    if set(features.columns) != expected_columns:
        raise RuntimeError("NEXT43 computed feature schema differs")
    table = pd.concat([metadata.reset_index(drop=True), features], axis=1)
    for family in FAMILY_NAMES:
        column = f"{family}_supported"
        if not table[column].map(lambda value: type(value) is bool).all():
            raise RuntimeError(f"NEXT43 {family} support flags differ")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURE_NAME
        table.to_parquet(feature_path, index=False)
        repository = Path(__file__).resolve().parents[1]
        source_names = (
            "src/next19_valence_transport.py",
            "src/next32_inorganic_response_features.py",
            "src/next33_symmetry_steric_features.py",
            "src/next34_analytic_field_features.py",
            "src/next35_coulomb_steric_balance_features.py",
            "src/next36_charge_spectrum_features.py",
            "src/next37_self_stress_compatibility_features.py",
            "src/next38_bond_valence_transport_compatibility_features.py",
            "src/next43_analytic_feature_bank.py",
        )
        support_counts = {
            family: int(table[f"{family}_supported"].sum())
            for family in FAMILY_NAMES
        }
        failure_counts = {
            family: dict(
                sorted(
                    Counter(
                        str(value)
                        for value in table.loc[
                            ~table[f"{family}_supported"],
                            f"{family}_failure",
                        ]
                        if value is not None
                    ).items()
                )
            )
            for family in FAMILY_NAMES
        }
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "development analytic descriptor bank from one raw x0",
            "input_role": "one_raw_pre_dft_pre_mlip_x0_only",
            "rows": len(table),
            "candidate_feature_count": len(CANDIDATE_FEATURE_NAMES),
            "candidate_features": list(CANDIDATE_FEATURE_NAMES),
            "family_support_counts": support_counts,
            "family_failure_counts": failure_counts,
            "labels_opened": False,
            "endpoint_fields_read": False,
            "dft_values_used": False,
            "mlip_or_model_potential_used": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "same_composition_alternatives_used": False,
            "classical_analytic_electrostatics_used": True,
            "deterministic_linear_algebra_used": True,
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
        for path, expected in (
            (metadata_path, manifest["inputs_sha256"]["metadata"]),
            (geometry_path, manifest["inputs_sha256"]["geometry"]),
            (cohort_manifest_path, manifest["inputs_sha256"]["cohort_manifest"]),
        ):
            if _sha256(path) != expected:
                raise RuntimeError(f"NEXT43 input changed during build: {path}")
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
    manifest = build_feature_bank(
        metadata_path=args.metadata,
        geometry_path=args.geometry,
        cohort_manifest_path=args.cohort_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({"rows": manifest["rows"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
