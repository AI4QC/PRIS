"""Dimensionless analytic Ewald-field imbalance features for NEXT34.

The module accepts one supplied periodic geometry and a neutral analytic charge
assignment.  It evaluates a classical point-charge lattice sum without moving
coordinates and emits no energy-like quantity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import (
    _publish_directory_no_replace,
    _sha256,
    _strict_json,
)
from src.next19_valence_transport import infer_valence_assignment
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next33_symmetry_steric_features import (
    FEATURE_NAME as NEXT33_FEATURE_NAME,
    PROTOCOL as NEXT33_FEATURE_PROTOCOL,
)


COULOMB_EV_ANGSTROM = 14.3996454784255
CANDIDATE_FEATURE_NAMES = (
    "aefi_field_rms",
    "aefi_field_q95",
    "aefi_field_max",
    "aefi_residual_rms",
    "aefi_residual_q95",
    "aefi_residual_max",
    "aefi_field_tensor_deviator",
)
DIAGNOSTIC_FEATURE_NAMES = ("aefi_active_site_fraction",)
REUSED_FEATURE_NAMES = (
    "steric_rep12_vector_rms",
    "steric_rep12_vector_q95",
    "steric_rep12_vector_max",
    "steric_overlap2_vector_rms",
    "steric_rep12_tensor_deviator",
    "sivr_site_imbalance_rms",
    "sivr_edge_mismatch_q95",
    "cov_q05",
)
PROTOCOL = "2026-08-03-next34-analytic-electrostatic-field-v1"
FEATURE_NAME = "next34_analytic_field_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class AnalyticFieldResult:
    """Fail-open result for one classical analytic point-charge field."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> AnalyticFieldResult:
    return AnalyticFieldResult(False, reason, {})


def _inverted_q95(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.95, method="inverted_cdf"))


def compute_analytic_field_features(
    structure,
    charges: Sequence[float] | np.ndarray,
) -> AnalyticFieldResult:
    """Evaluate charge-scale and length-scale invariant Ewald-field residuals."""

    try:
        charge = np.asarray(charges, dtype=float)
        n_sites = len(structure)
        if charge.shape != (n_sites,):
            return _failure("charges must match the structure sites")
        if n_sites < 2 or not np.isfinite(charge).all():
            return _failure("charges must be finite and describe at least two sites")
        total_magnitude = float(np.abs(charge).sum())
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, total_magnitude):
            return _failure("charges must be neutral")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            return _failure("charges must have nonzero values of both signs")
        volume = float(structure.volume)
        if not np.isfinite(volume) or volume <= 1.0e-10:
            return _failure("periodic structure volume must be finite and positive")
        q_rms = float(np.sqrt(np.mean(charge**2)))
        if not np.isfinite(q_rms) or q_rms <= 0.0:
            return _failure("charge magnitude must be nonzero")
        tolerance = max(1.0e-12, 1.0e-12 * float(np.max(np.abs(charge))))
        active = np.abs(charge) > tolerance
        if not active.any():
            return _failure("charge assignment has no active sites")

        decorated = structure.copy()
        try:
            decorated.remove_oxidation_states()
        except Exception:
            pass
        decorated.add_oxidation_state_by_site(charge.tolist())
        ewald = EwaldSummation(decorated, compute_forces=True)
        derivative = np.asarray(ewald.forces, dtype=float)
        if derivative.shape != (n_sites, 3) or not np.isfinite(derivative).all():
            return _failure("analytic Ewald derivative has invalid shape or values")

        length = float((volume / n_sites) ** (1.0 / 3.0))
        field_vectors = (
            derivative[active]
            * length**2
            / (
                COULOMB_EV_ANGSTROM
                * np.abs(charge[active])[:, None]
                * q_rms
            )
        )
        residual_vectors = (
            derivative[active]
            * length**2
            / (COULOMB_EV_ANGSTROM * q_rms**2)
        )
        field = np.linalg.norm(field_vectors, axis=1)
        residual = np.linalg.norm(residual_vectors, axis=1)
        if not np.isfinite(field).all() or not np.isfinite(residual).all():
            return _failure("dimensionless analytic field is non-finite")

        tensor_denominator = float(np.sum(field_vectors**2))
        if tensor_denominator <= 1.0e-28:
            tensor_deviator = 0.0
        else:
            tensor = field_vectors.T @ field_vectors / tensor_denominator
            deviator = tensor - np.eye(3) * float(np.trace(tensor)) / 3.0
            tensor_deviator = float(np.linalg.norm(deviator))
        values = {
            "aefi_field_rms": float(np.sqrt(np.mean(field**2))),
            "aefi_field_q95": _inverted_q95(field),
            "aefi_field_max": float(np.max(field)),
            "aefi_residual_rms": float(np.sqrt(np.mean(residual**2))),
            "aefi_residual_q95": _inverted_q95(residual),
            "aefi_residual_max": float(np.max(residual)),
            "aefi_field_tensor_deviator": tensor_deviator,
            "aefi_active_site_fraction": float(np.mean(active)),
        }
        if tuple(values) != CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES:
            return _failure("analytic field feature schema differs")
        if not np.isfinite(list(values.values())).all():
            return _failure("analytic field feature values are non-finite")
        return AnalyticFieldResult(True, None, values)
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _validate_batch_inputs(
    *,
    archive: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next33_feature_path: Path,
    next33_feature_manifest_path: Path,
) -> tuple[pd.DataFrame, list[object], pd.DataFrame]:
    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("NEXT34 geometry archive path/name is invalid")
    if metadata_path.name != "next32_cohort.parquet" or not metadata_path.is_file():
        raise ValueError("NEXT34 cohort metadata path/name is invalid")
    if next33_feature_path.name != NEXT33_FEATURE_NAME or not next33_feature_path.is_file():
        raise ValueError("NEXT34 upstream feature path/name is invalid")

    cohort_manifest = _strict_json(
        cohort_manifest_path, role="NEXT34 cohort manifest"
    )
    cohort_outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("output_role") != "unrelaxed_x0_geometry_only"
        or cohort_manifest.get("endpoint_numeric_fields_parsed") is not False
        or cohort_manifest.get("label_values_exported") is not False
        or cohort_manifest.get("labels_opened") is not False
    ):
        raise ValueError("NEXT34 cohort is not a label-free geometry projection")
    if not isinstance(cohort_outputs, Mapping) or any(
        cohort_outputs.get(path.name) != _sha256(path)
        for path in (archive, metadata_path)
    ):
        raise ValueError("NEXT34 cohort geometry or metadata hash differs")

    upstream_manifest = _strict_json(
        next33_feature_manifest_path, role="NEXT34 upstream feature manifest"
    )
    upstream_outputs = upstream_manifest.get("outputs_sha256")
    if (
        upstream_manifest.get("protocol") != NEXT33_FEATURE_PROTOCOL
        or upstream_manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or upstream_manifest.get("labels_opened") is not False
        or upstream_manifest.get("endpoint_fields_read") is not False
        or upstream_manifest.get("sid_metadata_used") is not False
        or upstream_manifest.get("model_or_proxy_potential_used") is not False
        or upstream_manifest.get("coordinates_or_cell_modified") is not False
    ):
        raise ValueError("NEXT34 upstream features crossed the label-free boundary")
    if (
        not isinstance(upstream_outputs, Mapping)
        or upstream_outputs.get(NEXT33_FEATURE_NAME) != _sha256(next33_feature_path)
    ):
        raise ValueError("NEXT34 upstream feature hash differs")

    metadata = pd.read_parquet(metadata_path)
    identity = ["material_id", "source_name", "parent_id", "natoms"]
    required_metadata = {*identity, "input_role"}
    if not required_metadata.issubset(metadata):
        raise ValueError("NEXT34 cohort metadata lacks required identity columns")
    metadata = metadata.loc[:, identity + ["input_role"]].copy()
    for column in ("material_id", "source_name", "parent_id"):
        metadata[column] = metadata[column].astype(str)
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        metadata.material_id.duplicated().any()
        or metadata.parent_id.duplicated().any()
        or not metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT34 cohort identities or input roles differ")
    material_ids = tuple(metadata.material_id)
    loaded_ids, structures = _load_archive_only(archive, material_ids)
    if loaded_ids != list(material_ids) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT34 geometry identity or atom counts differ")

    upstream = pd.read_parquet(next33_feature_path)
    required_upstream = {*identity, *REUSED_FEATURE_NAMES}
    if not required_upstream.issubset(upstream):
        raise ValueError("NEXT34 upstream feature table lacks frozen columns")
    upstream = upstream.loc[:, identity + list(REUSED_FEATURE_NAMES)].copy()
    for column in ("material_id", "source_name", "parent_id"):
        upstream[column] = upstream[column].astype(str)
    upstream = upstream.sort_values("material_id", kind="stable", ignore_index=True)
    if upstream.material_id.duplicated().any() or not upstream[identity].equals(
        metadata[identity]
    ):
        raise ValueError("NEXT34 upstream identities differ from geometry")
    return metadata, structures, upstream


def build_analytic_field_feature_batch(
    *,
    archive_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next33_feature_path: Path,
    next33_feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Seal NEXT34 fields from hash-locked geometry while labels stay unopened."""

    paths = {
        "geometry": Path(archive_path).resolve(),
        "metadata": Path(metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "next33_features": Path(next33_feature_path).resolve(),
        "next33_feature_manifest": Path(next33_feature_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    metadata, structures, upstream = _validate_batch_inputs(
        archive=paths["geometry"],
        metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        next33_feature_path=paths["next33_features"],
        next33_feature_manifest_path=paths["next33_feature_manifest"],
    )

    rows: list[dict[str, object]] = []
    policy_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    for meta, old, atoms in zip(
        metadata.to_dict("records"),
        upstream.to_dict("records"),
        structures,
        strict=True,
    ):
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
        if assignment.supported and assignment.values is not None:
            result = compute_analytic_field_features(structure, assignment.values)
        else:
            result = _failure(assignment.failure_reason or "valence assignment failed")
        policy = assignment.policy if assignment.supported else None
        if policy is not None:
            policy_counts[str(policy)] += 1
        if not result.supported:
            failure_counts[result.failure_reason or "unknown"] += 1
        row: dict[str, object] = {
            "material_id": str(meta["material_id"]),
            "source_name": str(meta["source_name"]),
            "parent_id": str(meta["parent_id"]),
            "natoms": int(meta["natoms"]),
            "aefi_supported": bool(result.supported),
            "aefi_failure": result.failure_reason,
            "valence_policy": policy,
        }
        for name in REUSED_FEATURE_NAMES:
            row[name] = float(old[name])
        for name in CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES:
            row[name] = float(result.features[name]) if result.supported else math.nan
        rows.append(row)
    features = pd.DataFrame(rows)
    forbidden = [
        column
        for column in features
        if column.lower() == "sid"
        or any(
            token in column.lower()
            for token in (
                "energy",
                "force",
                "stress",
                "dft",
                "endpoint",
                "label",
                "target",
                "relax",
                "mattersim",
                "mlip",
            )
        )
    ]
    if forbidden:
        raise ValueError(f"NEXT34 feature output crossed no-DFT contract: {forbidden}")
    if len(features) != len(metadata) or features.material_id.duplicated().any():
        raise ValueError("NEXT34 feature identity accounting differs")

    source_dir = Path(__file__).resolve().parent
    source_names = (
        "next11_geometry_only_frames.py",
        "next19_valence_transport.py",
        "next33_symmetry_steric_features.py",
        "next34_analytic_field_features.py",
    )
    source_paths = {f"src/{name}": source_dir / name for name in source_names}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "dft_values_used": False,
        "classical_analytic_electrostatics_used": True,
        "electronic_structure_calculation_used": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "sid_metadata_used": False,
        "feature_names": list(
            REUSED_FEATURE_NAMES
            + CANDIDATE_FEATURE_NAMES
            + DIAGNOSTIC_FEATURE_NAMES
        ),
        "counts": {
            "rows": len(features),
            "atoms": int(features.natoms.sum()),
            "aefi_supported": int(features.aefi_supported.sum()),
        },
        "valence_policy_counts": dict(sorted(policy_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "package_versions": {
            "pymatgen": importlib.metadata.version("pymatgen"),
        },
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURE_NAME
        features.to_parquet(feature_path, index=False)
        manifest["outputs_sha256"] = {FEATURE_NAME: _sha256(feature_path)}
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for role, path in paths.items():
            if _sha256(path) != input_hashes[role]:
                raise RuntimeError(f"NEXT34 input changed before publication: {role}")
        for name, path in source_paths.items():
            if _sha256(path) != source_hashes[name]:
                raise RuntimeError(f"NEXT34 source changed before publication: {name}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "AnalyticFieldResult",
    "CANDIDATE_FEATURE_NAMES",
    "DIAGNOSTIC_FEATURE_NAMES",
    "FEATURE_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "REUSED_FEATURE_NAMES",
    "build_analytic_field_feature_batch",
    "compute_analytic_field_features",
]
