"""Closed-form balance between analytic Coulomb and steric vector fields."""

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
from src.next32_inorganic_response_features import _resolve_radii
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next33_symmetry_steric_features import _periodic_steric_pairs
from src.next34_analytic_field_features import (
    COULOMB_EV_ANGSTROM,
    FEATURE_NAME as NEXT34_FEATURE_NAME,
    PROTOCOL as NEXT34_FEATURE_PROTOCOL,
)


CANDIDATE_FEATURE_NAMES = (
    "acsb_opposition_deficit",
    "acsb_global_residual",
    "acsb_site_residual_rms",
    "acsb_site_residual_q95",
    "acsb_site_residual_max",
    "acsb_site_direction_deficit_q95",
    "acsb_active_disagreement_fraction",
)
DIAGNOSTIC_FEATURE_NAMES = (
    "acsb_optimal_repulsion_scale",
    "acsb_coulomb_norm_per_site",
    "acsb_steric_norm_per_site",
    "acsb_joint_active_site_fraction",
)
REUSED_FEATURE_NAMES = (
    "aefi_residual_max",
    "steric_rep12_vector_rms",
    "steric_rep12_vector_max",
    "sivr_site_imbalance_rms",
)
PROTOCOL = "2026-08-03-next35-coulomb-steric-balance-features-v1"
FEATURE_NAME = "next35_coulomb_steric_balance_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class BalanceFeatureResult:
    """Fail-open result for one pair of analytic vector fields."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> BalanceFeatureResult:
    return BalanceFeatureResult(False, reason, {})


def _q95(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.95, method="inverted_cdf"))


def analytic_vector_balance_features(
    coulomb_vectors: Sequence[Sequence[float]] | np.ndarray,
    steric_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> BalanceFeatureResult:
    """Measure whether two supplied vector fields can oppose after scalar matching."""

    try:
        coulomb = np.asarray(coulomb_vectors, dtype=float)
        steric = np.asarray(steric_vectors, dtype=float)
        if (
            coulomb.ndim != 2
            or coulomb.shape[1:] != (3,)
            or steric.shape != coulomb.shape
            or len(coulomb) < 1
        ):
            return _failure("analytic vector fields must have matching nonempty shape (N,3)")
        if not np.isfinite(coulomb).all() or not np.isfinite(steric).all():
            return _failure("analytic vector fields must be finite")

        n_sites = len(coulomb)
        c_norm = float(np.linalg.norm(coulomb))
        s_norm = float(np.linalg.norm(steric))
        c_global = c_norm > 1.0e-12
        s_global = s_norm > 1.0e-12
        if c_global and s_global:
            dot = float(np.sum(coulomb * steric))
            cosine = float(np.clip(dot / (c_norm * s_norm), -1.0, 1.0))
            opposition_deficit = (1.0 + cosine) / 2.0
            denominator = float(np.sum(steric**2))
            optimal_scale = max(0.0, -dot / denominator)
            residual_vectors = coulomb + optimal_scale * steric
            global_denominator = math.sqrt(
                c_norm**2 + (optimal_scale * s_norm) ** 2
            )
            global_residual = float(
                np.linalg.norm(residual_vectors) / global_denominator
            )
        elif c_global or s_global:
            opposition_deficit = 1.0
            optimal_scale = 0.0
            residual_vectors = coulomb.copy()
            global_residual = 1.0
        else:
            opposition_deficit = 0.0
            optimal_scale = 0.0
            residual_vectors = np.zeros_like(coulomb)
            global_residual = 0.0

        c_site_norm = np.linalg.norm(coulomb, axis=1)
        s_site_norm = np.linalg.norm(steric, axis=1)
        c_tolerance = max(1.0e-12, c_norm * 1.0e-12)
        s_tolerance = max(1.0e-12, s_norm * 1.0e-12)
        c_active = c_site_norm > c_tolerance
        s_active = s_site_norm > s_tolerance
        union = c_active | s_active
        joint = c_active & s_active
        site_residual = np.zeros(n_sites, dtype=float)
        direction_deficit = np.zeros(n_sites, dtype=float)
        disagreement = np.zeros(n_sites, dtype=bool)
        for index in range(n_sites):
            if joint[index]:
                local_dot = float(np.dot(coulomb[index], steric[index]))
                local_cosine = float(
                    np.clip(
                        local_dot / (c_site_norm[index] * s_site_norm[index]),
                        -1.0,
                        1.0,
                    )
                )
                direction_deficit[index] = (1.0 + local_cosine) / 2.0
                local_denominator = float(
                    c_site_norm[index] + optimal_scale * s_site_norm[index]
                )
                if local_denominator > 1.0e-14:
                    site_residual[index] = min(
                        1.0,
                        float(np.linalg.norm(residual_vectors[index]))
                        / local_denominator,
                    )
                disagreement[index] = local_dot >= 0.0
            elif union[index]:
                direction_deficit[index] = 1.0
                site_residual[index] = 1.0
                disagreement[index] = True
        active_disagreement = (
            float(np.mean(disagreement[union])) if union.any() else 0.0
        )
        values = {
            "acsb_opposition_deficit": float(opposition_deficit),
            "acsb_global_residual": float(np.clip(global_residual, 0.0, 1.0)),
            "acsb_site_residual_rms": float(np.sqrt(np.mean(site_residual**2))),
            "acsb_site_residual_q95": _q95(site_residual),
            "acsb_site_residual_max": float(np.max(site_residual)),
            "acsb_site_direction_deficit_q95": _q95(direction_deficit),
            "acsb_active_disagreement_fraction": active_disagreement,
            "acsb_optimal_repulsion_scale": float(optimal_scale),
            "acsb_coulomb_norm_per_site": c_norm / math.sqrt(n_sites),
            "acsb_steric_norm_per_site": s_norm / math.sqrt(n_sites),
            "acsb_joint_active_site_fraction": float(np.mean(joint)),
        }
        if tuple(values) != CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES:
            return _failure("Coulomb--steric balance feature schema differs")
        if not np.isfinite(list(values.values())).all():
            return _failure("Coulomb--steric balance features are non-finite")
        return BalanceFeatureResult(True, None, values)
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _analytic_coulomb_vectors(structure, charges: np.ndarray) -> np.ndarray:
    volume = float(structure.volume)
    n_sites = len(structure)
    q_rms = float(np.sqrt(np.mean(charges**2)))
    decorated = structure.copy()
    try:
        decorated.remove_oxidation_states()
    except Exception:
        pass
    decorated.add_oxidation_state_by_site(charges.tolist())
    ewald = EwaldSummation(decorated, compute_forces=True)
    derivative = np.asarray(ewald.forces, dtype=float)
    if derivative.shape != (n_sites, 3) or not np.isfinite(derivative).all():
        raise ValueError("analytic Ewald derivative is invalid")
    length = float((volume / n_sites) ** (1.0 / 3.0))
    return derivative * length**2 / (COULOMB_EV_ANGSTROM * q_rms**2)


def _analytic_steric_vectors(structure) -> np.ndarray:
    atoms = AseAtomsAdaptor.get_atoms(structure)
    numbers = np.asarray(atoms.numbers, dtype=int)
    radii = _resolve_radii(numbers, None)
    pairs = _periodic_steric_pairs(atoms, radii)
    vectors = np.zeros((len(atoms), 3), dtype=float)
    for left, right, unit, ratio in pairs:
        weight = max(0.0, max(float(ratio), 0.45) ** -12 - 1.0)
        if weight <= 0.0:
            continue
        vectors[left] -= weight * unit
        vectors[right] += weight * unit
    return vectors


def compute_coulomb_steric_balance_features(
    structure,
    charges: Sequence[float] | np.ndarray,
) -> BalanceFeatureResult:
    """Build both analytic fields at x0 and evaluate their closed-form balance."""

    try:
        charge = np.asarray(charges, dtype=float)
        if charge.shape != (len(structure),):
            return _failure("charges must match the structure sites")
        if len(structure) < 2 or not np.isfinite(charge).all():
            return _failure("charges must be finite and describe at least two sites")
        magnitude = float(np.abs(charge).sum())
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            return _failure("charges must be neutral")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            return _failure("charges must contain nonzero values of both signs")
        if not np.isfinite(float(structure.volume)) or float(structure.volume) <= 1.0e-10:
            return _failure("periodic structure volume must be finite and positive")
        coulomb = _analytic_coulomb_vectors(structure, charge)
        steric = _analytic_steric_vectors(structure)
        return analytic_vector_balance_features(coulomb, steric)
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _validate_batch_inputs(
    *,
    archive: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next34_feature_path: Path,
    next34_feature_manifest_path: Path,
) -> tuple[pd.DataFrame, list[object], pd.DataFrame]:
    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("NEXT35 geometry archive path/name is invalid")
    if metadata_path.name != "next32_cohort.parquet" or not metadata_path.is_file():
        raise ValueError("NEXT35 cohort metadata path/name is invalid")
    if next34_feature_path.name != NEXT34_FEATURE_NAME or not next34_feature_path.is_file():
        raise ValueError("NEXT35 upstream feature path/name is invalid")
    cohort_manifest = _strict_json(cohort_manifest_path, role="NEXT35 cohort manifest")
    cohort_outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("output_role") != "unrelaxed_x0_geometry_only"
        or cohort_manifest.get("endpoint_numeric_fields_parsed") is not False
        or cohort_manifest.get("label_values_exported") is not False
        or cohort_manifest.get("labels_opened") is not False
    ):
        raise ValueError("NEXT35 cohort is not a label-free geometry projection")
    if not isinstance(cohort_outputs, Mapping) or any(
        cohort_outputs.get(path.name) != _sha256(path)
        for path in (archive, metadata_path)
    ):
        raise ValueError("NEXT35 cohort geometry or metadata hash differs")
    upstream_manifest = _strict_json(
        next34_feature_manifest_path, role="NEXT35 upstream feature manifest"
    )
    upstream_outputs = upstream_manifest.get("outputs_sha256")
    if (
        upstream_manifest.get("protocol") != NEXT34_FEATURE_PROTOCOL
        or upstream_manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or upstream_manifest.get("labels_opened") is not False
        or upstream_manifest.get("endpoint_fields_read") is not False
        or upstream_manifest.get("dft_values_used") is not False
        or upstream_manifest.get("classical_analytic_electrostatics_used") is not True
        or upstream_manifest.get("electronic_structure_calculation_used") is not False
        or upstream_manifest.get("model_or_proxy_potential_used") is not False
        or upstream_manifest.get("coordinates_or_cell_modified") is not False
    ):
        raise ValueError("NEXT35 upstream features crossed the label-free boundary")
    if (
        not isinstance(upstream_outputs, Mapping)
        or upstream_outputs.get(NEXT34_FEATURE_NAME) != _sha256(next34_feature_path)
    ):
        raise ValueError("NEXT35 upstream feature hash differs")

    identity = ["material_id", "source_name", "parent_id", "natoms"]
    metadata = pd.read_parquet(metadata_path)
    if not {*identity, "input_role"}.issubset(metadata):
        raise ValueError("NEXT35 cohort metadata lacks required identity columns")
    metadata = metadata.loc[:, identity + ["input_role"]].copy()
    for column in ("material_id", "source_name", "parent_id"):
        metadata[column] = metadata[column].astype(str)
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        metadata.material_id.duplicated().any()
        or metadata.parent_id.duplicated().any()
        or not metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT35 cohort identities or roles differ")
    material_ids = tuple(metadata.material_id)
    loaded_ids, structures = _load_archive_only(archive, material_ids)
    if loaded_ids != list(material_ids) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT35 geometry identity or atom counts differ")
    upstream = pd.read_parquet(next34_feature_path)
    if not {*identity, *REUSED_FEATURE_NAMES}.issubset(upstream):
        raise ValueError("NEXT35 upstream table lacks frozen comparator columns")
    upstream = upstream.loc[:, identity + list(REUSED_FEATURE_NAMES)].copy()
    for column in ("material_id", "source_name", "parent_id"):
        upstream[column] = upstream[column].astype(str)
    upstream = upstream.sort_values("material_id", kind="stable", ignore_index=True)
    if upstream.material_id.duplicated().any() or not upstream[identity].equals(
        metadata[identity]
    ):
        raise ValueError("NEXT35 upstream identities differ from geometry")
    return metadata, structures, upstream


def build_coulomb_steric_balance_feature_batch(
    *,
    archive_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next34_feature_path: Path,
    next34_feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Seal NEXT35 balance features from exact geometry-only inputs."""

    paths = {
        "geometry": Path(archive_path).resolve(),
        "metadata": Path(metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "next34_features": Path(next34_feature_path).resolve(),
        "next34_feature_manifest": Path(next34_feature_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    metadata, structures, upstream = _validate_batch_inputs(
        archive=paths["geometry"],
        metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        next34_feature_path=paths["next34_features"],
        next34_feature_manifest_path=paths["next34_feature_manifest"],
    )
    rows: list[dict[str, object]] = []
    policies: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for meta, old, atoms in zip(
        metadata.to_dict("records"), upstream.to_dict("records"), structures, strict=True
    ):
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
        if assignment.supported and assignment.values is not None:
            result = compute_coulomb_steric_balance_features(
                structure, assignment.values
            )
        else:
            result = _failure(assignment.failure_reason or "valence assignment failed")
        policy = assignment.policy if assignment.supported else None
        if policy is not None:
            policies[str(policy)] += 1
        if not result.supported:
            failures[result.failure_reason or "unknown"] += 1
        row: dict[str, object] = {
            "material_id": str(meta["material_id"]),
            "source_name": str(meta["source_name"]),
            "parent_id": str(meta["parent_id"]),
            "natoms": int(meta["natoms"]),
            "acsb_supported": bool(result.supported),
            "acsb_failure": result.failure_reason,
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
        raise ValueError(f"NEXT35 feature output crossed no-DFT contract: {forbidden}")
    if len(features) != len(metadata) or features.material_id.duplicated().any():
        raise ValueError("NEXT35 feature identity accounting differs")

    source_dir = Path(__file__).resolve().parent
    source_names = (
        "next11_geometry_only_frames.py",
        "next19_valence_transport.py",
        "next32_inorganic_response_features.py",
        "next33_symmetry_steric_features.py",
        "next34_analytic_field_features.py",
        "next35_coulomb_steric_balance_features.py",
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
        "analytic_steric_field_used": True,
        "electronic_structure_calculation_used": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "sid_metadata_used": False,
        "feature_names": list(
            REUSED_FEATURE_NAMES + CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES
        ),
        "counts": {
            "rows": len(features),
            "atoms": int(features.natoms.sum()),
            "acsb_supported": int(features.acsb_supported.sum()),
        },
        "valence_policy_counts": dict(sorted(policies.items())),
        "failure_counts": dict(sorted(failures.items())),
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "package_versions": {"pymatgen": importlib.metadata.version("pymatgen")},
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
                raise RuntimeError(f"NEXT35 input changed before publication: {role}")
        for name, path in source_paths.items():
            if _sha256(path) != source_hashes[name]:
                raise RuntimeError(f"NEXT35 source changed before publication: {name}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "BalanceFeatureResult",
    "CANDIDATE_FEATURE_NAMES",
    "DIAGNOSTIC_FEATURE_NAMES",
    "FEATURE_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "REUSED_FEATURE_NAMES",
    "analytic_vector_balance_features",
    "build_coulomb_steric_balance_feature_batch",
    "compute_coulomb_steric_balance_features",
]
