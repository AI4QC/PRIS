"""Finite-cell weighted charge-spectrum descriptors from one raw crystal x0."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import (
    _publish_directory_no_replace,
    _sha256,
    _strict_json,
)
from src.next19_valence_transport import infer_valence_assignment
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next35_coulomb_steric_balance_features import (
    FEATURE_NAME as NEXT35_FEATURE_NAME,
    PROTOCOL as NEXT35_FEATURE_PROTOCOL,
)


SMOOTHING_SCALES = (0.25, 0.40, 0.60)
DIMENSIONLESS_CUTOFF = 18.0
MAX_ENUMERATED_RECIPROCAL_POINTS = 2_000_000
CANDIDATE_FEATURE_NAMES = (
    "csf_gaussian_t025",
    "csf_gaussian_t040",
    "csf_gaussian_t060",
    "csf_long_fraction",
    "csf_long_peak_fraction",
    "csf_long_anisotropy",
)
DIAGNOSTIC_FEATURE_NAMES = (
    "csf_reciprocal_vector_count",
    "csf_min_dimensionless_wavenumber",
    "csf_truncated_intensity",
)
REUSED_FEATURE_NAMES = (
    "aefi_residual_max",
    "steric_rep12_vector_rms",
    "steric_rep12_vector_max",
    "sivr_site_imbalance_rms",
)
PROTOCOL = "2026-08-03-next36-weighted-charge-spectrum-features-v1"
FEATURE_NAME = "next36_charge_spectrum_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class ChargeSpectrumFeatureResult:
    """Fail-open finite-cell weighted charge-spectrum result."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> ChargeSpectrumFeatureResult:
    return ChargeSpectrumFeatureResult(False, reason, {})


def _reciprocal_integer_vectors(
    lattice: np.ndarray,
    *,
    length_per_site: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    maximum_wave_number = DIMENSIONLESS_CUTOFF / length_per_site
    bounds = np.ceil(
        maximum_wave_number * np.linalg.norm(lattice, axis=1) / (2.0 * np.pi)
    ).astype(int)
    if np.any(bounds < 1):
        bounds = np.maximum(bounds, 1)
    count = int(np.prod(2 * bounds + 1, dtype=np.int64))
    if count > MAX_ENUMERATED_RECIPROCAL_POINTS:
        raise ValueError(
            f"reciprocal enumeration would require {count} integer points"
        )
    axes = [np.arange(-bound, bound + 1, dtype=int) for bound in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    integers = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    cartesian = integers @ reciprocal
    dimensionless = np.linalg.norm(cartesian, axis=1) * length_per_site
    selected = (dimensionless > 1.0e-12) & (
        dimensionless <= DIMENSIONLESS_CUTOFF + 1.0e-12
    )
    if not selected.any():
        raise ValueError("reciprocal cutoff contains no nonzero vectors")
    return integers[selected], cartesian[selected], dimensionless[selected]


def charge_spectrum_features(
    lattice_matrix: Sequence[Sequence[float]] | np.ndarray,
    fractional_coordinates: Sequence[Sequence[float]] | np.ndarray,
    charges: Sequence[float] | np.ndarray,
) -> ChargeSpectrumFeatureResult:
    """Compute scale-free Gaussian charge-spectrum metrics for one periodic cell."""

    try:
        lattice = np.asarray(lattice_matrix, dtype=float)
        fractional = np.asarray(fractional_coordinates, dtype=float)
        charge = np.asarray(charges, dtype=float)
        if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
            return _failure("lattice must be a finite 3x3 matrix")
        if (
            fractional.ndim != 2
            or fractional.shape[1:] != (3,)
            or len(fractional) < 2
            or not np.isfinite(fractional).all()
        ):
            return _failure("fractional coordinates must have finite shape (N,3)")
        if charge.shape != (len(fractional),) or not np.isfinite(charge).all():
            return _failure("charges must be finite and match all sites")
        magnitude = float(np.abs(charge).sum())
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            return _failure("charges must be neutral")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            return _failure("charges must contain nonzero values of both signs")
        q2 = float(np.sum(charge**2))
        volume = abs(float(np.linalg.det(lattice)))
        if not np.isfinite(q2) or q2 <= 0.0:
            return _failure("charges must have positive squared magnitude")
        if not np.isfinite(volume) or volume <= 1.0e-12:
            return _failure("periodic cell volume must be finite and positive")
        n_sites = len(charge)
        length_per_site = float((volume / n_sites) ** (1.0 / 3.0))
        integers, cartesian, dimensionless = _reciprocal_integer_vectors(
            lattice, length_per_site=length_per_site
        )

        fractional = np.mod(fractional, 1.0)
        intensity = np.empty(len(integers), dtype=float)
        for start in range(0, len(integers), 16_384):
            stop = min(start + 16_384, len(integers))
            phase = 2.0 * np.pi * (integers[start:stop] @ fractional.T)
            amplitude = np.exp(-1j * phase) @ charge
            intensity[start:stop] = np.abs(amplitude) ** 2 / (n_sites * q2)
        intensity = np.maximum(intensity, 0.0)
        if not np.isfinite(intensity).all():
            return _failure("weighted reciprocal intensities are non-finite")

        weights = {
            tau: np.exp(-((tau * dimensionless) ** 2))
            for tau in SMOOTHING_SCALES
        }
        smoothed = {
            tau: float(np.dot(intensity, weights[tau]))
            for tau in SMOOTHING_SCALES
        }
        short = smoothed[0.25]
        long = smoothed[0.60]
        if not np.isfinite(short) or short <= 0.0:
            return _failure("truncated charge spectrum has zero support")
        long_fraction = float(long / short)
        long_weighted = intensity * weights[0.60]
        if long > 0.0:
            peak_fraction = float(np.max(long_weighted) / long)
            norms = np.linalg.norm(cartesian, axis=1)
            directions = cartesian / norms[:, None]
            tensor = np.einsum(
                "i,ij,ik->jk", long_weighted, directions, directions
            ) / long
            deviator = tensor - np.eye(3) / 3.0
            anisotropy = float(
                np.clip(np.sqrt(1.5) * np.linalg.norm(deviator), 0.0, 1.0)
            )
        else:
            peak_fraction = 0.0
            anisotropy = 0.0
        values = {
            "csf_gaussian_t025": short,
            "csf_gaussian_t040": smoothed[0.40],
            "csf_gaussian_t060": long,
            "csf_long_fraction": float(np.clip(long_fraction, 0.0, 1.0)),
            "csf_long_peak_fraction": float(np.clip(peak_fraction, 0.0, 1.0)),
            "csf_long_anisotropy": anisotropy,
            "csf_reciprocal_vector_count": float(len(integers)),
            "csf_min_dimensionless_wavenumber": float(np.min(dimensionless)),
            "csf_truncated_intensity": float(np.sum(intensity)),
        }
        if tuple(values) != CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES:
            return _failure("charge-spectrum feature schema differs")
        if not np.isfinite(list(values.values())).all():
            return _failure("charge-spectrum features are non-finite")
        return ChargeSpectrumFeatureResult(True, None, values)
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def compute_charge_spectrum_features(
    structure,
    charges: Sequence[float] | np.ndarray,
) -> ChargeSpectrumFeatureResult:
    """Apply the pure charge-spectrum kernel to one pymatgen structure."""

    try:
        return charge_spectrum_features(
            np.asarray(structure.lattice.matrix, dtype=float),
            np.asarray(structure.frac_coords, dtype=float),
            charges,
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _validate_batch_inputs(
    *,
    archive: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next35_feature_path: Path,
    next35_feature_manifest_path: Path,
) -> tuple[pd.DataFrame, list[object], pd.DataFrame]:
    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("NEXT36 geometry archive path/name is invalid")
    if metadata_path.name != "next32_cohort.parquet" or not metadata_path.is_file():
        raise ValueError("NEXT36 cohort metadata path/name is invalid")
    if next35_feature_path.name != NEXT35_FEATURE_NAME or not next35_feature_path.is_file():
        raise ValueError("NEXT36 upstream feature path/name is invalid")
    cohort_manifest = _strict_json(cohort_manifest_path, role="NEXT36 cohort manifest")
    cohort_outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("output_role") != "unrelaxed_x0_geometry_only"
        or cohort_manifest.get("endpoint_numeric_fields_parsed") is not False
        or cohort_manifest.get("label_values_exported") is not False
        or cohort_manifest.get("labels_opened") is not False
    ):
        raise ValueError("NEXT36 cohort is not a label-free geometry projection")
    if not isinstance(cohort_outputs, Mapping) or any(
        cohort_outputs.get(path.name) != _sha256(path)
        for path in (archive, metadata_path)
    ):
        raise ValueError("NEXT36 cohort geometry or metadata hash differs")

    upstream_manifest = _strict_json(
        next35_feature_manifest_path, role="NEXT36 upstream feature manifest"
    )
    upstream_outputs = upstream_manifest.get("outputs_sha256")
    if (
        upstream_manifest.get("protocol") != NEXT35_FEATURE_PROTOCOL
        or upstream_manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or upstream_manifest.get("labels_opened") is not False
        or upstream_manifest.get("endpoint_fields_read") is not False
        or upstream_manifest.get("dft_values_used") is not False
        or upstream_manifest.get("classical_analytic_electrostatics_used") is not True
        or upstream_manifest.get("analytic_steric_field_used") is not True
        or upstream_manifest.get("electronic_structure_calculation_used") is not False
        or upstream_manifest.get("model_or_proxy_potential_used") is not False
        or upstream_manifest.get("coordinates_or_cell_modified") is not False
    ):
        raise ValueError("NEXT36 upstream features crossed the label-free boundary")
    if (
        not isinstance(upstream_outputs, Mapping)
        or upstream_outputs.get(NEXT35_FEATURE_NAME) != _sha256(next35_feature_path)
    ):
        raise ValueError("NEXT36 upstream feature hash differs")

    identity = ["material_id", "source_name", "parent_id", "natoms"]
    metadata = pd.read_parquet(metadata_path)
    if not {*identity, "input_role"}.issubset(metadata):
        raise ValueError("NEXT36 cohort metadata lacks required identity columns")
    metadata = metadata.loc[:, identity + ["input_role"]].copy()
    for column in ("material_id", "source_name", "parent_id"):
        metadata[column] = metadata[column].astype(str)
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        metadata.material_id.duplicated().any()
        or metadata.parent_id.duplicated().any()
        or not metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT36 cohort identities or roles differ")
    material_ids = tuple(metadata.material_id)
    loaded_ids, structures = _load_archive_only(archive, material_ids)
    if loaded_ids != list(material_ids) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT36 geometry identity or atom counts differ")

    upstream = pd.read_parquet(next35_feature_path)
    if not {*identity, *REUSED_FEATURE_NAMES}.issubset(upstream):
        raise ValueError("NEXT36 upstream table lacks frozen comparator columns")
    upstream = upstream.loc[:, identity + list(REUSED_FEATURE_NAMES)].copy()
    for column in ("material_id", "source_name", "parent_id"):
        upstream[column] = upstream[column].astype(str)
    upstream = upstream.sort_values("material_id", kind="stable", ignore_index=True)
    if upstream.material_id.duplicated().any() or not upstream[identity].equals(
        metadata[identity]
    ):
        raise ValueError("NEXT36 upstream identities differ from geometry")
    return metadata, structures, upstream


def build_charge_spectrum_feature_batch(
    *,
    archive_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next35_feature_path: Path,
    next35_feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Seal NEXT36 weighted charge-spectrum features from geometry-only inputs."""

    paths = {
        "geometry": Path(archive_path).resolve(),
        "metadata": Path(metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "next35_features": Path(next35_feature_path).resolve(),
        "next35_feature_manifest": Path(next35_feature_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    metadata, structures, upstream = _validate_batch_inputs(
        archive=paths["geometry"],
        metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        next35_feature_path=paths["next35_features"],
        next35_feature_manifest_path=paths["next35_feature_manifest"],
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
            result = compute_charge_spectrum_features(structure, assignment.values)
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
            "csf_supported": bool(result.supported),
            "csf_failure": result.failure_reason,
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
        raise ValueError(f"NEXT36 feature output crossed no-DFT contract: {forbidden}")
    if len(features) != len(metadata) or features.material_id.duplicated().any():
        raise ValueError("NEXT36 feature identity accounting differs")

    source_dir = Path(__file__).resolve().parent
    source_names = (
        "next11_geometry_only_frames.py",
        "next19_valence_transport.py",
        "next35_coulomb_steric_balance_features.py",
        "next36_charge_spectrum_features.py",
    )
    source_paths = {f"src/{name}": source_dir / name for name in source_names}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "dft_values_used": False,
        "weighted_charge_spectrum_used": True,
        "thermodynamic_limit_hyperuniformity_claimed": False,
        "electronic_structure_calculation_used": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "sid_metadata_used": False,
        "smoothing_scales": list(SMOOTHING_SCALES),
        "dimensionless_reciprocal_cutoff": DIMENSIONLESS_CUTOFF,
        "feature_names": list(
            REUSED_FEATURE_NAMES + CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES
        ),
        "counts": {
            "rows": len(features),
            "atoms": int(features.natoms.sum()),
            "csf_supported": int(features.csf_supported.sum()),
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
                raise RuntimeError(f"NEXT36 input changed before publication: {role}")
        for name, path in source_paths.items():
            if _sha256(path) != source_hashes[name]:
                raise RuntimeError(f"NEXT36 source changed before publication: {name}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "ChargeSpectrumFeatureResult",
    "DIAGNOSTIC_FEATURE_NAMES",
    "DIMENSIONLESS_CUTOFF",
    "FEATURE_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "REUSED_FEATURE_NAMES",
    "SMOOTHING_SCALES",
    "build_charge_spectrum_feature_batch",
    "charge_spectrum_features",
    "compute_charge_spectrum_features",
]
