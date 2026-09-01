"""Single-x0 approximate-symmetry and directional-steric features for NEXT33."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ase import Atoms
from ase.neighborlist import neighbor_list
import numpy as np
import pandas as pd
from pymatgen.core import Lattice
from pymatgen.util.coord import pbc_shortest_vectors
from scipy.optimize import linear_sum_assignment
import spglib

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import (
    _publish_directory_no_replace,
    _sha256,
    _strict_json,
)
from src.next32_inorganic_response_features import _resolve_radii
from src.next32_inorganic_response_features import (
    FEATURE_NAME as NEXT32_FEATURE_NAME,
    PROTOCOL as NEXT32_FEATURE_PROTOCOL,
)
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL


RELATIVE_SYMPREC_GRID = (0.003, 0.01, 0.02, 0.04, 0.08, 0.12)
SYMMETRY_FEATURE_NAMES = (
    "sym_recovery_onset_rel",
    "sym_recovery_gain_log2",
    "sym_orbit_collapse",
    "sym_recovery_residual_rms_rel",
    "sym_recovery_residual_q95_rel",
    "sym_recovery_residual_max_rel",
)
STERIC_FEATURE_NAMES = (
    "steric_rep12_pa",
    "steric_rep12_site_q95",
    "steric_rep12_site_max",
    "steric_rep12_vector_rms",
    "steric_rep12_vector_q95",
    "steric_rep12_vector_max",
    "steric_rep12_tensor_deviator",
    "steric_overlap2_vector_rms",
    "steric_overlap2_vector_q95",
    "steric_overlap2_tensor_deviator",
)
REUSED_NEXT32_FEATURE_NAMES = (
    "cov_q01",
    "cov_q05",
    "sivr_edge_mismatch_q95",
    "sivr_site_imbalance_rms",
)
PROTOCOL = "2026-08-03-next33-symmetry-steric-features-v1"
FEATURE_NAME = "next33_symmetry_steric_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class AnalyticFeatureResult:
    """One fail-open geometry-only descriptor result."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _strict_geometry(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(atoms) < 1 or not bool(np.all(atoms.pbc)):
        raise ValueError("three-dimensional periodic geometry is required")
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    numbers = np.asarray(atoms.numbers, dtype=int)
    if (
        cell.shape != (3, 3)
        or positions.shape != (len(atoms), 3)
        or numbers.shape != (len(atoms),)
        or np.any(numbers <= 0)
        or not np.all(np.isfinite(cell))
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("periodic geometry contains invalid values")
    volume = abs(float(np.linalg.det(cell)))
    if not np.isfinite(volume) or volume <= 1e-10:
        raise ValueError("periodic cell has zero volume")
    length = float((volume / len(atoms)) ** (1.0 / 3.0))
    fractional = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    return cell, fractional, numbers, length


def _symmetry_dataset(
    cell: np.ndarray,
    fractional: np.ndarray,
    numbers: np.ndarray,
    symprec: float,
):
    dataset = spglib.get_symmetry_dataset(
        (cell, fractional, numbers), symprec=float(symprec)
    )
    if dataset is None:
        raise ValueError("spglib returned no symmetry dataset")
    return dataset


def _primitive_representation(
    cell: np.ndarray,
    fractional: np.ndarray,
    numbers: np.ndarray,
    strict_symprec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove representational supercell translations without idealizing x0."""

    primitive = spglib.standardize_cell(
        (cell, fractional, numbers),
        to_primitive=True,
        no_idealize=True,
        symprec=float(strict_symprec),
    )
    if primitive is None:
        raise ValueError("spglib primitive reduction failed")
    primitive_cell, primitive_fractional, primitive_numbers = primitive
    return (
        np.asarray(primitive_cell, dtype=float),
        np.mod(np.asarray(primitive_fractional, dtype=float), 1.0),
        np.asarray(primitive_numbers, dtype=int),
    )


def _normalized_operation_count(dataset, n_atoms: int) -> float:
    primitive_sites = len({int(value) for value in dataset.mapping_to_primitive})
    if primitive_sites < 1 or n_atoms % primitive_sites != 0:
        raise ValueError("spglib primitive mapping is inconsistent")
    translation_multiplicity = n_atoms / primitive_sites
    value = len(dataset.rotations) / translation_multiplicity
    if not np.isfinite(value) or value < 1.0:
        raise ValueError("spglib operation count is inconsistent")
    return float(value)


def _orbit_fraction(dataset, n_atoms: int) -> float:
    value = len({int(index) for index in dataset.equivalent_atoms}) / n_atoms
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("spglib orbit accounting is inconsistent")
    return float(value)


def _operation_displacements(
    *,
    lattice: Lattice,
    fractional: np.ndarray,
    numbers: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    transformed = np.mod(fractional @ np.asarray(rotation, dtype=int).T + translation, 1.0)
    displacements = np.empty(len(fractional), dtype=float)
    for number in sorted({int(value) for value in numbers}):
        indices = np.flatnonzero(numbers == number)
        _vectors, squared = pbc_shortest_vectors(
            lattice,
            transformed[indices],
            fractional[indices],
            return_d2=True,
        )
        rows, columns = linear_sum_assignment(np.asarray(squared, dtype=float))
        if len(rows) != len(indices):
            raise ValueError("symmetry operation assignment is incomplete")
        displacements[indices[rows]] = np.sqrt(np.maximum(0.0, squared[rows, columns]))
    if not np.all(np.isfinite(displacements)):
        raise ValueError("symmetry operation residual is invalid")
    return displacements


def compute_symmetry_recovery_features(atoms: Atoms) -> AnalyticFeatureResult:
    """Measure approximate operations recoverable from x0 without symmetrization."""

    try:
        cell, fractional, numbers, length = _strict_geometry(atoms)
        cell, fractional, numbers = _primitive_representation(
            cell,
            fractional,
            numbers,
            RELATIVE_SYMPREC_GRID[0] * length,
        )
        datasets = [
            _symmetry_dataset(cell, fractional, numbers, relative * length)
            for relative in RELATIVE_SYMPREC_GRID
        ]
        operation_counts = [
            _normalized_operation_count(dataset, len(numbers)) for dataset in datasets
        ]
        strict_count = operation_counts[0]
        loose_count = operation_counts[-1]
        onset = next(
            (
                relative
                for relative, count in zip(
                    RELATIVE_SYMPREC_GRID[1:], operation_counts[1:], strict=True
                )
                if count > strict_count + 1e-8
            ),
            0.0,
        )
        gain = math.log2(loose_count / strict_count) if loose_count > strict_count else 0.0
        orbit_collapse = max(
            0.0,
            _orbit_fraction(datasets[0], len(numbers))
            - _orbit_fraction(datasets[-1], len(numbers)),
        )

        recovered: list[np.ndarray] = []
        if gain > 0.0:
            lattice = Lattice(cell)
            for rotation, translation in zip(
                datasets[-1].rotations, datasets[-1].translations, strict=True
            ):
                distances = _operation_displacements(
                    lattice=lattice,
                    fractional=fractional,
                    numbers=numbers,
                    rotation=rotation,
                    translation=translation,
                )
                relative_distances = distances / length
                if float(np.sqrt(np.mean(relative_distances**2))) > RELATIVE_SYMPREC_GRID[0]:
                    recovered.append(relative_distances)
        if recovered:
            residuals = np.concatenate(recovered)
            residual_rms = float(np.sqrt(np.mean(residuals**2)))
            residual_q95 = float(np.quantile(residuals, 0.95, method="inverted_cdf"))
            residual_max = float(np.max(residuals))
        else:
            residual_rms = residual_q95 = residual_max = 0.0
        values = {
            "sym_recovery_onset_rel": float(onset),
            "sym_recovery_gain_log2": float(gain),
            "sym_orbit_collapse": float(orbit_collapse),
            "sym_recovery_residual_rms_rel": residual_rms,
            "sym_recovery_residual_q95_rel": residual_q95,
            "sym_recovery_residual_max_rel": residual_max,
        }
        if tuple(values) != SYMMETRY_FEATURE_NAMES or not np.all(
            np.isfinite(list(values.values()))
        ):
            raise ValueError("NEXT33 symmetry feature schema or values are invalid")
        return AnalyticFeatureResult(True, None, values)
    except Exception as exc:
        return AnalyticFeatureResult(False, f"{type(exc).__name__}: {exc}", {})


def _periodic_steric_pairs(
    atoms: Atoms, radii: np.ndarray
) -> list[tuple[int, int, np.ndarray, float]]:
    cutoff = 3.2 * float(np.max(radii))
    first, second, shifts, vectors, distances = neighbor_list(
        "ijSDd", atoms, cutoff, self_interaction=True
    )
    pairs: list[tuple[int, int, tuple[int, int, int], np.ndarray, float]] = []
    for raw_i, raw_j, raw_shift, raw_vector, raw_distance in zip(
        first, second, shifts, vectors, distances, strict=True
    ):
        i = int(raw_i)
        j = int(raw_j)
        shift = tuple(int(value) for value in raw_shift)
        if i == j and shift == (0, 0, 0):
            continue
        reverse = tuple(-value for value in shift)
        if (i, j, *shift) >= (j, i, *reverse):
            continue
        distance = float(raw_distance)
        ratio = distance / float(radii[i] + radii[j])
        if np.isfinite(ratio) and ratio <= 1.6 and distance > 1e-12:
            unit = np.asarray(raw_vector, dtype=float) / distance
            pairs.append((i, j, shift, unit, float(ratio)))
    pairs.sort(key=lambda row: (row[0], row[1], row[2], row[4]))
    return [(i, j, unit, ratio) for i, j, _shift, unit, ratio in pairs]


def _steric_kernel_features(
    *,
    pairs: list[tuple[int, int, np.ndarray, float]],
    n_atoms: int,
    kernel: str,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    scalar = np.zeros(n_atoms, dtype=float)
    vector = np.zeros((n_atoms, 3), dtype=float)
    tensor = np.zeros((3, 3), dtype=float)
    total = 0.0
    for i, j, unit, ratio in pairs:
        if kernel == "rep12":
            weight = max(0.0, max(ratio, 0.45) ** -12 - 1.0)
        elif kernel == "overlap2":
            weight = max(0.0, 1.0 - ratio) ** 2
        else:
            raise RuntimeError(f"unknown NEXT33 steric kernel: {kernel}")
        if weight == 0.0:
            continue
        total += weight
        scalar[i] += weight
        scalar[j] += weight
        vector[i] -= weight * unit
        vector[j] += weight * unit
        tensor += weight * np.outer(unit, unit)
    trace = float(np.trace(tensor))
    if trace > 0.0:
        deviator = tensor - np.eye(3) * trace / 3.0
        tensor_deviator = float(np.linalg.norm(deviator) / trace)
    else:
        tensor_deviator = 0.0
    return 2.0 * total / n_atoms, scalar, np.linalg.norm(vector, axis=1), tensor_deviator


def compute_directional_steric_features(
    atoms: Atoms, *, radii: Mapping[int, float] | None = None
) -> AnalyticFeatureResult:
    """Compute dimensionless directional cancellation residuals from x0 geometry."""

    try:
        _cell, _fractional, numbers, _length = _strict_geometry(atoms)
        resolved_radii = _resolve_radii(numbers, radii)
        pairs = _periodic_steric_pairs(atoms, resolved_radii)
        if not pairs:
            raise ValueError("periodic geometry has no radius-scaled steric pairs")
        rep_pa, rep_site, rep_vector, rep_tensor = _steric_kernel_features(
            pairs=pairs, n_atoms=len(atoms), kernel="rep12"
        )
        _overlap_pa, _overlap_site, overlap_vector, overlap_tensor = (
            _steric_kernel_features(
                pairs=pairs, n_atoms=len(atoms), kernel="overlap2"
            )
        )
        values = {
            "steric_rep12_pa": float(rep_pa),
            "steric_rep12_site_q95": float(
                np.quantile(rep_site, 0.95, method="inverted_cdf")
            ),
            "steric_rep12_site_max": float(np.max(rep_site)),
            "steric_rep12_vector_rms": float(np.sqrt(np.mean(rep_vector**2))),
            "steric_rep12_vector_q95": float(
                np.quantile(rep_vector, 0.95, method="inverted_cdf")
            ),
            "steric_rep12_vector_max": float(np.max(rep_vector)),
            "steric_rep12_tensor_deviator": float(rep_tensor),
            "steric_overlap2_vector_rms": float(np.sqrt(np.mean(overlap_vector**2))),
            "steric_overlap2_vector_q95": float(
                np.quantile(overlap_vector, 0.95, method="inverted_cdf")
            ),
            "steric_overlap2_tensor_deviator": float(overlap_tensor),
        }
        if tuple(values) != STERIC_FEATURE_NAMES or not np.all(
            np.isfinite(list(values.values()))
        ):
            raise ValueError("NEXT33 steric feature schema or values are invalid")
        return AnalyticFeatureResult(True, None, values)
    except Exception as exc:
        return AnalyticFeatureResult(False, f"{type(exc).__name__}: {exc}", {})


def _validate_batch_inputs(
    *,
    archive: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next32_feature_path: Path,
    next32_feature_manifest_path: Path,
) -> tuple[pd.DataFrame, list[Atoms], pd.DataFrame]:
    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("NEXT33 geometry archive path/name is invalid")
    if metadata_path.name != "next32_cohort.parquet" or not metadata_path.is_file():
        raise ValueError("NEXT33 cohort metadata path/name is invalid")
    if next32_feature_path.name != NEXT32_FEATURE_NAME or not next32_feature_path.is_file():
        raise ValueError("NEXT33 upstream feature path/name is invalid")
    cohort_manifest = _strict_json(
        cohort_manifest_path, role="NEXT33 cohort manifest"
    )
    cohort_outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("output_role") != "unrelaxed_x0_geometry_only"
        or cohort_manifest.get("endpoint_numeric_fields_parsed") is not False
        or cohort_manifest.get("label_values_exported") is not False
        or cohort_manifest.get("labels_opened") is not False
    ):
        raise ValueError("NEXT33 cohort is not a label-free geometry projection")
    if not isinstance(cohort_outputs, Mapping) or any(
        cohort_outputs.get(path.name) != _sha256(path)
        for path in (archive, metadata_path)
    ):
        raise ValueError("NEXT33 cohort geometry or metadata hash differs")
    next32_manifest = _strict_json(
        next32_feature_manifest_path, role="NEXT33 upstream feature manifest"
    )
    next32_outputs = next32_manifest.get("outputs_sha256")
    if (
        next32_manifest.get("protocol") != NEXT32_FEATURE_PROTOCOL
        or next32_manifest.get("labels_opened") is not False
        or next32_manifest.get("endpoint_fields_read") is not False
        or next32_manifest.get("model_or_proxy_potential_used") is not False
    ):
        raise ValueError("NEXT33 upstream features crossed the label-free boundary")
    if (
        not isinstance(next32_outputs, Mapping)
        or next32_outputs.get(NEXT32_FEATURE_NAME) != _sha256(next32_feature_path)
    ):
        raise ValueError("NEXT33 upstream feature hash differs")

    metadata = pd.read_parquet(metadata_path)
    metadata_required = {
        "material_id",
        "source_name",
        "parent_id",
        "natoms",
        "input_role",
    }
    if not metadata_required.issubset(metadata):
        raise ValueError("NEXT33 cohort metadata lacks required identity columns")
    metadata = metadata.loc[:, sorted(metadata_required)].copy()
    metadata["material_id"] = metadata.material_id.astype(str)
    metadata["source_name"] = metadata.source_name.astype(str)
    metadata["parent_id"] = metadata.parent_id.astype(str)
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        metadata.material_id.duplicated().any()
        or metadata.parent_id.duplicated().any()
        or not metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT33 cohort identities or input roles differ")
    identities = tuple(metadata.material_id)
    loaded_ids, structures = _load_archive_only(archive, identities)
    if loaded_ids != list(identities) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT33 geometry identity or atom counts differ")

    old = pd.read_parquet(next32_feature_path)
    old_required = {
        "material_id",
        "source_name",
        "parent_id",
        "natoms",
        *REUSED_NEXT32_FEATURE_NAMES,
    }
    if not old_required.issubset(old):
        raise ValueError("NEXT33 upstream feature table lacks required columns")
    old = old.loc[:, [
        "material_id",
        "source_name",
        "parent_id",
        "natoms",
        *REUSED_NEXT32_FEATURE_NAMES,
    ]].copy()
    for column in ("material_id", "source_name", "parent_id"):
        old[column] = old[column].astype(str)
    old = old.sort_values("material_id", kind="stable", ignore_index=True)
    identity_columns = ["material_id", "source_name", "parent_id", "natoms"]
    if old.material_id.duplicated().any() or not old[identity_columns].equals(
        metadata[identity_columns]
    ):
        raise ValueError("NEXT33 upstream feature identities differ from geometry")
    return metadata, structures, old


def build_symmetry_steric_feature_batch(
    *,
    archive_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next32_feature_path: Path,
    next32_feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Seal NEXT33 analytic features from hash-locked geometry-only inputs."""

    paths = {
        "geometry": Path(archive_path).resolve(),
        "metadata": Path(metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "next32_features": Path(next32_feature_path).resolve(),
        "next32_feature_manifest": Path(next32_feature_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    metadata, structures, old = _validate_batch_inputs(
        archive=paths["geometry"],
        metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        next32_feature_path=paths["next32_features"],
        next32_feature_manifest_path=paths["next32_feature_manifest"],
    )
    rows: list[dict[str, object]] = []
    failure_counts: dict[str, dict[str, int]] = {"symmetry": {}, "steric": {}}
    for upstream, old_row, atoms in zip(
        metadata.to_dict("records"), old.to_dict("records"), structures, strict=True
    ):
        symmetry = compute_symmetry_recovery_features(atoms)
        steric = compute_directional_steric_features(atoms)
        row: dict[str, object] = {
            "material_id": str(upstream["material_id"]),
            "source_name": str(upstream["source_name"]),
            "parent_id": str(upstream["parent_id"]),
            "natoms": int(upstream["natoms"]),
            "symmetry_supported": symmetry.supported,
            "symmetry_failure": symmetry.failure_reason,
            "steric_supported": steric.supported,
            "steric_failure": steric.failure_reason,
        }
        for name in REUSED_NEXT32_FEATURE_NAMES:
            row[name] = float(old_row[name])
        for family, result, names in (
            ("symmetry", symmetry, SYMMETRY_FEATURE_NAMES),
            ("steric", steric, STERIC_FEATURE_NAMES),
        ):
            if not result.supported:
                reason = result.failure_reason or "unknown"
                failure_counts[family][reason] = failure_counts[family].get(reason, 0) + 1
            for name in names:
                row[name] = float(result.features[name]) if result.supported else math.nan
        rows.append(row)
    features = pd.DataFrame(rows)
    forbidden = [
        column
        for column in features
        if column.lower() == "sid"
        or any(
            token in column.lower()
            for token in ("energy", "force", "stress", "dft", "endpoint", "label", "target")
        )
    ]
    if forbidden:
        raise ValueError(f"NEXT33 feature output crossed no-DFT contract: {forbidden}")
    if len(features) != len(metadata) or features.material_id.duplicated().any():
        raise ValueError("NEXT33 feature identity accounting differs")

    source_dir = Path(__file__).resolve().parent
    source_names = (
        "next11_geometry_only_frames.py",
        "next32_inorganic_response_features.py",
        "next33_symmetry_steric_features.py",
    )
    source_paths = {f"src/{name}": source_dir / name for name in source_names}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "sid_metadata_used": False,
        "symmetrized_or_refined_structure_constructed": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "feature_names": list(
            REUSED_NEXT32_FEATURE_NAMES + SYMMETRY_FEATURE_NAMES + STERIC_FEATURE_NAMES
        ),
        "counts": {
            "rows": len(features),
            "atoms": int(features.natoms.sum()),
            "symmetry_supported": int(features.symmetry_supported.sum()),
            "steric_supported": int(features.steric_supported.sum()),
        },
        "failure_counts": {
            family: dict(sorted(values.items()))
            for family, values in failure_counts.items()
        },
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
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
                raise RuntimeError(f"NEXT33 input changed before publication: {role}")
        for name, path in source_paths.items():
            if _sha256(path) != source_hashes[name]:
                raise RuntimeError(f"NEXT33 source changed before publication: {name}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "AnalyticFeatureResult",
    "FEATURE_NAME",
    "MANIFEST_NAME",
    "RELATIVE_SYMPREC_GRID",
    "REUSED_NEXT32_FEATURE_NAMES",
    "STERIC_FEATURE_NAMES",
    "SYMMETRY_FEATURE_NAMES",
    "build_symmetry_steric_feature_batch",
    "compute_directional_steric_features",
    "compute_symmetry_recovery_features",
]
