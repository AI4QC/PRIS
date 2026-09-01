"""Rigidity-column/self-stress compatibility projections for one raw x0."""

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
from src.next19_valence_transport import (
    build_periodic_edge_geometry,
    infer_valence_assignment,
)
from src.next20_valence_rigidity import _tabulated_radius
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next36_charge_spectrum_features import (
    FEATURE_NAME as NEXT36_FEATURE_NAME,
    PROTOCOL as NEXT36_FEATURE_PROTOCOL,
)


CANDIDATE_FEATURE_NAMES = (
    "sscp_load_fraction",
    "sscp_load_rms",
    "sscp_load_q95",
    "sscp_atomic_load_fraction",
    "sscp_cell_load_fraction",
    "sscp_load_localization",
)
DIAGNOSTIC_FEATURE_NAMES = (
    "sscp_balanced_fraction",
    "sscp_cokernel_dimension_fraction",
    "sscp_matrix_rank",
    "sscp_constraint_count",
)
REUSED_FEATURE_NAMES = (
    "aefi_residual_max",
    "steric_rep12_vector_rms",
    "steric_rep12_vector_max",
    "sivr_site_imbalance_rms",
)
PROTOCOL = "2026-08-03-next37-self-stress-compatibility-features-v1"
FEATURE_NAME = "next37_self_stress_compatibility_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class CompatibilityFeatureResult:
    """Fail-open projection of analytic edge mismatch onto rigidity subspaces."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> CompatibilityFeatureResult:
    return CompatibilityFeatureResult(False, reason, {})


def _projection(
    matrix: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, int]:
    if matrix.shape[0] != len(values):
        raise ValueError("projection matrix row count differs")
    if not matrix.size:
        return np.zeros_like(values), 0
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or float(singular[0]) <= 0.0:
        return np.zeros_like(values), 0
    tolerance = (
        np.finfo(float).eps * max(matrix.shape) * float(singular[0])
    )
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        return np.zeros_like(values), 0
    basis = left[:, :rank]
    return basis @ (basis.T @ values), rank


def _weighted_inverted_cdf(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = float(quantile) * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def self_stress_compatibility_features(
    *,
    n_sites: int,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
    residuals: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> CompatibilityFeatureResult:
    """Decompose frozen edge mismatch into load and self-balanced components."""

    try:
        if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
            return _failure("compatibility graph needs at least two sites")
        n_sites = int(n_sites)
        pair = np.asarray(endpoints, dtype=int)
        vector = np.asarray(vectors, dtype=float)
        residual = np.asarray(residuals, dtype=float)
        weight = np.asarray(weights, dtype=float)
        if pair.ndim != 2 or pair.shape[1:] != (2,) or len(pair) < 1:
            return _failure("endpoints must have nonempty shape (E,2)")
        n_edges = len(pair)
        if vector.shape != (n_edges, 3):
            return _failure("vectors must have shape (E,3)")
        if residual.shape != (n_edges,) or weight.shape != (n_edges,):
            return _failure("residuals and weights must match all edges")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            return _failure("endpoints contain invalid site indices")
        if not np.isfinite(vector).all() or not np.isfinite(residual).all():
            return _failure("edge vectors and residuals must be finite")
        if not np.isfinite(weight).all() or np.any(weight <= 0.0):
            return _failure("edge weights must be finite and positive")
        distance = np.linalg.norm(vector, axis=1)
        if not np.isfinite(distance).all() or np.any(distance <= 0.0):
            return _failure("edge distances must be finite and positive")
        direction = vector / distance[:, None]
        root_weight = np.sqrt(weight)

        atomic = np.zeros((n_edges, 3 * n_sites), dtype=float)
        derivative = root_weight[:, None] * direction / distance[:, None]
        rows = np.arange(n_edges)
        for axis in range(3):
            atomic[rows, 3 * pair[:, 0] + axis] -= derivative[:, axis]
            atomic[rows, 3 * pair[:, 1] + axis] += derivative[:, axis]
        nx, ny, nz = direction.T
        affine = root_weight[:, None] * np.column_stack(
            (nx**2, ny**2, nz**2, 2.0 * ny * nz, 2.0 * nx * nz, 2.0 * nx * ny)
        )
        full = np.column_stack((atomic, affine))
        tension = root_weight * residual
        atomic_load, _atomic_rank = _projection(atomic, tension)
        full_load, full_rank = _projection(full, tension)
        self_balanced = tension - full_load

        total_norm = float(np.linalg.norm(tension))
        full_norm = float(np.linalg.norm(full_load))
        atomic_norm = float(np.linalg.norm(atomic_load))
        numerical_zero = 1.0e-12 * max(total_norm, np.finfo(float).tiny)
        if full_norm <= numerical_zero:
            full_load = np.zeros_like(full_load)
            full_norm = 0.0
        if atomic_norm <= numerical_zero:
            atomic_load = np.zeros_like(atomic_load)
            atomic_norm = 0.0
        self_balanced = tension - full_load
        self_norm = float(np.linalg.norm(self_balanced))
        if total_norm > 0.0:
            load_fraction = float(np.clip(full_norm / total_norm, 0.0, 1.0))
            atomic_fraction = float(np.clip(atomic_norm / total_norm, 0.0, 1.0))
            cell_square = max(0.0, full_norm**2 - atomic_norm**2)
            cell_fraction = float(
                np.clip(math.sqrt(cell_square) / total_norm, 0.0, 1.0)
            )
            self_fraction = float(np.clip(self_norm / total_norm, 0.0, 1.0))
        else:
            load_fraction = 0.0
            atomic_fraction = 0.0
            cell_fraction = 0.0
            self_fraction = 0.0
        load_square = full_load**2
        localization = (
            float(n_edges * np.max(load_square) / np.sum(load_square))
            if float(np.sum(load_square)) > 0.0
            else 0.0
        )
        load_edge_residual = np.abs(full_load) / root_weight
        values = {
            "sscp_load_fraction": load_fraction,
            "sscp_load_rms": float(full_norm / math.sqrt(float(weight.sum()))),
            "sscp_load_q95": float(
                _weighted_inverted_cdf(load_edge_residual, weight, 0.95)
            ),
            "sscp_atomic_load_fraction": atomic_fraction,
            "sscp_cell_load_fraction": cell_fraction,
            "sscp_load_localization": localization,
            "sscp_balanced_fraction": self_fraction,
            "sscp_cokernel_dimension_fraction": float(
                (n_edges - full_rank) / n_edges
            ),
            "sscp_matrix_rank": float(full_rank),
            "sscp_constraint_count": float(n_edges),
        }
        if tuple(values) != CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES:
            return _failure("self-stress compatibility feature schema differs")
        if not np.isfinite(list(values.values())).all():
            return _failure("self-stress compatibility features are non-finite")
        return CompatibilityFeatureResult(True, None, values)
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def compute_self_stress_compatibility_features(
    structure,
    charges: Sequence[float] | np.ndarray,
) -> CompatibilityFeatureResult:
    """Build the frozen NEXT20 graph and apply the NEXT37 projection kernel."""

    try:
        charge = np.asarray(charges, dtype=float)
        if charge.shape != (len(structure),) or not np.isfinite(charge).all():
            return _failure("charges must be finite and match all sites")
        magnitude = float(np.abs(charge).sum())
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            return _failure("charges must be neutral")
        geometry = build_periodic_edge_geometry(
            structure, charge, graph_mode="voronoi"
        )
        if not geometry.supported:
            return _failure(geometry.failure_reason or "periodic graph is unsupported")
        radii: list[float] = []
        for site in structure:
            value = _tabulated_radius(site.specie.symbol)
            if value is None:
                return _failure(f"tabulated radius is missing for {site.specie.symbol}")
            radii.append(float(value))
        endpoints: list[tuple[int, int]] = []
        vectors: list[np.ndarray] = []
        radius_sums: list[float] = []
        weights: list[float] = []
        for edge in geometry.edges:
            left = int(edge.cation)
            right = int(edge.anion)
            fractional = (
                np.asarray(structure[right].frac_coords, dtype=float)
                + np.asarray(edge.image, dtype=float)
                - np.asarray(structure[left].frac_coords, dtype=float)
            )
            displacement = np.asarray(
                structure.lattice.get_cartesian_coords(fractional), dtype=float
            )
            endpoints.append((left, right))
            vectors.append(displacement)
            radius_sums.append(radii[left] + radii[right])
            weights.append(float(edge.neighbor_weight))
        vector_array = np.asarray(vectors, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        distances = np.linalg.norm(vector_array, axis=1)
        log_ratio = np.log(distances / np.asarray(radius_sums, dtype=float))
        center = _weighted_inverted_cdf(log_ratio, weight_array, 0.5)
        residuals = log_ratio - center
        return self_stress_compatibility_features(
            n_sites=len(structure),
            endpoints=np.asarray(endpoints, dtype=int),
            vectors=vector_array,
            residuals=residuals,
            weights=weight_array,
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _validate_batch_inputs(
    *,
    archive: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next36_feature_path: Path,
    next36_feature_manifest_path: Path,
) -> tuple[pd.DataFrame, list[object], pd.DataFrame]:
    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("NEXT37 geometry archive path/name is invalid")
    if metadata_path.name != "next32_cohort.parquet" or not metadata_path.is_file():
        raise ValueError("NEXT37 cohort metadata path/name is invalid")
    if next36_feature_path.name != NEXT36_FEATURE_NAME or not next36_feature_path.is_file():
        raise ValueError("NEXT37 upstream feature path/name is invalid")
    cohort_manifest = _strict_json(cohort_manifest_path, role="NEXT37 cohort manifest")
    cohort_outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("output_role") != "unrelaxed_x0_geometry_only"
        or cohort_manifest.get("endpoint_numeric_fields_parsed") is not False
        or cohort_manifest.get("label_values_exported") is not False
        or cohort_manifest.get("labels_opened") is not False
    ):
        raise ValueError("NEXT37 cohort is not a label-free geometry projection")
    if not isinstance(cohort_outputs, Mapping) or any(
        cohort_outputs.get(path.name) != _sha256(path)
        for path in (archive, metadata_path)
    ):
        raise ValueError("NEXT37 cohort geometry or metadata hash differs")

    upstream_manifest = _strict_json(
        next36_feature_manifest_path, role="NEXT37 upstream feature manifest"
    )
    upstream_outputs = upstream_manifest.get("outputs_sha256")
    if (
        upstream_manifest.get("protocol") != NEXT36_FEATURE_PROTOCOL
        or upstream_manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or upstream_manifest.get("labels_opened") is not False
        or upstream_manifest.get("endpoint_fields_read") is not False
        or upstream_manifest.get("dft_values_used") is not False
        or upstream_manifest.get("weighted_charge_spectrum_used") is not True
        or upstream_manifest.get("electronic_structure_calculation_used") is not False
        or upstream_manifest.get("model_or_proxy_potential_used") is not False
        or upstream_manifest.get("coordinates_or_cell_modified") is not False
    ):
        raise ValueError("NEXT37 upstream features crossed the label-free boundary")
    if (
        not isinstance(upstream_outputs, Mapping)
        or upstream_outputs.get(NEXT36_FEATURE_NAME) != _sha256(next36_feature_path)
    ):
        raise ValueError("NEXT37 upstream feature hash differs")

    identity = ["material_id", "source_name", "parent_id", "natoms"]
    metadata = pd.read_parquet(metadata_path)
    if not {*identity, "input_role"}.issubset(metadata):
        raise ValueError("NEXT37 cohort metadata lacks required identity columns")
    metadata = metadata.loc[:, identity + ["input_role"]].copy()
    for column in ("material_id", "source_name", "parent_id"):
        metadata[column] = metadata[column].astype(str)
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        metadata.material_id.duplicated().any()
        or metadata.parent_id.duplicated().any()
        or not metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT37 cohort identities or roles differ")
    material_ids = tuple(metadata.material_id)
    loaded_ids, structures = _load_archive_only(archive, material_ids)
    if loaded_ids != list(material_ids) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT37 geometry identity or atom counts differ")

    upstream = pd.read_parquet(next36_feature_path)
    if not {*identity, *REUSED_FEATURE_NAMES}.issubset(upstream):
        raise ValueError("NEXT37 upstream table lacks frozen comparator columns")
    upstream = upstream.loc[:, identity + list(REUSED_FEATURE_NAMES)].copy()
    for column in ("material_id", "source_name", "parent_id"):
        upstream[column] = upstream[column].astype(str)
    upstream = upstream.sort_values("material_id", kind="stable", ignore_index=True)
    if upstream.material_id.duplicated().any() or not upstream[identity].equals(
        metadata[identity]
    ):
        raise ValueError("NEXT37 upstream identities differ from geometry")
    return metadata, structures, upstream


def build_self_stress_compatibility_feature_batch(
    *,
    archive_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next36_feature_path: Path,
    next36_feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Seal NEXT37 compatibility projections from exact geometry-only inputs."""

    paths = {
        "geometry": Path(archive_path).resolve(),
        "metadata": Path(metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "next36_features": Path(next36_feature_path).resolve(),
        "next36_feature_manifest": Path(next36_feature_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    metadata, structures, upstream = _validate_batch_inputs(
        archive=paths["geometry"],
        metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        next36_feature_path=paths["next36_features"],
        next36_feature_manifest_path=paths["next36_feature_manifest"],
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
            result = compute_self_stress_compatibility_features(
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
            "sscp_supported": bool(result.supported),
            "sscp_failure": result.failure_reason,
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
        raise ValueError(f"NEXT37 feature output crossed no-DFT contract: {forbidden}")
    if len(features) != len(metadata) or features.material_id.duplicated().any():
        raise ValueError("NEXT37 feature identity accounting differs")

    source_dir = Path(__file__).resolve().parent
    source_names = (
        "next11_geometry_only_frames.py",
        "next19_valence_transport.py",
        "next20_valence_rigidity.py",
        "next36_charge_spectrum_features.py",
        "next37_self_stress_compatibility_features.py",
    )
    source_paths = {f"src/{name}": source_dir / name for name in source_names}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "dft_values_used": False,
        "self_stress_compatibility_projection_used": True,
        "coordinate_displacement_solved_or_applied": False,
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
            "sscp_supported": int(features.sscp_supported.sum()),
        },
        "valence_policy_counts": dict(sorted(policies.items())),
        "failure_counts": dict(sorted(failures.items())),
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "package_versions": {"numpy": importlib.metadata.version("numpy")},
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
                raise RuntimeError(f"NEXT37 input changed before publication: {role}")
        for name, path in source_paths.items():
            if _sha256(path) != source_hashes[name]:
                raise RuntimeError(f"NEXT37 source changed before publication: {name}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "CompatibilityFeatureResult",
    "DIAGNOSTIC_FEATURE_NAMES",
    "FEATURE_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "REUSED_FEATURE_NAMES",
    "build_self_stress_compatibility_feature_batch",
    "compute_self_stress_compatibility_features",
    "self_stress_compatibility_features",
]
