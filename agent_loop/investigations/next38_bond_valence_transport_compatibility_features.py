"""Bond-valence transport correction and differential compatibility for raw x0."""

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

from src.advanced_local_features import resolve_bond_valence_parameter
from src.elec_feat import bv_table
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
from src.next22_bond_valence_equilibrium import PARAMETER_SOURCES
from src.next32_inorganic_response_features import (
    FEATURE_NAME as NEXT32_FEATURE_NAME,
    PROTOCOL as NEXT32_FEATURE_PROTOCOL,
)
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next37_self_stress_compatibility_features import (
    FEATURE_NAME as NEXT37_FEATURE_NAME,
    PROTOCOL as NEXT37_FEATURE_PROTOCOL,
)


CANDIDATE_FEATURE_NAMES = (
    "bvtc_correction_rms",
    "bvtc_compatible_rms",
    "bvtc_compatible_q95",
    "bvtc_incompatible_rms",
    "bvtc_incompatible_fraction",
    "bvtc_compatible_localization",
)
DIAGNOSTIC_FEATURE_NAMES = (
    "bvtc_site_deficit_rms",
    "bvtc_site_deficit_max",
    "bvtc_jacobian_rank",
    "bvtc_jacobian_rank_fraction",
    "bvtc_negative_corrected_edge_fraction",
    "bvtc_parameter_exact_fraction",
    "bvtc_parameter_generic_fraction",
    "bvtc_edge_count",
)
REUSED_FEATURE_NAMES = (
    "scbv_mismatch_q95",
    "steric_rep12_vector_rms",
    "steric_rep12_vector_max",
    "sivr_site_imbalance_rms",
)
PROTOCOL = "2026-08-03-next38-bond-valence-transport-compatibility-features-v1"
FEATURE_NAME = "next38_bond_valence_transport_compatibility_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


@dataclass(frozen=True)
class BondValenceTransportCompatibilityResult:
    """Fail-open result for one analytic bond-valence correction projection."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(reason: str) -> BondValenceTransportCompatibilityResult:
    return BondValenceTransportCompatibilityResult(False, reason, {})


def _svd_tolerance(matrix: np.ndarray, singular_max: float) -> float:
    return np.finfo(float).eps * max(matrix.shape) * float(singular_max)


def _projection(matrix: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, int]:
    if matrix.ndim != 2 or matrix.shape[0] != len(values):
        raise ValueError("projection matrix row count differs")
    if not matrix.size:
        return np.zeros_like(values), 0
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or float(singular[0]) <= 0.0:
        return np.zeros_like(values), 0
    rank = int(np.sum(singular > _svd_tolerance(matrix, float(singular[0]))))
    if rank == 0:
        return np.zeros_like(values), 0
    basis = left[:, :rank]
    return basis @ (basis.T @ values), rank


def _minimum_norm_solution(
    matrix: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, int, np.ndarray]:
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise ValueError("minimum-norm system shape differs")
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or float(singular[0]) <= 0.0:
        solution = np.zeros(matrix.shape[1], dtype=float)
        return solution, 0, values.copy()
    rank = int(np.sum(singular > _svd_tolerance(matrix, float(singular[0]))))
    if rank == 0:
        solution = np.zeros(matrix.shape[1], dtype=float)
    else:
        solution = right[:rank].T @ ((left[:, :rank].T @ values) / singular[:rank])
    return solution, rank, values - matrix @ solution


def _inverted_cdf(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(values, kind="stable")
    index = max(0, int(math.ceil(float(quantile) * len(ordered))) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def transport_compatibility_from_jacobian(
    *,
    charges: Sequence[float] | np.ndarray,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    priors: Sequence[float] | np.ndarray,
    jacobian: Sequence[Sequence[float]] | np.ndarray,
    parameter_sources: Sequence[str],
) -> BondValenceTransportCompatibilityResult:
    """Project the unique minimum-norm site-valence correction through J."""

    try:
        charge = np.asarray(charges, dtype=float)
        pair = np.asarray(endpoints, dtype=int)
        prior = np.asarray(priors, dtype=float)
        matrix = np.asarray(jacobian, dtype=float)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            return _failure("charges must be a finite site vector")
        n_sites = len(charge)
        magnitude = float(np.abs(charge).sum())
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            return _failure("charges need both signs")
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            return _failure("charges must be neutral")
        if pair.ndim != 2 or pair.shape[1:] != (2,) or len(pair) < 1:
            return _failure("endpoints must have nonempty shape (E,2)")
        n_edges = len(pair)
        if prior.shape != (n_edges,) or matrix.ndim != 2 or matrix.shape[0] != n_edges:
            return _failure("priors and Jacobian must match all edges")
        if matrix.shape[1] < 1:
            return _failure("Jacobian needs at least one generalized column")
        if len(parameter_sources) != n_edges or any(
            source not in PARAMETER_SOURCES for source in parameter_sources
        ):
            return _failure("parameter sources are invalid")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            return _failure("endpoints contain invalid site indices")
        if not np.all(charge[pair[:, 0]] > 0.0) or not np.all(
            charge[pair[:, 1]] < 0.0
        ):
            return _failure("edges must be ordered cation to anion")
        if not np.isfinite(prior).all() or np.any(prior <= 0.0):
            return _failure("edge priors must be finite and positive")
        if not np.isfinite(matrix).all():
            return _failure("Jacobian must be finite")

        for cation in np.flatnonzero(charge > 0.0):
            selected = pair[:, 0] == cation
            if not selected.any():
                return _failure("cation has no edge prior")
            supply = float(charge[cation])
            if not math.isclose(
                float(prior[selected].sum()), supply, rel_tol=1.0e-8, abs_tol=1.0e-8
            ):
                return _failure("edge priors do not sum to cation supply")
            star_sum = np.sum(matrix[selected], axis=0)
            scale = max(1.0, float(np.linalg.norm(matrix[selected])))
            if float(np.linalg.norm(star_sum)) > 1.0e-8 * scale:
                return _failure("Jacobian does not preserve cation-star supply")

        incidence = np.zeros((n_sites, n_edges), dtype=float)
        columns = np.arange(n_edges)
        incidence[pair[:, 0], columns] = 1.0
        incidence[pair[:, 1], columns] = 1.0
        site_deficit = np.abs(charge) - incidence @ prior
        correction, _incidence_rank, conservation_residual = _minimum_norm_solution(
            incidence, site_deficit
        )
        conservation_scale = max(1.0, float(np.linalg.norm(site_deficit)))
        if float(np.linalg.norm(conservation_residual)) > 1.0e-8 * conservation_scale:
            return _failure("periodic graph cannot carry the site-valence correction")

        exact_zero_scale = max(
            1.0,
            float(np.linalg.norm(prior)),
            float(np.linalg.norm(np.abs(charge))),
        )
        if float(np.linalg.norm(site_deficit)) <= 1.0e-12 * exact_zero_scale:
            site_deficit = np.zeros_like(site_deficit)
            correction = np.zeros_like(correction)

        compatible, jacobian_rank = _projection(matrix, correction)
        correction_norm = float(np.linalg.norm(correction))
        compatible_norm = float(np.linalg.norm(compatible))
        numerical_zero = 1.0e-12 * max(correction_norm, np.finfo(float).tiny)
        if compatible_norm <= numerical_zero:
            compatible = np.zeros_like(compatible)
            compatible_norm = 0.0
        incompatible = correction - compatible
        incompatible_norm = float(np.linalg.norm(incompatible))
        if incompatible_norm <= numerical_zero:
            incompatible = np.zeros_like(incompatible)
            incompatible_norm = 0.0
        incompatible_fraction = (
            float(np.clip(incompatible_norm / correction_norm, 0.0, 1.0))
            if correction_norm > 0.0
            else 0.0
        )
        compatible_square = compatible**2
        compatible_localization = (
            float(n_edges * np.max(compatible_square) / np.sum(compatible_square))
            if float(np.sum(compatible_square)) > 0.0
            else 0.0
        )
        corrected = prior + correction
        negative_tolerance = 1.0e-10 * max(1.0, float(np.max(prior)))
        sources = np.asarray(parameter_sources, dtype=object)
        values = {
            "bvtc_correction_rms": float(correction_norm / math.sqrt(n_edges)),
            "bvtc_compatible_rms": float(compatible_norm / math.sqrt(n_edges)),
            "bvtc_compatible_q95": _inverted_cdf(np.abs(compatible), 0.95),
            "bvtc_incompatible_rms": float(incompatible_norm / math.sqrt(n_edges)),
            "bvtc_incompatible_fraction": incompatible_fraction,
            "bvtc_compatible_localization": compatible_localization,
            "bvtc_site_deficit_rms": float(np.sqrt(np.mean(site_deficit**2))),
            "bvtc_site_deficit_max": float(np.max(np.abs(site_deficit))),
            "bvtc_jacobian_rank": float(jacobian_rank),
            "bvtc_jacobian_rank_fraction": float(jacobian_rank / n_edges),
            "bvtc_negative_corrected_edge_fraction": float(
                np.mean(corrected < -negative_tolerance)
            ),
            "bvtc_parameter_exact_fraction": float(np.mean(sources == "exact")),
            "bvtc_parameter_generic_fraction": float(
                np.mean(np.isin(sources, ("brown_generic", "radius_generic")))
            ),
            "bvtc_edge_count": float(n_edges),
        }
        if tuple(values) != CANDIDATE_FEATURE_NAMES + DIAGNOSTIC_FEATURE_NAMES:
            return _failure("bond-valence transport compatibility schema differs")
        if not np.isfinite(list(values.values())).all():
            return _failure("bond-valence transport compatibility features are non-finite")
        return BondValenceTransportCompatibilityResult(True, None, values)
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def bond_valence_transport_compatibility_features(
    *,
    charges: Sequence[float] | np.ndarray,
    endpoints: Sequence[Sequence[int]] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
    strengths: Sequence[float] | np.ndarray,
    decays: Sequence[float] | np.ndarray,
    parameter_sources: Sequence[str],
) -> BondValenceTransportCompatibilityResult:
    """Build the normalized bond-valence prior and its exact analytic Jacobian."""

    try:
        charge = np.asarray(charges, dtype=float)
        pair = np.asarray(endpoints, dtype=int)
        vector = np.asarray(vectors, dtype=float)
        strength = np.asarray(strengths, dtype=float)
        decay = np.asarray(decays, dtype=float)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            return _failure("charges must be a finite site vector")
        n_sites = len(charge)
        if pair.ndim != 2 or pair.shape[1:] != (2,) or len(pair) < 1:
            return _failure("endpoints must have nonempty shape (E,2)")
        n_edges = len(pair)
        if vector.shape != (n_edges, 3):
            return _failure("vectors must have shape (E,3)")
        if strength.shape != (n_edges,) or decay.shape != (n_edges,):
            return _failure("strengths and decays must match all edges")
        if (
            np.any(pair < 0)
            or np.any(pair >= n_sites)
            or np.any(pair[:, 0] == pair[:, 1])
        ):
            return _failure("endpoints contain invalid site indices")
        if not np.all(charge[pair[:, 0]] > 0.0) or not np.all(
            charge[pair[:, 1]] < 0.0
        ):
            return _failure("edges must be ordered cation to anion")
        if not np.isfinite(vector).all():
            return _failure("edge vectors must be finite")
        if not np.isfinite(strength).all() or np.any(strength <= 0.0):
            return _failure("bond strengths must be finite and positive")
        if not np.isfinite(decay).all() or np.any(decay <= 0.0):
            return _failure("bond decays must be finite and positive")
        distance = np.linalg.norm(vector, axis=1)
        if not np.isfinite(distance).all() or np.any(distance <= 0.0):
            return _failure("edge distances must be finite and positive")
        direction = vector / distance[:, None]

        prior = np.zeros(n_edges, dtype=float)
        for cation in np.flatnonzero(charge > 0.0):
            selected = pair[:, 0] == cation
            if not selected.any():
                return _failure("cation has no bond-valence edge")
            normalizer = float(strength[selected].sum())
            if not math.isfinite(normalizer) or normalizer <= 0.0:
                return _failure("cation bond-valence normalizer is invalid")
            prior[selected] = float(charge[cation]) * strength[selected] / normalizer

        distance_jacobian = np.zeros((n_edges, 3 * n_sites + 6), dtype=float)
        rows = np.arange(n_edges)
        for axis in range(3):
            distance_jacobian[rows, 3 * pair[:, 0] + axis] -= direction[:, axis]
            distance_jacobian[rows, 3 * pair[:, 1] + axis] += direction[:, axis]
        nx, ny, nz = direction.T
        distance_jacobian[:, -6:] = distance[:, None] * np.column_stack(
            (nx**2, ny**2, nz**2, 2.0 * ny * nz, 2.0 * nx * nz, 2.0 * ny * nx)
        )
        log_strength_jacobian = -distance_jacobian / decay[:, None]
        normalized_jacobian = np.zeros_like(log_strength_jacobian)
        for cation in np.flatnonzero(charge > 0.0):
            selected = pair[:, 0] == cation
            weights = prior[selected] / float(charge[cation])
            star_mean = weights @ log_strength_jacobian[selected]
            normalized_jacobian[selected] = prior[selected, None] * (
                log_strength_jacobian[selected] - star_mean[None, :]
            )
        return transport_compatibility_from_jacobian(
            charges=charge,
            endpoints=pair,
            priors=prior,
            jacobian=normalized_jacobian,
            parameter_sources=parameter_sources,
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def compute_bond_valence_transport_compatibility_features(
    structure,
    charges: Sequence[float] | np.ndarray,
) -> BondValenceTransportCompatibilityResult:
    """Resolve the frozen periodic graph/parameters and evaluate NEXT38."""

    try:
        charge = np.asarray(charges, dtype=float)
        if charge.shape != (len(structure),) or not np.isfinite(charge).all():
            return _failure("charges must be finite and match all sites")
        magnitude = float(np.abs(charge).sum())
        if abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            return _failure("charges must be neutral")
        geometry = build_periodic_edge_geometry(structure, charge, graph_mode="voronoi")
        if not geometry.supported:
            return _failure(geometry.failure_reason or "periodic graph is unsupported")
        parameters = bv_table()
        endpoints: list[tuple[int, int]] = []
        vectors: list[np.ndarray] = []
        strengths: list[float] = []
        decays: list[float] = []
        sources: list[str] = []
        for edge in geometry.edges:
            left = int(edge.cation)
            right = int(edge.anion)
            key = (
                structure[left].specie.symbol,
                int(round(float(charge[left]))),
                structure[right].specie.symbol,
                int(round(float(charge[right]))),
            )
            resolved = resolve_bond_valence_parameter(
                key, parameters, policy="frozen-fallback"
            )
            if resolved is None:
                left_radius = _tabulated_radius(structure[left].specie.symbol)
                right_radius = _tabulated_radius(structure[right].specie.symbol)
                if left_radius is None or right_radius is None:
                    return _failure(
                        "bond-valence and radius-generic parameters are missing for "
                        f"{structure[left].specie.symbol}-{structure[right].specie.symbol}"
                    )
                resolved = (left_radius + right_radius, 0.37, "radius_generic")
            r0, decay, source = resolved
            if (
                not np.isfinite(r0)
                or not np.isfinite(decay)
                or float(decay) <= 0.0
                or source not in PARAMETER_SOURCES
            ):
                return _failure("bond-valence parameter is invalid")
            fractional = (
                np.asarray(structure[right].frac_coords, dtype=float)
                + np.asarray(edge.image, dtype=float)
                - np.asarray(structure[left].frac_coords, dtype=float)
            )
            displacement = np.asarray(
                structure.lattice.get_cartesian_coords(fractional), dtype=float
            )
            distance = float(np.linalg.norm(displacement))
            try:
                strength = math.exp((float(r0) - distance) / float(decay))
            except OverflowError:
                return _failure("bond strength overflowed")
            if not np.isfinite(strength) or strength <= 0.0:
                return _failure("bond strength is invalid")
            endpoints.append((left, right))
            vectors.append(displacement)
            strengths.append(strength)
            decays.append(float(decay))
            sources.append(str(source))
        return bond_valence_transport_compatibility_features(
            charges=charge,
            endpoints=np.asarray(endpoints, dtype=int),
            vectors=np.asarray(vectors, dtype=float),
            strengths=np.asarray(strengths, dtype=float),
            decays=np.asarray(decays, dtype=float),
            parameter_sources=tuple(sources),
        )
    except Exception as exc:
        return _failure(f"{type(exc).__name__}: {exc}")


def _validate_batch_inputs(
    *,
    archive: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next32_feature_path: Path,
    next32_feature_manifest_path: Path,
    next37_feature_path: Path,
    next37_feature_manifest_path: Path,
) -> tuple[pd.DataFrame, list[object], pd.DataFrame]:
    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("NEXT38 geometry archive path/name is invalid")
    if metadata_path.name != "next32_cohort.parquet" or not metadata_path.is_file():
        raise ValueError("NEXT38 cohort metadata path/name is invalid")
    if next32_feature_path.name != NEXT32_FEATURE_NAME or not next32_feature_path.is_file():
        raise ValueError("NEXT38 NEXT32 feature path/name is invalid")
    if next37_feature_path.name != NEXT37_FEATURE_NAME or not next37_feature_path.is_file():
        raise ValueError("NEXT38 NEXT37 feature path/name is invalid")

    cohort_manifest = _strict_json(cohort_manifest_path, role="NEXT38 cohort manifest")
    cohort_outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("output_role") != "unrelaxed_x0_geometry_only"
        or cohort_manifest.get("endpoint_numeric_fields_parsed") is not False
        or cohort_manifest.get("label_values_exported") is not False
        or cohort_manifest.get("labels_opened") is not False
    ):
        raise ValueError("NEXT38 cohort is not a label-free geometry projection")
    if not isinstance(cohort_outputs, Mapping) or any(
        cohort_outputs.get(path.name) != _sha256(path) for path in (archive, metadata_path)
    ):
        raise ValueError("NEXT38 cohort geometry or metadata hash differs")

    next32_manifest = _strict_json(
        next32_feature_manifest_path, role="NEXT38 NEXT32 feature manifest"
    )
    next32_outputs = next32_manifest.get("outputs_sha256")
    if (
        next32_manifest.get("protocol") != NEXT32_FEATURE_PROTOCOL
        or next32_manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or next32_manifest.get("labels_opened") is not False
        or next32_manifest.get("endpoint_fields_read") is not False
        or next32_manifest.get("dft_values_used") is True
        or next32_manifest.get("electronic_structure_calculation_used") is True
        or next32_manifest.get("model_or_proxy_potential_used") is not False
        or next32_manifest.get("coordinates_or_cell_modified") is not False
        or not isinstance(next32_outputs, Mapping)
        or next32_outputs.get(NEXT32_FEATURE_NAME) != _sha256(next32_feature_path)
    ):
        raise ValueError("NEXT38 NEXT32 features crossed the label-free boundary")

    next37_manifest = _strict_json(
        next37_feature_manifest_path, role="NEXT38 NEXT37 feature manifest"
    )
    next37_outputs = next37_manifest.get("outputs_sha256")
    if (
        next37_manifest.get("protocol") != NEXT37_FEATURE_PROTOCOL
        or next37_manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or next37_manifest.get("labels_opened") is not False
        or next37_manifest.get("endpoint_fields_read") is not False
        or next37_manifest.get("dft_values_used") is not False
        or next37_manifest.get("self_stress_compatibility_projection_used") is not True
        or next37_manifest.get("coordinate_displacement_solved_or_applied") is not False
        or next37_manifest.get("electronic_structure_calculation_used") is not False
        or next37_manifest.get("model_or_proxy_potential_used") is not False
        or next37_manifest.get("coordinates_or_cell_modified") is not False
        or not isinstance(next37_outputs, Mapping)
        or next37_outputs.get(NEXT37_FEATURE_NAME) != _sha256(next37_feature_path)
    ):
        raise ValueError("NEXT38 NEXT37 features crossed the label-free boundary")

    identity = ["material_id", "source_name", "parent_id", "natoms"]
    metadata = pd.read_parquet(metadata_path)
    if not {*identity, "input_role"}.issubset(metadata):
        raise ValueError("NEXT38 cohort metadata lacks required identity columns")
    metadata = metadata.loc[:, identity + ["input_role"]].copy()
    for column in ("material_id", "source_name", "parent_id"):
        metadata[column] = metadata[column].astype(str)
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        metadata.material_id.duplicated().any()
        or metadata.parent_id.duplicated().any()
        or not metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT38 cohort identities or roles differ")
    material_ids = tuple(metadata.material_id)
    loaded_ids, structures = _load_archive_only(archive, material_ids)
    if loaded_ids != list(material_ids) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT38 geometry identity or atom counts differ")

    old32 = pd.read_parquet(next32_feature_path)
    if not {"material_id", "scbv_mismatch_q95"}.issubset(old32):
        raise ValueError("NEXT38 NEXT32 table lacks frozen comparator")
    old32 = old32.loc[:, ["material_id", "scbv_mismatch_q95"]].copy()
    old32["material_id"] = old32.material_id.astype(str)
    old32 = old32.sort_values("material_id", kind="stable", ignore_index=True)
    if old32.material_id.duplicated().any() or not old32.material_id.equals(
        metadata.material_id
    ):
        raise ValueError("NEXT38 NEXT32 identities differ from geometry")

    old37_names = (
        "steric_rep12_vector_rms",
        "steric_rep12_vector_max",
        "sivr_site_imbalance_rms",
    )
    old37 = pd.read_parquet(next37_feature_path)
    if not {*identity, *old37_names}.issubset(old37):
        raise ValueError("NEXT38 NEXT37 table lacks frozen comparators")
    old37 = old37.loc[:, identity + list(old37_names)].copy()
    for column in ("material_id", "source_name", "parent_id"):
        old37[column] = old37[column].astype(str)
    old37 = old37.sort_values("material_id", kind="stable", ignore_index=True)
    if old37.material_id.duplicated().any() or not old37[identity].equals(
        metadata[identity]
    ):
        raise ValueError("NEXT38 NEXT37 identities differ from geometry")
    upstream = old37.copy()
    upstream.insert(4, "scbv_mismatch_q95", old32.scbv_mismatch_q95.to_numpy(float))
    return metadata, structures, upstream


def build_bond_valence_transport_compatibility_feature_batch(
    *,
    archive_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    next32_feature_path: Path,
    next32_feature_manifest_path: Path,
    next37_feature_path: Path,
    next37_feature_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Seal NEXT38 features from exact geometry-only upstream artifacts."""

    paths = {
        "geometry": Path(archive_path).resolve(),
        "metadata": Path(metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "next32_features": Path(next32_feature_path).resolve(),
        "next32_feature_manifest": Path(next32_feature_manifest_path).resolve(),
        "next37_features": Path(next37_feature_path).resolve(),
        "next37_feature_manifest": Path(next37_feature_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    metadata, structures, upstream = _validate_batch_inputs(
        archive=paths["geometry"],
        metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        next32_feature_path=paths["next32_features"],
        next32_feature_manifest_path=paths["next32_feature_manifest"],
        next37_feature_path=paths["next37_features"],
        next37_feature_manifest_path=paths["next37_feature_manifest"],
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
            result = compute_bond_valence_transport_compatibility_features(
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
            "bvtc_supported": bool(result.supported),
            "bvtc_failure": result.failure_reason,
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
        raise ValueError(f"NEXT38 feature output crossed no-DFT contract: {forbidden}")
    if len(features) != len(metadata) or features.material_id.duplicated().any():
        raise ValueError("NEXT38 feature identity accounting differs")

    source_dir = Path(__file__).resolve().parent
    source_names = (
        "advanced_local_features.py",
        "elec_feat.py",
        "next11_geometry_only_frames.py",
        "next19_valence_transport.py",
        "next20_valence_rigidity.py",
        "next22_bond_valence_equilibrium.py",
        "next32_inorganic_response_features.py",
        "next37_self_stress_compatibility_features.py",
        "next38_bond_valence_transport_compatibility_features.py",
    )
    source_paths = {f"src/{name}": source_dir / name for name in source_names}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "dft_values_used": False,
        "bond_valence_transport_compatibility_used": True,
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
            "bvtc_supported": int(features.bvtc_supported.sum()),
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
                raise RuntimeError(f"NEXT38 input changed before publication: {role}")
        for name, path in source_paths.items():
            if _sha256(path) != source_hashes[name]:
                raise RuntimeError(f"NEXT38 source changed before publication: {name}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "BondValenceTransportCompatibilityResult",
    "CANDIDATE_FEATURE_NAMES",
    "DIAGNOSTIC_FEATURE_NAMES",
    "FEATURE_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "REUSED_FEATURE_NAMES",
    "bond_valence_transport_compatibility_features",
    "build_bond_valence_transport_compatibility_feature_batch",
    "compute_bond_valence_transport_compatibility_features",
    "transport_compatibility_from_jacobian",
]
