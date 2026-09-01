"""DFT-free absolute periodic-contact features for NEXT32.

Only the raw, unrelaxed cell, species, and coordinates enter this module.  The
features deliberately use a frozen elemental-radius table and analytic
periodic geometry; no calculated property or alternative structure is read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.neighborlist import neighbor_list
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _load_archive_only
from src.next12_pauling_controls import (
    DECISIONS,
    RULES,
    _classical_features,
    _combined_decision,
    _rule_decision,
)
from src.next19_feature_build import (
    _publish_directory_no_replace,
    _sha256,
    _strict_json,
)
from src.next19_valence_transport import (
    build_periodic_edge_geometry,
    infer_valence_assignment,
)
from src.next20_valence_rigidity import rigidity_features_from_periodic_geometry
from src.next21_normalized_madelung import normalized_madelung_features
from src.next22_bond_valence_equilibrium import (
    bond_valence_features_from_periodic_geometry,
)
from src.next32_omat24_cohort import PROTOCOL as COHORT_PROTOCOL


CONTACT_FEATURE_NAMES = (
    "cov_q01",
    "cov_q05",
    "cov_contact085_pa",
    "cov_overlap2_pa",
    "cov_site_overlap_q95",
    "cov_site_overlap_max",
)

INORGANIC_FEATURE_NAMES = CONTACT_FEATURE_NAMES + (
    "sivr_edge_mismatch_q95",
    "sivr_site_imbalance_rms",
    "sivr_cell_anisotropy",
    "nm_total_reduced",
    "nm_site_spread",
    "scbv_mismatch_q95",
    "scbv_vector_asymmetry_rms",
    "scbv_global_scale",
)
PROTOCOL = "2026-08-03-next32-omat24-inorganic-response-features-v1"
FEATURE_NAME = "next32_inorganic_response_features.parquet"
PAULING_NAME = "next32_pauling_controls.parquet"
MANIFEST_NAME = "MANIFEST.json"
_FORBIDDEN_COLUMN_TOKENS = (
    "energy",
    "force",
    "stress",
    "relax",
    "mattersim",
    "dft",
    "endpoint",
    "label",
    "target",
)


@dataclass(frozen=True)
class ContactFeatureResult:
    """One fail-open result from the absolute periodic-contact calculation."""

    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


@dataclass(frozen=True)
class InorganicFeatureResult:
    """Selected analytic feature families with explicit fail-open accounting."""

    features: Mapping[str, float]
    family_supported: Mapping[str, bool]
    family_failures: Mapping[str, str | None]


def _resolve_radii(
    numbers: np.ndarray, supplied: Mapping[int, float] | None
) -> np.ndarray:
    """Resolve one positive frozen covalent radius for every atomic number."""

    values: list[float] = []
    for raw_number in numbers:
        number = int(raw_number)
        if supplied is not None:
            if number not in supplied:
                raise ValueError(f"missing covalent radius for atomic number {number}")
            radius = float(supplied[number])
        else:
            radius = float(covalent_radii[number])
            if not np.isfinite(radius) or radius <= 0:
                from src.next20_valence_rigidity import _tabulated_radius

                fallback = _tabulated_radius(chemical_symbols[number])
                if fallback is None:
                    raise ValueError(
                        f"missing covalent radius for atomic number {number}"
                    )
                radius = float(fallback)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"invalid covalent radius for atomic number {number}")
        values.append(radius)
    return np.asarray(values, dtype=float)


def _canonical_periodic_ratios(atoms: Atoms, radii: np.ndarray) -> list[tuple[int, int, float]]:
    """Return each periodic undirected pair once, up to a radius ratio of 1.6."""

    cutoff = 3.2 * float(np.max(radii))
    first, second, shifts, distances = neighbor_list(
        "ijSd", atoms, cutoff, self_interaction=True
    )
    rows: list[tuple[int, int, tuple[int, int, int], float]] = []
    for raw_i, raw_j, raw_shift, raw_distance in zip(
        first, second, shifts, distances, strict=True
    ):
        i = int(raw_i)
        j = int(raw_j)
        shift = tuple(int(value) for value in raw_shift)
        if i == j and shift == (0, 0, 0):
            continue
        reverse = tuple(-value for value in shift)
        if (i, j, *shift) >= (j, i, *reverse):
            continue
        ratio = float(raw_distance / (radii[i] + radii[j]))
        if np.isfinite(ratio) and ratio <= 1.6:
            rows.append((i, j, shift, ratio))
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return [(i, j, ratio) for i, j, _shift, ratio in rows]


def compute_periodic_contact_features(
    atoms: Atoms, *, radii: Mapping[int, float] | None = None
) -> ContactFeatureResult:
    """Compute size-extensive covalent-contact features from one raw structure."""

    try:
        if len(atoms) == 0:
            raise ValueError("periodic geometry has no atoms")
        if not bool(np.all(atoms.pbc)):
            raise ValueError("periodic geometry requires three periodic axes")
        cell = np.asarray(atoms.cell.array, dtype=float)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError("periodic cell is invalid")
        if abs(float(np.linalg.det(cell))) <= 1e-10:
            raise ValueError("periodic cell has zero volume")
        positions = np.asarray(atoms.positions, dtype=float)
        if not np.all(np.isfinite(positions)):
            raise ValueError("periodic coordinates are invalid")

        numbers = np.asarray(atoms.numbers, dtype=int)
        resolved_radii = _resolve_radii(numbers, radii)
        pairs = _canonical_periodic_ratios(atoms, resolved_radii)
        if not pairs:
            raise ValueError("periodic geometry has no covalent-radius contacts")

        ratios = np.asarray([row[2] for row in pairs], dtype=float)
        overlap2 = np.maximum(0.0, 1.0 - ratios) ** 2
        site_load = np.zeros(len(atoms), dtype=float)
        for (i, j, _ratio), value in zip(pairs, overlap2, strict=True):
            site_load[i] += value
            site_load[j] += value

        values = {
            "cov_q01": float(np.quantile(ratios, 0.01)),
            "cov_q05": float(np.quantile(ratios, 0.05)),
            "cov_contact085_pa": float(2.0 * np.sum(ratios < 0.85) / len(atoms)),
            "cov_overlap2_pa": float(2.0 * np.sum(overlap2) / len(atoms)),
            "cov_site_overlap_q95": float(np.quantile(site_load, 0.95)),
            "cov_site_overlap_max": float(np.max(site_load)),
        }
        if tuple(values) != CONTACT_FEATURE_NAMES or not np.all(
            np.isfinite(list(values.values()))
        ):
            raise ValueError("periodic contact feature schema or values are invalid")
        return ContactFeatureResult(True, None, values)
    except Exception as exc:
        return ContactFeatureResult(False, f"{type(exc).__name__}: {exc}", {})


def compute_inorganic_response_features(atoms: Atoms) -> InorganicFeatureResult:
    """Reuse the frozen analytic kernels on a single unmodified x0 structure."""

    values = {name: math.nan for name in INORGANIC_FEATURE_NAMES}
    supported = {name: False for name in ("contact", "sivr", "madelung", "scbve")}
    failures: dict[str, str | None] = {name: None for name in supported}

    contact = compute_periodic_contact_features(atoms)
    supported["contact"] = contact.supported
    failures["contact"] = contact.failure_reason
    if contact.supported:
        values.update(contact.features)

    try:
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
    except Exception as exc:
        reason = f"valence assignment failed: {type(exc).__name__}: {exc}"
        for family in ("sivr", "madelung", "scbve"):
            failures[family] = reason
        return InorganicFeatureResult(values, supported, failures)
    if not assignment.supported or assignment.values is None:
        reason = assignment.failure_reason or "valence assignment is unsupported"
        for family in ("sivr", "madelung", "scbve"):
            failures[family] = reason
        return InorganicFeatureResult(values, supported, failures)

    try:
        geometry = build_periodic_edge_geometry(
            structure, assignment.values, graph_mode="voronoi"
        )
    except Exception as exc:
        geometry = None
        geometry_failure = f"periodic edge geometry failed: {type(exc).__name__}: {exc}"
    else:
        geometry_failure = getattr(geometry, "failure_reason", None)

    if geometry is not None and geometry.supported:
        try:
            rigidity = rigidity_features_from_periodic_geometry(
                structure,
                assignment.values,
                geometry,
                charge_weight_exponent=0.0,
            )
        except Exception as exc:
            failures["sivr"] = f"SIVR failed: {type(exc).__name__}: {exc}"
        else:
            supported["sivr"] = rigidity.supported
            failures["sivr"] = rigidity.failure_reason
            if rigidity.supported:
                for name in (
                    "sivr_edge_mismatch_q95",
                    "sivr_site_imbalance_rms",
                    "sivr_cell_anisotropy",
                ):
                    values[name] = float(rigidity.features[name])
        try:
            bond_valence = bond_valence_features_from_periodic_geometry(
                structure, assignment.values, geometry
            )
        except Exception as exc:
            failures["scbve"] = f"SCBVE failed: {type(exc).__name__}: {exc}"
        else:
            supported["scbve"] = bond_valence.supported
            failures["scbve"] = bond_valence.failure_reason
            if bond_valence.supported:
                for name in (
                    "scbv_mismatch_q95",
                    "scbv_vector_asymmetry_rms",
                    "scbv_global_scale",
                ):
                    values[name] = float(bond_valence.features[name])
    else:
        reason = geometry_failure or "periodic edge geometry is unsupported"
        failures["sivr"] = reason
        failures["scbve"] = reason

    try:
        madelung = normalized_madelung_features(structure, assignment.values)
    except Exception as exc:
        failures["madelung"] = f"normalized Madelung failed: {type(exc).__name__}: {exc}"
    else:
        supported["madelung"] = madelung.supported
        failures["madelung"] = madelung.failure_reason
        if madelung.supported:
            for name in ("nm_total_reduced", "nm_site_spread"):
                values[name] = float(madelung.features[name])

    if tuple(values) != INORGANIC_FEATURE_NAMES:
        raise RuntimeError("NEXT32 inorganic feature schema changed")
    return InorganicFeatureResult(values, supported, failures)


def _validate_geometry_inputs(
    *, archive: Path, metadata_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[Atoms], dict[str, object]]:
    """Load only a label-free, hash-locked NEXT32 geometry projection."""

    if archive.name != "geometry_only_frames.zip" or not archive.is_file():
        raise ValueError("geometry-only archive path/name is invalid")
    if metadata_path.name != "next32_cohort.parquet" or not metadata_path.is_file():
        raise ValueError("NEXT32 cohort metadata path/name is invalid")
    manifest = _strict_json(manifest_path, role="NEXT32 cohort manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != COHORT_PROTOCOL
        or manifest.get("output_role") != "unrelaxed_x0_geometry_only"
        or manifest.get("endpoint_numeric_fields_parsed") is not False
        or manifest.get("label_values_exported") is not False
        or manifest.get("labels_opened") is not False
    ):
        raise ValueError("NEXT32 source is not a label-free geometry projection")
    if not isinstance(outputs, MappingABC) or any(
        outputs.get(path.name) != _sha256(path) for path in (archive, metadata_path)
    ):
        raise ValueError("NEXT32 geometry or metadata hash differs from manifest")

    metadata = pd.read_parquet(metadata_path)
    required = {
        "material_id",
        "source_name",
        "sid",
        "parent_id",
        "record_key",
        "natoms",
        "input_role",
    }
    if not required.issubset(metadata.columns):
        raise ValueError(f"NEXT32 cohort metadata lacks columns: {sorted(required - set(metadata))}")
    metadata = metadata.loc[:, sorted(required)].copy()
    metadata["material_id"] = metadata["material_id"].astype(str)
    metadata = metadata.sort_values("material_id", kind="stable", ignore_index=True)
    if (
        metadata.material_id.duplicated().any()
        or metadata.parent_id.astype(str).duplicated().any()
        or not metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("NEXT32 cohort identity, parent uniqueness, or input role differs")
    identities = tuple(metadata.material_id)
    loaded, structures = _load_archive_only(archive, identities)
    if loaded != list(identities) or any(
        len(atoms) != int(natoms)
        for atoms, natoms in zip(structures, metadata.natoms, strict=True)
    ):
        raise ValueError("NEXT32 geometry identity or atom count differs")
    return metadata, structures, manifest


def _pauling_row(material_id: str, atoms: Atoms) -> dict[str, object]:
    """Apply unchanged repository Pauling 2--5 controls with fail-open abstention."""

    try:
        raw_features, error = _classical_features(atoms)
    except Exception as exc:
        raw_features, error = None, f"calculator failed: {type(exc).__name__}: {exc}"
    pauling_features = dict(raw_features) if isinstance(raw_features, MappingABC) else {}
    row: dict[str, object] = {
        "material_id": material_id,
        "pauling_feature_error": error,
    }
    decisions: list[str] = []
    for name, rule in RULES.items():
        value = pauling_features.get(str(rule["feature"]), np.nan)
        decision = _rule_decision(
            value,
            operator=str(rule["operator"]),
            threshold=float(rule["threshold"]),
        )
        row[f"{name}_value"] = value
        row[f"{name}_decision"] = decision
        decisions.append(decision)
    row["pauling_p2_p5_decision"] = _combined_decision(decisions)
    return row


def build_inorganic_response_feature_batch(
    *,
    archive_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Seal analytic NEXT32 features and Pauling controls before label opening."""

    paths = {
        "geometry": Path(archive_path).resolve(),
        "metadata": Path(metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    input_hashes = {name: _sha256(path) for name, path in paths.items()}
    metadata, structures, _source_manifest = _validate_geometry_inputs(
        archive=paths["geometry"],
        metadata_path=paths["metadata"],
        manifest_path=paths["cohort_manifest"],
    )

    failures: dict[str, Counter[str]] = {
        family: Counter() for family in ("contact", "sivr", "madelung", "scbve")
    }
    feature_rows: list[dict[str, object]] = []
    pauling_rows: list[dict[str, object]] = []
    for upstream, atoms in zip(metadata.to_dict("records"), structures, strict=True):
        result = compute_inorganic_response_features(atoms)
        row: dict[str, object] = {
            "material_id": str(upstream["material_id"]),
            "source_name": str(upstream["source_name"]),
            "parent_id": str(upstream["parent_id"]),
            "natoms": int(upstream["natoms"]),
        }
        for family in failures:
            row[f"{family}_supported"] = bool(result.family_supported[family])
            row[f"{family}_failure"] = result.family_failures[family]
            if not result.family_supported[family]:
                failures[family][result.family_failures[family] or "unknown"] += 1
        row.update(result.features)
        feature_rows.append(row)
        pauling = _pauling_row(str(upstream["material_id"]), atoms)
        pauling.update(
            {
                "source_name": str(upstream["source_name"]),
                "parent_id": str(upstream["parent_id"]),
                "natoms": int(upstream["natoms"]),
            }
        )
        pauling_rows.append(pauling)

    features = pd.DataFrame(feature_rows)
    pauling = pd.DataFrame(pauling_rows)
    for table, role in ((features, "features"), (pauling, "Pauling controls")):
        forbidden = [
            column
            for column in table
            if any(token in column.lower() for token in _FORBIDDEN_COLUMN_TOKENS)
        ]
        if forbidden:
            raise ValueError(f"NEXT32 {role} crossed no-DFT contract: {forbidden}")
        if len(table) != len(metadata) or table.material_id.duplicated().any():
            raise ValueError(f"NEXT32 {role} identity accounting differs")

    source_dir = Path(__file__).resolve().parent
    source_names = (
        "next11_geometry_only_frames.py",
        "next12_pauling_controls.py",
        "next19_valence_transport.py",
        "next20_valence_rigidity.py",
        "next21_normalized_madelung.py",
        "next22_bond_valence_equilibrium.py",
        "next32_omat24_cohort.py",
        Path(__file__).name,
    )
    source_paths = {f"src/{name}": source_dir / name for name in source_names}
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    counts: dict[str, object] = {
        "rows": len(features),
        "atoms": int(features.natoms.sum()),
        "families_supported": {
            family: int(features[f"{family}_supported"].sum()) for family in failures
        },
        "pauling": {
            decision: int(pauling.pauling_p2_p5_decision.eq(decision).sum())
            for decision in DECISIONS
        },
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "endpoint_fields_read": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "feature_names": list(INORGANIC_FEATURE_NAMES),
        "pauling_rules": RULES,
        "counts": counts,
        "failure_counts": {
            family: dict(sorted(counter.items())) for family, counter in failures.items()
        },
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURE_NAME
        pauling_path = staging / PAULING_NAME
        features.to_parquet(feature_path, index=False)
        pauling.to_parquet(pauling_path, index=False)
        manifest["outputs_sha256"] = {
            FEATURE_NAME: _sha256(feature_path),
            PAULING_NAME: _sha256(pauling_path),
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for role, path in paths.items():
            if _sha256(path) != input_hashes[role]:
                raise RuntimeError(f"NEXT32 input changed before publication: {role}")
        for name, path in source_paths.items():
            if _sha256(path) != source_hashes[name]:
                raise RuntimeError(f"NEXT32 source changed before publication: {name}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


__all__ = [
    "CONTACT_FEATURE_NAMES",
    "FEATURE_NAME",
    "INORGANIC_FEATURE_NAMES",
    "MANIFEST_NAME",
    "PAULING_NAME",
    "build_inorganic_response_feature_batch",
    "compute_inorganic_response_features",
    "compute_periodic_contact_features",
]
