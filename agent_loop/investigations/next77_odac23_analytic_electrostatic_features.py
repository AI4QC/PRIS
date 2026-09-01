#!/usr/bin/env python3
"""Label-free analytic electrostatic feature families for ODAC23 framework x0."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.io.ase import AseAtomsAdaptor

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next19_valence_transport import infer_valence_assignment
from src.next34_analytic_field_features import (
    CANDIDATE_FEATURE_NAMES as AEFI_CANDIDATE_FEATURE_NAMES,
    COULOMB_EV_ANGSTROM,
    DIAGNOSTIC_FEATURE_NAMES as AEFI_DIAGNOSTIC_FEATURE_NAMES,
)
from src.next35_coulomb_steric_balance_features import (
    CANDIDATE_FEATURE_NAMES as ACSB_CANDIDATE_FEATURE_NAMES,
    DIAGNOSTIC_FEATURE_NAMES as ACSB_DIAGNOSTIC_FEATURE_NAMES,
    _analytic_steric_vectors,
    analytic_vector_balance_features,
)
from src.next36_charge_spectrum_features import (
    CANDIDATE_FEATURE_NAMES as CSF_CANDIDATE_FEATURE_NAMES,
    compute_charge_spectrum_features,
)
from src.next49_framework_topology import _environment_versions
from src.next54_odac23_train_selection import (
    GEOMETRY_NAME as SOURCE_GEOMETRY_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    METADATA_NAME as SOURCE_METADATA_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)
from src.next55_odac23_analytic_features import _load_archive
from src.next75_odac23_metal_ligand_rigidity_features import (
    FEATURES_NAME as BASE_FEATURES_NAME,
    PROTOCOL as BASE_FEATURE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next77-odac23-analytic-electrostatic-features-v1"
DESIGN_SHA256 = "7ea418fbb3ac0b0c59db62df22148be621bd555cde233e0a8e09826a9d78b824"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
EXPECTED_BASE_MANIFEST_SHA256 = (
    "7ee6197a08da3fafc1aa46d6ed5f5cb82aeb55480466974892af111dc685a758"
)
EXPECTED_BASE_FEATURE_SHA256 = (
    "7d7079571d2f6be5c2e21835108ba6fb43e95cb0fae1edaacfbabe7dc71b813d"
)
NM_INVARIANT_FEATURE_NAMES = (
    "nm_site_positive_fraction",
    "nm_charge_concentration",
)
ANALYTIC_ELECTROSTATIC_FEATURE_NAMES = (
    NM_INVARIANT_FEATURE_NAMES
    + tuple(AEFI_CANDIDATE_FEATURE_NAMES)
    + tuple(AEFI_DIAGNOSTIC_FEATURE_NAMES)
    + tuple(ACSB_CANDIDATE_FEATURE_NAMES)
    + tuple(ACSB_DIAGNOSTIC_FEATURE_NAMES)
    + tuple(CSF_CANDIDATE_FEATURE_NAMES)
)
FEATURES_NAME = "next77_odac23_analytic_electrostatic_features.parquet"
MANIFEST_NAME = "MANIFEST.json"


if len(ANALYTIC_ELECTROSTATIC_FEATURE_NAMES) != len(
    set(ANALYTIC_ELECTROSTATIC_FEATURE_NAMES)
):
    raise RuntimeError("NEXT77 analytic electrostatic feature names overlap")


@dataclass(frozen=True)
class ODACAnalyticElectrostaticResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _shared_ewald_features(
    structure,
    charges: np.ndarray,
) -> dict[str, float]:
    """Derive all retained Ewald families from one analytic lattice sum."""

    charge = np.asarray(charges, dtype=float)
    n_sites = len(structure)
    if charge.shape != (n_sites,):
        raise ValueError("charges must match the structure sites")
    if n_sites < 2 or not np.isfinite(charge).all():
        raise ValueError("charges must be finite and describe at least two sites")
    total_magnitude = float(np.abs(charge).sum())
    if abs(float(charge.sum())) > 1.0e-8 * max(1.0, total_magnitude):
        raise ValueError("charges must be neutral")
    if not np.any(charge > 0.0) or not np.any(charge < 0.0):
        raise ValueError("charges must have nonzero values of both signs")
    volume = float(structure.volume)
    if not np.isfinite(volume) or volume <= 1.0e-10:
        raise ValueError("periodic structure volume must be finite and positive")
    q2 = float(np.sum(charge**2))
    q_rms = float(np.sqrt(q2 / n_sites))
    if not np.isfinite(q_rms) or q_rms <= 0.0:
        raise ValueError("charge magnitude must be nonzero")

    decorated = structure.copy()
    try:
        decorated.remove_oxidation_states()
    except Exception:
        pass
    decorated.add_oxidation_state_by_site(charge.tolist())
    ewald = EwaldSummation(decorated, compute_forces=True)
    derivative = np.asarray(ewald.forces, dtype=float)
    site_terms = np.asarray(
        [float(ewald.get_site_energy(index)) for index in range(n_sites)],
        dtype=float,
    )
    if (
        derivative.shape != (n_sites, 3)
        or not np.isfinite(derivative).all()
        or not np.isfinite(site_terms).all()
    ):
        raise ValueError("analytic Ewald sum returned invalid values")

    tolerance = max(1.0e-12, 1.0e-12 * float(np.max(np.abs(charge))))
    active = np.abs(charge) > tolerance
    if not active.any():
        raise ValueError("charge assignment has no active sites")
    length = float((volume / n_sites) ** (1.0 / 3.0))
    field_vectors = (
        derivative[active]
        * length**2
        / (COULOMB_EV_ANGSTROM * np.abs(charge[active])[:, None] * q_rms)
    )
    residual_vectors = (
        derivative[active]
        * length**2
        / (COULOMB_EV_ANGSTROM * q_rms**2)
    )
    field = np.linalg.norm(field_vectors, axis=1)
    residual = np.linalg.norm(residual_vectors, axis=1)
    if not np.isfinite(field).all() or not np.isfinite(residual).all():
        raise ValueError("dimensionless analytic field is non-finite")
    tensor_denominator = float(np.sum(field_vectors**2))
    if tensor_denominator <= 1.0e-28:
        tensor_deviator = 0.0
    else:
        tensor = field_vectors.T @ field_vectors / tensor_denominator
        deviator = tensor - np.eye(3) * float(np.trace(tensor)) / 3.0
        tensor_deviator = float(np.linalg.norm(deviator))
    aefi = {
        "aefi_field_rms": float(np.sqrt(np.mean(field**2))),
        "aefi_field_q95": float(np.quantile(field, 0.95, method="inverted_cdf")),
        "aefi_field_max": float(np.max(field)),
        "aefi_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "aefi_residual_q95": float(
            np.quantile(residual, 0.95, method="inverted_cdf")
        ),
        "aefi_residual_max": float(np.max(residual)),
        "aefi_field_tensor_deviator": tensor_deviator,
        "aefi_active_site_fraction": float(np.mean(active)),
    }

    coulomb_vectors = derivative * length**2 / (
        COULOMB_EV_ANGSTROM * q_rms**2
    )
    balance = analytic_vector_balance_features(
        coulomb_vectors,
        _analytic_steric_vectors(structure),
    )
    if not balance.supported:
        raise ValueError(balance.failure_reason or "Coulomb--steric balance unsupported")
    values = {
        "nm_site_positive_fraction": float(np.mean(site_terms > 0.0)),
        "nm_charge_concentration": float(
            n_sites * np.sum(charge**4) / q2**2
        ),
        **aefi,
        **{
            name: float(balance.features[name])
            for name in tuple(ACSB_CANDIDATE_FEATURE_NAMES)
            + tuple(ACSB_DIAGNOSTIC_FEATURE_NAMES)
        },
    }
    expected = (
        NM_INVARIANT_FEATURE_NAMES
        + tuple(AEFI_CANDIDATE_FEATURE_NAMES)
        + tuple(AEFI_DIAGNOSTIC_FEATURE_NAMES)
        + tuple(ACSB_CANDIDATE_FEATURE_NAMES)
        + tuple(ACSB_DIAGNOSTIC_FEATURE_NAMES)
    )
    if tuple(values) != expected or not np.isfinite(list(values.values())).all():
        raise ValueError("shared Ewald feature schema differs")
    return values


def compute_odac23_analytic_electrostatic_features(
    atoms: Atoms,
) -> ODACAnalyticElectrostaticResult:
    """Evaluate four frozen classical analytic families on one unchanged x0."""

    try:
        if len(atoms) < 2 or not np.all(atoms.pbc):
            raise ValueError("periodic structure must contain at least two atoms")
        structure = AseAtomsAdaptor.get_structure(atoms)
        assignment = infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "neutral charge assignment unsupported")
        charges = assignment.values
        shared = _shared_ewald_features(structure, charges)
        spectrum = compute_charge_spectrum_features(structure, charges)
        if not spectrum.supported:
            raise ValueError(
                f"charge_spectrum: {spectrum.failure_reason or 'unsupported'}"
            )
        values = {
            **shared,
            **{
                name: float(spectrum.features[name]) for name in CSF_CANDIDATE_FEATURE_NAMES
            },
        }
        if tuple(values) != ANALYTIC_ELECTROSTATIC_FEATURE_NAMES or not np.isfinite(
            list(values.values())
        ).all():
            raise ValueError("analytic electrostatic feature schema differs")
        return ODACAnalyticElectrostaticResult(True, None, values)
    except Exception as exc:
        return ODACAnalyticElectrostaticResult(False, f"{type(exc).__name__}: {exc}", {})


def _feature_record(atoms: Atoms) -> dict[str, object]:
    result = compute_odac23_analytic_electrostatic_features(atoms)
    row: dict[str, object] = {
        "analytic_electrostatic_supported": result.supported,
        "analytic_electrostatic_failure": result.failure_reason,
    }
    row.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in ANALYTIC_ELECTROSTATIC_FEATURE_NAMES
        }
    )
    return row


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT77 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_analytic_electrostatic_batch(
    *,
    source_dir: Path,
    base_feature_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, object]:
    """Build all partitions without labels or opened-validation artifacts."""

    source_dir = Path(source_dir).resolve()
    base_feature_dir = Path(base_feature_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT77 workers must be 1 through 64")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": source_dir / SOURCE_METADATA_NAME,
        "geometry": source_dir / SOURCE_GEOMETRY_NAME,
        "source_manifest": source_dir / SOURCE_MANIFEST_NAME,
        "base_features": base_feature_dir / BASE_FEATURES_NAME,
        "base_manifest": base_feature_dir / MANIFEST_NAME,
        "design": design_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT77 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["source_manifest"] != EXPECTED_SOURCE_MANIFEST_SHA256
        or hashes["base_manifest"] != EXPECTED_BASE_MANIFEST_SHA256
        or hashes["base_features"] != EXPECTED_BASE_FEATURE_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT77 frozen input hash differs")
    source_manifest = _read_json(paths["source_manifest"])
    base_manifest = _read_json(paths["base_manifest"])
    source_outputs = source_manifest.get("outputs_sha256")
    base_outputs = base_manifest.get("outputs_sha256")
    if (
        source_manifest.get("protocol") != SOURCE_PROTOCOL
        or source_manifest.get("selection_frozen_before_row_labels_opened") is not True
        or source_manifest.get("validation_or_test_payload_deserialized") is not False
        or not isinstance(source_outputs, Mapping)
        or source_outputs.get(paths["metadata"].name) != hashes["metadata"]
        or source_outputs.get(paths["geometry"].name) != hashes["geometry"]
        or base_manifest.get("protocol") != BASE_FEATURE_PROTOCOL
        or base_manifest.get("labels_opened") is not False
        or base_manifest.get("opened_internal_validation_result_used") is not False
        or base_manifest.get("internal_replication_labels_opened") is not False
        or not isinstance(base_outputs, Mapping)
        or base_outputs.get(BASE_FEATURES_NAME) != hashes["base_features"]
    ):
        raise ValueError("NEXT77 label-free provenance differs")
    metadata = pd.read_parquet(paths["metadata"])
    base = pd.read_parquet(paths["base_features"])
    material_ids = tuple(metadata["material_id"].astype(str))
    if (
        len(metadata) != len(base)
        or metadata["material_id"].duplicated().any()
        or base["material_id"].duplicated().any()
        or tuple(base["material_id"].astype(str)) != material_ids
    ):
        raise ValueError("NEXT77 base feature identity differs")
    structures = _load_archive(paths["geometry"], material_ids)
    rows = []
    failures: Counter[str] = Counter()
    if workers == 1:
        iterator = map(_feature_record, structures)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_feature_record, structures, chunksize=1)
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if not bool(row["analytic_electrostatic_supported"]):
                failures[str(row["analytic_electrostatic_failure"])] += 1
            if index % 50 == 0 or index == len(structures):
                print(f"NEXT77 ODAC23 analytic electrostatics: {index}/{len(structures)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    additions = pd.DataFrame(rows)
    if tuple(additions.loc[:, ANALYTIC_ELECTROSTATIC_FEATURE_NAMES].columns) != (
        ANALYTIC_ELECTROSTATIC_FEATURE_NAMES
    ):
        raise ValueError("NEXT77 output feature schema differs")
    table = pd.concat([base.reset_index(drop=True), additions], axis=1)
    if not table["combined_supported"].equals(base["combined_supported"]):
        raise RuntimeError("NEXT77 changed pre-existing support")
    source_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "selected_odac23_all_partitions_label_free_analytic_electrostatics",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "opened_internal_validation_result_used": False,
        "internal_replication_labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "classical_analytic_electrostatics_used": True,
        "electronic_structure_calculation_used": False,
        "dft_calculation_or_value_used": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "missing_policy": "optional_family_fail_open_keep",
        "feature_columns": list(ANALYTIC_ELECTROSTATIC_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "base_supported": int(table["combined_supported"].sum()),
            "analytic_electrostatic_supported": int(
                table["analytic_electrostatic_supported"].sum()
            ),
            "failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next77_odac23_analytic_electrostatic_features.py": _sha256(source_path)
        },
        "environment_versions": {
            **_environment_versions(),
            "ase": importlib.metadata.version("ase"),
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURES_NAME
        table.to_parquet(feature_path, index=False)
        manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(feature_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != manifest["executed_source_sha256"][
            "src/next77_odac23_analytic_electrostatic_features.py"
        ]:
            raise RuntimeError("NEXT77 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT77 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-feature-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_analytic_electrostatic_batch(
        source_dir=args.source_dir,
        base_feature_dir=args.base_feature_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "ANALYTIC_ELECTROSTATIC_FEATURE_NAMES",
    "PROTOCOL",
    "build_analytic_electrostatic_batch",
    "compute_odac23_analytic_electrostatic_features",
]


if __name__ == "__main__":
    main()
