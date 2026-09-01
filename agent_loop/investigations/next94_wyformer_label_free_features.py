"""Freeze all WyFormer raw-x0 analytic features before endpoint opening."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.next85_scigen_label_free_features import (
    ALL_ANALYTIC_FEATURES,
    ALWAYS_PRESENT_STATUS_COLUMNS,
    DECISIONS,
    IDENTITY_ONLY_COLUMNS,
    NEXT43_FEATURE_NAMES,
    NEXT44_FEATURE_NAMES,
    PRLR_FEATURE_NAMES,
    RULES,
    compute_scigen_feature_row,
)
from src.next93b_wyformer_blind_lockbox import (
    GEOMETRY_NAMES,
    INPUT_ROLE,
    MANIFEST_NAME as COHORT_MANIFEST_NAME,
    METADATA_NAME as COHORT_METADATA_NAME,
    PARTITIONS,
    PROTOCOL as COHORT_PROTOCOL,
)
from src.next93_wyformer_source_lockbox import _sha256_file, _write_json


PROTOCOL = "2026-08-03-next94-wyformer-all-partition-label-free-feature-freeze-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "WYFORMER_X0_FEATURE_CATALOGUE.json"
FEATURE_NAMES = {role: f"wyformer_x0_features_{role}.parquet" for role in PARTITIONS}
EXPECTED_UPSTREAM_FEATURE_SOURCE_SHA256 = (
    "2caf0fa0aafe6df6732c3b8ed02cd19d96076314273331f32a449b6bd3b41335"
)
EXPECTED_INPUT_SHA256 = {
    "cohort_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "geometry_discovery": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
    "geometry_internal_validation": "fa2c017b8ece8600d0810f9851013a7015688e6b8c87545d7accc07901682fb8",
    "geometry_internal_replication": "485f88ce5798acf37b27688b04109bc1c47da7637091196636cccca65983455d",
    "design": "db9e05470132d57002b62b408b4c0ed3ee39201a61fe6586610b70f1123cbc77",
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _error_row(message: str) -> dict[str, object]:
    row: dict[str, object] = {name: math.nan for name in ALL_ANALYTIC_FEATURES}
    row.update(
        {
            "next43_error": message,
            "next44_error": message,
            "next80_error": message,
            "next80_supported": False,
            "next80_failure": message,
            "pauling_feature_error": message,
            "pauling_p2_p5_decision": "KEEP",
        }
    )
    for name in RULES:
        row[f"pauling_{name}_value"] = math.nan
        row[f"pauling_{name}_decision"] = "KEEP"
    return row


def _compute_structure_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        atoms = AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_scigen_feature_row(atoms)
    except Exception as exc:
        message = f"structure_parse: {type(exc).__name__}: {exc}"
        return material_id, _error_row(message)


def _compute_many(
    payloads: Sequence[tuple[str, str]], *, workers: int
) -> list[tuple[str, dict[str, object]]]:
    if workers == 1:
        return [_compute_structure_payload(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_compute_structure_payload, payloads, chunksize=8))


def _payloads(path: Path, expected_ids: Sequence[str]) -> list[tuple[str, str]]:
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"material_id", "structure_json"}:
        raise ValueError(f"{path.name} geometry columns differ")
    if frame["material_id"].duplicated().any():
        raise ValueError(f"{path.name} material ids are duplicated")
    mapping = dict(zip(frame["material_id"].astype(str), frame["structure_json"].astype(str)))
    if set(mapping) != set(expected_ids):
        raise ValueError(f"{path.name} material ids differ from metadata")
    return [(material_id, mapping[material_id]) for material_id in expected_ids]


def _publish_directory(staging: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    os.replace(staging, target)


def build_wyformer_label_free_features(
    *,
    cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Freeze x0-only features; deliberately has no endpoint argument."""

    cohort = Path(cohort_dir).resolve()
    design = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive exact integer")
    paths = {
        "cohort_manifest": cohort / COHORT_MANIFEST_NAME,
        "metadata": cohort / COHORT_METADATA_NAME,
        **{f"geometry_{role}": cohort / GEOMETRY_NAMES[role] for role in PARTITIONS},
        "design": design,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT94 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT94 formal input identity differs")

    cohort_manifest = _read_json(paths["cohort_manifest"])
    outputs = cohort_manifest.get("outputs_sha256")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("labels_opened_by_feature_builder") is not False
        or cohort_manifest.get("discovery_endpoint_opened") is not False
        or cohort_manifest.get("validation_endpoint_opened") is not False
        or cohort_manifest.get("replication_endpoint_opened") is not False
        or cohort_manifest.get("relaxed_structures_published") is not False
        or cohort_manifest.get("learned_proxy_execution_input") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(COHORT_METADATA_NAME) != input_hashes["metadata"]
        or any(
            outputs.get(GEOMETRY_NAMES[role]) != input_hashes[f"geometry_{role}"]
            for role in PARTITIONS
        )
    ):
        raise ValueError("NEXT93b blind cohort provenance differs")

    metadata = pd.read_parquet(paths["metadata"])
    required = {
        "material_id",
        "reduced_formula",
        "chemical_system",
        "natoms",
        "generated_space_group",
        "crystal_system",
        "partition_role",
        "input_role",
    }
    if required - set(metadata.columns):
        raise ValueError("NEXT93b metadata columns differ")
    if (
        metadata["material_id"].duplicated().any()
        or set(metadata["partition_role"]) - set(PARTITIONS)
        or not metadata["input_role"].eq(INPUT_ROLE).all()
    ):
        raise ValueError("NEXT93b metadata identity differs")

    repository_root = Path(__file__).resolve().parents[1]
    upstream_path = repository_root / "src/next85_scigen_label_free_features.py"
    upstream_hash = _sha256_file(upstream_path)
    if require_formal_inputs and upstream_hash != EXPECTED_UPSTREAM_FEATURE_SOURCE_SHA256:
        raise ValueError("NEXT94 upstream feature implementation differs")
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    started = time.perf_counter()
    output_paths: list[Path] = []
    counts: dict[str, object] = {}
    try:
        for role in PARTITIONS:
            part_meta = metadata[metadata["partition_role"].eq(role)].copy()
            part_meta = part_meta.sort_values("material_id", kind="stable", ignore_index=True)
            payloads = _payloads(
                paths[f"geometry_{role}"], part_meta["material_id"].astype(str).tolist()
            )
            computed = _compute_many(payloads, workers=workers)
            computed_frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed]
            )
            if computed_frame.empty:
                computed_frame = pd.DataFrame(columns=["material_id", *ALL_ANALYTIC_FEATURES])
            table = part_meta.merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(part_meta):
                raise RuntimeError(f"NEXT94 {role} row accounting differs")
            for name in ALWAYS_PRESENT_STATUS_COLUMNS:
                if name not in table:
                    table[name] = None
            feature_path = staging / FEATURE_NAMES[role]
            table.to_parquet(feature_path, index=False)
            output_paths.append(feature_path)
            finite_counts = {
                name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
                for name in ALL_ANALYTIC_FEATURES
            }
            counts[role] = {
                "rows": int(len(table)),
                "full_row_errors": int(
                    table[["next43_error", "next44_error", "next80_error"]]
                    .notna()
                    .any(axis=1)
                    .sum()
                ),
                "pauling_feature_errors": int(table["pauling_feature_error"].notna().sum()),
                "pauling_decisions": {
                    decision: int(table["pauling_p2_p5_decision"].eq(decision).sum())
                    for decision in DECISIONS
                },
                "finite_feature_counts": finite_counts,
            }

        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(ALL_ANALYTIC_FEATURES),
            "feature_count": len(ALL_ANALYTIC_FEATURES),
            "families": {
                "NEXT43": list(NEXT43_FEATURE_NAMES),
                "NEXT44": list(NEXT44_FEATURE_NAMES),
                "NEXT80": list(PRLR_FEATURE_NAMES),
                "Pauling_controls": list(RULES),
            },
            "identity_only_columns_excluded_from_candidate_laws": list(
                IDENTITY_ONLY_COLUMNS
            ),
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "all_partition_raw_x0_only_analytic_and_pauling_feature_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "labels_opened": False,
            "endpoint_payloads_opened": False,
            "relaxed_structures_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_features": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in paths.items()
            },
            "upstream_source_sha256": {
                "src/next85_scigen_label_free_features.py": upstream_hash
            },
            "executed_source_sha256": {"src/next94_wyformer_label_free_features.py": source_hash},
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "scientific_improvement_claim": False,
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT94 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT94 source changed before publication")
        _publish_directory(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CATALOGUE_NAME",
    "FEATURE_NAMES",
    "MANIFEST_NAME",
    "PROTOCOL",
    "build_wyformer_label_free_features",
]
