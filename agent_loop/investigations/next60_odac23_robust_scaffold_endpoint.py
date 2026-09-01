#!/usr/bin/env python3
"""Build adsorbate-robust, translation-aligned ODAC23 train endpoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile

import lmdb
import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next53_odac23_train_cohort import (
    EXPECTED_TRAIN_SHARDS,
    PROTOCOL as TRAIN_SOURCE_PROTOCOL,
    _directory_digest,
    _geometry_sha256,
    _numpy,
    sanitize_odac23_train_record,
)
from src.next54_odac23_train_selection import (
    MANIFEST_NAME as SELECTION_MANIFEST_NAME,
    METADATA_NAME as SELECTION_METADATA_NAME,
    PROTOCOL as SELECTION_PROTOCOL,
)


PROTOCOL = "2026-08-03-next60-odac23-robust-scaffold-endpoint-v1"
DESIGN_SHA256 = "42fce7e4e16f23ee8c45077ab5f5e7f44c8b3e72e4d71951faa42407e9514aeb"
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
EXPECTED_TRAIN_COHORT_MANIFEST_SHA256 = (
    "8fe4798f7df5ec8ddc5e748f8820cb1616133a9b3502a72a86a239a6b2d9e9ce"
)
CELL_GRID_ANGSTROM = 0.001
MIN_RELAXATIONS = 4
ENDPOINT_COLUMN = "robust_aligned_framework_displacement_p95_median"
ROLES = ("discovery", "internal_validation", "internal_replication")
ROLE_LABELS_NAME = "robust_offline_labels.parquet"
ROLE_MANIFEST_NAME = "MANIFEST.json"
COHORT_METADATA_NAME = "next60_robust_scaffold_cohort_metadata.parquet"
TOP_MANIFEST_NAME = "FIREWALL_MANIFEST.json"


def translation_aligned_displacements(
    *, initial: np.ndarray, relaxed: np.ndarray, cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual distances and robust periodic common translation."""

    initial = np.asarray(initial, dtype=float)
    relaxed = np.asarray(relaxed, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if (
        initial.ndim != 2
        or initial.shape[1:] != (3,)
        or relaxed.shape != initial.shape
        or len(initial) < 1
        or cell.shape != (3, 3)
        or not np.isfinite(initial).all()
        or not np.isfinite(relaxed).all()
        or not np.isfinite(cell).all()
        or abs(float(np.linalg.det(cell))) <= 1.0e-12
    ):
        raise ValueError("NEXT60 alignment geometry differs")
    fractional = (relaxed - initial) @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    circular_center = np.angle(
        np.mean(np.exp(2.0j * math.pi * fractional), axis=0)
    ) / (2.0 * math.pi)
    unwrapped = fractional - np.round(fractional - circular_center)
    translation = np.median(unwrapped, axis=0)
    residual = unwrapped - translation
    residual -= np.round(residual)
    distances = np.linalg.norm(residual @ cell, axis=1)
    if not np.isfinite(distances).all() or not np.isfinite(translation).all():
        raise ValueError("NEXT60 alignment result is non-finite")
    return distances, np.asarray(translation, dtype=float)


def scaffold_condition_key(
    *,
    framework_name: str,
    supercell: tuple[int, int, int],
    numbers: np.ndarray,
    cell: np.ndarray,
) -> str:
    """Hash the frozen adsorbate-independent scaffold-condition identity."""

    numbers = np.asarray(numbers, dtype=int).reshape(-1)
    cell = np.asarray(cell, dtype=float)
    if (
        not framework_name
        or len(supercell) != 3
        or any(type(value) is not int or value < 1 for value in supercell)
        or len(numbers) < 1
        or np.any(numbers <= 0)
        or cell.shape != (3, 3)
        or not np.isfinite(cell).all()
    ):
        raise ValueError("NEXT60 scaffold identity differs")
    cell_grid = np.rint(cell / CELL_GRID_ANGSTROM).astype("<i8")
    number_bytes = np.asarray(numbers, dtype="<i4").tobytes(order="C")
    payload = {
        "framework_name": framework_name,
        "supercell": list(supercell),
        "natoms": len(numbers),
        "numbers_sha256": hashlib.sha256(number_bytes).hexdigest(),
        "cell_grid_0p001_angstrom": cell_grid.reshape(-1).tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _strict_json(path: Path, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_robust_scaffold_endpoint(
    *,
    train_dir: Path,
    train_cohort_manifest_path: Path,
    selection_dir: Path,
    design_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Scan official train only, aggregate robust labels, and publish role firewalls."""

    train_dir = Path(train_dir).resolve()
    selection_dir = Path(selection_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "train_cohort_manifest": Path(train_cohort_manifest_path).resolve(),
        "selection_metadata": selection_dir / SELECTION_METADATA_NAME,
        "selection_manifest": selection_dir / SELECTION_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not train_dir.is_dir() or any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT60 input is missing")
    shards = sorted(train_dir.glob("*.lmdb"))
    if len(shards) != EXPECTED_TRAIN_SHARDS:
        raise ValueError("NEXT60 train shard inventory differs")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["train_cohort_manifest"] != EXPECTED_TRAIN_COHORT_MANIFEST_SHA256
        or hashes["selection_manifest"] != EXPECTED_SELECTION_MANIFEST_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT60 frozen input hash differs")
    train_manifest = _strict_json(paths["train_cohort_manifest"], "NEXT53 manifest")
    selection_manifest = _strict_json(paths["selection_manifest"], "NEXT54 manifest")
    selection_outputs = selection_manifest.get("outputs_sha256")
    shard_digest = _directory_digest(shards)
    if (
        train_manifest.get("protocol") != TRAIN_SOURCE_PROTOCOL
        or train_manifest.get("validation_or_test_payload_deserialized") is not False
        or train_manifest.get("train_shard_set_sha256") != shard_digest
        or selection_manifest.get("protocol") != SELECTION_PROTOCOL
        or not isinstance(selection_outputs, Mapping)
        or selection_outputs.get(paths["selection_metadata"].name)
        != hashes["selection_metadata"]
    ):
        raise ValueError("NEXT60 train-only provenance differs")

    selected = pd.read_parquet(paths["selection_metadata"])
    required = {
        "material_id",
        "framework_name",
        "geometry_sha256",
        "partition_role",
    }
    if (
        selected.empty
        or not required.issubset(selected.columns)
        or selected["material_id"].duplicated().any()
        or selected["framework_name"].duplicated().any()
        or set(selected["partition_role"]) != set(ROLES)
    ):
        raise ValueError("NEXT60 selection metadata differs")
    geometry_to_material = dict(
        zip(selected["geometry_sha256"].astype(str), selected["material_id"].astype(str), strict=True)
    )
    if len(geometry_to_material) != len(selected):
        raise ValueError("NEXT60 selected geometry hashes are not unique")

    groups: dict[str, dict[str, object]] = {}
    selected_links: dict[str, set[str]] = {material_id: set() for material_id in selected["material_id"].astype(str)}
    total_records = 0
    for shard_index, shard in enumerate(shards, start=1):
        environment = lmdb.open(
            str(shard), readonly=True, lock=False, readahead=False, meminit=False, subdir=False
        )
        try:
            with environment.begin() as transaction:
                for key, payload in transaction.cursor():
                    if key == b"length":
                        continue
                    try:
                        raw_record = pickle.loads(payload)
                        sanitized = sanitize_odac23_train_record(raw_record)
                        tags = _numpy(raw_record.tags, name="tags").reshape(-1).astype(int)
                        keep = tags == 0
                        initial = _numpy(raw_record.pos, name="initial positions").astype(float)[keep]
                        relaxed = _numpy(raw_record.pos_relaxed, name="relaxed positions").astype(float)[keep]
                        cell = np.asarray(sanitized.atoms.cell.array, dtype=float)
                        distances, translation = translation_aligned_displacements(
                            initial=initial, relaxed=relaxed, cell=cell
                        )
                        raw_supercell = _numpy(raw_record.supercell, name="supercell").reshape(-1)
                        rounded_supercell = np.rint(raw_supercell).astype(int)
                        if (
                            raw_supercell.shape != (3,)
                            or not np.array_equal(raw_supercell, rounded_supercell.astype(raw_supercell.dtype))
                            or np.any(rounded_supercell < 1)
                        ):
                            raise ValueError("ODAC23 supercell differs")
                        supercell = tuple(int(value) for value in rounded_supercell)
                        group_key = scaffold_condition_key(
                            framework_name=sanitized.framework_name,
                            supercell=supercell,
                            numbers=np.asarray(sanitized.atoms.numbers, dtype=int),
                            cell=cell,
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"NEXT60 record failed in {shard.name}:{key!r}"
                        ) from exc
                    group = groups.setdefault(
                        group_key,
                        {
                            "framework_name": sanitized.framework_name,
                            "supercell": supercell,
                            "natoms": len(sanitized.atoms),
                            "raw_p95": [],
                            "aligned_p95": [],
                            "translation_norm": [],
                        },
                    )
                    if (
                        group["framework_name"] != sanitized.framework_name
                        or group["supercell"] != supercell
                        or group["natoms"] != len(sanitized.atoms)
                    ):
                        raise RuntimeError("NEXT60 scaffold key collision")
                    group["raw_p95"].append(sanitized.framework_displacement_p95)
                    group["aligned_p95"].append(float(np.quantile(distances, 0.95)))
                    group["translation_norm"].append(float(np.linalg.norm(translation @ cell)))
                    geometry_hash = _geometry_sha256(sanitized.atoms)
                    material_id = geometry_to_material.get(geometry_hash)
                    if material_id is not None:
                        selected_links[material_id].add(group_key)
                    total_records += 1
        finally:
            environment.close()
        if shard_index % 10 == 0 or shard_index == len(shards):
            linked = sum(bool(values) for values in selected_links.values())
            print(
                f"NEXT60 ODAC23 robust endpoint: {shard_index}/{len(shards)} shards, "
                f"{total_records} records, {len(groups)} scaffold conditions, "
                f"{linked}/{len(selected_links)} selected linked",
                flush=True,
            )

    if any(len(keys) != 1 for keys in selected_links.values()):
        missing = sum(not keys for keys in selected_links.values())
        conflicts = sum(len(keys) > 1 for keys in selected_links.values())
        raise ValueError(
            f"NEXT60 selected-to-scaffold links differ: missing={missing}, conflicts={conflicts}"
        )
    selected_index = selected.set_index("material_id")
    label_rows = []
    cohort_rows = []
    excluded_low_count = 0
    for material_id in sorted(selected_links):
        group_key = next(iter(selected_links[material_id]))
        group = groups[group_key]
        aligned = np.asarray(group["aligned_p95"], dtype=float)
        raw = np.asarray(group["raw_p95"], dtype=float)
        translations = np.asarray(group["translation_norm"], dtype=float)
        role = str(selected_index.at[material_id, "partition_role"])
        included = len(aligned) >= MIN_RELAXATIONS
        cohort_rows.append(
            {
                "material_id": material_id,
                "framework_name": str(selected_index.at[material_id, "framework_name"]),
                "partition_role": role,
                "scaffold_condition_sha256": group_key,
                "relaxation_count": len(aligned),
                "included_minimum_four": included,
            }
        )
        if not included:
            excluded_low_count += 1
            continue
        label_rows.append(
            {
                "material_id": material_id,
                "partition_role": role,
                "scaffold_condition_sha256": group_key,
                "relaxation_count": len(aligned),
                ENDPOINT_COLUMN: float(np.median(aligned)),
                "aligned_p95_q25": float(np.quantile(aligned, 0.25)),
                "aligned_p95_q75": float(np.quantile(aligned, 0.75)),
                "raw_p95_median": float(np.median(raw)),
                "common_translation_norm_median": float(np.median(translations)),
                "offline_label_role": "PBE+D3_adsorbate_robust_translation_aligned_framework_response",
            }
        )
    cohort = pd.DataFrame(cohort_rows).sort_values("material_id", kind="mergesort")
    labels = pd.DataFrame(label_rows).sort_values("material_id", kind="mergesort")
    if labels.empty or labels["material_id"].duplicated().any():
        raise ValueError("NEXT60 robust endpoint cohort is empty or duplicated")

    role_tables = {
        role: labels[labels["partition_role"].eq(role)].reset_index(drop=True)
        for role in ROLES
    }
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    top_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "official_train_only_robust_scaffold_endpoint_and_role_firewall",
        "cell_grid_angstrom": CELL_GRID_ANGSTROM,
        "minimum_relaxations": MIN_RELAXATIONS,
        "protected_max_angstrom": 0.05,
        "severe_min_angstrom": 0.20,
        "formula_or_feature_search_executed": False,
        "internal_validation_endpoint_values_summarized_or_inspected": False,
        "internal_replication_endpoint_values_summarized_or_inspected": False,
        "official_validation_or_test_payload_deserialized": False,
        "counts": {
            "raw_train_relaxations": total_records,
            "scaffold_conditions": len(groups),
            "selected_linked": len(selected_links),
            "excluded_below_minimum_relaxations": excluded_low_count,
            **{role: len(role_tables[role]) for role in ROLES},
        },
        "train_shard_set_sha256": shard_digest,
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next60_odac23_robust_scaffold_endpoint.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        cohort_path = staging / COHORT_METADATA_NAME
        cohort.to_parquet(cohort_path, index=False)
        outputs = {COHORT_METADATA_NAME: _sha256(cohort_path)}
        for role in ROLES:
            role_dir = staging / role
            role_dir.mkdir()
            label_path = role_dir / ROLE_LABELS_NAME
            role_tables[role].to_parquet(label_path, index=False)
            role_manifest = {
                "protocol": PROTOCOL,
                "partition_role": role,
                "rows": len(role_tables[role]),
                "endpoint_values_summarized_or_inspected": role == "discovery",
                "outputs_sha256": {ROLE_LABELS_NAME: _sha256(label_path)},
            }
            role_manifest_path = role_dir / ROLE_MANIFEST_NAME
            role_manifest_path.write_bytes(_json_bytes(role_manifest))
            outputs[f"{role}/{ROLE_LABELS_NAME}"] = _sha256(label_path)
            outputs[f"{role}/{ROLE_MANIFEST_NAME}"] = _sha256(role_manifest_path)
        top_manifest["outputs_sha256"] = outputs
        (staging / TOP_MANIFEST_NAME).write_bytes(_json_bytes(top_manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT60 source changed before publication")
        if _directory_digest(shards) != shard_digest:
            raise RuntimeError("NEXT60 train shard set changed")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT60 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return top_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--train-cohort-manifest", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_robust_scaffold_endpoint(
        train_dir=args.train_dir,
        train_cohort_manifest_path=args.train_cohort_manifest,
        selection_dir=args.selection_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "ENDPOINT_COLUMN",
    "PROTOCOL",
    "ROLE_LABELS_NAME",
    "ROLE_MANIFEST_NAME",
    "TOP_MANIFEST_NAME",
    "build_robust_scaffold_endpoint",
    "scaffold_condition_key",
    "translation_aligned_displacements",
]


if __name__ == "__main__":
    main()
