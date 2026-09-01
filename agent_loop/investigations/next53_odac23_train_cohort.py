#!/usr/bin/env python3
"""Sanitize ODAC23 train records into framework-only x0 and offline labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile

from ase import Atoms
import lmdb
import numpy as np
import pandas as pd

from src.next11_geometry_only_frames import _ParsedFrame, _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next51_odac23_lockbox import (
    EXPECTED_SHA256 as ODAC23_ARCHIVE_SHA256,
    PROTOCOL as SOURCE_RECEIPT_PROTOCOL,
    RECEIPT_NAME,
)


PROTOCOL = "2026-08-03-next53-odac23-train-framework-cohort-v1"
PARTITION_PROTOCOL_SHA256 = (
    "4444239f7a8c1a7057414309162ff5af8ecd99efb6439b456aa357b8cd80a82f"
)
EXPECTED_TRAIN_SHARDS = 200
METADATA_NAME = "next53_odac23_train_framework_metadata.parquet"
GEOMETRY_NAME = "next53_odac23_train_framework_x0.zip"
LABELS_NAME = "next53_odac23_train_framework_labels.parquet"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_RECORD_FIELDS = frozenset(
    (
        "atomic_numbers",
        "cell",
        "defective",
        "fid",
        "fixed",
        "nads",
        "name",
        "natoms",
        "nco2",
        "nh2o",
        "oms",
        "pos",
        "pos_relaxed",
        "raw_y",
        "sid",
        "supercell",
        "tags",
        "y_init",
        "y_relaxed",
    )
)


@dataclass(frozen=True)
class SanitizedODACRecord:
    atoms: Atoms
    framework_name: str
    sample_name: str
    framework_displacement_p95: float
    framework_displacement_mean: float
    framework_displacement_max: float
    adsorbate_atoms_removed: int
    defective: bool
    open_metal_site: bool


def _numpy(value: object, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError(f"ODAC23 {name} is non-finite")
    return result


def sanitize_odac23_train_record(record: object) -> SanitizedODACRecord:
    """Remove tagged adsorbates and derive a framework-only DFT displacement."""

    try:
        fields = frozenset(str(value) for value in record.keys())
    except Exception as exc:
        raise ValueError("ODAC23 record has no inspectable field schema") from exc
    if fields != EXPECTED_RECORD_FIELDS:
        raise ValueError("ODAC23 record field schema differs")
    name = str(record.name)
    if "_w_" not in name:
        raise ValueError("ODAC23 sample name has no adsorbate delimiter")
    framework_name = name.split("_w_", 1)[0]
    numbers_float = _numpy(record.atomic_numbers, name="atomic numbers").reshape(-1)
    numbers = np.rint(numbers_float).astype(int)
    tags = _numpy(record.tags, name="tags").reshape(-1).astype(int)
    fixed = _numpy(record.fixed, name="fixed").reshape(-1)
    initial = _numpy(record.pos, name="initial positions").astype(float)
    relaxed = _numpy(record.pos_relaxed, name="relaxed positions").astype(float)
    cell = _numpy(record.cell, name="cell").astype(float).reshape(3, 3)
    natoms = int(record.natoms)
    if (
        natoms < 1
        or numbers.shape != (natoms,)
        or tags.shape != (natoms,)
        or fixed.shape != (natoms,)
        or initial.shape != (natoms, 3)
        or relaxed.shape != (natoms, 3)
        or not np.array_equal(numbers_float, numbers.astype(numbers_float.dtype))
        or np.any(numbers <= 0)
        or np.any(fixed != 0.0)
        or abs(float(np.linalg.det(cell))) <= 1e-12
    ):
        raise ValueError("ODAC23 record geometry schema differs")
    framework = tags == 0
    adsorbate_count = int((~framework).sum())
    expected_adsorbate_count = 3 * (int(record.nco2) + int(record.nh2o))
    if (
        not framework.any()
        or adsorbate_count != expected_adsorbate_count
        or int(record.nads) != int(record.nco2) + int(record.nh2o)
    ):
        raise ValueError("ODAC23 atom tags do not match adsorbate counts")
    inverse_cell = np.linalg.inv(cell)
    fractional_delta = (relaxed - initial) @ inverse_cell
    fractional_delta -= np.round(fractional_delta)
    distances = np.linalg.norm(fractional_delta @ cell, axis=1)[framework]
    if not len(distances) or not np.isfinite(distances).all():
        raise ValueError("ODAC23 framework displacement is invalid")
    atoms = Atoms(
        numbers=numbers[framework],
        positions=initial[framework],
        cell=cell,
        pbc=True,
    )
    atoms.info.clear()
    if set(atoms.arrays) != {"numbers", "positions"}:
        raise ValueError("ODAC23 sanitized x0 retained non-geometric arrays")
    return SanitizedODACRecord(
        atoms=atoms,
        framework_name=framework_name,
        sample_name=name,
        framework_displacement_p95=float(np.quantile(distances, 0.95)),
        framework_displacement_mean=float(np.mean(distances)),
        framework_displacement_max=float(np.max(distances)),
        adsorbate_atoms_removed=adsorbate_count,
        defective=bool(record.defective),
        open_metal_site=bool(record.oms),
    )


def _geometry_sha256(atoms: Atoms) -> str:
    digest = hashlib.sha256()
    for array in (
        np.asarray(atoms.numbers, dtype="<i4"),
        np.asarray(atoms.positions, dtype="<f8"),
        np.asarray(atoms.cell.array, dtype="<f8"),
    ):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    digest.update(b"pbc=111")
    return digest.hexdigest()


def _strict_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _directory_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def build_odac23_train_framework_cohort(
    *,
    train_dir: Path,
    source_receipt_path: Path,
    source_receipt_manifest_path: Path,
    partition_protocol_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish unique framework x0 geometries and separately aggregated labels."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    train_dir = Path(train_dir).resolve()
    inputs = {
        "source_receipt": Path(source_receipt_path).resolve(),
        "source_receipt_manifest": Path(source_receipt_manifest_path).resolve(),
        "partition_protocol": Path(partition_protocol_path).resolve(),
    }
    if not train_dir.is_dir() or any(not path.is_file() for path in inputs.values()):
        raise FileNotFoundError("NEXT53 train cohort input is missing")
    shards = sorted(train_dir.glob("*.lmdb"))
    if len(shards) != EXPECTED_TRAIN_SHARDS or any(not path.is_file() for path in shards):
        raise ValueError("ODAC23 train shard inventory differs")
    if _sha256(inputs["partition_protocol"]) != PARTITION_PROTOCOL_SHA256:
        raise ValueError("NEXT53 partition protocol hash differs")
    receipt = _strict_json(inputs["source_receipt"], role="ODAC23 source receipt")
    receipt_manifest = _strict_json(
        inputs["source_receipt_manifest"], role="ODAC23 receipt manifest"
    )
    receipt_outputs = receipt_manifest.get("outputs_sha256")
    source = receipt.get("source")
    if (
        receipt.get("protocol") != SOURCE_RECEIPT_PROTOCOL
        or receipt.get("labels_opened") is not False
        or receipt.get("archive_members_opened") is not False
        or not isinstance(source, dict)
        or source.get("sha256") != ODAC23_ARCHIVE_SHA256
        or receipt_manifest.get("protocol") != SOURCE_RECEIPT_PROTOCOL
        or not isinstance(receipt_outputs, dict)
        or receipt_outputs.get(RECEIPT_NAME) != _sha256(inputs["source_receipt"])
    ):
        raise ValueError("NEXT53 opaque source receipt differs")

    groups: dict[str, dict[str, object]] = {}
    total_records = 0
    removed_adsorbates = 0
    for shard_index, shard in enumerate(shards, start=1):
        environment = lmdb.open(
            str(shard),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=False,
        )
        try:
            with environment.begin() as transaction:
                for key, payload in transaction.cursor():
                    if key == b"length":
                        continue
                    try:
                        record = sanitize_odac23_train_record(pickle.loads(payload))
                    except Exception as exc:
                        raise ValueError(
                            f"ODAC23 train record failed in {shard.name}:{key!r}"
                        ) from exc
                    geometry_hash = _geometry_sha256(record.atoms)
                    group = groups.setdefault(
                        geometry_hash,
                        {
                            "atoms": record.atoms,
                            "framework_names": set(),
                            "p95": [],
                            "mean": [],
                            "maximum": [],
                            "defective": set(),
                            "oms": set(),
                        },
                    )
                    group["framework_names"].add(record.framework_name)
                    group["p95"].append(record.framework_displacement_p95)
                    group["mean"].append(record.framework_displacement_mean)
                    group["maximum"].append(record.framework_displacement_max)
                    group["defective"].add(record.defective)
                    group["oms"].add(record.open_metal_site)
                    total_records += 1
                    removed_adsorbates += record.adsorbate_atoms_removed
        finally:
            environment.close()
        if shard_index % 10 == 0 or shard_index == len(shards):
            print(
                f"NEXT53 ODAC23 ingest: {shard_index}/{len(shards)} shards, "
                f"{total_records} records, {len(groups)} exact frameworks",
                flush=True,
            )
    if not groups or total_records < len(groups):
        raise ValueError("NEXT53 ODAC23 grouping is empty")

    frames: dict[str, _ParsedFrame] = {}
    metadata_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for geometry_hash, group in sorted(groups.items()):
        names = sorted(group["framework_names"])
        defective = sorted(group["defective"])
        oms = sorted(group["oms"])
        if len(names) != 1 or len(defective) != 1 or len(oms) != 1:
            raise ValueError("ODAC23 exact geometry identity metadata conflicts")
        material_id = f"odac23-train-{geometry_hash[:20]}"
        atoms = group["atoms"]
        frames[material_id] = _ParsedFrame(atoms, (), ())
        p95 = np.asarray(group["p95"], dtype=float)
        means = np.asarray(group["mean"], dtype=float)
        maxima = np.asarray(group["maximum"], dtype=float)
        metadata_rows.append(
            {
                "material_id": material_id,
                "framework_name": names[0],
                "geometry_sha256": geometry_hash,
                "natoms": len(atoms),
                "records": len(p95),
                "defective": defective[0],
                "open_metal_site": oms[0],
                "input_role": "raw_unrelaxed_framework_x0_geometry_only",
            }
        )
        label_rows.append(
            {
                "material_id": material_id,
                "records": len(p95),
                "framework_displacement_p95_median": float(np.median(p95)),
                "framework_displacement_p95_q25": float(np.quantile(p95, 0.25)),
                "framework_displacement_p95_q75": float(np.quantile(p95, 0.75)),
                "framework_displacement_p95_max": float(np.max(p95)),
                "framework_displacement_mean_median": float(np.median(means)),
                "framework_displacement_max_median": float(np.median(maxima)),
                "offline_label_role": "PBE+D3_relaxed_geometry_only",
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    labels = pd.DataFrame(label_rows)
    if (
        metadata["material_id"].duplicated().any()
        or labels["material_id"].duplicated().any()
        or metadata["material_id"].tolist() != labels["material_id"].tolist()
    ):
        raise ValueError("NEXT53 ODAC23 output identity differs")

    shard_digest = _directory_digest(shards)
    input_hashes = {name: _sha256(path) for name, path in inputs.items()}
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "official_train_only_framework_sanitization_and_label_aggregation",
        "source_archive_sha256": ODAC23_ARCHIVE_SHA256,
        "partition_protocol_sha256": PARTITION_PROTOCOL_SHA256,
        "train_shard_count": len(shards),
        "train_shard_set_sha256": shard_digest,
        "validation_or_test_payload_deserialized": False,
        "development_labels_opened": True,
        "law_execution_dft_values_read": False,
        "law_execution_relaxed_geometry_read": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "adsorbate_removal": "keep tags == 0; require other count == 3*(nco2+nh2o)",
        "counts": {
            "raw_relaxations": total_records,
            "unique_framework_x0": len(metadata),
            "adsorbate_atoms_removed": removed_adsorbates,
            "protected_at_0p05": int(
                (labels["framework_displacement_p95_median"] <= 0.05).sum()
            ),
            "severe_at_0p20": int(
                (labels["framework_displacement_p95_median"] >= 0.20).sum()
            ),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": input_hashes[name]}
            for name, path in inputs.items()
        },
        "executed_source_sha256": {
            "src/next53_odac23_train_cohort.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        metadata_path = staging / METADATA_NAME
        geometry_path = staging / GEOMETRY_NAME
        labels_path = staging / LABELS_NAME
        metadata.to_parquet(metadata_path, index=False)
        _write_deterministic_archive(geometry_path, frames)
        labels.to_parquet(labels_path, index=False)
        manifest["outputs_sha256"] = {
            path.name: _sha256(path)
            for path in (metadata_path, geometry_path, labels_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _directory_digest(shards) != shard_digest:
            raise RuntimeError("NEXT53 train shard set changed before publication")
        for name, path in inputs.items():
            if _sha256(path) != input_hashes[name]:
                raise RuntimeError(f"NEXT53 input {name} changed before publication")
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT53 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seal ODAC23 train records as framework-only x0 plus offline labels."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt-manifest", type=Path, required=True)
    parser.add_argument("--partition-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_odac23_train_framework_cohort(
        train_dir=args.train_dir,
        source_receipt_path=args.source_receipt,
        source_receipt_manifest_path=args.source_receipt_manifest,
        partition_protocol_path=args.partition_protocol,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "GEOMETRY_NAME",
    "LABELS_NAME",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "PROTOCOL",
    "SanitizedODACRecord",
    "build_odac23_train_framework_cohort",
    "sanitize_odac23_train_record",
]


if __name__ == "__main__":
    main()
