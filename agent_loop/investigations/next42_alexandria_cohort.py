#!/usr/bin/env python3
"""Freeze every source-qualified Alexandria raw x0 as geometry-only data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import pandas as pd

from src.next11_geometry_only_frames import _canonical_frame
from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next18_alexandria_holdout import (
    _clean_atoms,
    _initial_structure,
    iter_bz2_object,
)
from src.next42_alexandria_source_audit import (
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    OUTPUT_NAME as SOURCE_TABLE_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
    RAW_X0_SOURCE_FAMILIES,
)


PROTOCOL = "2026-08-03-next42-alexandria-raw-x0-cohort-v1"
COHORT_NAME = "next42_alexandria_raw_x0_metadata.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
MANIFEST_NAME = "MANIFEST.json"
INPUT_ROLE = "raw_pre_dft_pre_mlip_x0_geometry_only"


def _read_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _validate_source_audit(
    *,
    source_table_path: Path,
    source_manifest_path: Path,
    path_hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = _read_json(source_manifest_path, role="NEXT42 source audit manifest")
    if (
        manifest.get("protocol") != SOURCE_PROTOCOL
        or manifest.get("scientific_labels_emitted") is not False
        or manifest.get("source_qualification_frozen_before_endpoint_evaluation")
        is not True
        or manifest.get("trajectory_endpoint_values_accessed_for_qualification")
        is not False
    ):
        raise ValueError("NEXT42 source audit contract differs")
    if source_table_path.name != SOURCE_TABLE_NAME or source_manifest_path.name != SOURCE_MANIFEST_NAME:
        raise ValueError("NEXT42 source audit filenames differ")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(SOURCE_TABLE_NAME) != _sha256_file(
        source_table_path
    ):
        raise ValueError("NEXT42 source audit output hash differs")
    inputs = manifest.get("inputs")
    fixed = inputs.get("fixed_sha256") if isinstance(inputs, Mapping) else None
    if not isinstance(fixed, Mapping) or any(
        fixed.get(role) != digest for role, digest in path_hashes.items()
    ):
        raise ValueError("NEXT42 source audit path hashes differ")
    if sorted(manifest.get("raw_x0_source_families", [])) != sorted(
        RAW_X0_SOURCE_FAMILIES
    ):
        raise ValueError("NEXT42 raw-x0 source allowlist differs")

    table = pd.read_parquet(source_table_path)
    required = {
        "material_id",
        "source_family",
        "location",
        "official_benchmark",
        "raw_x0_eligible",
        "qualification_reason",
    }
    if set(table.columns) != required:
        raise ValueError("NEXT42 source table schema differs")
    table = table.sort_values("material_id", kind="stable", ignore_index=True)
    if table.material_id.astype(str).duplicated().any():
        raise ValueError("NEXT42 source table IDs are duplicated")
    if not table.official_benchmark.map(lambda value: type(value) is bool).all():
        raise ValueError("NEXT42 benchmark flags must be exact booleans")
    if not table.raw_x0_eligible.map(lambda value: type(value) is bool).all():
        raise ValueError("NEXT42 eligibility flags must be exact booleans")
    eligible = table.loc[table.raw_x0_eligible].copy()
    if eligible.empty:
        raise ValueError("NEXT42 source audit has no eligible rows")
    if (
        eligible.official_benchmark.any()
        or not eligible.qualification_reason.eq("eligible_round1_raw_x0").all()
        or not eligible.source_family.isin(RAW_X0_SOURCE_FAMILIES).all()
    ):
        raise ValueError("NEXT42 source audit contains an ineligible selected row")
    return eligible, manifest


def build_next42_cohort(
    *,
    shard_0000_path: Path,
    shard_0001_path: Path,
    source_table_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish every qualified initial geometry without accessing endpoints."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    shards = {
        "pbe_0000": Path(shard_0000_path).resolve(),
        "pbe_0001": Path(shard_0001_path).resolve(),
    }
    source_table_path = Path(source_table_path).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    if any(not path.is_file() for path in shards.values()) or not source_table_path.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("NEXT42 cohort input is missing")
    path_hashes = {role: _sha256_file(path) for role, path in shards.items()}
    eligible, _source_manifest = _validate_source_audit(
        source_table_path=source_table_path,
        source_manifest_path=source_manifest_path,
        path_hashes=path_hashes,
    )
    expected_ids = set(eligible.material_id.astype(str))
    structures: dict[str, bytes] = {}
    structure_metadata: dict[str, dict[str, object]] = {}
    for source_shard, path in shards.items():
        for material_id, calculations in iter_bz2_object(path):
            if material_id not in expected_ids:
                continue
            if material_id in structures:
                raise ValueError(f"duplicate eligible path identity: {material_id}")
            structure = _initial_structure(calculations)
            atoms = _clean_atoms(structure)
            structures[material_id] = _canonical_frame(atoms)
            structure_metadata[material_id] = {
                "source_shard": source_shard,
                "formula": structure.composition.formula.replace(" ", ""),
                "reduced_formula": structure.composition.reduced_formula,
                "natoms": len(structure),
            }
    missing = expected_ids - set(structures)
    if missing:
        raise ValueError(f"{len(missing)} eligible identities are missing from path shards")
    if set(structures) != expected_ids:
        raise ValueError("NEXT42 cohort geometry identity differs")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        geometry_path = staging / GEOMETRY_NAME
        with zipfile.ZipFile(
            geometry_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            archive.comment = b""
            for material_id in sorted(structures):
                name = f"{material_id}.extxyz"
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.extra = b""
                info.comment = b""
                archive.writestr(
                    info,
                    structures[material_id],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        eligible_index = eligible.set_index("material_id")
        rows: list[dict[str, object]] = []
        for material_id in sorted(expected_ids):
            upstream = eligible_index.loc[material_id]
            rows.append(
                {
                    "material_id": material_id,
                    "source_family": str(upstream.source_family),
                    "source_shard": str(structure_metadata[material_id]["source_shard"]),
                    "formula": str(structure_metadata[material_id]["formula"]),
                    "reduced_formula": str(
                        structure_metadata[material_id]["reduced_formula"]
                    ),
                    "natoms": int(structure_metadata[material_id]["natoms"]),
                    "input_role": INPUT_ROLE,
                }
            )
        metadata = pd.DataFrame(rows)
        metadata_path = staging / COHORT_NAME
        metadata.to_parquet(metadata_path, index=False)

        repository = Path(__file__).resolve().parents[1]
        source_names = (
            "src/next11_geometry_only_frames.py",
            "src/next18_alexandria_holdout.py",
            "src/next42_alexandria_source_audit.py",
            "src/next42_alexandria_cohort.py",
        )
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "source-qualified raw pre-DFT pre-MLIP x0 cohort",
            "input_role": INPUT_ROLE,
            "selection": {
                "rule": "all source-audit rows with raw_x0_eligible true",
                "sampled": False,
                "endpoint_fields_used": False,
            },
            "raw_container_endpoint_bytes_present": True,
            "raw_container_records_decoded": True,
            "endpoint_fields_accessed_by_sanitizer": False,
            "later_geometry_accessed": False,
            "dft_values_read": False,
            "mlip_prerelaxation_used": False,
            "physical_relaxation_executed": False,
            "counts": {
                "selected_rows": len(metadata),
                "selected_atoms": int(metadata.natoms.sum()),
                "source_families": {
                    str(key): int(value)
                    for key, value in metadata.source_family.value_counts().sort_index().items()
                },
            },
            "inputs_sha256": {
                **{role: digest for role, digest in path_hashes.items()},
                "source_table": _sha256_file(source_table_path),
                "source_manifest": _sha256_file(source_manifest_path),
            },
            "executed_source_sha256": {
                name: _sha256_file(repository / name) for name in source_names
            },
            "scientific_improvement_claim": False,
        }
        manifest["outputs_sha256"] = {
            COHORT_NAME: _sha256_file(metadata_path),
            GEOMETRY_NAME: _sha256_file(geometry_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        input_paths = {
            **shards,
            "source_table": source_table_path,
            "source_manifest": source_manifest_path,
        }
        expected_hashes = {
            **path_hashes,
            "source_table": manifest["inputs_sha256"]["source_table"],
            "source_manifest": manifest["inputs_sha256"]["source_manifest"],
        }
        if any(
            _sha256_file(path) != expected_hashes[role]
            for role, path in input_paths.items()
        ):
            raise RuntimeError("NEXT42 cohort input changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbe-0000", type=Path, required=True)
    parser.add_argument("--pbe-0001", type=Path, required=True)
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_next42_cohort(
        shard_0000_path=args.pbe_0000,
        shard_0001_path=args.pbe_0001,
        source_table_path=args.source_table,
        source_manifest_path=args.source_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
