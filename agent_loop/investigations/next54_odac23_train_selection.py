#!/usr/bin/env python3
"""Select one label-independent ODAC23 train x0 per official framework name."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import pandas as pd

from src.next11_geometry_only_frames import _parse_frame, _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next53_odac23_train_cohort import (
    GEOMETRY_NAME as SOURCE_GEOMETRY_NAME,
    LABELS_NAME as SOURCE_LABELS_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    METADATA_NAME as SOURCE_METADATA_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next54-odac23-train-representative-selection-v1"
DESIGN_SHA256 = "9be52914d9ebc347ed650c01e716c6dda0e2a2935e509ee18237767f3341ccbc"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "8fe4798f7df5ec8ddc5e748f8820cb1616133a9b3502a72a86a239a6b2d9e9ce"
)
REPRESENTATIVE_SALT = "NEXT54-REP-v1\0"
PARTITION_SALT = "NEXT54-SPLIT-v1\0"
METADATA_NAME = "next54_odac23_selected_metadata.parquet"
GEOMETRY_NAME = "next54_odac23_selected_x0.zip"
LABELS_NAME = "next54_odac23_selected_offline_labels.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _digest_text(salt: str, value: str) -> str:
    return hashlib.sha256((salt + value).encode("utf-8")).hexdigest()


def _partition(framework_name: str) -> tuple[str, str, float]:
    digest = _digest_text(PARTITION_SALT, framework_name)
    fraction = int(digest[:16], 16) / float(2**64)
    if fraction < 0.60:
        role = "discovery"
    elif fraction < 0.80:
        role = "internal_validation"
    else:
        role = "internal_replication"
    return role, digest, fraction


def select_representatives(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return one geometry-hash-selected row per framework, without labels."""

    required = {"material_id", "framework_name", "geometry_sha256", "natoms"}
    if not required.issubset(metadata.columns):
        raise ValueError("NEXT54 metadata schema differs")
    work = metadata.copy()
    if (
        work.empty
        or work["material_id"].duplicated().any()
        or work[list(required)].isna().any().any()
        or not work["geometry_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise ValueError("NEXT54 metadata identity differs")
    work["representative_sha256"] = work["geometry_sha256"].map(
        lambda value: _digest_text(REPRESENTATIVE_SALT, str(value))
    )
    work = work.sort_values(
        ["framework_name", "representative_sha256", "material_id"],
        kind="mergesort",
    )
    selected = work.drop_duplicates("framework_name", keep="first").copy()
    partitions = selected["framework_name"].map(_partition)
    selected["partition_role"] = [value[0] for value in partitions]
    selected["partition_sha256"] = [value[1] for value in partitions]
    selected["partition_fraction"] = [value[2] for value in partitions]
    selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
    if selected["framework_name"].duplicated().any():
        raise RuntimeError("NEXT54 framework isolation failed")
    return selected


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid NEXT53 source manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("NEXT53 source manifest must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_selected_train_cohort(
    *, source_dir: Path, design_path: Path, output_dir: Path
) -> dict[str, object]:
    """Publish selected geometry/roles and a separately stored offline label table."""

    source_dir = Path(source_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": source_dir / SOURCE_METADATA_NAME,
        "geometry": source_dir / SOURCE_GEOMETRY_NAME,
        "labels": source_dir / SOURCE_LABELS_NAME,
        "manifest": source_dir / SOURCE_MANIFEST_NAME,
        "design": design_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT54 source artifact is incomplete")
    if _sha256(paths["manifest"]) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("NEXT54 source manifest hash differs")
    if _sha256(design_path) != DESIGN_SHA256:
        raise ValueError("NEXT54 partition design hash differs")
    source_manifest = _strict_json(paths["manifest"])
    source_outputs = source_manifest.get("outputs_sha256")
    if (
        source_manifest.get("protocol") != SOURCE_PROTOCOL
        or not isinstance(source_outputs, dict)
        or any(
            source_outputs.get(paths[key].name) != _sha256(paths[key])
            for key in ("metadata", "geometry", "labels")
        )
    ):
        raise ValueError("NEXT54 NEXT53 provenance differs")

    # Selection and role assignment are completed before row labels are opened.
    metadata = pd.read_parquet(paths["metadata"])
    selected = select_representatives(metadata)
    selected_ids = tuple(selected["material_id"].astype(str))
    if len(selected) != int(metadata["framework_name"].nunique()):
        raise RuntimeError("NEXT54 representative coverage differs")

    frames = {}
    with zipfile.ZipFile(paths["geometry"]) as archive:
        infos = archive.infolist()
        names = {info.filename: info for info in infos}
        if len(names) != len(infos):
            raise ValueError("NEXT54 source geometry has duplicate members")
        for material_id in selected_ids:
            member_name = f"{material_id}.extxyz"
            info = names.get(member_name)
            if info is None or PurePosixPath(member_name).parent != PurePosixPath("."):
                raise ValueError("NEXT54 selected geometry member is missing")
            frames[material_id] = _parse_frame(
                archive.read(info), strict_output=True
            )

    # The selected row identities are now frozen; only now open offline labels.
    labels = pd.read_parquet(paths["labels"])
    selected_labels = labels[labels["material_id"].isin(selected_ids)].copy()
    selected_labels = selected_labels.sort_values("material_id", kind="mergesort")
    if (
        len(selected_labels) != len(selected)
        or selected_labels["material_id"].duplicated().any()
        or tuple(selected_labels["material_id"].astype(str)) != selected_ids
    ):
        raise ValueError("NEXT54 selected label identity differs")

    input_hashes = {name: _sha256(path) for name, path in paths.items()}
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    counts = {
        "source_exact_x0": len(metadata),
        "source_framework_names": int(metadata["framework_name"].nunique()),
        "selected_x0": len(selected),
        "discovery": int(selected["partition_role"].eq("discovery").sum()),
        "internal_validation": int(
            selected["partition_role"].eq("internal_validation").sum()
        ),
        "internal_replication": int(
            selected["partition_role"].eq("internal_replication").sum()
        ),
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "one_real_x0_per_framework_name_selected_without_labels",
        "partition_design_sha256": DESIGN_SHA256,
        "selection_frozen_before_row_labels_opened": True,
        "validation_or_test_payload_deserialized": False,
        "law_execution_dft_values_read": False,
        "law_execution_relaxed_geometry_read": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "counts": counts,
        "inputs_sha256": {
            name: {"path": str(paths[name]), "sha256": input_hashes[name]}
            for name in paths
        },
        "executed_source_sha256": {"src/next54_odac23_train_selection.py": source_hash},
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        metadata_path = staging / METADATA_NAME
        geometry_path = staging / GEOMETRY_NAME
        labels_path = staging / LABELS_NAME
        selected.to_parquet(metadata_path, index=False)
        _write_deterministic_archive(geometry_path, frames)
        selected_labels.to_parquet(labels_path, index=False)
        manifest["outputs_sha256"] = {
            path.name: _sha256(path) for path in (metadata_path, geometry_path, labels_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT54 source changed before publication")
        if any(_sha256(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT54 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_selected_train_cohort(
        source_dir=args.source_dir, design_path=args.design, output_dir=args.output_dir
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "GEOMETRY_NAME",
    "LABELS_NAME",
    "MANIFEST_NAME",
    "METADATA_NAME",
    "PROTOCOL",
    "build_selected_train_cohort",
    "select_representatives",
]


if __name__ == "__main__":
    main()
