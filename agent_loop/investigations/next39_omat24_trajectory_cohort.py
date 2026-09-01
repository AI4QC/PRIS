#!/usr/bin/env python3
"""Freeze a parent-unique OMat24 trajectory cohort from step-0 geometry only."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zlib

import lmdb
import pandas as pd

from src.next11_geometry_only_frames import _ParsedFrame, _write_deterministic_archive
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next32_omat24_cohort import (
    _project_identity,
    _record_payload,
    project_identity_geometry,
)


PROTOCOL = "2026-08-03-next39-omat24-trajectory-cohort-v1"
COHORT_NAME = "next39_trajectory_cohort.parquet"
GEOMETRY_NAME = "geometry_only_frames.zip"
PARENT_NAME = "parent_ids.txt"
MANIFEST_NAME = "MANIFEST.json"
TASK_TYPE = "Structure Optimization"
_SID_PATTERN = re.compile(r"^(?P<stem>.+)_(?P<step>0|[1-9][0-9]*)$")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def parse_trajectory_sid(sid: str) -> tuple[str, int]:
    """Split an OMat24 SID at its final integer trajectory-step suffix."""

    match = _SID_PATTERN.fullmatch(str(sid))
    if match is None:
        raise ValueError(f"invalid OMat24 trajectory step suffix: {sid!r}")
    return match.group("stem"), int(match.group("step"))


def select_eligible_trajectories(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_latest_step: int,
    salt: str,
) -> list[dict[str, object]]:
    """Select step-0/late, optimization-only, parent-unique trajectories."""

    if type(minimum_latest_step) is not int or minimum_latest_step <= 0 or not salt:
        raise ValueError("trajectory selection settings are invalid")
    by_trajectory: dict[str, dict[str, object]] = {}
    seen_sids: set[str] = set()
    seen_keys: set[int] = set()
    for raw in rows:
        try:
            sid = str(raw["sid"])
            parent_id = str(raw["parent_id"])
            task_type = str(raw["task_type"])
            record_key = int(raw["record_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid OMat24 trajectory identity row") from exc
        if not sid or not parent_id or record_key <= 0:
            raise ValueError("invalid OMat24 trajectory identity")
        if sid in seen_sids or record_key in seen_keys:
            raise ValueError("duplicate OMat24 SID or record key")
        seen_sids.add(sid)
        seen_keys.add(record_key)
        stem, step = parse_trajectory_sid(sid)
        entry = by_trajectory.setdefault(
            stem,
            {"parent_id": parent_id, "task_type": task_type, "steps": {}},
        )
        if entry["parent_id"] != parent_id or entry["task_type"] != task_type:
            raise ValueError("OMat24 trajectory identity changes across steps")
        steps = entry["steps"]
        assert isinstance(steps, dict)
        if step in steps:
            raise ValueError("duplicate OMat24 trajectory step")
        steps[step] = {"sid": sid, "record_key": record_key}

    eligible: list[dict[str, object]] = []
    for stem, raw_entry in by_trajectory.items():
        if raw_entry["task_type"] != TASK_TYPE:
            continue
        steps = raw_entry["steps"]
        assert isinstance(steps, dict)
        if 0 not in steps:
            continue
        latest_step = max(int(value) for value in steps)
        if latest_step < minimum_latest_step:
            continue
        initial = steps[0]
        latest = steps[latest_step]
        assert isinstance(initial, dict) and isinstance(latest, dict)
        eligible.append(
            {
                "trajectory_stem": stem,
                "parent_id": str(raw_entry["parent_id"]),
                "task_type": str(raw_entry["task_type"]),
                "initial_sid": str(initial["sid"]),
                "initial_step": 0,
                "initial_record_key": int(initial["record_key"]),
                "latest_sid": str(latest["sid"]),
                "latest_step": latest_step,
                "latest_record_key": int(latest["record_key"]),
                "observed_step_count": len(steps),
            }
        )

    by_parent: dict[str, tuple[str, dict[str, object]]] = {}
    for row in eligible:
        parent_id = str(row["parent_id"])
        digest = hashlib.sha256(
            f"{salt}|{parent_id}|{row['trajectory_stem']}".encode()
        ).hexdigest()
        previous = by_parent.get(parent_id)
        if previous is None or digest < previous[0]:
            by_parent[parent_id] = (digest, row)
    return [
        dict(row)
        for _digest, row in sorted(
            by_parent.values(), key=lambda item: str(item[1]["trajectory_stem"])
        )
    ]


def freeze_trajectory_cohort(
    *,
    db_path: Path,
    source_name: str,
    salt: str,
    minimum_latest_step: int,
    output_dir: Path,
) -> dict[str, object]:
    """Publish only selected step-0 geometries; later records stay unopened."""

    database = Path(db_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not database.is_file():
        raise FileNotFoundError(str(database))
    if not source_name:
        raise ValueError("source name is empty")
    database_hash = _sha256(database)

    env = lmdb.open(
        str(database), subdir=False, readonly=True, lock=False, readahead=False
    )
    scanned: list[dict[str, object]] = []
    try:
        with env.begin() as transaction:
            for key, compressed in transaction.cursor():
                if key in {b"nextid", b"metadata", b"deleted_ids"}:
                    continue
                try:
                    record_key = int(key)
                    payload = zlib.decompress(compressed)
                except (ValueError, zlib.error) as exc:
                    raise ValueError(f"invalid OMat24 LMDB key {key!r}") from exc
                scanned.append({**_project_identity(payload), "record_key": record_key})
        selected = select_eligible_trajectories(
            scanned, minimum_latest_step=minimum_latest_step, salt=salt
        )
        if not selected:
            raise ValueError("no eligible OMat24 trajectories")
        frames: dict[str, _ParsedFrame] = {}
        output_rows: list[dict[str, object]] = []
        with env.begin() as transaction:
            for selected_row in selected:
                payload = _record_payload(
                    transaction, int(selected_row["initial_record_key"])
                )
                identity, atoms = project_identity_geometry(payload)
                if (
                    identity["sid"] != selected_row["initial_sid"]
                    or identity["parent_id"] != selected_row["parent_id"]
                    or identity["task_type"] != TASK_TYPE
                ):
                    raise ValueError("OMat24 step-0 identity changed after selection")
                material_id = f"{source_name}::{selected_row['trajectory_stem']}"
                frames[material_id] = _ParsedFrame(atoms, (), ())
                output_rows.append(
                    {
                        "material_id": material_id,
                        "source_name": source_name,
                        **selected_row,
                        "natoms": len(atoms),
                        "input_role": "step0_unrelaxed_x0_geometry_only",
                    }
                )
    finally:
        env.close()

    metadata = pd.DataFrame(output_rows).sort_values(
        "material_id", kind="stable", ignore_index=True
    )
    if (
        metadata.material_id.duplicated().any()
        or metadata.parent_id.duplicated().any()
        or metadata.trajectory_stem.duplicated().any()
    ):
        raise ValueError("NEXT39 cohort identities are not unique")
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "source_name": source_name,
        "input_role": "omat24_records_containing_geometry_and_dft_values",
        "output_role": "step0_unrelaxed_x0_geometry_only",
        "raw_records_contain_dft_values": True,
        "identity_and_step_fields_parsed": True,
        "step0_geometry_parsed": True,
        "later_record_identity_recorded": True,
        "later_geometry_opened": False,
        "dft_numeric_fields_parsed": False,
        "dft_values_read": False,
        "selection_uses_later_geometry": False,
        "selection_uses_dft_values": False,
        "model_or_proxy_potential_used": False,
        "coordinates_or_cell_modified": False,
        "same_composition_candidates_used": False,
        "task_type": TASK_TYPE,
        "minimum_latest_step": minimum_latest_step,
        "salt": salt,
        "counts": {
            "raw_records": len(scanned),
            "selected_trajectories": len(metadata),
            "selected_parents": int(metadata.parent_id.nunique()),
            "selected_atoms": int(metadata.natoms.sum()),
        },
        "inputs_sha256": {"aselmdb": {"path": str(database), "sha256": database_hash}},
        "executed_source_sha256": {"src/next39_omat24_trajectory_cohort.py": source_hash},
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        metadata_path = staging / COHORT_NAME
        geometry_path = staging / GEOMETRY_NAME
        parent_path = staging / PARENT_NAME
        metadata.to_parquet(metadata_path, index=False)
        _write_deterministic_archive(geometry_path, frames)
        parent_path.write_text(
            "\n".join(sorted(metadata.parent_id.astype(str))) + "\n", encoding="utf-8"
        )
        manifest["outputs_sha256"] = {
            name: _sha256(staging / name)
            for name in (COHORT_NAME, GEOMETRY_NAME, PARENT_NAME)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(database) != database_hash or _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT39 cohort input or source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--minimum-latest-step", type=int, default=20)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    freeze_trajectory_cohort(
        db_path=args.db,
        source_name=args.source_name,
        salt=args.salt,
        minimum_latest_step=args.minimum_latest_step,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COHORT_NAME",
    "GEOMETRY_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "freeze_trajectory_cohort",
    "parse_trajectory_sid",
    "select_eligible_trajectories",
]
