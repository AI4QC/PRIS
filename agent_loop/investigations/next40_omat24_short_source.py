#!/usr/bin/env python3
"""Build an opaque, parent-disjoint OMat24 short-trajectory raw source."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zlib

import lmdb

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next32_omat24_cohort import _project_identity
from src.next39_omat24_trajectory_cohort import TASK_TYPE, parse_trajectory_sid


PROTOCOL = "2026-08-03-next40-omat24-short-horizon-source-v1"
FILTERED_DB_NAME = "short-trajectories.aselmdb"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_excluded(path: Path) -> frozenset[str]:
    try:
        values = [line.strip() for line in path.read_text("utf-8").splitlines()]
    except (OSError, UnicodeError) as exc:
        raise ValueError("invalid excluded-parent file") from exc
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("excluded-parent identities are empty or duplicated")
    return frozenset(values)


def _select_short_trajectories(
    rows: Sequence[Mapping[str, object]],
    *,
    excluded_parents: frozenset[str],
    salt: str,
    minimum_latest_step: int,
    maximum_latest_step: int,
) -> list[dict[str, object]]:
    if (
        not salt
        or type(minimum_latest_step) is not int
        or type(maximum_latest_step) is not int
        or minimum_latest_step < 1
        or maximum_latest_step < minimum_latest_step
    ):
        raise ValueError("NEXT40 horizon settings are invalid")
    groups: dict[str, dict[str, object]] = {}
    seen_sids: set[str] = set()
    seen_keys: set[int] = set()
    for raw in rows:
        sid = str(raw["sid"])
        parent = str(raw["parent_id"])
        task = str(raw["task_type"])
        record_key = int(raw["record_key"])
        if sid in seen_sids or record_key in seen_keys:
            raise ValueError("duplicate OMat24 identity in NEXT40 source")
        seen_sids.add(sid)
        seen_keys.add(record_key)
        stem, step = parse_trajectory_sid(sid)
        entry = groups.setdefault(
            stem, {"parent_id": parent, "task_type": task, "steps": {}}
        )
        if entry["parent_id"] != parent or entry["task_type"] != task:
            raise ValueError("OMat24 trajectory identity changes across steps")
        steps = entry["steps"]
        assert isinstance(steps, dict)
        if step in steps:
            raise ValueError("duplicate OMat24 trajectory step")
        steps[step] = record_key

    candidates: list[dict[str, object]] = []
    for stem, entry in groups.items():
        parent = str(entry["parent_id"])
        steps = entry["steps"]
        assert isinstance(steps, dict)
        if (
            parent in excluded_parents
            or entry["task_type"] != TASK_TYPE
            or 0 not in steps
        ):
            continue
        latest = max(int(step) for step in steps)
        if latest < minimum_latest_step or latest > maximum_latest_step:
            continue
        candidates.append(
            {
                "trajectory_stem": stem,
                "parent_id": parent,
                "latest_step": latest,
                "record_keys": tuple(int(steps[step]) for step in sorted(steps)),
            }
        )

    by_parent: dict[str, tuple[str, dict[str, object]]] = {}
    for candidate in candidates:
        parent = str(candidate["parent_id"])
        digest = hashlib.sha256(
            f"{salt}|{parent}|{candidate['trajectory_stem']}".encode()
        ).hexdigest()
        previous = by_parent.get(parent)
        if previous is None or digest < previous[0]:
            by_parent[parent] = (digest, candidate)
    return [
        dict(row)
        for _digest, row in sorted(
            by_parent.values(), key=lambda item: str(item[1]["trajectory_stem"])
        )
    ]


def build_short_horizon_source(
    *,
    db_path: Path,
    excluded_parent_ids_path: Path,
    salt: str,
    minimum_latest_step: int,
    maximum_latest_step: int,
    output_dir: Path,
) -> dict[str, object]:
    """Copy selected compressed records without decoding geometry or DFT values."""

    source = Path(db_path).resolve()
    excluded_path = Path(excluded_parent_ids_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not source.is_file() or not excluded_path.is_file():
        raise FileNotFoundError("NEXT40 source input is missing")
    input_hashes = {"aselmdb": _sha256(source), "excluded_parents": _sha256(excluded_path)}
    excluded = _read_excluded(excluded_path)

    source_env = lmdb.open(
        str(source), subdir=False, readonly=True, lock=False, readahead=False
    )
    scanned: list[dict[str, object]] = []
    try:
        with source_env.begin() as transaction:
            for key, compressed in transaction.cursor():
                if key in {b"nextid", b"metadata", b"deleted_ids"}:
                    continue
                try:
                    record_key = int(key)
                    payload = zlib.decompress(compressed)
                except (ValueError, zlib.error) as exc:
                    raise ValueError(f"invalid OMat24 LMDB record {key!r}") from exc
                scanned.append({**_project_identity(payload), "record_key": record_key})
        selected = _select_short_trajectories(
            scanned,
            excluded_parents=excluded,
            salt=salt,
            minimum_latest_step=minimum_latest_step,
            maximum_latest_step=maximum_latest_step,
        )
        if not selected:
            raise ValueError("no eligible NEXT40 short trajectories")
        selected_keys = sorted(
            {key for row in selected for key in row["record_keys"]}
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
        )
        try:
            database_path = staging / FILTERED_DB_NAME
            map_size = max(16 * 1024 * 1024, source.stat().st_size)
            output_env = lmdb.open(str(database_path), subdir=False, map_size=map_size)
            try:
                with source_env.begin() as source_transaction, output_env.begin(
                    write=True
                ) as output_transaction:
                    for record_key in selected_keys:
                        key = str(record_key).encode()
                        compressed = source_transaction.get(key)
                        if compressed is None:
                            raise ValueError(f"missing selected record {record_key}")
                        output_transaction.put(key, compressed)
                    output_transaction.put(
                        b"nextid", zlib.compress(str(max(selected_keys) + 1).encode())
                    )
            finally:
                output_env.close()

            source_path = Path(__file__).resolve()
            source_hash = _sha256(source_path)
            manifest: dict[str, object] = {
                "protocol": PROTOCOL,
                "evidence_role": "adaptive_parent_disjoint_short_horizon_raw_source",
                "adaptation_trigger": "NEXT39 had no protected endpoint examples",
                "selection_fields": ["sid", "parent_id", "task_type", "trajectory_step"],
                "excluded_parent_ids_sha256": input_hashes["excluded_parents"],
                "minimum_latest_step": minimum_latest_step,
                "maximum_latest_step": maximum_latest_step,
                "salt": salt,
                "raw_payloads_copied_opaque": True,
                "later_geometry_opened": False,
                "step0_geometry_opened": False,
                "dft_values_read": False,
                "selection_uses_geometry": False,
                "selection_uses_dft_values": False,
                "counts": {
                    "raw_records_scanned": len(scanned),
                    "excluded_parents": len(excluded),
                    "selected_trajectories": len(selected),
                    "selected_parents": len({str(row["parent_id"]) for row in selected}),
                    "copied_records": len(selected_keys),
                },
                "latest_step_counts": {
                    str(step): sum(int(row["latest_step"]) == step for row in selected)
                    for step in range(minimum_latest_step, maximum_latest_step + 1)
                },
                "inputs_sha256": input_hashes,
                "executed_source_sha256": {
                    "src/next40_omat24_short_source.py": source_hash
                },
                "outputs_sha256": {FILTERED_DB_NAME: _sha256(database_path)},
                "scientific_improvement_claim": False,
            }
            (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
            if (
                _sha256(source) != input_hashes["aselmdb"]
                or _sha256(excluded_path) != input_hashes["excluded_parents"]
                or _sha256(source_path) != source_hash
            ):
                raise RuntimeError("NEXT40 source input or code changed during publication")
            _publish_directory_no_replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    finally:
        source_env.close()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--excluded-parent-ids", required=True, type=Path)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--minimum-latest-step", type=int, default=1)
    parser.add_argument("--maximum-latest-step", type=int, default=19)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    build_short_horizon_source(
        db_path=args.db,
        excluded_parent_ids_path=args.excluded_parent_ids,
        salt=args.salt,
        minimum_latest_step=args.minimum_latest_step,
        maximum_latest_step=args.maximum_latest_step,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FILTERED_DB_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "build_short_horizon_source",
]
