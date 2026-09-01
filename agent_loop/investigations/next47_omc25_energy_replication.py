#!/usr/bin/env python3
"""Sealed second confirmation of the frozen NEXT31 OMC25 energy rule."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace
from src.next31_omc25_energy_evaluate import (
    ENERGY_POSITIVE_MIN,
    GATES,
    PROTECTED_MAX,
    energy_metrics,
)


PROTOCOL = "2026-08-03-next47-omc25-energy-replication-v1"
SOURCE_URL = (
    "https://dl.fbaipublicfiles.com/opencatalystproject/data/omc/250802/"
    "omc_val_250802.tar.gz"
)
ARCHIVE_SKIP_MAIN = 24
ARCHIVE_TAKE_MAIN = 16
EXPECTED_NEXT31_RULE_SHA256 = (
    "993d64b851c755fc5cc0d4b68ca7ca6994d4bdb7ed666f860d43a04925e254a8"
)
EXPECTED_START_EXCLUSION_SHA256 = (
    "f05c044297c2287ec18abf3a91bfe57ad3016f32b395f780d7768d0748e7ff3e"
)
EXPECTED_START_EXCLUSION_COUNT = 1732
PROTOCOL_NAME = "NEXT47_REPLICATION_PROTOCOL.json"
SUMMARY_NAME = "NEXT47_REPLICATION_SUMMARY.json"
REFCODES_NAME = "refcodes.txt"
MANIFEST_NAME = "MANIFEST.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(_safe_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_refcodes(path: Path) -> tuple[str, ...]:
    try:
        lines = path.read_text("utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("invalid refcode exclusion file") from exc
    values = tuple(line.strip() for line in lines if line.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("refcode exclusions must be unique and non-empty")
    return values


def _publish_json_directory(
    *, target: Path, output_name: str, value: object, manifest: dict[str, object]
) -> dict[str, object]:
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / output_name
        output_path.write_bytes(_json_bytes(value))
        manifest["outputs_sha256"] = {output_name: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def freeze_replication_protocol(
    *,
    frozen_rule_path: Path,
    exclusion_path: Path,
    output_dir: Path,
    expected_rule_sha256: str = EXPECTED_NEXT31_RULE_SHA256,
    expected_exclusion_sha256: str = EXPECTED_START_EXCLUSION_SHA256,
    expected_exclusion_count: int = EXPECTED_START_EXCLUSION_COUNT,
) -> dict[str, object]:
    """Bind the tail cohort, old rule, and old gates before raw records are read."""

    rule_path = Path(frozen_rule_path).resolve()
    refcode_path = Path(exclusion_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not rule_path.is_file() or not refcode_path.is_file():
        raise FileNotFoundError("frozen rule or starting exclusion file is missing")
    rule_hash = _sha256(rule_path)
    exclusion_hash = _sha256(refcode_path)
    exclusions = _read_refcodes(refcode_path)
    if rule_hash != expected_rule_sha256:
        raise ValueError("NEXT31 frozen rule hash differs")
    if exclusion_hash != expected_exclusion_sha256:
        raise ValueError("NEXT31 starting exclusion hash differs")
    if len(exclusions) != expected_exclusion_count:
        raise ValueError("NEXT31 starting exclusion count differs")
    source_path = Path(__file__).resolve()
    inputs = {
        "frozen_rule": {"path": str(rule_path), "sha256": rule_hash},
        "starting_refcodes": {
            "path": str(refcode_path),
            "sha256": exclusion_hash,
            "count": len(exclusions),
        },
    }
    protocol = {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "thresholds_refit": False,
        "source_url": SOURCE_URL,
        "archive_selection": {
            "skip_main": ARCHIVE_SKIP_MAIN,
            "take_main": ARCHIVE_TAKE_MAIN,
        },
        "cohort_selection": (
            "all 16 OMC25 val main members after the 24 members used by NEXT26-31, "
            "processed in archive order with cumulative CSD-refcode exclusion"
        ),
        "frozen_rule": inputs["frozen_rule"],
        "starting_refcodes": inputs["starting_refcodes"],
        "energy_positive_min_ev_per_atom": ENERGY_POSITIVE_MIN,
        "protected_max_ev_per_atom": PROTECTED_MAX,
        "gates": GATES,
        "second_cohort_must_pass_independently": True,
        "pooled_result_can_rescue_second_failure": False,
        "physical_never_read_lockbox": False,
        "execution_contract": (
            "one unrelaxed x0 plus frozen analytic tables and deterministic geometry only"
        ),
        "forbidden_at_execution": [
            "DFT values or calculations",
            "relaxed structures or trajectories",
            "MLIP or learned energy-force-stress proxies",
            "physical relaxation",
            "same-composition alternatives",
        ],
        "claim_boundary": (
            "OMC25 DFT relaxation-energy response only; not formation energy, "
            "convex-hull stability, thermodynamic stability, or replacement of DFT"
        ),
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "pre_record_and_pre_label_replication_freeze",
        "labels_opened": False,
        "inputs_sha256": inputs,
        "executed_source_sha256": {
            "src/next47_omc25_energy_replication.py": _sha256(source_path)
        },
    }
    result = _publish_json_directory(
        target=target,
        output_name=PROTOCOL_NAME,
        value=protocol,
        manifest=manifest,
    )
    if _sha256(rule_path) != rule_hash or _sha256(refcode_path) != exclusion_hash:
        raise RuntimeError("replication protocol input changed during publication")
    return result


def extend_refcode_exclusions(
    *,
    previous_refcodes_path: Path,
    metadata_path: Path,
    metadata_manifest_path: Path,
    source_shard: str,
    output_dir: Path,
) -> dict[str, object]:
    """Extend cumulative identities from a sealed label-free x0 cohort."""

    previous_path = Path(previous_refcodes_path).resolve()
    table_path = Path(metadata_path).resolve()
    upstream_path = Path(metadata_manifest_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in (previous_path, table_path, upstream_path)):
        raise FileNotFoundError("refcode extension input is missing")
    previous = _read_refcodes(previous_path)
    try:
        upstream = json.loads(upstream_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid x0 metadata manifest") from exc
    outputs = upstream.get("outputs_sha256")
    if (
        upstream.get("labels_opened") is not False
        or upstream.get("endpoint_numeric_fields_parsed") is not False
        or upstream.get("relaxed_structures_opened") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(table_path.name) != _sha256(table_path)
    ):
        raise ValueError("x0 metadata crossed the label-free boundary")
    metadata = pd.read_parquet(table_path, columns=["material_id", "csd_refcode"])
    if (
        metadata.empty
        or metadata["material_id"].isna().any()
        or metadata["material_id"].astype(str).duplicated().any()
        or metadata["csd_refcode"].isna().any()
        or metadata["csd_refcode"].astype(str).str.len().eq(0).any()
    ):
        raise ValueError("x0 identity metadata is invalid")
    new_refcodes = tuple(metadata["csd_refcode"].astype(str))
    union = tuple(sorted(set(previous) | set(new_refcodes)))
    input_hashes = {
        "previous_refcodes": {"path": str(previous_path), "sha256": _sha256(previous_path)},
        "metadata": {"path": str(table_path), "sha256": _sha256(table_path)},
        "metadata_manifest": {"path": str(upstream_path), "sha256": _sha256(upstream_path)},
    }
    source_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "cumulative_label_free_refcode_exclusion",
        "source_shard": str(source_shard),
        "labels_opened": False,
        "endpoint_fields_read": False,
        "counts": {
            "previous": len(previous),
            "new_rows": len(metadata),
            "union": len(union),
        },
        "inputs_sha256": input_hashes,
        "executed_source_sha256": {
            "src/next47_omc25_energy_replication.py": _sha256(source_path)
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / REFCODES_NAME
        output_path.write_text("".join(f"{value}\n" for value in union), encoding="utf-8")
        manifest["outputs_sha256"] = {REFCODES_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for value in input_hashes.values():
            assert isinstance(value, Mapping)
            if _sha256(Path(str(value["path"]))) != value["sha256"]:
                raise RuntimeError("refcode extension input changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def _joined_metrics(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "material_id",
        "source_shard",
        "analytic_supported",
        "next31_risk_score",
        "reject",
        "energy_drop_pa",
    }
    if not required.issubset(frame.columns):
        raise ValueError("joined replication table lacks required columns")
    if frame.empty or frame["material_id"].isna().any() or frame["material_id"].astype(str).duplicated().any():
        raise ValueError("joined replication identities are invalid")
    return energy_metrics(
        energy=frame["energy_drop_pa"].to_numpy(float),
        supported=frame["analytic_supported"].to_numpy(bool),
        reject=frame["reject"].to_numpy(bool),
        score=frame["next31_risk_score"].to_numpy(float),
    )


def summarize_replications(
    *,
    first_joined_path: Path,
    second_joined_path: Path,
    frozen_rule_path: Path,
    output_dir: Path,
    expected_rule_sha256: str = EXPECTED_NEXT31_RULE_SHA256,
    verify_rule_file: bool = True,
) -> dict[str, object]:
    """Recompute first, second, and pooled metrics with no fitting or rescue rule."""

    first_path = Path(first_joined_path).resolve()
    second_path = Path(second_joined_path).resolve()
    rule_path = Path(frozen_rule_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if not first_path.is_file() or not second_path.is_file():
        raise FileNotFoundError("joined replication input is missing")
    if verify_rule_file:
        if not rule_path.is_file():
            raise FileNotFoundError(str(rule_path))
        if _sha256(rule_path) != expected_rule_sha256:
            raise ValueError("NEXT31 frozen rule hash differs")
    first = pd.read_parquet(first_path)
    second = pd.read_parquet(second_path)
    first_metrics = _joined_metrics(first)
    second_metrics = _joined_metrics(second)
    first_ids = set(first["material_id"].astype(str))
    second_ids = set(second["material_id"].astype(str))
    if first_ids & second_ids:
        raise ValueError("first and second confirmation identities overlap")
    pooled = pd.concat([first, second], ignore_index=True)
    pooled_metrics = _joined_metrics(pooled)
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "opened_labels_used_for_evaluation_only": True,
        "thresholds_refit": False,
        "features_added": False,
        "second_confirmation_independent": True,
        "second_confirmation_pass": bool(second_metrics["prospective_gate_pass"]),
        "pooled_is_descriptive_only": True,
        "pooled_can_rescue_second_failure": False,
        "frozen_rule_sha256": expected_rule_sha256,
        "first_confirmation": first_metrics,
        "second_confirmation": second_metrics,
        "pooled": pooled_metrics,
        "cohort_rows": {"first": len(first), "second": len(second), "pooled": len(pooled)},
        "identity_overlap": 0,
        "claim_boundary": (
            "same-source OMC25 DFT relaxation-energy response only; not external "
            "transfer, convex-hull stability, thermodynamic stability, or DFT replacement"
        ),
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "opened_label_no_refit_replication_summary",
        "labels_opened": True,
        "inputs_sha256": {
            "first_joined": {"path": str(first_path), "sha256": _sha256(first_path)},
            "second_joined": {"path": str(second_path), "sha256": _sha256(second_path)},
            "frozen_rule": {
                "path": str(rule_path),
                "sha256": expected_rule_sha256,
                "verified": verify_rule_file,
            },
        },
        "executed_source_sha256": {
            "src/next47_omc25_energy_replication.py": _sha256(Path(__file__).resolve())
        },
    }
    return_value = _publish_json_directory(
        target=target,
        output_name=SUMMARY_NAME,
        value=result,
        manifest=manifest,
    )
    del return_value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-protocol")
    freeze.add_argument("--frozen-rule", required=True, type=Path)
    freeze.add_argument("--starting-refcodes", required=True, type=Path)
    freeze.add_argument("--output-dir", required=True, type=Path)
    extend = subparsers.add_parser("extend-refcodes")
    extend.add_argument("--previous-refcodes", required=True, type=Path)
    extend.add_argument("--metadata", required=True, type=Path)
    extend.add_argument("--metadata-manifest", required=True, type=Path)
    extend.add_argument("--source-shard", required=True)
    extend.add_argument("--output-dir", required=True, type=Path)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--first-joined", required=True, type=Path)
    summary.add_argument("--second-joined", required=True, type=Path)
    summary.add_argument("--frozen-rule", required=True, type=Path)
    summary.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-protocol":
        result = freeze_replication_protocol(
            frozen_rule_path=args.frozen_rule,
            exclusion_path=args.starting_refcodes,
            output_dir=args.output_dir,
        )
    elif args.command == "extend-refcodes":
        result = extend_refcode_exclusions(
            previous_refcodes_path=args.previous_refcodes,
            metadata_path=args.metadata,
            metadata_manifest_path=args.metadata_manifest,
            source_shard=args.source_shard,
            output_dir=args.output_dir,
        )
    else:
        result = summarize_replications(
            first_joined_path=args.first_joined,
            second_joined_path=args.second_joined,
            frozen_rule_path=args.frozen_rule,
            output_dir=args.output_dir,
        )
    print(json.dumps(_safe_json(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_NEXT31_RULE_SHA256",
    "EXPECTED_START_EXCLUSION_SHA256",
    "EXPECTED_START_EXCLUSION_COUNT",
    "extend_refcode_exclusions",
    "freeze_replication_protocol",
    "summarize_replications",
]
