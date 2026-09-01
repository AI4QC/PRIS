#!/usr/bin/env python3
"""Freeze and apply the one-term NEXT28 periodic contact-coordination law."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next26_evaluate import decision_metrics
from src.next26_omc25 import severe_dft_response
from src.next27_development import PROSPECTIVE_GATES


PROTOCOL = "2026-08-03-next28-fixed-contact-coordination-law-v1"
APPLICATION_PROTOCOL = "2026-08-03-next28-label-free-application-v1"
FROZEN_RULE_NAME = "FROZEN_RULE.json"
PREDICTIONS_NAME = "next28_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"
FEATURE = "periodic_contact_coord105"
THRESHOLD = 6.3


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _passes(metrics: Mapping[str, object]) -> bool:
    return all(float(metrics[name]) >= cutoff for name, cutoff in PROSPECTIVE_GATES.items())


def freeze_rule(
    *, features: pd.DataFrame, endpoints: pd.DataFrame, output_dir: Path
) -> dict[str, object]:
    """Audit the predeclared 6.3 threshold and freeze without threshold search."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    required = {"material_id", "development_shard", FEATURE, "analytic_supported"}
    if not required.issubset(features.columns):
        raise ValueError(f"features lack columns: {sorted(required-set(features.columns))}")
    for frame, role in ((features, "features"), (endpoints, "endpoints")):
        if (
            "material_id" not in frame
            or frame["material_id"].isna().any()
            or frame["material_id"].duplicated().any()
        ):
            raise ValueError(f"{role} material IDs are invalid")
    if set(features["material_id"].astype(str)) != set(endpoints["material_id"].astype(str)):
        raise ValueError("development feature and endpoint IDs differ")
    if features["development_shard"].nunique() < 6:
        raise ValueError("six distinct development shards are required")
    merged = features.merge(endpoints, on="material_id", how="inner", validate="one_to_one")
    score = pd.to_numeric(merged[FEATURE], errors="coerce").to_numpy(float)
    declared_support = merged["analytic_supported"].to_numpy(bool)
    support = declared_support & np.isfinite(score)
    reject = support & (score >= THRESHOLD)
    positive = severe_dft_response(merged).to_numpy(bool)
    aggregate = decision_metrics(supported=support, reject=reject, endpoint_positive=positive)
    aggregate["passes_prospective_gates"] = _passes(aggregate)
    by_shard: dict[str, dict[str, object]] = {}
    stable_shards = 0
    for shard in sorted(merged["development_shard"].astype(str).unique()):
        mask = merged["development_shard"].astype(str).to_numpy() == shard
        metrics = decision_metrics(
            supported=support[mask], reject=reject[mask], endpoint_positive=positive[mask]
        )
        by_shard[shard] = metrics
        if int(metrics["rejected"]) >= 5:
            stable_shards += 1
            if (
                float(metrics["endpoint_positive_precision"]) < 0.70
                or float(metrics["endpoint_negative_protection"]) < 0.95
            ):
                stable_shards = -10_000
    eligible = bool(aggregate["passes_prospective_gates"] and stable_shards >= 4)
    frozen_at = datetime.now(timezone.utc).isoformat()
    rule: dict[str, object] = {
        "protocol": PROTOCOL,
        "eligible": eligible,
        "frozen_at_utc": frozen_at,
        "selected_candidate": "periodic_contact_coordination_105",
        "feature": FEATURE,
        "threshold": THRESHOLD,
        "formula": f"reject iff {FEATURE} >= {THRESHOLD}",
        "maximum_terms": 1,
        "threshold_search_performed": False,
        "threshold_refit": False,
        "development_metrics": aggregate,
        "development_shard_metrics": by_shard,
        "prospective_gates": dict(PROSPECTIVE_GATES),
        "missing_policy": "fail_open_do_not_reject",
        "model_or_proxy_potential_used": False,
        "prospective_labels_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        rule_path = staging / FROZEN_RULE_NAME
        rule_path.write_bytes(_json_bytes(rule))
        source_hashes = {"src/next28_contact_coordination.py": _sha256(Path(__file__).resolve())}
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "eligible": eligible,
            "frozen_at_utc": frozen_at,
            "development_labels_opened": True,
            "prospective_labels_opened": False,
            "development_shards": sorted(by_shard),
            "counts": {"rows": len(merged), "stable_shards": stable_shards},
            "outputs_sha256": {FROZEN_RULE_NAME: _sha256(rule_path)},
            "executed_source_sha256": source_hashes,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"eligible": eligible, "rule": rule, "manifest": manifest}


def apply_frozen_rule(
    *,
    frozen_rule_path: Path,
    rule_manifest_path: Path,
    features: pd.DataFrame,
    output_dir: Path,
    feature_source_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Apply NEXT28 to x0-derived features without any endpoint access."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    rule_path = Path(frozen_rule_path).resolve()
    manifest_path = Path(rule_manifest_path).resolve()
    rule = _read_json(rule_path, "frozen rule")
    rule_manifest = _read_json(manifest_path, "rule manifest")
    outputs = rule_manifest.get("outputs_sha256")
    if (
        rule.get("protocol") != PROTOCOL
        or rule.get("eligible") is not True
        or rule.get("prospective_labels_opened") is not False
        or rule.get("feature") != FEATURE
        or rule.get("threshold") != THRESHOLD
        or rule_manifest.get("protocol") != PROTOCOL
        or rule_manifest.get("eligible") is not True
        or not isinstance(outputs, Mapping)
        or outputs.get(FROZEN_RULE_NAME) != _sha256(rule_path)
    ):
        raise ValueError("NEXT28 frozen rule is ineligible")
    if "material_id" not in features or features["material_id"].isna().any() or features["material_id"].duplicated().any():
        raise ValueError("prospective feature IDs are invalid")
    if FEATURE not in features:
        raise ValueError(f"prospective features lack {FEATURE}")
    score = pd.to_numeric(features[FEATURE], errors="coerce").to_numpy(float)
    declared = features.get("analytic_supported", pd.Series(True, index=features.index)).to_numpy(bool)
    support = declared & np.isfinite(score)
    reject = support & (score >= THRESHOLD)
    data: dict[str, object] = {
        "material_id": features["material_id"].astype(str),
        "analytic_supported": support,
        "next28_risk_score": score,
        "reject": reject,
        "input_role": "unrelaxed_x0_geometry_only",
    }
    if "source_shard" in features:
        data["source_shard"] = features["source_shard"].astype(str)
    predictions = pd.DataFrame(data)
    sources = [Path(path).resolve() for path in feature_source_paths]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    frozen_at = datetime.now(timezone.utc).isoformat()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        prediction_path = staging / PREDICTIONS_NAME
        predictions.to_parquet(prediction_path, index=False)
        application_manifest: dict[str, object] = {
            "protocol": APPLICATION_PROTOCOL,
            "frozen_at_utc": frozen_at,
            "rule_protocol": PROTOCOL,
            "threshold": THRESHOLD,
            "threshold_refit": False,
            "labels_opened": False,
            "endpoint_fields_read": False,
            "relaxed_structures_opened": False,
            "model_or_proxy_potential_used": False,
            "counts": {"rows": len(predictions), "supported": int(support.sum()), "rejected": int(reject.sum())},
            "inputs_sha256": {
                "frozen_rule": {"path": str(rule_path), "sha256": _sha256(rule_path)},
                "rule_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
                "feature_sources": [{"path": str(path), "sha256": _sha256(path)} for path in sources],
            },
            "outputs_sha256": {PREDICTIONS_NAME: _sha256(prediction_path)},
            "executed_source_sha256": {"src/next28_contact_coordination.py": _sha256(Path(__file__).resolve())},
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(application_manifest))
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return application_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--features", nargs="+", required=True, type=Path)
    freeze.add_argument("--endpoints", required=True, type=Path)
    freeze.add_argument("--shards", nargs="+", required=True)
    freeze.add_argument("--output-dir", required=True, type=Path)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--frozen-rule", required=True, type=Path)
    apply.add_argument("--rule-manifest", required=True, type=Path)
    apply.add_argument("--features", nargs="+", required=True, type=Path)
    apply.add_argument("--shards", nargs="+", required=True)
    apply.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.features) != len(args.shards):
        raise ValueError("features and shards must have equal lengths")
    parts: list[pd.DataFrame] = []
    for path, shard in zip(args.features, args.shards, strict=True):
        frame = pd.read_parquet(path)
        frame["development_shard" if args.command == "freeze" else "source_shard"] = str(shard)
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    if args.command == "freeze":
        result = freeze_rule(features=combined, endpoints=pd.read_parquet(args.endpoints), output_dir=args.output_dir)
    else:
        result = apply_frozen_rule(
            frozen_rule_path=args.frozen_rule,
            rule_manifest_path=args.rule_manifest,
            features=combined,
            feature_source_paths=args.features,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FROZEN_RULE_NAME", "PREDICTIONS_NAME", "apply_frozen_rule", "freeze_rule"]
