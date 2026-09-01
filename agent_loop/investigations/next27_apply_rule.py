#!/usr/bin/env python3
"""Apply frozen NEXT27 periodic pressure law before opening DFT labels."""

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
from src.next26_development import score_rule
from src.next27_development import PROTOCOL as RULE_PROTOCOL


PROTOCOL = "2026-08-03-next27-periodic-pressure-label-free-application-v1"
PREDICTIONS_NAME = "next27_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


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


def _rule(rule_path: Path, manifest_path: Path) -> dict[str, object]:
    rule = _read_json(rule_path, "NEXT27 frozen rule")
    manifest = _read_json(manifest_path, "NEXT27 rule manifest")
    outputs = manifest.get("outputs_sha256")
    if (
        rule.get("protocol") != RULE_PROTOCOL
        or rule.get("eligible") is not True
        or rule.get("prospective_labels_opened") is not False
        or rule.get("model_or_proxy_potential_used") is not False
        or manifest.get("protocol") != RULE_PROTOCOL
        or manifest.get("eligible") is not True
        or manifest.get("prospective_labels_opened") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(rule_path.name) != _sha256(rule_path)
    ):
        raise ValueError("NEXT27 frozen rule is ineligible")
    if (
        not isinstance(rule.get("terms"), list)
        or not 1 <= len(rule["terms"]) <= 2  # type: ignore[arg-type]
        or not isinstance(rule.get("parameters"), Mapping)
        or not isinstance(rule.get("threshold"), (int, float))
        or not math.isfinite(float(rule["threshold"]))
    ):
        raise ValueError("NEXT27 frozen rule schema is invalid")
    return rule


def apply_frozen_rule(
    *,
    frozen_rule_path: Path,
    rule_manifest_path: Path,
    features: pd.DataFrame,
    output_dir: Path,
    feature_source_paths: Sequence[Path] = (),
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    rule_path = Path(frozen_rule_path).resolve()
    manifest_path = Path(rule_manifest_path).resolve()
    rule = _rule(rule_path, manifest_path)
    features = features.copy()
    if "material_id" not in features or features["material_id"].isna().any() or features["material_id"].duplicated().any():
        raise ValueError("prospective feature IDs are invalid")
    terms = rule["terms"]
    parameters = rule["parameters"]
    assert isinstance(terms, list) and isinstance(parameters, Mapping)
    score, support = score_rule(
        features,
        terms=terms,
        parameters=parameters,
        formula_family=str(rule["formula_family"]),
    )
    threshold = float(rule["threshold"])
    reject = support & (score >= threshold)
    data: dict[str, object] = {
        "material_id": features["material_id"].astype(str),
        "analytic_supported": support,
        "next27_risk_score": score,
        "reject": reject,
        "input_role": "unrelaxed_x0_geometry_only",
    }
    if "source_shard" in features:
        data["source_shard"] = features["source_shard"].astype(str)
    predictions = pd.DataFrame(data)
    source_paths = [Path(path).resolve() for path in feature_source_paths]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"feature source is not a file: {path}")
    inputs: dict[str, object] = {
        "frozen_rule": {"path": str(rule_path), "sha256": _sha256(rule_path)},
        "rule_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "feature_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in source_paths
        ],
    }
    frozen_at = datetime.now(timezone.utc).isoformat()
    source_hashes = {"src/next27_apply_rule.py": _sha256(Path(__file__).resolve())}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "frozen_at_utc": frozen_at,
        "rule_protocol": RULE_PROTOCOL,
        "selected_candidate": rule.get("selected_candidate"),
        "formula_family": rule.get("formula_family"),
        "terms": terms,
        "threshold": threshold,
        "threshold_refit": False,
        "labels_opened": False,
        "endpoint_fields_read": False,
        "relaxed_structures_opened": False,
        "model_or_proxy_potential_used": False,
        "same_composition_candidates_used": False,
        "missing_policy": "fail_open_do_not_reject",
        "counts": {
            "rows": len(predictions),
            "supported": int(support.sum()),
            "rejected": int(reject.sum()),
        },
        "inputs_sha256": inputs,
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        predictions_path = staging / PREDICTIONS_NAME
        predictions.to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {PREDICTIONS_NAME: _sha256(predictions_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(rule_path) != inputs["frozen_rule"]["sha256"]:  # type: ignore[index]
            raise RuntimeError("frozen rule changed before publication")
        for source, record in zip(source_paths, inputs["feature_sources"], strict=True):  # type: ignore[arg-type]
            if _sha256(source) != record["sha256"]:
                raise RuntimeError("feature source changed before publication")
        if _sha256(Path(__file__).resolve()) != source_hashes["src/next27_apply_rule.py"]:
            raise RuntimeError("NEXT27 application source changed")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-rule", required=True, type=Path)
    parser.add_argument("--rule-manifest", required=True, type=Path)
    parser.add_argument("--features", nargs="+", required=True, type=Path)
    parser.add_argument("--shards", nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.shards is not None and len(args.shards) != len(args.features):
        raise ValueError("shards and features must have equal lengths")
    parts: list[pd.DataFrame] = []
    for index, path in enumerate(args.features):
        frame = pd.read_parquet(path)
        if args.shards is not None:
            frame["source_shard"] = str(args.shards[index])
        parts.append(frame)
    result = apply_frozen_rule(
        frozen_rule_path=args.frozen_rule,
        rule_manifest_path=args.rule_manifest,
        features=pd.concat(parts, ignore_index=True),
        feature_source_paths=args.features,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PREDICTIONS_NAME", "apply_frozen_rule"]
