#!/usr/bin/env python3
"""Evaluate checksum-frozen NEXT28 decisions after OMC25 endpoint opening."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
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
from src.next27_evaluate import _continuous_diagnostics
from src.next28_contact_coordination import APPLICATION_PROTOCOL, PREDICTIONS_NAME


PROTOCOL = "2026-08-03-next28-omc25-dft-response-evaluation-v1"
RESULT_NAME = "NEXT28_DFT_RESPONSE_EVALUATION.json"
JOINED_NAME = "next28_joined.parquet"
LABEL_OPENING_NAME = "LABEL_OPENING.json"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid prediction manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("prediction manifest must be an object")
    return value


def _passes(metrics: Mapping[str, object]) -> bool:
    return all(float(metrics[name]) >= cutoff for name, cutoff in PROSPECTIVE_GATES.items())


def evaluate_predictions(
    *, predictions_path: Path, prediction_manifest_path: Path, endpoints_path: Path, output_dir: Path
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "predictions": Path(predictions_path).resolve(),
        "prediction_manifest": Path(prediction_manifest_path).resolve(),
        "endpoints": Path(endpoints_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role}: {path}")
    hashes = {role: _sha256(path) for role, path in paths.items()}
    manifest = _read_json(paths["prediction_manifest"])
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != APPLICATION_PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_fields_read") is not False
        or manifest.get("relaxed_structures_opened") is not False
        or manifest.get("threshold_refit") is not False
        or manifest.get("frozen_at_utc") is None
        or not isinstance(outputs, Mapping)
        or outputs.get(PREDICTIONS_NAME) != hashes["predictions"]
    ):
        raise ValueError("predictions are not an eligible NEXT28 pre-opening freeze")
    predictions = pd.read_parquet(paths["predictions"])
    endpoints = pd.read_parquet(paths["endpoints"])
    required = {"material_id", "source_shard", "analytic_supported", "next28_risk_score", "reject"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"prediction columns are incomplete: {sorted(required-set(predictions.columns))}")
    for frame, role in ((predictions, "predictions"), (endpoints, "endpoints")):
        if "material_id" not in frame or frame.material_id.isna().any() or frame.material_id.duplicated().any():
            raise ValueError(f"{role} material IDs are invalid")
    if set(predictions.material_id.astype(str)) != set(endpoints.material_id.astype(str)):
        raise ValueError("prediction and endpoint identities differ")
    if "source_shard" in endpoints:
        check = predictions[["material_id", "source_shard"]].merge(
            endpoints[["material_id", "source_shard"]], on="material_id", suffixes=("_p", "_e"), validate="one_to_one"
        )
        if not (check.source_shard_p.astype(str) == check.source_shard_e.astype(str)).all():
            raise ValueError("prediction and endpoint source shards differ")
        endpoints = endpoints.drop(columns="source_shard")
    joined = predictions.merge(endpoints, on="material_id", validate="one_to_one")
    joined = joined.sort_values("material_id", kind="stable").reset_index(drop=True)
    positive = severe_dft_response(joined).to_numpy(bool)
    analytic = decision_metrics(
        supported=joined.analytic_supported.to_numpy(bool),
        reject=joined.reject.to_numpy(bool),
        endpoint_positive=positive,
    )
    analytic["passes_prospective_gates"] = _passes(analytic)
    by_shard: dict[str, dict[str, object]] = {}
    shard_values = joined.source_shard.astype(str).to_numpy()
    for shard in sorted(set(shard_values)):
        mask = shard_values == shard
        by_shard[shard] = decision_metrics(
            supported=joined.loc[mask, "analytic_supported"].to_numpy(bool),
            reject=joined.loc[mask, "reject"].to_numpy(bool),
            endpoint_positive=positive[mask],
        )
    diagnostics_frame = joined.rename(columns={"next28_risk_score": "next27_risk_score"})
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "prospective_gates": dict(PROSPECTIVE_GATES),
        "analytic": analytic,
        "by_source_shard": by_shard,
        "continuous_diagnostics": _continuous_diagnostics(diagnostics_frame, positive),
        "passes_all_prospective_gates": bool(analytic["passes_prospective_gates"]),
        "beyond_pauling_or_dft_claim": False,
        "claim_boundary": "substantial DFT response, not thermodynamic instability",
    }
    joined["endpoint_positive"] = positive
    opened_at = datetime.now(timezone.utc).isoformat()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        joined_path = staging / JOINED_NAME
        result_path = staging / RESULT_NAME
        opening_path = staging / LABEL_OPENING_NAME
        joined.to_parquet(joined_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        opening_path.write_bytes(_json_bytes({
            "protocol": PROTOCOL,
            "opened_at_utc": opened_at,
            "prediction_frozen_at_utc": manifest["frozen_at_utc"],
            "prediction_sha256": hashes["predictions"],
            "endpoint_sha256": hashes["endpoints"],
            "prediction_preceded_label_opening": True,
            "physical_never_read_lockbox": False,
        }))
        source_hashes = {"src/next28_evaluate.py": _sha256(Path(__file__).resolve())}
        publication = {
            "protocol": PROTOCOL,
            "labels_opened": True,
            "opened_at_utc": opened_at,
            "inputs_sha256": {role: {"path": str(path), "sha256": hashes[role]} for role, path in paths.items()},
            "outputs_sha256": {
                JOINED_NAME: _sha256(joined_path),
                RESULT_NAME: _sha256(result_path),
                LABEL_OPENING_NAME: _sha256(opening_path),
            },
            "executed_source_sha256": source_hashes,
            "passes_all_prospective_gates": bool(analytic["passes_prospective_gates"]),
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(publication))
        for role, path in paths.items():
            if _sha256(path) != hashes[role]:
                raise RuntimeError(f"input {role} changed during evaluation")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--prediction-manifest", required=True, type=Path)
    parser.add_argument("--endpoints", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_predictions(
        predictions_path=args.predictions,
        prediction_manifest_path=args.prediction_manifest,
        endpoints_path=args.endpoints,
        output_dir=args.output_dir,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RESULT_NAME", "evaluate_predictions"]
