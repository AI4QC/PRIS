#!/usr/bin/env python3
"""Evaluate checksum-frozen NEXT27 decisions after OMC25 endpoint opening."""

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
from scipy.stats import spearmanr

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next23_evaluate import _roc_auc
from src.next26_evaluate import decision_metrics
from src.next26_omc25 import severe_dft_response
from src.next27_development import PROSPECTIVE_GATES


PROTOCOL = "2026-08-03-next27-omc25-dft-response-evaluation-v1"
RESULT_NAME = "NEXT27_DFT_RESPONSE_EVALUATION.json"
JOINED_NAME = "next27_joined.parquet"
LABEL_OPENING_NAME = "LABEL_OPENING.json"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _passes(metrics: Mapping[str, object]) -> bool:
    return all(float(metrics[name]) >= cutoff for name, cutoff in PROSPECTIVE_GATES.items())


def _continuous_diagnostics(
    joined: pd.DataFrame, endpoint_positive: np.ndarray
) -> dict[str, object]:
    score = pd.to_numeric(joined["next27_risk_score"], errors="coerce").to_numpy(float)
    finite = joined["analytic_supported"].to_numpy(bool) & np.isfinite(score)
    result: dict[str, object] = {
        "endpoint_auc": _roc_auc(score[finite], endpoint_positive[finite]) if finite.any() else None
    }
    endpoints = (
        "force0_max",
        "force0_rms",
        "energy_drop_pa",
        "stress0_norm",
        "disp_p90",
        "cell_logstrain_max",
    )
    for endpoint in endpoints:
        if endpoint not in joined or finite.sum() < 3:
            continue
        endpoint_values = joined.loc[finite, endpoint].to_numpy(float)
        if np.ptp(endpoint_values) == 0.0 or np.ptp(score[finite]) == 0.0:
            result[f"risk_spearman_{endpoint}"] = None
            continue
        correlation = float(
            spearmanr(score[finite], endpoint_values).statistic
        )
        result[f"risk_spearman_{endpoint}"] = correlation if math.isfinite(correlation) else None
    return result


def evaluate_predictions(
    *,
    predictions_path: Path,
    prediction_manifest_path: Path,
    endpoints_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Open endpoints only after validating the immutable label-free prediction freeze."""

    from src.next27_apply_rule import PREDICTIONS_NAME, PROTOCOL as PREDICTION_PROTOCOL

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
            raise FileNotFoundError(f"{role} is not a file: {path}")
    hashes = {role: _sha256(path) for role, path in paths.items()}
    prediction_manifest = _read_json(paths["prediction_manifest"], role="prediction manifest")
    outputs = prediction_manifest.get("outputs_sha256")
    if (
        prediction_manifest.get("protocol") != PREDICTION_PROTOCOL
        or prediction_manifest.get("labels_opened") is not False
        or prediction_manifest.get("endpoint_fields_read") is not False
        or prediction_manifest.get("relaxed_structures_opened") is not False
        or prediction_manifest.get("frozen_at_utc") is None
        or not isinstance(outputs, Mapping)
        or outputs.get(PREDICTIONS_NAME) != hashes["predictions"]
    ):
        raise ValueError("predictions are not an eligible pre-opening freeze")

    predictions = pd.read_parquet(paths["predictions"])
    endpoints = pd.read_parquet(paths["endpoints"])
    required_prediction_columns = {
        "material_id",
        "analytic_supported",
        "next27_risk_score",
        "reject",
        "source_shard",
    }
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError(
            f"predictions lack columns: {sorted(required_prediction_columns-set(predictions.columns))}"
        )
    for frame, role in ((predictions, "predictions"), (endpoints, "endpoints")):
        if (
            "material_id" not in frame
            or frame["material_id"].isna().any()
            or frame["material_id"].duplicated().any()
        ):
            raise ValueError(f"{role} material IDs are invalid")
    if set(predictions["material_id"].astype(str)) != set(endpoints["material_id"].astype(str)):
        raise ValueError("prediction and endpoint identities differ")
    if "source_shard" in endpoints:
        shard_check = predictions[["material_id", "source_shard"]].merge(
            endpoints[["material_id", "source_shard"]],
            on="material_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_prediction", "_endpoint"),
        )
        if (
            len(shard_check) != len(predictions)
            or not (
                shard_check["source_shard_prediction"].astype(str)
                == shard_check["source_shard_endpoint"].astype(str)
            ).all()
        ):
            raise ValueError("prediction and endpoint source shards differ")
        endpoints = endpoints.drop(columns=["source_shard"])
    joined = predictions.merge(endpoints, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(predictions) or set(joined["material_id"]) != set(endpoints["material_id"]):
        raise ValueError("prediction and endpoint identities differ")
    joined = joined.sort_values("material_id", kind="stable").reset_index(drop=True)

    endpoint_positive = severe_dft_response(joined).to_numpy(bool)
    analytic = decision_metrics(
        supported=joined["analytic_supported"].to_numpy(bool),
        reject=joined["reject"].to_numpy(bool),
        endpoint_positive=endpoint_positive,
    )
    analytic["passes_prospective_gates"] = _passes(analytic)
    by_shard: dict[str, dict[str, object]] = {}
    for shard in sorted(joined["source_shard"].astype(str).unique()):
        mask = joined["source_shard"].astype(str).to_numpy() == shard
        metrics = decision_metrics(
            supported=joined.loc[mask, "analytic_supported"].to_numpy(bool),
            reject=joined.loc[mask, "reject"].to_numpy(bool),
            endpoint_positive=endpoint_positive[mask],
        )
        metrics["passes_prospective_gates"] = _passes(metrics)
        by_shard[shard] = metrics

    opened_at = datetime.now(timezone.utc).isoformat()
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "endpoint_definition": {
            "force0_max_ge": 1.0,
            "force0_rms_ge": 0.40,
            "energy_drop_pa_ge": 0.040,
            "stress0_norm_ge": 0.030,
            "combination": "logical_or",
        },
        "prospective_gates": dict(PROSPECTIVE_GATES),
        "analytic": analytic,
        "by_source_shard": by_shard,
        "continuous_diagnostics": _continuous_diagnostics(joined, endpoint_positive),
        "passes_all_prospective_gates": bool(analytic["passes_prospective_gates"]),
        "beyond_pauling_or_dft_claim": False,
        "claim_boundary": "substantial DFT response, not thermodynamic instability",
    }
    joined["endpoint_positive"] = endpoint_positive

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        joined_path = staging / JOINED_NAME
        result_path = staging / RESULT_NAME
        joined.to_parquet(joined_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        opening = {
            "protocol": PROTOCOL,
            "opened_at_utc": opened_at,
            "prediction_frozen_at_utc": prediction_manifest["frozen_at_utc"],
            "prediction_sha256": hashes["predictions"],
            "endpoint_sha256": hashes["endpoints"],
            "prediction_preceded_label_opening": True,
            "physical_never_read_lockbox": False,
        }
        opening_path = staging / LABEL_OPENING_NAME
        opening_path.write_bytes(_json_bytes(opening))
        source_hashes = {"src/next27_evaluate.py": _sha256(Path(__file__).resolve())}
        manifest = {
            "protocol": PROTOCOL,
            "labels_opened": True,
            "opened_at_utc": opened_at,
            "inputs_sha256": {
                role: {"path": str(path), "sha256": hashes[role]}
                for role, path in paths.items()
            },
            "outputs_sha256": {
                JOINED_NAME: _sha256(joined_path),
                RESULT_NAME: _sha256(result_path),
                LABEL_OPENING_NAME: _sha256(opening_path),
            },
            "executed_source_sha256": source_hashes,
            "passes_all_prospective_gates": bool(analytic["passes_prospective_gates"]),
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in paths.items():
            if _sha256(path) != hashes[role]:
                raise RuntimeError(f"input {role} changed during evaluation")
        if _sha256(Path(__file__).resolve()) != source_hashes["src/next27_evaluate.py"]:
            raise RuntimeError("NEXT27 evaluator source changed during evaluation")
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
    result = evaluate_predictions(
        predictions_path=args.predictions,
        prediction_manifest_path=args.prediction_manifest,
        endpoints_path=args.endpoints,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RESULT_NAME", "evaluate_predictions"]
