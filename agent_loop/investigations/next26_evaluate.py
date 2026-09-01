#!/usr/bin/env python3
"""Evaluate checksum-frozen NEXT26 decisions after DFT endpoint opening."""

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
from src.next23_relaxation_rule import wilson_lower_bound
from src.next26_omc25 import severe_dft_response


PROTOCOL = "2026-08-03-next26-omc25-dft-response-evaluation-v1"
RESULT_NAME = "NEXT26_DFT_RESPONSE_EVALUATION.json"
JOINED_NAME = "next26_joined.parquet"
LABEL_OPENING_NAME = "LABEL_OPENING.json"
MANIFEST_NAME = "MANIFEST.json"
PRIMARY_GATES: Mapping[str, float] = {
    "coverage_lower": 0.95,
    "endpoint_negative_protection_lower": 0.95,
    "endpoint_positive_precision_lower": 0.90,
    "savings_lower": 0.10,
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, endpoint_positive: np.ndarray
) -> dict[str, object]:
    supported = np.asarray(supported, dtype=bool)
    reject = np.asarray(reject, dtype=bool) & supported
    endpoint_positive = np.asarray(endpoint_positive, dtype=bool)
    if not (supported.shape == reject.shape == endpoint_positive.shape) or supported.ndim != 1:
        raise ValueError("decision arrays must be aligned one-dimensional arrays")
    rows = len(supported)
    negatives = ~endpoint_positive
    negative_count = int(negatives.sum())
    rejected = int(reject.sum())
    supported_count = int(supported.sum())
    positives_rejected = int((endpoint_positive & reject).sum())
    negatives_kept = int((negatives & ~reject).sum())
    metrics: dict[str, object] = {
        "rows": rows,
        "supported": supported_count,
        "rejected": rejected,
        "endpoint_positives": int(endpoint_positive.sum()),
        "endpoint_negatives": negative_count,
        "endpoint_positives_rejected": positives_rejected,
        "endpoint_negatives_kept": negatives_kept,
        "coverage": supported_count / rows if rows else 0.0,
        "coverage_lower": wilson_lower_bound(supported_count, rows),
        "endpoint_negative_protection": negatives_kept / negative_count if negative_count else 0.0,
        "endpoint_negative_protection_lower": wilson_lower_bound(negatives_kept, negative_count),
        "endpoint_positive_precision": positives_rejected / rejected if rejected else 0.0,
        "endpoint_positive_precision_lower": wilson_lower_bound(positives_rejected, rejected),
        "savings": rejected / rows if rows else 0.0,
        "savings_lower": wilson_lower_bound(rejected, rows),
        "endpoint_positive_recall": positives_rejected / int(endpoint_positive.sum()) if int(endpoint_positive.sum()) else 0.0,
    }
    metrics["passes_primary_gates"] = all(
        float(metrics[name]) >= cutoff for name, cutoff in PRIMARY_GATES.items()
    )
    return metrics


def _read_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def evaluate_predictions(
    *,
    predictions_path: Path,
    prediction_manifest_path: Path,
    endpoints_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Open the frozen endpoint once and publish the complete audit trail."""

    from src.next26_apply_rule import PREDICTIONS_NAME, PROTOCOL as PREDICTION_PROTOCOL

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
        or prediction_manifest.get("frozen_at_utc") is None
        or not isinstance(outputs, Mapping)
        or outputs.get(PREDICTIONS_NAME) != hashes["predictions"]
    ):
        raise ValueError("predictions are not an eligible pre-opening freeze")
    predictions = pd.read_parquet(paths["predictions"])
    endpoints = pd.read_parquet(paths["endpoints"])
    for frame, role in ((predictions, "predictions"), (endpoints, "endpoints")):
        if "material_id" not in frame or frame["material_id"].isna().any() or frame["material_id"].duplicated().any():
            raise ValueError(f"{role} material IDs are invalid")
    joined = predictions.merge(endpoints, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(predictions) or set(joined["material_id"]) != set(endpoints["material_id"]):
        raise ValueError("prediction and endpoint identities differ")
    endpoint_positive = severe_dft_response(joined).to_numpy(bool)
    metrics = decision_metrics(
        supported=joined["analytic_supported"].to_numpy(bool),
        reject=joined["reject"].to_numpy(bool),
        endpoint_positive=endpoint_positive,
    )
    score = pd.to_numeric(joined["next26_risk_score"], errors="coerce").to_numpy(float)
    finite = joined["analytic_supported"].to_numpy(bool) & np.isfinite(score)
    auc = _roc_auc(score[finite], endpoint_positive[finite]) if finite.any() else None
    continuous: dict[str, object] = {"endpoint_auc": auc}
    for endpoint in ("force0_max", "force0_rms", "energy_drop_pa", "stress0_norm", "disp_p90", "cell_logstrain_max"):
        if endpoint in joined and finite.sum() >= 3:
            correlation = spearmanr(score[finite], joined.loc[finite, endpoint].to_numpy(float)).statistic
            continuous[f"risk_spearman_{endpoint}"] = float(correlation) if math.isfinite(float(correlation)) else None
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
        "primary_gates": dict(PRIMARY_GATES),
        "analytic": metrics,
        "continuous_diagnostics": continuous,
        "passes_all_primary_gates": bool(metrics["passes_primary_gates"]),
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
        (staging / LABEL_OPENING_NAME).write_bytes(_json_bytes(opening))
        source_hashes = {"src/next26_evaluate.py": _sha256(Path(__file__).resolve())}
        manifest = {
            "protocol": PROTOCOL,
            "labels_opened": True,
            "opened_at_utc": opened_at,
            "inputs_sha256": {role: {"path": str(path), "sha256": hashes[role]} for role, path in paths.items()},
            "outputs_sha256": {
                JOINED_NAME: _sha256(joined_path),
                RESULT_NAME: _sha256(result_path),
                LABEL_OPENING_NAME: _sha256(staging / LABEL_OPENING_NAME),
            },
            "executed_source_sha256": source_hashes,
            "passes_all_primary_gates": bool(metrics["passes_primary_gates"]),
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
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


__all__ = ["PRIMARY_GATES", "decision_metrics", "evaluate_predictions"]
