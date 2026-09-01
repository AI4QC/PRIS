#!/usr/bin/env python3
"""Evaluate frozen NEXT23 and Pauling decisions after one blind label opening."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next23_apply_rule import PROTOCOL as PREDICTION_PROTOCOL
from src.next23_pauling_controls import PROTOCOL as PAULING_PROTOCOL
from src.next23_relaxation_rule import (
    ENDPOINT_COLUMN,
    PRIMARY_GATES,
    PROTECTED_MAX,
    SEVERE_MIN,
    SUBSTANTIAL_MIN,
    wilson_lower_bound,
)
from src.next23_wbm_holdout import PROTOCOL as COHORT_PROTOCOL


PROTOCOL = "2026-08-02-next23-relaxation-change-blind-evaluation-v1"
RESULT_NAME = "NEXT23_RELAXATION_CHANGE_EVALUATION.json"
MANIFEST_NAME = "MANIFEST.json"
PRIVATE_JOIN_NAME = "joined_predictions_labels.parquet"
LABEL_OPENING_NAME = "LABEL_OPENING.json"


def _read_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _validate_output_hash(
    manifest: Mapping[str, object], path: Path, *, role: str
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != _sha256(path):
        raise ValueError(f"{role} hash differs")


def _decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, endpoint: np.ndarray
) -> dict[str, object]:
    n_rows = len(endpoint)
    supported = np.asarray(supported, dtype=bool)
    reject = np.asarray(reject, dtype=bool) & supported
    protected = endpoint <= PROTECTED_MAX
    changed = endpoint > PROTECTED_MAX
    substantial = endpoint >= SUBSTANTIAL_MIN
    severe = endpoint >= SEVERE_MIN
    n_supported = int(supported.sum())
    n_rejected = int(reject.sum())
    n_protected = int(protected.sum())
    protected_kept = int((protected & ~reject).sum())
    changed_rejected = int((changed & reject).sum())
    substantial_total = int(substantial.sum())
    severe_total = int(severe.sum())
    metrics: dict[str, object] = {
        "rows": n_rows,
        "supported": n_supported,
        "rejected": n_rejected,
        "protected": n_protected,
        "protected_kept": protected_kept,
        "changed_rejected": changed_rejected,
        "coverage": n_supported / n_rows if n_rows else 0.0,
        "coverage_lower": wilson_lower_bound(n_supported, n_rows),
        "protected_recall": protected_kept / n_protected if n_protected else 0.0,
        "protected_recall_lower": wilson_lower_bound(protected_kept, n_protected),
        "rejection_precision": changed_rejected / n_rejected if n_rejected else 0.0,
        "rejection_precision_lower": wilson_lower_bound(changed_rejected, n_rejected),
        "savings": n_rejected / n_rows if n_rows else 0.0,
        "savings_lower": wilson_lower_bound(n_rejected, n_rows),
        "substantial_total": substantial_total,
        "substantial_recall": (
            int((substantial & reject).sum()) / substantial_total
            if substantial_total
            else 0.0
        ),
        "severe_total": severe_total,
        "severe_recall": (
            int((severe & reject).sum()) / severe_total if severe_total else 0.0
        ),
    }
    metrics["passes_primary_gates"] = all(
        float(metrics[name]) >= cutoff for name, cutoff in PRIMARY_GATES.items()
    )
    return metrics


def _roc_auc(score: np.ndarray, positive: np.ndarray) -> float | None:
    score = np.asarray(score, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    finite = np.isfinite(score)
    score = score[finite]
    positive = positive[finite]
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if not n_positive or not n_negative:
        return None
    ranks = rankdata(score, method="average")
    value = (
        float(ranks[positive].sum())
        - n_positive * (n_positive + 1.0) / 2.0
    ) / (n_positive * n_negative)
    return float(value)


def _continuous_diagnostics(
    score: np.ndarray, supported: np.ndarray, endpoint: np.ndarray
) -> dict[str, object]:
    mask = np.asarray(supported, dtype=bool) & np.isfinite(score)
    if int(mask.sum()) < 2:
        rho = None
    else:
        value = spearmanr(score[mask], endpoint[mask]).statistic
        rho = float(value) if math.isfinite(float(value)) else None
    return {
        "supported_rows": int(mask.sum()),
        "spearman_rho": rho,
        "auc_changed_gt_0_10": _roc_auc(score[mask], endpoint[mask] > PROTECTED_MAX),
        "auc_substantial_ge_0_20": _roc_auc(
            score[mask], endpoint[mask] >= SUBSTANTIAL_MIN
        ),
        "auc_severe_ge_0_50": _roc_auc(score[mask], endpoint[mask] >= SEVERE_MIN),
    }


def _bootstrap_intervals(
    supported: np.ndarray,
    reject: np.ndarray,
    endpoint: np.ndarray,
    *,
    repetitions: int = 500,
) -> dict[str, object]:
    rng = np.random.default_rng(230823)
    n_rows = len(endpoint)
    values: dict[str, list[float]] = {
        "coverage": [],
        "protected_recall": [],
        "rejection_precision": [],
        "savings": [],
    }
    for _ in range(repetitions):
        indices = rng.integers(0, n_rows, size=n_rows)
        metrics = _decision_metrics(
            supported=supported[indices],
            reject=reject[indices],
            endpoint=endpoint[indices],
        )
        for name in values:
            values[name].append(float(metrics[name]))
    return {
        "method": "deterministic nonparametric row bootstrap",
        "seed": 230823,
        "repetitions": repetitions,
        "interval_level": 0.95,
        "intervals": {
            name: [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ]
            for name, samples in values.items()
        },
    }


def _stratified_metrics(joined: pd.DataFrame) -> dict[str, object]:
    material_ids = joined["material_id"].astype(str)
    steps = material_ids.str.extract(r"^wbm-(\d+)-", expand=False).fillna("unknown")
    atom_bins = pd.cut(
        joined["natoms"],
        bins=[0, 4, 8, 12, np.inf],
        labels=["2-4", "5-8", "9-12", "13+"],
        include_lowest=True,
    ).astype(str)
    result: dict[str, object] = {}
    for name, strata in (("substitution_step", steps), ("atom_count", atom_bins)):
        rows: dict[str, object] = {}
        for value in sorted(strata.unique()):
            mask = strata.eq(value).to_numpy()
            rows[str(value)] = _decision_metrics(
                supported=joined.loc[mask, "analytic_supported"].to_numpy(bool),
                reject=joined.loc[mask, "reject"].to_numpy(bool),
                endpoint=joined.loc[mask, ENDPOINT_COLUMN].to_numpy(float),
            )
        result[name] = rows
    return result


def _validate_label_free_inputs(
    *,
    predictions_path: Path,
    prediction_manifest_path: Path,
    pauling_path: Path,
    pauling_manifest_path: Path,
    metadata_path: Path,
    cohort_manifest_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_manifest = _read_json(
        prediction_manifest_path, role="prediction manifest"
    )
    _validate_output_hash(prediction_manifest, predictions_path, role="prediction")
    if (
        prediction_manifest.get("protocol") != PREDICTION_PROTOCOL
        or prediction_manifest.get("blind_labels_opened") is not False
        or prediction_manifest.get("endpoint_fields_read") is not False
        or not isinstance(prediction_manifest.get("frozen_at_utc"), str)
    ):
        raise ValueError("prediction was not frozen before blind label opening")

    pauling_manifest = _read_json(pauling_manifest_path, role="Pauling manifest")
    _validate_output_hash(pauling_manifest, pauling_path, role="Pauling control")
    if (
        pauling_manifest.get("protocol") != PAULING_PROTOCOL
        or pauling_manifest.get("labels_opened") is not False
        or pauling_manifest.get("endpoint_artifacts_opened") is not False
        or pauling_manifest.get("thresholds_refit") is not False
    ):
        raise ValueError("Pauling controls crossed the blind-label boundary")

    cohort_manifest = _read_json(cohort_manifest_path, role="cohort manifest")
    _validate_output_hash(cohort_manifest, metadata_path, role="cohort metadata")
    if (
        cohort_manifest.get("protocol") != COHORT_PROTOCOL
        or cohort_manifest.get("labels_opened") is not False
        or cohort_manifest.get("relaxed_structures_opened") is not False
    ):
        raise ValueError("cohort crossed the blind-label boundary")

    metadata = pd.read_parquet(metadata_path)
    predictions = pd.read_parquet(predictions_path)
    pauling = pd.read_parquet(pauling_path)
    for role, frame in (
        ("metadata", metadata),
        ("predictions", predictions),
        ("Pauling", pauling),
    ):
        if "material_id" not in frame or frame["material_id"].isna().any():
            raise ValueError(f"{role} lacks material IDs")
        frame["material_id"] = frame["material_id"].astype(str)
        if frame["material_id"].duplicated().any():
            raise ValueError(f"{role} material IDs must be unique")
    expected = set(metadata["material_id"])
    if set(predictions["material_id"]) != expected or set(pauling["material_id"]) != expected:
        raise ValueError("label-free input IDs do not join one-to-one")
    required_prediction = {
        "analytic_supported",
        "next23_risk_score",
        "reject",
        "input_role",
    }
    if required_prediction - set(predictions.columns):
        raise ValueError("prediction table lacks frozen decision columns")
    if not predictions["input_role"].eq("unrelaxed_x0_geometry_only").all():
        raise ValueError("prediction input role differs")
    return metadata, predictions, pauling


def evaluate_frozen_predictions(
    *,
    predictions_path: Path,
    prediction_manifest_path: Path,
    pauling_controls_path: Path,
    pauling_manifest_path: Path,
    cohort_metadata_path: Path,
    cohort_manifest_path: Path,
    labels_path: Path,
    public_output_dir: Path,
    private_output_dir: Path,
) -> dict[str, object]:
    """Open blind labels once, evaluate without refit, and publish aggregates."""

    public_target = Path(public_output_dir).resolve()
    private_target = Path(private_output_dir).resolve()
    for target in (public_target, private_target):
        if os.path.lexists(target):
            raise FileExistsError(str(target))
    paths = {
        "predictions": Path(predictions_path).resolve(),
        "prediction_manifest": Path(prediction_manifest_path).resolve(),
        "pauling_controls": Path(pauling_controls_path).resolve(),
        "pauling_manifest": Path(pauling_manifest_path).resolve(),
        "cohort_metadata": Path(cohort_metadata_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "blind_labels": Path(labels_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")

    # Everything above this boundary is label-free and hash-validated first.
    metadata, predictions, pauling = _validate_label_free_inputs(
        predictions_path=paths["predictions"],
        prediction_manifest_path=paths["prediction_manifest"],
        pauling_path=paths["pauling_controls"],
        pauling_manifest_path=paths["pauling_manifest"],
        metadata_path=paths["cohort_metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
    )
    preopening_hashes = {
        role: _sha256(paths[role])
        for role in paths
        if role != "blind_labels"
    }
    opened_at = datetime.now(timezone.utc).isoformat()

    # The blind endpoint is first read here. No formula or threshold changes follow.
    label_hash = _sha256(paths["blind_labels"])
    labels = pd.read_parquet(
        paths["blind_labels"], columns=["material_id", ENDPOINT_COLUMN]
    )
    labels["material_id"] = labels["material_id"].astype(str)
    expected_ids = set(metadata["material_id"])
    labels = labels.loc[labels["material_id"].isin(expected_ids)].copy()
    if (
        labels["material_id"].duplicated().any()
        or set(labels["material_id"]) != expected_ids
    ):
        raise ValueError("blind label IDs do not join one-to-one")
    labels[ENDPOINT_COLUMN] = pd.to_numeric(labels[ENDPOINT_COLUMN], errors="coerce")
    if not np.isfinite(labels[ENDPOINT_COLUMN].to_numpy(float)).all():
        raise ValueError("blind endpoint must be finite")

    joined = (
        metadata.loc[:, ["material_id", "rk", "formula", "natoms"]]
        .merge(predictions, on="material_id", validate="one_to_one")
        .merge(labels, on="material_id", validate="one_to_one")
        .merge(pauling, on="material_id", validate="one_to_one", suffixes=("", "_pauling"))
        .sort_values("material_id", kind="stable", ignore_index=True)
    )
    endpoint = joined[ENDPOINT_COLUMN].to_numpy(float)
    support = joined["analytic_supported"].to_numpy(bool)
    reject = joined["reject"].to_numpy(bool)
    score = joined["next23_risk_score"].to_numpy(float)
    next23_metrics = _decision_metrics(
        supported=support, reject=reject, endpoint=endpoint
    )

    pauling_columns = {
        "pauling_p2": "pauling_p2_decision",
        "pauling_p3": "pauling_p3_decision",
        "pauling_p4": "pauling_p4_decision",
        "pauling_p5": "pauling_p5_decision",
        "pauling_p2_p5_combined": "pauling_p2_p5_decision",
    }
    pauling_metrics: dict[str, object] = {}
    for name, column in pauling_columns.items():
        if column not in joined:
            raise ValueError(f"Pauling controls lack {column}")
        decisions = joined[column].astype(str).to_numpy()
        if not set(decisions) <= {"KEEP", "REJECT", "ABSTAIN"}:
            raise ValueError(f"Pauling decision vocabulary differs: {column}")
        pauling_metrics[name] = _decision_metrics(
            supported=decisions != "ABSTAIN",
            reject=decisions == "REJECT",
            endpoint=endpoint,
        )
    safe_pauling_savings = [
        float(metrics["savings_lower"])
        for metrics in pauling_metrics.values()
        if metrics["passes_primary_gates"]
    ]
    best_safe_pauling = max(safe_pauling_savings, default=0.0)
    beyond_pauling = bool(
        next23_metrics["passes_primary_gates"]
        and float(next23_metrics["savings_lower"]) > best_safe_pauling
    )

    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "evaluation_role": "physically_disjoint_frozen_blind_evaluation",
        "claim_scope": "relaxation_change_screening_only",
        "not_claimed": [
            "convex_hull_thermodynamic_stability",
            "formation_energy_prediction",
            "synthesizability",
            "replacement_for_DFT_energy_calculation",
        ],
        "law_execution_used_dft_or_relaxed_inputs": False,
        "dft_used_only_as_offline_blind_endpoint": True,
        "no_refit_after_blind_opening": True,
        "blind_labels_opened_at_utc": opened_at,
        "endpoint": {
            "column": ENDPOINT_COLUMN,
            "protected_max": PROTECTED_MAX,
            "substantial_min": SUBSTANTIAL_MIN,
            "severe_min": SEVERE_MIN,
            "cutoffs_are_operational_not_universal_constants": True,
            "distribution": {
                "rows": len(joined),
                "minimum": float(endpoint.min()),
                "q10": float(np.quantile(endpoint, 0.10)),
                "q25": float(np.quantile(endpoint, 0.25)),
                "median": float(np.median(endpoint)),
                "q75": float(np.quantile(endpoint, 0.75)),
                "q90": float(np.quantile(endpoint, 0.90)),
                "maximum": float(endpoint.max()),
            },
        },
        "primary_gates": dict(PRIMARY_GATES),
        "next23": next23_metrics,
        "pauling_controls": pauling_metrics,
        "best_safe_pauling_savings_lower": best_safe_pauling,
        "beyond_pauling_on_this_endpoint": beyond_pauling,
        "continuous_diagnostics": _continuous_diagnostics(score, support, endpoint),
        "bootstrap": _bootstrap_intervals(support, reject, endpoint),
        "stratified": _stratified_metrics(joined),
        "scientific_goal_achieved": False,
        "reason_goal_remains_open": (
            "This endpoint tests structural relaxation change, not DFT convex-hull "
            "thermodynamic stability or universal generated-structure validity."
        ),
    }
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next23_evaluate.py": Path(__file__).resolve(),
        "src/next23_apply_rule.py": repository_root / "src/next23_apply_rule.py",
        "src/next23_relaxation_rule.py": repository_root
        / "src/next23_relaxation_rule.py",
        "src/next23_pauling_controls.py": repository_root
        / "src/next23_pauling_controls.py",
    }
    source_hashes = {
        relative: _sha256(path) for relative, path in source_paths.items()
    }
    input_hashes = {
        **preopening_hashes,
        "blind_labels": label_hash,
    }
    opening = {
        "protocol": PROTOCOL,
        "labels_opened": True,
        "opened_at_utc": opened_at,
        "predictions_sha256_before_label_opening": preopening_hashes["predictions"],
        "prediction_manifest_sha256_before_label_opening": preopening_hashes[
            "prediction_manifest"
        ],
        "pauling_controls_sha256_before_label_opening": preopening_hashes[
            "pauling_controls"
        ],
        "blind_labels_sha256": label_hash,
        "no_refit_after_opening": True,
    }

    public_target.parent.mkdir(parents=True, exist_ok=True)
    private_target.parent.mkdir(parents=True, exist_ok=True)
    public_staging = Path(
        tempfile.mkdtemp(
            prefix=f".{public_target.name}.staging-", dir=public_target.parent
        )
    )
    private_staging = Path(
        tempfile.mkdtemp(
            prefix=f".{private_target.name}.staging-", dir=private_target.parent
        )
    )
    try:
        public_result_path = public_staging / RESULT_NAME
        public_result_path.write_bytes(_json_bytes(result))
        public_manifest = {
            "protocol": PROTOCOL,
            "mode": "aggregate_only_blind_evaluation",
            "identifier_bearing_rows_published_in_repository": False,
            "blind_labels_opened": True,
            "no_refit_after_blind_opening": True,
            "inputs_sha256": {
                role: {"path": str(paths[role]), "sha256": digest}
                for role, digest in input_hashes.items()
            },
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {RESULT_NAME: _sha256(public_result_path)},
            "scientific_goal_achieved": False,
        }
        (public_staging / MANIFEST_NAME).write_bytes(_json_bytes(public_manifest))

        private_join_path = private_staging / PRIVATE_JOIN_NAME
        private_opening_path = private_staging / LABEL_OPENING_NAME
        joined.to_parquet(private_join_path, index=False)
        private_opening_path.write_bytes(_json_bytes(opening))
        private_manifest = {
            "protocol": PROTOCOL,
            "mode": "private_identifier_bearing_blind_join",
            "labels_opened": True,
            "no_refit_after_blind_opening": True,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                PRIVATE_JOIN_NAME: _sha256(private_join_path),
                LABEL_OPENING_NAME: _sha256(private_opening_path),
            },
        }
        (private_staging / MANIFEST_NAME).write_bytes(_json_bytes(private_manifest))

        for role, path in paths.items():
            expected = input_hashes[role]
            if _sha256(path) != expected:
                raise RuntimeError(f"input {role} changed before publication")
        for relative, path in source_paths.items():
            if _sha256(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before publication")
        _publish_directory_no_replace(private_staging, private_target)
        _publish_directory_no_replace(public_staging, public_target)
    except Exception:
        shutil.rmtree(public_staging, ignore_errors=True)
        shutil.rmtree(private_staging, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--prediction-manifest", required=True, type=Path)
    parser.add_argument("--pauling-controls", required=True, type=Path)
    parser.add_argument("--pauling-manifest", required=True, type=Path)
    parser.add_argument("--cohort-metadata", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--public-output", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    args = parser.parse_args(argv)
    evaluate_frozen_predictions(
        predictions_path=args.predictions,
        prediction_manifest_path=args.prediction_manifest,
        pauling_controls_path=args.pauling_controls,
        pauling_manifest_path=args.pauling_manifest,
        cohort_metadata_path=args.cohort_metadata,
        cohort_manifest_path=args.cohort_manifest,
        labels_path=args.labels,
        public_output_dir=args.public_output,
        private_output_dir=args.private_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LABEL_OPENING_NAME",
    "PRIVATE_JOIN_NAME",
    "RESULT_NAME",
    "evaluate_frozen_predictions",
]
