#!/usr/bin/env python3
"""One-shot evaluation of the label-free frozen PRLR rule on discovery labels."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next57_odac23_discovery_search import (
    GATES,
    PROTECTED_MAX,
    SEVERE_MIN,
    _auc_diagnostics,
    _decision_metrics,
    _gate_rank,
)
from src.next60_odac23_robust_scaffold_endpoint import (
    ENDPOINT_COLUMN,
    PROTOCOL as ENDPOINT_PROTOCOL,
    ROLE_LABELS_NAME,
    ROLE_MANIFEST_NAME,
)
from src.next72_odac23_anchored_tail_correction_search import _strata
from src.next76_odac23_rigidity_tail_search import (
    EXPECTED_DISCOVERY_LABEL_SHA256,
    EXPECTED_DISCOVERY_MANIFEST_SHA256,
    EXPECTED_ENDPOINT_FIREWALL_SHA256,
)
from src.next80_periodic_repulsive_load_resolvability import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    MANIFEST_NAME as FEATURE_MANIFEST_NAME,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)
from src.next81_prlr_label_free_rule_freeze import (
    FORMULA_NAME,
    MANIFEST_NAME as RULE_MANIFEST_NAME,
    PREDICTIONS_NAME as FROZEN_PREDICTIONS_NAME,
    PROTOCOL as RULE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next82-one-shot-prlr-discovery-evaluation-v1"
DESIGN_SHA256 = "43f871f77f471b9e2615358128286b6168cddda923d91a383be70e06ec012c23"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "5966354373f5bd8d5df3664b57e588759eee3f92aa12de5d8de68d5ce98acda6"
)
EXPECTED_FEATURE_SHA256 = (
    "3ad1a2db4a61edbf7ed27aee36b0f52dbbd2cb734222957c97ebe6ba359dce35"
)
EXPECTED_RULE_MANIFEST_SHA256 = (
    "d704801bb5b23b367a6c9b1507cec6038e686a347529f4b9a02a04981c1ba8df"
)
EXPECTED_FORMULA_SHA256 = (
    "bc9f070e6cec67973fc95cf8e105d0e988119762c768d4e59102c1c05ade791c"
)
EXPECTED_PREDICTIONS_SHA256 = (
    "11b1851612f02fa21c53f4ecc21b255a45e967411b68c13eece574eba96622f8"
)
REPLICATION_PRECISION_LOWER_MIN = 0.80
EVALUATION_NAME = "NEXT82_PRLR_DISCOVERY_EVALUATION.json"
PREDICTIONS_NAME = "next82_prlr_discovery_evaluation.parquet"
MANIFEST_NAME = "MANIFEST.json"


def replication_ready(metrics: Mapping[str, object]) -> bool:
    checks = {
        "coverage_lower_at_least": float(metrics["coverage_lower"]),
        "protected_recall_lower_at_least": float(metrics["protected_recall_lower"]),
        "reject_precision_lower_at_least": float(metrics["reject_precision_lower"]),
        "savings_lower_at_least": float(metrics["savings_lower"]),
        "pooled_extreme_auc_at_least": float(metrics["pooled_extreme_auc"]),
        "macro_stratum_auc_at_least": float(metrics["macro_stratum_auc"]),
        "worst_stratum_auc_at_least": float(metrics["worst_stratum_auc"]),
    }
    return all(checks[name] >= float(threshold) for name, threshold in GATES.items()) and (
        checks["reject_precision_lower_at_least"]
        >= REPLICATION_PRECISION_LOWER_MIN
    )


def evaluate_frozen_prlr_discovery(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate already-frozen predictions with no score or threshold changes."""

    required_features = {"material_id", "partition_role", "defective", "open_metal_site"}
    required_predictions = {
        "material_id",
        "partition_role",
        "prlr_risk",
        "supported",
        "reject",
    }
    required_labels = {"material_id", "partition_role", ENDPOINT_COLUMN}
    if (
        not required_features.issubset(features)
        or not required_predictions.issubset(predictions)
        or not required_labels.issubset(labels)
    ):
        raise ValueError("NEXT82 evaluation input columns differ")
    frames = []
    for name, frame, columns in (
        ("features", features, required_features),
        ("predictions", predictions, required_predictions),
        ("labels", labels, required_labels),
    ):
        selected = frame.loc[:, sorted(columns)].copy()
        selected["material_id"] = selected["material_id"].astype(str)
        selected["partition_role"] = selected["partition_role"].astype(str)
        if selected["material_id"].duplicated().any():
            raise ValueError(f"NEXT82 {name} identity is duplicated")
        if set(selected["partition_role"]) != {"discovery"}:
            raise ValueError(f"NEXT82 {name} must contain discovery only")
        frames.append(selected)
    feature_frame, prediction_frame, label_frame = frames
    joined = feature_frame.merge(
        prediction_frame,
        on=["material_id", "partition_role"],
        how="inner",
        validate="one_to_one",
    ).merge(
        label_frame,
        on=["material_id", "partition_role"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(label_frame):
        raise ValueError("NEXT82 discovery identity differs")
    endpoint = pd.to_numeric(joined[ENDPOINT_COLUMN], errors="coerce").to_numpy(float)
    score = pd.to_numeric(joined["prlr_risk"], errors="coerce").to_numpy(float)
    supported = joined["supported"].to_numpy(bool)
    reject = joined["reject"].to_numpy(bool)
    if not np.isfinite(endpoint).all() or not np.isfinite(score[supported]).all():
        raise ValueError("NEXT82 endpoint or supported score is non-finite")
    strata = _strata(joined)
    metrics = _decision_metrics(supported=supported, reject=reject, endpoint=endpoint)
    aucs = _auc_diagnostics(
        score=score,
        supported=supported,
        endpoint=endpoint,
        strata=strata,
    )
    combined = {
        **metrics,
        **{
            key: aucs[key]
            for key in (
                "pooled_extreme_auc",
                "macro_stratum_auc",
                "worst_stratum_auc",
                "evaluable_strata",
            )
        },
    }
    rank = _gate_rank(metrics, aucs, 1)
    original = bool(rank[0] == 1.0)
    ready = replication_ready(combined)
    evaluated = joined.assign(
        protected=endpoint <= PROTECTED_MAX,
        severe=endpoint >= SEVERE_MIN,
        risk_score=score,
    )
    return {
        "rows": len(joined),
        "metrics": combined,
        "stratum_diagnostics": aucs["strata"],
        "rank": list(rank),
        "passes_original_gates": original,
        "passes_replication_readiness_margin": ready,
        "evaluated_predictions": evaluated,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT82 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_prlr_discovery_evaluation(
    *,
    feature_dir: Path,
    rule_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Open robust discovery labels once for the already-frozen PRLR rule."""

    feature_dir = Path(feature_dir).resolve()
    rule_dir = Path(rule_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / FEATURE_MANIFEST_NAME,
        "rule_manifest": rule_dir / RULE_MANIFEST_NAME,
        "formula": rule_dir / FORMULA_NAME,
        "frozen_predictions": rule_dir / FROZEN_PREDICTIONS_NAME,
        "endpoint_firewall": Path(endpoint_firewall_manifest_path).resolve(),
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT82 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "features": EXPECTED_FEATURE_SHA256,
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "rule_manifest": EXPECTED_RULE_MANIFEST_SHA256,
        "formula": EXPECTED_FORMULA_SHA256,
        "frozen_predictions": EXPECTED_PREDICTIONS_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "design": DESIGN_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT82 frozen input hash differs")
    feature_manifest = _read_json(paths["feature_manifest"])
    rule_manifest = _read_json(paths["rule_manifest"])
    endpoint_firewall = _read_json(paths["endpoint_firewall"])
    discovery_manifest = _read_json(paths["discovery_manifest"])
    formula = _read_json(paths["formula"])
    feature_outputs = feature_manifest.get("outputs_sha256")
    rule_outputs = rule_manifest.get("outputs_sha256")
    discovery_outputs = discovery_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != SOURCE_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
        or rule_manifest.get("protocol") != RULE_PROTOCOL
        or rule_manifest.get("labels_opened") is not False
        or rule_manifest.get("endpoint_columns_read") is not False
        or rule_manifest.get("internal_replication_labels_opened") is not False
        or not isinstance(rule_outputs, Mapping)
        or rule_outputs.get(FORMULA_NAME) != hashes["formula"]
        or rule_outputs.get(FROZEN_PREDICTIONS_NAME) != hashes["frozen_predictions"]
        or formula.get("protocol") != RULE_PROTOCOL
        or float(formula.get("replication_precision_lower_min", -1.0))
        != REPLICATION_PRECISION_LOWER_MIN
        or endpoint_firewall.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_firewall.get(
            "internal_replication_endpoint_values_summarized_or_inspected"
        )
        is not False
        or discovery_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or discovery_manifest.get("partition_role") != "discovery"
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get(ROLE_LABELS_NAME) != hashes["discovery_labels"]
    ):
        raise ValueError("NEXT82 prospective provenance differs")

    feature_columns = [
        "material_id",
        "partition_role",
        "defective",
        "open_metal_site",
    ]
    features_all = pd.read_parquet(paths["features"], columns=feature_columns)
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    predictions_all = pd.read_parquet(paths["frozen_predictions"])
    predictions = predictions_all[
        predictions_all["partition_role"].eq("discovery")
    ].copy()
    threshold = float(formula["threshold"])
    recomputed = predictions["supported"].to_numpy(bool) & (
        predictions["prlr_risk"].to_numpy(float) >= threshold
    )
    if not np.array_equal(recomputed, predictions["reject"].to_numpy(bool)):
        raise ValueError("NEXT82 frozen prediction decisions differ from formula")
    labels = pd.read_parquet(paths["discovery_labels"])
    result = evaluate_frozen_prlr_discovery(features, predictions, labels)
    status = (
        "advance_to_one_shot_internal_replication"
        if result["passes_replication_readiness_margin"]
        else "independent_hypothesis_discovery_failure_stop"
    )
    evaluation = {
        key: value
        for key, value in result.items()
        if key != "evaluated_predictions"
    }
    evaluation.update(
        {
            "protocol": PROTOCOL,
            "partition_role": "discovery",
            "formula_sha256": hashes["formula"],
            "frozen_predictions_sha256": hashes["frozen_predictions"],
            "formula_or_threshold_modified": False,
            "replication_precision_lower_min": REPLICATION_PRECISION_LOWER_MIN,
            "scientific_status": status,
        }
    )
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "one_shot_frozen_prlr_discovery_evaluation",
        "robust_discovery_labels_opened": True,
        "previous_internal_validation_artifact_used": False,
        "internal_validation_labels_opened_here": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "formula_or_threshold_modified": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "passes_original_discovery_gates": result["passes_original_gates"],
        "passes_replication_readiness_margin": result[
            "passes_replication_readiness_margin"
        ],
        "counts": {
            "rows": result["rows"],
            "supported": int(result["evaluated_predictions"]["supported"].sum()),
            "rejected": int(result["evaluated_predictions"]["reject"].sum()),
            "protected": int(result["evaluated_predictions"]["protected"].sum()),
            "severe": int(result["evaluated_predictions"]["severe"].sum()),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next82_prlr_discovery_evaluation.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        evaluation_path = staging / EVALUATION_NAME
        predictions_path = staging / PREDICTIONS_NAME
        evaluation_path.write_bytes(_json_bytes(evaluation))
        result["evaluated_predictions"].to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {
            path.name: _sha256(path) for path in (evaluation_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT82 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT82 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--rule-dir", type=Path, required=True)
    parser.add_argument("--endpoint-firewall", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_prlr_discovery_evaluation(
        feature_dir=args.feature_dir,
        rule_dir=args.rule_dir,
        endpoint_firewall_manifest_path=args.endpoint_firewall,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "passes_original": manifest["passes_original_discovery_gates"],
                "passes_replication_margin": manifest[
                    "passes_replication_readiness_margin"
                ],
                **manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "PROTOCOL",
    "evaluate_frozen_prlr_discovery",
    "replication_ready",
    "run_prlr_discovery_evaluation",
]


if __name__ == "__main__":
    main()
