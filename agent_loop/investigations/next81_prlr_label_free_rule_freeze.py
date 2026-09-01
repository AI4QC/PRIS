#!/usr/bin/env python3
"""Freeze one PRLR threshold from unlabeled discovery-x0 scores only."""

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
from src.next60_odac23_robust_scaffold_endpoint import ENDPOINT_COLUMN
from src.next80_periodic_repulsive_load_resolvability import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next81-label-free-prlr-rule-freeze-v2"
DESIGN_SHA256 = "2dd045c85c435dbceb042c87d5ae1a13af6815b7fef41294e544db34ff69483d"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "5966354373f5bd8d5df3664b57e588759eee3f92aa12de5d8de68d5ce98acda6"
)
EXPECTED_FEATURE_SHA256 = (
    "3ad1a2db4a61edbf7ed27aee36b0f52dbbd2cb734222957c97ebe6ba359dce35"
)
THRESHOLD_QUANTILE = 0.95
FORMULA_NAME = "NEXT81_LABEL_FREE_PRLR_RULE.json"
PREDICTIONS_NAME = "next81_label_free_prlr_predictions.parquet"
SUMMARY_NAME = "NEXT81_LABEL_FREE_PRLR_FREEZE.json"
MANIFEST_NAME = "MANIFEST.json"


def freeze_label_free_prlr_rule(features: pd.DataFrame) -> dict[str, object]:
    """Freeze a q95 rule without accepting any endpoint-bearing table."""

    required = {
        "material_id",
        "partition_role",
        "repulsive_load_supported",
        "prlr_risk",
    }
    if ENDPOINT_COLUMN in features or any(
        str(column).lower() in {"endpoint", "protected", "severe"}
        for column in features.columns
    ):
        raise ValueError("NEXT81 endpoint columns are forbidden")
    if not required.issubset(features.columns):
        raise ValueError("NEXT81 feature table lacks required columns")
    table = features.loc[:, sorted(required)].copy()
    table["material_id"] = table["material_id"].astype(str)
    table["partition_role"] = table["partition_role"].astype(str)
    if table["material_id"].duplicated().any():
        raise ValueError("NEXT81 feature identity is duplicated")
    allowed_roles = {"discovery", "internal_validation", "internal_replication"}
    if set(table["partition_role"]) != allowed_roles:
        raise ValueError("NEXT81 partition roles differ")
    score = pd.to_numeric(table["prlr_risk"], errors="coerce").to_numpy(float)
    supported = table["repulsive_load_supported"].to_numpy(bool) & np.isfinite(score)
    discovery = table["partition_role"].eq("discovery").to_numpy(bool)
    if not discovery.any() or not supported[discovery].all():
        raise ValueError("NEXT81 discovery scores must all be supported and finite")
    threshold = float(
        np.quantile(score[discovery], THRESHOLD_QUANTILE, method="inverted_cdf")
    )
    reject = supported & (score >= threshold)
    predictions = pd.DataFrame(
        {
            "material_id": table["material_id"],
            "partition_role": table["partition_role"],
            "prlr_risk": score,
            "supported": supported,
            "reject": reject,
        }
    )
    role_counts = {}
    for role, group in predictions.groupby("partition_role", sort=True):
        role_counts[str(role)] = {
            "rows": len(group),
            "supported": int(group["supported"].sum()),
            "rejected": int(group["reject"].sum()),
        }
    formula = {
        "protocol": PROTOCOL,
        "kind": "direct_single_feature",
        "feature": "prlr_risk",
        "definition": (
            "prlr_residual_fraction * log1p(prlr_contact_weight_rms)"
        ),
        "operator": ">=",
        "threshold": threshold,
        "threshold_source": "unlabeled discovery-x0 inverted-CDF q95",
        "threshold_quantile": THRESHOLD_QUANTILE,
        "missing_policy": "KEEP",
        "replication_precision_lower_min": 0.80,
        "scientific_status": "hypothesis_predictions_frozen_without_endpoint_input",
    }
    summary = {
        "threshold": threshold,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold_fit_partition": "discovery x0 without endpoint labels",
        "threshold_fit_rows": int(discovery.sum()),
        "role_counts": role_counts,
    }
    return {"formula": formula, "predictions": predictions, "summary": summary}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT81 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze_prlr_rule_artifact(
    *,
    feature_dir: Path,
    design_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish formula and all predictions before any endpoint label read."""

    feature_dir = Path(feature_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / SOURCE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT81 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["features"] != EXPECTED_FEATURE_SHA256
        or hashes["feature_manifest"] != EXPECTED_FEATURE_MANIFEST_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT81 frozen input hash differs")
    upstream = _read_json(paths["feature_manifest"])
    outputs = upstream.get("outputs_sha256")
    if (
        upstream.get("protocol") != SOURCE_PROTOCOL
        or upstream.get("labels_opened") is not False
        or upstream.get("opened_internal_validation_result_used") is not False
        or upstream.get("internal_replication_labels_opened") is not False
        or upstream.get("dft_calculation_or_value_used") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
    ):
        raise ValueError("NEXT81 label-free feature provenance differs")
    columns = [
        "material_id",
        "partition_role",
        "repulsive_load_supported",
        "prlr_risk",
    ]
    features = pd.read_parquet(paths["features"], columns=columns)
    result = freeze_label_free_prlr_rule(features)
    formula = dict(result["formula"])
    formula["feature_artifact_sha256"] = hashes["features"]
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "label_free_discovery_x0_quantile_rule_freeze",
        "labels_opened": False,
        "endpoint_file_path_accepted": False,
        "endpoint_columns_read": False,
        "opened_internal_validation_result_used": False,
        "internal_validation_labels_opened": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "threshold_feature_count": 1,
        "threshold_candidate_count": 1,
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next81_prlr_label_free_rule_freeze.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        formula_path = staging / FORMULA_NAME
        predictions_path = staging / PREDICTIONS_NAME
        summary_path = staging / SUMMARY_NAME
        formula_path.write_bytes(_json_bytes(formula))
        result["predictions"].to_parquet(predictions_path, index=False)
        summary_path.write_bytes(_json_bytes(result["summary"]))
        manifest["outputs_sha256"] = {
            path.name: _sha256(path)
            for path in (formula_path, predictions_path, summary_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT81 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT81 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = freeze_prlr_rule_artifact(
        feature_dir=args.feature_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["outputs_sha256"], indent=2, sort_keys=True))


__all__ = ["PROTOCOL", "freeze_label_free_prlr_rule", "freeze_prlr_rule_artifact"]


if __name__ == "__main__":
    main()
