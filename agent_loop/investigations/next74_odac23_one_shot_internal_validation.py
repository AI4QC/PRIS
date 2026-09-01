#!/usr/bin/env python3
"""One-shot internal validation of the sealed NEXT73 explicit x0 formula."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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
from src.next68_odac23_sparse_stable_law import apply_sparse_formula
from src.next70_odac23_metal_donor_bond_valence_features import (
    FEATURES_NAME as SOURCE_FEATURES_NAME,
    PROTOCOL as SOURCE_FEATURE_PROTOCOL,
)
from src.next73_odac23_exact_threshold_calibration import PROTOCOL as FORMULA_PROTOCOL


PROTOCOL = "2026-08-03-next74-odac23-one-shot-internal-validation-v1"
DESIGN_SHA256 = "139578578df69ff7258e892688c936d583b5824cf48f0a83d648d8b153f5d469"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "928a0bbfa1120e2c92bac2e9d3f0046a1d440c24beb72f652e477eb827874f14"
)
EXPECTED_FEATURE_SHA256 = (
    "d3684af21c70e3be18ae4aed8dd9a505209cfb2d91e9639911aae72da77ca6dc"
)
EXPECTED_FORMULA_SHA256 = (
    "a2b1033ee1b8ad99254ed005215322306ce8e09897bd2426454df83c4c9143c8"
)
EXPECTED_ENDPOINT_FIREWALL_SHA256 = (
    "9dbd3f78d2505ba96b33715e6409cd8524e9b909f4134af0020b933dff2f769f"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "cb3aa1a4914d25e8b50b626386dda34fe59c5b692e130a10faa06d1083d058b9"
)
EXPECTED_VALIDATION_LABEL_SHA256 = (
    "ad964615c26fff9a7fcdaea7dc9d24042fa55f61eedd690758236d08cd6e2fc8"
)
EVALUATION_NAME = "NEXT74_ODAC23_INTERNAL_VALIDATION.json"
PREDICTIONS_NAME = "next74_odac23_internal_validation_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _strata(features: pd.DataFrame) -> np.ndarray:
    defective = np.asarray(features["defective"], dtype=bool)
    oms = np.asarray(features["open_metal_site"], dtype=bool)
    return np.asarray(
        [f"defective={int(left)}|oms={int(right)}" for left, right in zip(defective, oms, strict=True)],
        dtype=str,
    )


def evaluate_frozen_formula(
    *,
    features: pd.DataFrame,
    endpoint: Sequence[float],
    formula: Mapping[str, object],
) -> dict[str, object]:
    """Apply exact formula once; this function has no calibration operation."""

    endpoint = np.asarray(endpoint, dtype=float)
    if len(features) != len(endpoint) or not np.isfinite(endpoint).all():
        raise ValueError("NEXT74 evaluation arrays differ")
    score, supported, reject = apply_sparse_formula(features, formula)
    aucs = _auc_diagnostics(
        score=score, supported=supported, endpoint=endpoint, strata=_strata(features)
    )
    metrics = _decision_metrics(supported=supported, reject=reject, endpoint=endpoint)
    terms = formula.get("terms")
    if not isinstance(terms, list):
        raise ValueError("NEXT74 formula term list differs")
    rank = _gate_rank(metrics, aucs, len(terms))
    return {
        "metrics": {
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
        },
        "stratum_diagnostics": aucs["strata"],
        "passes_gates": bool(rank[0] == 1.0),
        "rank": list(rank),
        "evaluated_threshold": float(formula["threshold"]),
        "formula_recalibrated": False,
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT74 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_one_shot_internal_validation(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    validation_dir: Path,
    design_path: Path,
    formula_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Open the physically isolated validation endpoint exactly for this run."""

    feature_dir = Path(feature_dir).resolve()
    validation_dir = Path(validation_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / MANIFEST_NAME,
        "endpoint_firewall": Path(endpoint_firewall_manifest_path).resolve(),
        "validation_labels": validation_dir / ROLE_LABELS_NAME,
        "validation_manifest": validation_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
        "formula": Path(formula_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT74 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "features": EXPECTED_FEATURE_SHA256,
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "validation_labels": EXPECTED_VALIDATION_LABEL_SHA256,
        "validation_manifest": EXPECTED_VALIDATION_MANIFEST_SHA256,
        "design": DESIGN_SHA256,
        "formula": EXPECTED_FORMULA_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT74 frozen input hash differs")
    feature_manifest = _read_json(paths["feature_manifest"])
    endpoint_firewall = _read_json(paths["endpoint_firewall"])
    validation_manifest = _read_json(paths["validation_manifest"])
    formula = _read_json(paths["formula"])
    feature_outputs = feature_manifest.get("outputs_sha256")
    validation_outputs = validation_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != SOURCE_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
        or endpoint_firewall.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_firewall.get("internal_validation_endpoint_values_summarized_or_inspected") is not False
        or endpoint_firewall.get("internal_replication_endpoint_values_summarized_or_inspected") is not False
        or validation_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or validation_manifest.get("partition_role") != "internal_validation"
        or validation_manifest.get("endpoint_values_summarized_or_inspected") is not False
        or not isinstance(validation_outputs, Mapping)
        or validation_outputs.get(ROLE_LABELS_NAME) != hashes["validation_labels"]
        or formula.get("protocol") != FORMULA_PROTOCOL
        or formula.get("scientific_status") != "advance_to_internal_validation"
    ):
        raise ValueError("NEXT74 one-shot provenance differs")

    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("internal_validation")].copy()
    labels = pd.read_parquet(paths["validation_labels"])
    if set(labels["partition_role"]) != {"internal_validation"}:
        raise ValueError("NEXT74 received the wrong endpoint role")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels) or len(joined) != int(validation_manifest.get("rows", -1)):
        raise ValueError("NEXT74 internal-validation identity differs")
    result = evaluate_frozen_formula(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        formula=formula,
    )
    evaluation = {
        key: value for key, value in result.items() if key not in {"score", "supported", "reject"}
    }
    evaluation.update(
        {
            "protocol": PROTOCOL,
            "partition_role": "internal_validation",
            "formula_sha256": hashes["formula"],
            "formula_or_threshold_modified": False,
            "scientific_status": "advance_to_internal_replication"
            if result["passes_gates"]
            else "internal_validation_failure_stop",
        }
    )
    endpoint = joined[ENDPOINT_COLUMN].to_numpy(float)
    predictions = pd.DataFrame(
        {
            "material_id": joined["material_id"].astype(str),
            "partition_role": "internal_validation",
            ENDPOINT_COLUMN: endpoint,
            "protected": endpoint <= PROTECTED_MAX,
            "severe": endpoint >= SEVERE_MIN,
            "risk_score": result["score"],
            "supported": result["supported"],
            "reject": result["reject"],
        }
    )
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "one_shot_frozen_formula_internal_validation",
        "robust_discovery_labels_opened_here": False,
        "internal_validation_labels_opened": True,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "formula_or_threshold_modified": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "passes_internal_validation_gates": result["passes_gates"],
        "counts": {
            "rows": len(joined),
            "protected": int(predictions["protected"].sum()),
            "severe": int(predictions["severe"].sum()),
            "supported": int(predictions["supported"].sum()),
            "rejected": int(predictions["reject"].sum()),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next74_odac23_one_shot_internal_validation.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        evaluation_path = staging / EVALUATION_NAME
        predictions_path = staging / PREDICTIONS_NAME
        evaluation_path.write_bytes(_json_bytes(evaluation))
        predictions.to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {
            path.name: _sha256(path) for path in (evaluation_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT74 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT74 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--endpoint-firewall", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--formula", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_one_shot_internal_validation(
        feature_dir=args.feature_dir,
        endpoint_firewall_manifest_path=args.endpoint_firewall,
        validation_dir=args.validation_dir,
        design_path=args.design,
        formula_path=args.formula,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"passes": manifest["passes_internal_validation_gates"], **manifest["counts"]},
            indent=2,
            sort_keys=True,
        )
    )


__all__ = ["PROTOCOL", "evaluate_frozen_formula", "run_one_shot_internal_validation"]


if __name__ == "__main__":
    main()
