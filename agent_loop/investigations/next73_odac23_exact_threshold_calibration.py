#!/usr/bin/env python3
"""Exact score-boundary calibration for the frozen NEXT72 term list."""

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
    DOMAIN_GATE,
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
from src.next72_odac23_anchored_tail_correction_search import PROTOCOL as FROZEN_PROTOCOL


PROTOCOL = "2026-08-03-next73-odac23-exact-threshold-calibration-v1"
DESIGN_SHA256 = "cfe8fa0b40c30e85cffb7d3a303113e5cd4fae33afa8e13b6b7ea561bc4e0831"
EXPECTED_FEATURE_MANIFEST_SHA256 = (
    "928a0bbfa1120e2c92bac2e9d3f0046a1d440c24beb72f652e477eb827874f14"
)
EXPECTED_FEATURE_SHA256 = (
    "d3684af21c70e3be18ae4aed8dd9a505209cfb2d91e9639911aae72da77ca6dc"
)
EXPECTED_FROZEN_FORMULA_SHA256 = (
    "3b28dddeb978cd9607ab9f35a7bcaed145f5a489e843c94443b4a81a2a5cd23d"
)
EXPECTED_ENDPOINT_FIREWALL_SHA256 = (
    "9dbd3f78d2505ba96b33715e6409cd8524e9b909f4134af0020b933dff2f769f"
)
EXPECTED_DISCOVERY_MANIFEST_SHA256 = (
    "6ca39eb42629d626559618474f75aa6bb6571a38a928b3b16512b5d987b76137"
)
EXPECTED_DISCOVERY_LABEL_SHA256 = (
    "1a7c78fd87bb3f5795e59fa3c3799fbbb07a1629b90d472aef7e73740ce7f08a"
)
MIN_REJECTION_FRACTION = 0.02
MAX_REJECTION_FRACTION = 0.30
FORMULA_NAME = "NEXT73_ODAC23_EXACT_THRESHOLD_CANDIDATE.json"
SEARCH_NAME = "NEXT73_ODAC23_EXACT_THRESHOLD_SEARCH.json"
PREDICTIONS_NAME = "next73_odac23_exact_threshold_predictions.parquet"
MANIFEST_NAME = "MANIFEST.json"


def _strata(features: pd.DataFrame) -> np.ndarray:
    defective = np.asarray(features["defective"], dtype=bool)
    oms = np.asarray(features["open_metal_site"], dtype=bool)
    return np.asarray(
        [f"defective={int(left)}|oms={int(right)}" for left, right in zip(defective, oms, strict=True)],
        dtype=str,
    )


def calibrate_exact_threshold(
    *,
    features: pd.DataFrame,
    endpoint: Sequence[float],
    frozen_formula: Mapping[str, object],
) -> dict[str, object]:
    """Keep exact frozen terms and enumerate every in-range score boundary."""

    endpoint = np.asarray(endpoint, dtype=float)
    terms_raw = frozen_formula.get("terms")
    if (
        len(features) != len(endpoint)
        or not np.isfinite(endpoint).all()
        or not isinstance(terms_raw, list)
        or not 1 <= len(terms_raw) <= 8
        or not all(isinstance(term, Mapping) for term in terms_raw)
    ):
        raise ValueError("NEXT73 frozen discovery inputs differ")
    terms = [
        {
            "feature": str(term["feature"]),
            "direction": int(term["direction"]),
            "center": float(term["center"]),
            "scale": float(term["scale"]),
            "weight": float(term["weight"]),
        }
        for term in terms_raw
    ]
    provisional = {
        "kind": "additive",
        "terms": terms,
        "threshold": 0.0,
        "missing_policy": "KEEP",
        "domain_gate": dict(DOMAIN_GATE),
    }
    raw_score, supported, _reject = apply_sparse_formula(features, provisional)
    strata = _strata(features)
    aucs = _auc_diagnostics(
        score=raw_score, supported=supported, endpoint=endpoint, strata=strata
    )
    best = None
    candidate_count = 0
    n = len(features)
    for threshold in np.unique(raw_score[supported]):
        formula = {**provisional, "threshold": float(threshold)}
        score, formula_supported, reject = apply_sparse_formula(features, formula)
        fraction = float(reject.sum()) / n if n else 0.0
        if fraction < MIN_REJECTION_FRACTION or fraction > MAX_REJECTION_FRACTION:
            continue
        metrics = _decision_metrics(
            supported=formula_supported, reject=reject, endpoint=endpoint
        )
        rank = _gate_rank(metrics, aucs, len(terms))
        key = json.dumps(formula, sort_keys=True, separators=(",", ":"))
        candidate_count += 1
        record = (rank, key, formula, metrics, score, formula_supported, reject)
        if best is None or rank > best[0] or (rank == best[0] and key < best[1]):
            best = record
    if best is None:
        raise RuntimeError("NEXT73 has no in-range exact score boundary")
    rank, _key, formula, metrics, score, supported, reject = best
    return {
        "selected_formula": formula,
        "discovery_metrics": {
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
        "passes_discovery_gates": bool(rank[0] == 1.0),
        "candidate_threshold_count": candidate_count,
        "term_list_changed": terms != terms_raw,
        "rank": list(rank),
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NEXT73 JSON must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run_exact_threshold_calibration(
    *,
    feature_dir: Path,
    endpoint_firewall_manifest_path: Path,
    discovery_dir: Path,
    design_path: Path,
    frozen_formula_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Calibrate one scalar on discovery and publish the immutable candidate."""

    feature_dir = Path(feature_dir).resolve()
    discovery_dir = Path(discovery_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "features": feature_dir / SOURCE_FEATURES_NAME,
        "feature_manifest": feature_dir / MANIFEST_NAME,
        "endpoint_firewall": Path(endpoint_firewall_manifest_path).resolve(),
        "discovery_labels": discovery_dir / ROLE_LABELS_NAME,
        "discovery_manifest": discovery_dir / ROLE_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
        "frozen_formula": Path(frozen_formula_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT73 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "features": EXPECTED_FEATURE_SHA256,
        "feature_manifest": EXPECTED_FEATURE_MANIFEST_SHA256,
        "endpoint_firewall": EXPECTED_ENDPOINT_FIREWALL_SHA256,
        "discovery_manifest": EXPECTED_DISCOVERY_MANIFEST_SHA256,
        "discovery_labels": EXPECTED_DISCOVERY_LABEL_SHA256,
        "design": DESIGN_SHA256,
        "frozen_formula": EXPECTED_FROZEN_FORMULA_SHA256,
    }
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("NEXT73 frozen input hash differs")
    feature_manifest = _read_json(paths["feature_manifest"])
    endpoint_firewall = _read_json(paths["endpoint_firewall"])
    discovery_manifest = _read_json(paths["discovery_manifest"])
    frozen_formula = _read_json(paths["frozen_formula"])
    feature_outputs = feature_manifest.get("outputs_sha256")
    discovery_outputs = discovery_manifest.get("outputs_sha256")
    if (
        feature_manifest.get("protocol") != SOURCE_FEATURE_PROTOCOL
        or feature_manifest.get("labels_opened") is not False
        or not isinstance(feature_outputs, Mapping)
        or feature_outputs.get(SOURCE_FEATURES_NAME) != hashes["features"]
        or endpoint_firewall.get("protocol") != ENDPOINT_PROTOCOL
        or endpoint_firewall.get("internal_validation_endpoint_values_summarized_or_inspected") is not False
        or endpoint_firewall.get("internal_replication_endpoint_values_summarized_or_inspected") is not False
        or discovery_manifest.get("protocol") != ENDPOINT_PROTOCOL
        or discovery_manifest.get("partition_role") != "discovery"
        or not isinstance(discovery_outputs, Mapping)
        or discovery_outputs.get(ROLE_LABELS_NAME) != hashes["discovery_labels"]
        or frozen_formula.get("protocol") != FROZEN_PROTOCOL
    ):
        raise ValueError("NEXT73 discovery-only provenance differs")
    features_all = pd.read_parquet(paths["features"])
    features = features_all[features_all["partition_role"].eq("discovery")].copy()
    labels = pd.read_parquet(paths["discovery_labels"])
    if set(labels["partition_role"]) != {"discovery"}:
        raise ValueError("NEXT73 received non-discovery labels")
    joined = features.merge(labels, on="material_id", how="inner", validate="one_to_one")
    if len(joined) != len(labels):
        raise ValueError("NEXT73 discovery identity differs")
    result = calibrate_exact_threshold(
        features=joined,
        endpoint=joined[ENDPOINT_COLUMN].to_numpy(float),
        frozen_formula=frozen_formula,
    )
    formula = {
        **result["selected_formula"],
        "protocol": PROTOCOL,
        "training_partition": "ODAC23 official train / robust discovery only",
        "formula_family": "exact NEXT72 five-term list with scalar threshold calibration only",
        "frozen_formula_sha256": hashes["frozen_formula"],
        "endpoint_definition": {
            "column": ENDPOINT_COLUMN,
            "protected_max_angstrom": PROTECTED_MAX,
            "severe_min_angstrom": SEVERE_MIN,
            "minimum_adsorbate_configurations": 4,
            "common_translation_removed": True,
        },
        "gates": GATES,
        "feature_artifact_sha256": hashes["features"],
        "scientific_status": "advance_to_internal_validation"
        if result["passes_discovery_gates"]
        else "discovery_failure_diagnostic_only",
    }
    search_record = {
        key: value
        for key, value in result.items()
        if key not in {"score", "supported", "reject", "selected_formula"}
    }
    endpoint = joined[ENDPOINT_COLUMN].to_numpy(float)
    predictions = pd.DataFrame(
        {
            "material_id": joined["material_id"].astype(str),
            "partition_role": "discovery",
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
        "mode": "exact_threshold_only_robust_discovery_calibration",
        "robust_discovery_labels_opened": True,
        "internal_validation_labels_opened": False,
        "internal_replication_labels_opened": False,
        "official_validation_or_test_payload_deserialized": False,
        "term_list_or_weights_changed": False,
        "dft_calculation_executed": False,
        "dft_values_used_by_executable_formula": False,
        "relaxed_coordinates_used_by_executable_formula": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "executable_is_fixed_explicit_weighted_sum": True,
        "passes_discovery_gates": result["passes_discovery_gates"],
        "counts": {
            "rows": len(joined),
            "protected": int(predictions["protected"].sum()),
            "severe": int(predictions["severe"].sum()),
            "supported": int(predictions["supported"].sum()),
            "rejected": int(predictions["reject"].sum()),
            "candidate_thresholds": int(result["candidate_threshold_count"]),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next73_odac23_exact_threshold_calibration.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        formula_path = staging / FORMULA_NAME
        search_path = staging / SEARCH_NAME
        predictions_path = staging / PREDICTIONS_NAME
        formula_path.write_bytes(_json_bytes(formula))
        search_path.write_bytes(_json_bytes(search_record))
        predictions.to_parquet(predictions_path, index=False)
        manifest["outputs_sha256"] = {
            path.name: _sha256(path) for path in (formula_path, search_path, predictions_path)
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT73 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT73 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--endpoint-firewall", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--frozen-formula", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_exact_threshold_calibration(
        feature_dir=args.feature_dir,
        endpoint_firewall_manifest_path=args.endpoint_firewall,
        discovery_dir=args.discovery_dir,
        design_path=args.design,
        frozen_formula_path=args.frozen_formula,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"passes": manifest["passes_discovery_gates"], **manifest["counts"]},
            indent=2,
            sort_keys=True,
        )
    )


__all__ = ["PROTOCOL", "calibrate_exact_threshold", "run_exact_threshold_calibration"]


if __name__ == "__main__":
    main()
