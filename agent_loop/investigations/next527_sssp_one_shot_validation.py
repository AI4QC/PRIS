#!/usr/bin/env python3
"""One-shot dual-source internal validation of the frozen standalone SSSP law."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import src.next411_same_sign_shell_purity as n411
import src.next525_sssp_standalone_freeze as n525
import src.next526_sssp_holdout_feature_freeze as n526
from src.next23_evaluate import _roc_auc
from src.next87_scigen_sparse_law_search import (
    _pauling_baseline,
    assign_group_folds,
    decision_metrics,
)
from src.next95_wyformer_sparse_law_search import _endpoint_numeric
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next527-sssp-one-shot-dual-source-validation-v1"
ROLE = "internal_validation"
DESIGN_SHA256 = n525.DESIGN_SHA256
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_DRAWS = 1_000
MINIMUM_FOLD_CLASS_COUNT = 10
GATES = {
    "coverage_lower": 0.90,
    "protected_recall_lower": 0.95,
    "severe_rejection_precision_lower": 0.60,
    "savings_lower": 0.02,
    "pooled_auc": 0.60,
    "macro_fold_auc": 0.60,
    "worst_fold_auc": 0.55,
    "cluster_bootstrap_auc_lower": 0.50,
}
MANIFEST_NAME = "MANIFEST.json"
EVALUATION_NAME = "NEXT527_SSSP_INTERNAL_VALIDATION.json"
PREDICTION_NAMES = {
    source: f"next527_{source}_sssp_internal_validation_predictions.parquet"
    for source in ("scigen", "wyformer")
}
EXPECTED_INPUT_SHA256 = {
    "design": DESIGN_SHA256,
    "next525_manifest": "e15217dafaa1d86dc5a70640dd0ab96a99a9cc0bb04eff44ca850c88e4ff3140",
    "next525_formula": "e98f7cf1bf6d0947b653c133100495650a57265dddde46ce8e2c4dd9521e09cf",
    "next526_manifest": "2f9c196a04312679223bf1658670ba999fa640a1e8b84c841745a22261f9662b",
    "next526_catalogue": "7f63cd81f839c76b60e339ecc94a12004c7c0785fd02a5744bba4c838ee4e502",
    "next526_scigen": "a0901cf494d15ca277acf4d3fc5dc8ea617cfc09337c0cef65c431e1ad3273f7",
    "next526_wyformer": "8bdbc68b7db502a1af886637071553c900ca6916f0f4d5efff67098c9220782c",
    "scigen_feature_manifest": "8dcb8118f85ee4a3acbf0905f01c2b173d58742a1e16dcd6004adbbbedcf63cc",
    "scigen_features": "f266e6143bc23d9e131b5ec788676b520db928aa46a57a1fcba6fd8530a80c8a",
    "wyformer_feature_manifest": "fb66f7c5caade419a46b9a3fa6fef1bc5b3afa3eebeb95a4bc53baddabc0f659",
    "wyformer_features": "26d95746e8aa56087150737a62035f5d4c5ce51b1d2e10424ed6cb267ea1983c",
    "scigen_endpoint_manifest": "38b491ca3f1cc1143f2188c77de3124746ac557e7d38aac849a24dc47c2b399d",
    "scigen_endpoint": "22dd427bd63f0769c0bdb6ee786acfcaeb1b384e30a17fc0882bf0db40477807",
    "wyformer_endpoint_manifest": "d5b87e8e2902fb14112e20cf60fb0c59593c7586655f19e1b248131f8df2cd7f",
    "wyformer_endpoint": "514d8ba4ac9e335f9ffced15f021b76e4573ad263fe26a438aaeaece6ad128f5",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _cluster_bootstrap_auc(
    *, score: np.ndarray, endpoint: np.ndarray, supported: np.ndarray,
    groups: np.ndarray, draws: int, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if type(draws) is not int or draws <= 0:
        raise ValueError("NEXT527 bootstrap draws differ")
    extreme = supported & np.isfinite(score) & ((endpoint <= 1.0) | (endpoint >= 2.0))
    unique = np.unique(groups[extreme].astype(str))
    if len(unique) < 2:
        raise ValueError("NEXT527 bootstrap groups differ")
    row_groups = groups.astype(str)
    by_group = {group: np.flatnonzero(extreme & (row_groups == group)) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        auc = _roc_auc(score[indices], endpoint[indices] >= 2.0)
        if auc is not None:
            values.append(float(auc))
    if len(values) != draws:
        raise RuntimeError("NEXT527 bootstrap produced a one-class draw")
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def evaluate_sssp_source(
    *, frame: pd.DataFrame, threshold: float,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    """Evaluate the exact frozen direction and cutoff on one source."""

    required = {
        "material_id", "reduced_formula", n411.FEATURE_NAMES[0],
        "sssp_supported", "pauling_p2_p5_decision", "endpoint",
    }
    decisions = (
        frame["pauling_p2_p5_decision"].astype(str).to_numpy()
        if "pauling_p2_p5_decision" in frame else np.asarray([], dtype=str)
    )
    endpoint = (
        pd.to_numeric(frame["endpoint"], errors="coerce").to_numpy(float)
        if "endpoint" in frame else np.asarray([], dtype=float)
    )
    if (
        required - set(frame)
        or frame["material_id"].astype(str).duplicated().any()
        or not math.isfinite(float(threshold))
        or float(threshold) != n525.EXPECTED_THRESHOLD
        or not np.isfinite(endpoint).all()
        or set(np.unique(decisions)) - {"KEEP", "REJECT", "ABSTAIN"}
        or not (endpoint <= 1.0).any()
        or not (endpoint >= 2.0).any()
    ):
        raise ValueError("NEXT527 evaluation frame differs")
    sssp = pd.to_numeric(frame[n411.FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
    declared_support = frame["sssp_supported"].fillna(False).to_numpy(bool)
    supported = declared_support & np.isfinite(sssp)
    if not np.array_equal(declared_support, supported):
        raise ValueError("NEXT527 support mask differs")
    score = -sssp
    reject = supported & (sssp <= float(threshold))
    metrics = decision_metrics(supported=supported, reject=reject, distortion_ratio=endpoint)
    extreme = supported & ((endpoint <= 1.0) | (endpoint >= 2.0))
    pooled = _roc_auc(score[extreme], endpoint[extreme] >= 2.0)
    folds = assign_group_folds(frame["reduced_formula"].astype(str).to_numpy())
    fold_records = []
    fold_aucs = []
    all_fold_counts_pass = True
    for fold in range(5):
        mask = extreme & (folds == fold)
        protected = int((mask & (endpoint <= 1.0)).sum())
        severe = int((mask & (endpoint >= 2.0)).sum())
        auc = _roc_auc(score[mask], endpoint[mask] >= 2.0) if protected and severe else None
        count_pass = protected >= MINIMUM_FOLD_CLASS_COUNT and severe >= MINIMUM_FOLD_CLASS_COUNT
        all_fold_counts_pass &= bool(count_pass and auc is not None)
        fold_records.append(
            {
                "fold": fold, "supported_extremes": int(mask.sum()),
                "protected": protected, "severe": severe, "auc": auc,
                "passes_class_count": count_pass,
            }
        )
        if auc is not None:
            fold_aucs.append(float(auc))
    macro = float(np.mean(fold_aucs)) if len(fold_aucs) == 5 else None
    worst = float(np.min(fold_aucs)) if len(fold_aucs) == 5 else None
    interval = _cluster_bootstrap_auc(
        score=score, endpoint=endpoint, supported=supported,
        groups=frame["reduced_formula"].astype(str).to_numpy(),
        draws=bootstrap_draws,
    )
    pauling = _pauling_baseline(frame, endpoint)
    all_extremes = (endpoint <= 1.0) | (endpoint >= 2.0)
    binary_auc = _roc_auc(
        reject[all_extremes].astype(float), endpoint[all_extremes] >= 2.0
    )
    dominance = {
        "binary_reject_auc": bool(
            binary_auc is not None
            and pauling["binary_reject_auc_all_extremes"] is not None
            and float(binary_auc) > float(pauling["binary_reject_auc_all_extremes"])
        ),
        "coverage_lower": float(metrics["coverage_lower"]) > float(pauling["coverage_lower"]),
        "protected_recall_lower": float(metrics["protected_recall_lower"])
        > float(pauling["protected_recall_lower"]),
        "severe_rejection_precision_lower": float(metrics["severe_rejection_precision_lower"])
        > float(pauling["severe_rejection_precision_lower"]),
    }
    dominance["passes_all"] = all(dominance.values())
    gate_checks = {
        "coverage_lower": float(metrics["coverage_lower"]) >= GATES["coverage_lower"],
        "protected_recall_lower": float(metrics["protected_recall_lower"])
        >= GATES["protected_recall_lower"],
        "severe_rejection_precision_lower": float(metrics["severe_rejection_precision_lower"])
        >= GATES["severe_rejection_precision_lower"],
        "savings_lower": float(metrics["savings_lower"]) >= GATES["savings_lower"],
        "pooled_auc": pooled is not None and float(pooled) >= GATES["pooled_auc"],
        "macro_fold_auc": macro is not None and macro >= GATES["macro_fold_auc"],
        "worst_fold_auc": worst is not None and worst >= GATES["worst_fold_auc"],
        "fold_class_counts": bool(all_fold_counts_pass),
        "cluster_bootstrap_auc_lower": interval[0] > GATES["cluster_bootstrap_auc_lower"],
        "pauling_dominance": bool(dominance["passes_all"]),
    }
    return {
        "threshold": float(threshold),
        "formula_or_threshold_modified": False,
        "metrics": metrics,
        "pooled_auc": pooled,
        "macro_fold_auc": macro,
        "worst_fold_auc": worst,
        "fold_records": fold_records,
        "cluster_bootstrap_auc_95": list(interval),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sssp_binary_reject_auc_all_extremes": binary_auc,
        "pauling": pauling,
        "pauling_dominance": dominance,
        "gate_checks": gate_checks,
        "passes_source_gates": all(gate_checks.values()),
        "score": score,
        "supported": supported,
        "reject": reject,
    }


def _join_source(
    *, source: str, sssp: pd.DataFrame, base: pd.DataFrame, endpoint: pd.DataFrame
) -> pd.DataFrame:
    if source == "scigen":
        endpoint_column = "distortion_ratio"
    elif source == "wyformer":
        endpoint_column = "endpoint_stratum"
    else:
        raise ValueError("NEXT527 source differs")
    sssp_required = {
        "material_id", "reduced_formula", "partition_role",
        n411.FEATURE_NAMES[0], "sssp_supported",
    }
    if (
        sssp_required - set(sssp)
        or {"material_id", "pauling_p2_p5_decision"} - set(base)
        or {"material_id", endpoint_column} - set(endpoint)
        or any(frame["material_id"].astype(str).duplicated().any() for frame in (sssp, base, endpoint))
        or set(sssp["partition_role"].astype(str)) != {ROLE}
    ):
        raise ValueError(f"NEXT527 {source} input schema differs")
    joined = (
        sssp.merge(
            base[["material_id", "pauling_p2_p5_decision"]],
            on="material_id", validate="one_to_one",
        ).merge(
            endpoint[["material_id", endpoint_column]],
            on="material_id", validate="one_to_one",
        )
    )
    if len(joined) != len(sssp) or len(joined) != len(base) or len(joined) != len(endpoint):
        raise ValueError(f"NEXT527 {source} material identity differs")
    joined["endpoint"] = (
        pd.to_numeric(joined[endpoint_column], errors="coerce").to_numpy(float)
        if source == "scigen" else _endpoint_numeric(joined[endpoint_column])
    )
    return joined


def run_one_shot_validation(
    *,
    next525_dir: Path,
    next526_dir: Path,
    scigen_feature_dir: Path,
    wyformer_feature_dir: Path,
    scigen_endpoint_dir: Path,
    wyformer_endpoint_dir: Path,
    design_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Open validation endpoints once; replication remains physically unread."""

    freeze, holdout = Path(next525_dir).resolve(), Path(next526_dir).resolve()
    scigen_features, wyformer_features = Path(scigen_feature_dir).resolve(), Path(wyformer_feature_dir).resolve()
    scigen_endpoint, wyformer_endpoint = Path(scigen_endpoint_dir).resolve(), Path(wyformer_endpoint_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next525_manifest": freeze / n525.MANIFEST_NAME,
        "next525_formula": freeze / n525.FORMULA_NAME,
        "next526_manifest": holdout / n526.MANIFEST_NAME,
        "next526_catalogue": holdout / n526.CATALOGUE_NAME,
        "next526_scigen": holdout / n526.FEATURE_FILES["scigen"][ROLE],
        "next526_wyformer": holdout / n526.FEATURE_FILES["wyformer"][ROLE],
        "scigen_feature_manifest": scigen_features / "MANIFEST.json",
        "scigen_features": scigen_features / "features_internal_validation.parquet",
        "wyformer_feature_manifest": wyformer_features / "MANIFEST.json",
        "wyformer_features": wyformer_features / "wyformer_x0_features_internal_validation.parquet",
        "scigen_endpoint_manifest": scigen_endpoint / "MANIFEST.json",
        "scigen_endpoint": scigen_endpoint / "scigen_dft_distortion_endpoints.parquet",
        "wyformer_endpoint_manifest": wyformer_endpoint / "MANIFEST.json",
        "wyformer_endpoint": wyformer_endpoint / "wyformer_dft_screening_endpoints.parquet",
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT527 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT527 formal input identity differs: {differing}")
    manifests = {
        name: _read_json(paths[name])
        for name in (
            "next525_manifest", "next526_manifest", "scigen_feature_manifest",
            "wyformer_feature_manifest", "scigen_endpoint_manifest",
            "wyformer_endpoint_manifest",
        )
    }
    formula = _read_json(paths["next525_formula"])
    if (
        manifests["next525_manifest"].get("protocol") != n525.PROTOCOL
        or manifests["next526_manifest"].get("protocol") != n526.PROTOCOL
        or manifests["next526_manifest"].get("next527_internal_validation_authorized") is not True
        or manifests["next526_manifest"].get("validation_endpoint_opened") is not False
        or manifests["next526_manifest"].get("replication_endpoint_opened") is not False
        or manifests["scigen_feature_manifest"].get("labels_opened") is not False
        or manifests["wyformer_feature_manifest"].get("labels_opened") is not False
        or manifests["scigen_endpoint_manifest"].get("partition_role") != ROLE
        or manifests["scigen_endpoint_manifest"].get("endpoint_values_summarized_or_inspected") is not False
        or manifests["wyformer_endpoint_manifest"].get("partition_role") != ROLE
        or manifests["wyformer_endpoint_manifest"].get("endpoint_payload_opened") is not False
        or formula.get("protocol") != n525.PROTOCOL
        or float(formula.get("threshold", math.nan)) != n525.EXPECTED_THRESHOLD
        or formula.get("dft_inputs") != []
        or formula.get("learned_model_inputs") != []
        or formula.get("relaxation_inputs") != []
    ):
        raise ValueError("NEXT527 frozen provenance differs")
    # Keep all endpoint deserialization below the provenance and hash firewall.
    joined = {
        "scigen": _join_source(
            source="scigen",
            sssp=pd.read_parquet(paths["next526_scigen"]),
            base=pd.read_parquet(paths["scigen_features"]),
            endpoint=pd.read_parquet(paths["scigen_endpoint"]),
        ),
        "wyformer": _join_source(
            source="wyformer",
            sssp=pd.read_parquet(paths["next526_wyformer"]),
            base=pd.read_parquet(paths["wyformer_features"]),
            endpoint=pd.read_parquet(paths["wyformer_endpoint"]),
        ),
    }
    results = {
        source: evaluate_sssp_source(frame=frame, threshold=float(formula["threshold"]))
        for source, frame in joined.items()
    }
    passes = all(bool(result["passes_source_gates"]) for result in results.values())
    evaluation = {
        "protocol": PROTOCOL,
        "partition_role": ROLE,
        "gates": GATES,
        "minimum_fold_class_count": MINIMUM_FOLD_CLASS_COUNT,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "formula_or_threshold_modified": False,
        "sources": {
            source: {
                key: value for key, value in result.items()
                if key not in {"score", "supported", "reject"}
            }
            for source, result in results.items()
        },
        "passes_all_validation_gates": passes,
        "scientific_status": (
            "advance_to_internal_replication" if passes
            else "internal_validation_failure_stop"
        ),
    }
    predictions = {}
    for source, frame in joined.items():
        result = results[source]
        predictions[source] = pd.DataFrame(
            {
                "material_id": frame["material_id"].astype(str),
                "reduced_formula": frame["reduced_formula"].astype(str),
                "partition_role": ROLE,
                "endpoint": frame["endpoint"].to_numpy(float),
                n411.FEATURE_NAMES[0]: pd.to_numeric(
                    frame[n411.FEATURE_NAMES[0]], errors="coerce"
                ),
                "supported": result["supported"],
                "risk_score": result["score"],
                "reject": result["reject"],
                "pauling_p2_p5_decision": frame["pauling_p2_p5_decision"].astype(str),
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_hash = _sha256_file(Path(__file__).resolve())
    try:
        evaluation_path = staging / EVALUATION_NAME
        evaluation_path.write_bytes(_json_bytes(evaluation))
        output_paths = [evaluation_path]
        for source, frame in predictions.items():
            path = staging / PREDICTION_NAMES[source]
            frame.to_parquet(path, index=False)
            output_paths.append(path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "one_shot_frozen_sssp_dual_source_internal_validation",
            "internal_validation_endpoint_values_opened": True,
            "internal_replication_endpoint_values_opened": False,
            "formula_or_threshold_modified": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "passes_all_validation_gates": passes,
            "next528_internal_replication_authorized": passes,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next527_sssp_one_shot_validation.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(Path(__file__).resolve()) != source_hash:
            raise RuntimeError("NEXT527 source changed before publication")
        if any(_sha256_file(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT527 input changed before publication")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next525-dir", type=Path, required=True)
    parser.add_argument("--next526-dir", type=Path, required=True)
    parser.add_argument("--scigen-feature-dir", type=Path, required=True)
    parser.add_argument("--wyformer-feature-dir", type=Path, required=True)
    parser.add_argument("--scigen-endpoint-dir", type=Path, required=True)
    parser.add_argument("--wyformer-endpoint-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_one_shot_validation(
        next525_dir=args.next525_dir,
        next526_dir=args.next526_dir,
        scigen_feature_dir=args.scigen_feature_dir,
        wyformer_feature_dir=args.wyformer_feature_dir,
        scigen_endpoint_dir=args.scigen_endpoint_dir,
        wyformer_endpoint_dir=args.wyformer_endpoint_dir,
        design_path=args.design,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "passes": manifest["passes_all_validation_gates"],
        "replication_authorized": manifest["next528_internal_replication_authorized"],
    }, indent=2))


__all__ = [
    "BOOTSTRAP_SEED", "EVALUATION_NAME", "GATES", "MANIFEST_NAME", "PROTOCOL",
    "ROLE", "evaluate_sssp_source", "run_one_shot_validation",
]


if __name__ == "__main__":
    main()
