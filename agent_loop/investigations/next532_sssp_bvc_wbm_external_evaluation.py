#!/usr/bin/env python3
"""One-shot external WBM relaxation-change evaluation of frozen SSSP-BVC."""

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

import src.next529_sssp_bvc_development_freeze as n529
import src.next530_sssp_bvc_wbm_cohort as n530
import src.next531_sssp_bvc_wbm_features as n531
from src.next23_evaluate import _roc_auc
from src.next23_relaxation_rule import wilson_lower_bound
from src.next87_scigen_sparse_law_search import assign_group_folds
from src.next347_periodic_allocation_redistribution_capacity import _sha256_file


PROTOCOL = "2026-08-13-next532-sssp-bvc-wbm-one-shot-external-evaluation-v1"
ROLE = "external_validation"
ENDPOINT_COLUMN = "site_stats_fingerprint_init_final_norm_diff"
ENDPOINT_COLUMNS_OPENED = ("material_id", ENDPOINT_COLUMN)
PROTECTED_MAX = 0.10
SEVERE_MIN = 0.50
MINIMUM_FOLD_CLASS_COUNT = 20
EXECUTABLE_INPUT_BOUNDARY = ("composition", "one raw initial fully periodic geometry")
PROHIBITED_EXECUTABLE_INPUTS = (
    "DFT values", "relaxed or later geometry", "trajectory",
    "learned energy force stress", "MLIP or proxy potential", "relaxation",
)
GATES = {
    "coverage_lower": 0.90,
    "protected_recall_lower": 0.95,
    "severe_rejection_precision_lower": 0.60,
    "savings_lower": 0.02,
    "minimum_rejected_extremes": 25,
    "minimum_rejected_severe": 10,
    "binary_reject_auc": 0.50,
}
MANIFEST_NAME = "MANIFEST.json"
EVALUATION_NAME = "NEXT532_SSSP_BVC_WBM_EXTERNAL_EVALUATION.json"
JOINED_NAME = "next532_sssp_bvc_wbm_external_predictions_labels.parquet"
EXPECTED_INPUT_SHA256 = {
    "design": "208f1ee146397520e26a827169f418ebcb04df090c50c54535232eb3a8ec83cb",
    "next529_source": "b0d1355cfc067480df897a12fdaef6c6e7ba9b681b86f9b9a7aaf51df1185e40",
    "next529_manifest": "3f5bfa89726bfa7edc8daa898169c3e9259c5d3d29e1d12c2674fb4343f17705",
    "next529_formula": "b50e194273e83f06e26bd4f4e9c904cd692dc9fa9d874aebb0181c4fcfa849be",
    "next530_manifest": "c794f740d056816c9cefd3acef61a17a3ffb7aaf2143061cb93d41870ab9bb6b",
    "next530_metadata": "f9922c13dccd4b3f4b1fad8f991f16910ed92d8c88e6ccf437469c435da318b5",
    "next531_source": "755540fedeb49c1eb1637c079121e14c9fcd92bc9a419490321273ac5741eb07",
    "next531_manifest": "34a38c83a431ed7bc2484a02d3472b7bd2393c681e6bf441919038b02c9937cc",
    "next531_catalogue": "769eaa5a84be3f4c6c49c8f5451c2e097e01c1630e0db8ee3148024d10418748",
    "next531_predictions": "dbefd869cb92b3422703f78e9e23eed1d00f3d05defb935b08250c7aeeae7446",
    "wbm_summary": "ff19e59d74115de9762fbc868c9f35900ae099c18f23e9c89d10589af1418225",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, endpoint: np.ndarray
) -> dict[str, object]:
    support = np.asarray(supported, dtype=bool)
    rejected = np.asarray(reject, dtype=bool)
    values = np.asarray(endpoint, dtype=float)
    if (
        values.ndim != 1 or support.shape != values.shape or rejected.shape != values.shape
        or not np.isfinite(values).all()
    ):
        raise ValueError("NEXT532 decision arrays differ")
    rejected &= support
    protected = values <= PROTECTED_MAX
    severe = values >= SEVERE_MIN
    extremes = protected | severe
    rows = len(values)
    n_supported = int(support.sum())
    n_rejected = int(rejected.sum())
    n_protected = int(protected.sum())
    protected_kept = int((protected & ~rejected).sum())
    rejected_extremes = int((extremes & rejected).sum())
    severe_total = int(severe.sum())
    severe_rejected = int((severe & rejected).sum())
    return {
        "rows": rows,
        "supported": n_supported,
        "rejected": n_rejected,
        "protected": n_protected,
        "protected_kept": protected_kept,
        "rejected_extremes": rejected_extremes,
        "severe_total": severe_total,
        "severe_rejected": severe_rejected,
        "coverage": n_supported / rows if rows else 0.0,
        "coverage_lower": wilson_lower_bound(n_supported, rows),
        "protected_recall": protected_kept / n_protected if n_protected else 0.0,
        "protected_recall_lower": wilson_lower_bound(protected_kept, n_protected),
        "severe_rejection_precision": (
            severe_rejected / rejected_extremes if rejected_extremes else 0.0
        ),
        "severe_rejection_precision_lower": wilson_lower_bound(
            severe_rejected, rejected_extremes
        ),
        "savings": n_rejected / rows if rows else 0.0,
        "savings_lower": wilson_lower_bound(n_rejected, rows),
        "severe_recall": severe_rejected / severe_total if severe_total else 0.0,
    }


def _continuous_auc(
    *, score: np.ndarray, supported: np.ndarray, endpoint: np.ndarray
) -> dict[str, object]:
    values = np.asarray(score, dtype=float)
    support = np.asarray(supported, dtype=bool) & np.isfinite(values)
    extremes = support & ((endpoint <= PROTECTED_MAX) | (endpoint >= SEVERE_MIN))
    auc = _roc_auc(values[extremes], endpoint[extremes] >= SEVERE_MIN)
    return {"supported_extremes": int(extremes.sum()), "extreme_auc": auc}


def _pauling_baseline(frame: pd.DataFrame, endpoint: np.ndarray) -> dict[str, object]:
    decisions = frame["pauling_p2_p5_decision"].astype(str).to_numpy()
    if set(np.unique(decisions)) - {"KEEP", "REJECT", "ABSTAIN"}:
        raise ValueError("NEXT532 Pauling decision schema differs")
    supported = decisions != "ABSTAIN"
    reject = decisions == "REJECT"
    metrics = _decision_metrics(supported=supported, reject=reject, endpoint=endpoint)
    extremes = (endpoint <= PROTECTED_MAX) | (endpoint >= SEVERE_MIN)
    auc = _roc_auc(reject[extremes].astype(float), endpoint[extremes] >= SEVERE_MIN)
    return {
        **metrics,
        "binary_reject_auc_all_extremes": auc,
        "available": bool(supported.any()),
    }


def evaluate_wbm_external(*, frame: pd.DataFrame) -> dict[str, object]:
    """Evaluate the already-frozen predictions; never fit or repair a direction."""

    required = {
        "material_id", "rk", "sssp_same_sign_shell_purity_q10", "sssp_supported",
        n529.SCBV_FEATURE, "scbv_supported", "risk_score", "formula_supported",
        "reject", "sssp_bvc_decision", "pauling_p2_p5_decision", "endpoint",
    }
    endpoint = (
        pd.to_numeric(frame["endpoint"], errors="coerce").to_numpy(float)
        if "endpoint" in frame else np.asarray([], dtype=float)
    )
    decisions = (
        frame["sssp_bvc_decision"].astype(str).to_numpy()
        if "sssp_bvc_decision" in frame else np.asarray([], dtype=str)
    )
    if (
        required - set(frame) or not len(frame)
        or frame["material_id"].astype(str).duplicated().any()
        or not np.isfinite(endpoint).all()
        or not (endpoint <= PROTECTED_MAX).any()
        or not (endpoint >= SEVERE_MIN).any()
        or set(np.unique(decisions)) - {"KEEP", "REJECT", "ABSTAIN"}
    ):
        raise ValueError("NEXT532 evaluation frame differs")

    sssp = pd.to_numeric(
        frame["sssp_same_sign_shell_purity_q10"], errors="coerce"
    ).to_numpy(float)
    scbv = pd.to_numeric(frame[n529.SCBV_FEATURE], errors="coerce").to_numpy(float)
    sssp_supported = frame["sssp_supported"].fillna(False).to_numpy(bool)
    scbv_supported = frame["scbv_supported"].fillna(False).to_numpy(bool)
    applied = n529.apply_sssp_bvc(
        sssp=sssp,
        sssp_supported=sssp_supported,
        scbv=scbv,
        scbv_supported=scbv_supported,
        sssp_threshold=n529.SSSP_THRESHOLD,
        scbv_threshold=n529.EXPECTED_SCBV_THRESHOLD,
    )
    declared_supported = frame["formula_supported"].fillna(False).to_numpy(bool)
    declared_reject = frame["reject"].fillna(False).to_numpy(bool)
    declared_risk = pd.to_numeric(frame["risk_score"], errors="coerce").to_numpy(float)
    expected_decisions = np.where(
        ~applied["supported"], "ABSTAIN",
        np.where(applied["reject"], "REJECT", "KEEP"),
    )
    if (
        not np.array_equal(declared_supported, applied["supported"])
        or not np.array_equal(declared_reject, applied["reject"])
        or not np.array_equal(decisions, expected_decisions)
        or not np.allclose(declared_risk, applied["risk"], rtol=0.0, atol=0.0, equal_nan=True)
    ):
        raise ValueError("NEXT532 frozen predictions differ")

    metrics = _decision_metrics(
        supported=applied["supported"], reject=applied["reject"], endpoint=endpoint
    )
    extremes = (endpoint <= PROTECTED_MAX) | (endpoint >= SEVERE_MIN)
    binary_auc = _roc_auc(
        applied["reject"][extremes].astype(float), endpoint[extremes] >= SEVERE_MIN
    )
    folds = assign_group_folds(frame["rk"].astype(str).to_numpy())
    fold_records = []
    folds_pass = True
    for fold in range(5):
        mask = applied["supported"] & (folds == fold)
        protected = int((mask & (endpoint <= PROTECTED_MAX)).sum())
        severe = int((mask & (endpoint >= SEVERE_MIN)).sum())
        count_pass = (
            protected >= MINIMUM_FOLD_CLASS_COUNT and severe >= MINIMUM_FOLD_CLASS_COUNT
        )
        folds_pass &= count_pass
        extreme = mask & extremes
        fold_records.append(
            {
                "fold": fold,
                "formula_supported": int(mask.sum()),
                "protected": protected,
                "severe": severe,
                "binary_reject_auc": _roc_auc(
                    applied["reject"][extreme].astype(float),
                    endpoint[extreme] >= SEVERE_MIN,
                ) if protected and severe else None,
                "passes_class_counts": count_pass,
            }
        )

    pauling = _pauling_baseline(frame, endpoint)
    dominance = {
        "applicable": bool(pauling["available"]),
        "binary_reject_auc": bool(
            binary_auc is not None and pauling["binary_reject_auc_all_extremes"] is not None
            and float(binary_auc) > float(pauling["binary_reject_auc_all_extremes"])
        ),
        "coverage_lower": float(metrics["coverage_lower"]) > float(pauling["coverage_lower"]),
        "protected_recall_lower": float(metrics["protected_recall_lower"])
        > float(pauling["protected_recall_lower"]),
        "severe_rejection_precision_lower": float(
            metrics["severe_rejection_precision_lower"]
        ) > float(pauling["severe_rejection_precision_lower"]),
    }
    dominance["passes_all"] = bool(
        not dominance["applicable"]
        or all(dominance[name] for name in (
            "binary_reject_auc", "coverage_lower", "protected_recall_lower",
            "severe_rejection_precision_lower",
        ))
    )
    gate_checks = {
        "coverage_lower": float(metrics["coverage_lower"]) >= GATES["coverage_lower"],
        "protected_recall_lower": float(metrics["protected_recall_lower"])
        >= GATES["protected_recall_lower"],
        "severe_rejection_precision_lower": float(
            metrics["severe_rejection_precision_lower"]
        ) >= GATES["severe_rejection_precision_lower"],
        "savings_lower": float(metrics["savings_lower"]) >= GATES["savings_lower"],
        "minimum_rejected_extremes": int(metrics["rejected_extremes"])
        >= int(GATES["minimum_rejected_extremes"]),
        "minimum_rejected_severe": int(metrics["severe_rejected"])
        >= int(GATES["minimum_rejected_severe"]),
        "five_formula_folds": bool(folds_pass),
        "binary_reject_auc": binary_auc is not None
        and float(binary_auc) > GATES["binary_reject_auc"],
        "pauling_dominance": bool(dominance["passes_all"]),
    }
    continuous = {
        "sssp_negative_risk": _continuous_auc(
            score=-sssp, supported=sssp_supported, endpoint=endpoint
        ),
        "sssp_bvc_risk": _continuous_auc(
            score=applied["risk"], supported=applied["supported"], endpoint=endpoint
        ),
    }
    return {
        "formula_or_threshold_modified": False,
        "metrics": metrics,
        "binary_reject_auc_all_extremes": binary_auc,
        "fold_records": fold_records,
        "continuous_diagnostics_not_gates": continuous,
        "pauling": pauling,
        "pauling_dominance": dominance,
        "gate_checks": gate_checks,
        "passes_all_external_gates": all(gate_checks.values()),
    }


def run_one_shot_external_evaluation(
    *, next529_dir: Path, next530_dir: Path, next531_dir: Path,
    design_path: Path, wbm_summary_path: Path, output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    freeze = Path(next529_dir).resolve()
    cohort = Path(next530_dir).resolve()
    features = Path(next531_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next529_source": Path(n529.__file__).resolve(),
        "next529_manifest": freeze / n529.MANIFEST_NAME,
        "next529_formula": freeze / n529.FORMULA_NAME,
        "next530_manifest": cohort / n530.MANIFEST_NAME,
        "next530_metadata": cohort / n530.METADATA_NAME,
        "next531_source": Path(n531.__file__).resolve(),
        "next531_manifest": features / n531.MANIFEST_NAME,
        "next531_catalogue": features / n531.CATALOGUE_NAME,
        "next531_predictions": features / n531.TABLE_NAME,
        "wbm_summary": Path(wbm_summary_path).resolve(),
    }
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT532 input is missing")
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT532 formal input identity differs: {differing}")

    freeze_manifest = _read_json(paths["next529_manifest"])
    formula = _read_json(paths["next529_formula"])
    cohort_manifest = _read_json(paths["next530_manifest"])
    feature_manifest = _read_json(paths["next531_manifest"])
    if (
        freeze_manifest.get("wbm_external_endpoint_opened") is not False
        or formula.get("dft_inputs") != []
        or formula.get("learned_model_inputs") != []
        or formula.get("relaxation_inputs") != []
        or formula.get("sssp_threshold") != n529.SSSP_THRESHOLD
        or formula.get("scbv_threshold") != n529.EXPECTED_SCBV_THRESHOLD
        or cohort_manifest.get("labels_opened") is not False
        or cohort_manifest.get("wbm_summary_opened") is not False
        or cohort_manifest.get("relaxed_structures_opened") is not False
        or feature_manifest.get("protocol") != n531.PROTOCOL
        or feature_manifest.get("next532_external_evaluation_authorized") is not True
        or feature_manifest.get("wbm_summary_opened") is not False
        or feature_manifest.get("external_endpoint_opened") is not False
        or feature_manifest.get("relaxed_structures_opened") is not False
        or feature_manifest.get("dft_values_used_by_features") is not False
        or feature_manifest.get("formula_or_threshold_modified") is not False
        or not isinstance(feature_manifest.get("counts"), Mapping)
        or feature_manifest["counts"].get("passes") is not True
    ):
        raise ValueError("NEXT532 frozen provenance differs")

    metadata = pd.read_parquet(paths["next530_metadata"], columns=["material_id", "rk"])
    predictions = pd.read_parquet(paths["next531_predictions"])
    if (
        len(metadata) != n530.SAMPLE_SIZE or len(predictions) != n530.SAMPLE_SIZE
        or metadata["material_id"].astype(str).duplicated().any()
        or predictions["material_id"].astype(str).duplicated().any()
    ):
        raise ValueError("NEXT532 frozen cohort differs")
    identities = predictions[["material_id", "rk"]].merge(
        metadata, on="material_id", suffixes=("_prediction", "_cohort"),
        validate="one_to_one",
    )
    if (
        len(identities) != len(predictions)
        or not np.array_equal(
            identities["rk_prediction"].astype(str).to_numpy(),
            identities["rk_cohort"].astype(str).to_numpy(),
        )
    ):
        raise ValueError("NEXT532 frozen material identity differs")

    # Endpoint deserialization is deliberately below every hash and provenance check.
    endpoint = pd.read_csv(paths["wbm_summary"], usecols=list(ENDPOINT_COLUMNS_OPENED))
    if endpoint["material_id"].astype(str).duplicated().any():
        raise ValueError("NEXT532 WBM endpoint identity differs")
    joined = predictions.merge(endpoint, on="material_id", validate="one_to_one")
    if len(joined) != len(predictions):
        raise ValueError("NEXT532 WBM endpoint coverage differs")
    joined = joined.rename(columns={ENDPOINT_COLUMN: "endpoint"})
    result = evaluate_wbm_external(frame=joined)
    evaluation = {
        "protocol": PROTOCOL,
        "partition_role": ROLE,
        "endpoint_column": ENDPOINT_COLUMN,
        "endpoint_strata": {"protected_max": PROTECTED_MAX, "severe_min": SEVERE_MIN},
        "gates": GATES,
        "minimum_fold_class_count": MINIMUM_FOLD_CLASS_COUNT,
        "formula_or_threshold_modified": False,
        **result,
        "scientific_status": (
            "external_validation_pass_independent_report_authorized"
            if result["passes_all_external_gates"]
            else "external_validation_failure_continue_search"
        ),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        (temporary / EVALUATION_NAME).write_bytes(_json_bytes(evaluation))
        joined.to_parquet(temporary / JOINED_NAME, index=False)
        outputs = {
            name: _sha256_file(temporary / name)
            for name in (EVALUATION_NAME, JOINED_NAME)
        }
        manifest = {
            "protocol": PROTOCOL,
            "mode": "one_shot_external_wbm_relaxation_change_evaluation",
            "inputs_sha256": {
                name: {"path": str(paths[name]), "sha256": digest}
                for name, digest in hashes.items()
            },
            "executed_source_sha256": {
                "src/next532_sssp_bvc_wbm_external_evaluation.py": _sha256_file(
                    Path(__file__).resolve()
                )
            },
            "outputs_sha256": outputs,
            "endpoint_columns_opened": list(ENDPOINT_COLUMNS_OPENED),
            "wbm_summary_opened": True,
            "external_endpoint_opened": True,
            "relaxed_structures_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_executable_formula": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "formula_or_threshold_modified": False,
            "passes_all_external_gates": result["passes_all_external_gates"],
            "independent_report_authorized": result["passes_all_external_gates"],
            "canonical_content_modification_authorized": False,
        }
        (temporary / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        shutil.move(str(temporary), str(target))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next529-dir", type=Path, required=True)
    parser.add_argument("--next530-dir", type=Path, required=True)
    parser.add_argument("--next531-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--wbm-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = run_one_shot_external_evaluation(
        next529_dir=args.next529_dir,
        next530_dir=args.next530_dir,
        next531_dir=args.next531_dir,
        design_path=args.design,
        wbm_summary_path=args.wbm_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "passes_all_external_gates": manifest["passes_all_external_gates"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
