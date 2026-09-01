#!/usr/bin/env python3
"""Transport frozen NEXT28 contacts to OMatG and compare identical-cohort Pauling controls."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from ase import Atoms
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next23_evaluate import _roc_auc
from src.next23_relaxation_rule import wilson_lower_bound
from src.next27_periodic_packing import NEXT27_FEATURE_COLUMNS, compute_periodic_features
from src.next28_contact_coordination import (
    APPLICATION_PROTOCOL,
    PREDICTIONS_NAME,
    THRESHOLD,
)


FEATURE_PROTOCOL = "2026-08-03-next29-omatg-fixed-periodic-contact-features-v1"
EVALUATION_PROTOCOL = "2026-08-03-next29-omatg-fixed-contact-retrospective-v1"
NEXT25_HOLDOUT_PROTOCOL = "2026-08-03-next25-omatg-generated-x0-sanitize-v1"
NEXT25_EVALUATION_PROTOCOL = "2026-08-03-next25-omatg-dft-reference-csp-evaluation-v1"
FEATURES_NAME = "next29_periodic_contact_features.parquet"
RESULT_NAME = "NEXT29_OMATG_CONTACT_TRANSPORT.json"
JOINED_NAME = "next29_omatg_contact_joined.parquet"
MANIFEST_NAME = "MANIFEST.json"
FeatureCalculator = Callable[[Atoms], Mapping[str, float]]
PRIMARY_GATES: Mapping[str, float] = {
    "coverage_lower": 0.90,
    "match_protection_recall_lower": 0.95,
    "nonmatch_rejection_precision_lower": 0.90,
    "savings_lower": 0.10,
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _decision_metrics(
    *, supported: np.ndarray, reject: np.ndarray, reference_match: np.ndarray
) -> dict[str, object]:
    supported = np.asarray(supported, dtype=bool)
    reject = np.asarray(reject, dtype=bool) & supported
    reference_match = np.asarray(reference_match, dtype=bool)
    if not (supported.shape == reject.shape == reference_match.shape) or supported.ndim != 1:
        raise ValueError("decision arrays must align")
    rows = len(reference_match)
    supported_count = int(supported.sum())
    rejected = int(reject.sum())
    matches = int(reference_match.sum())
    matched_kept = int((reference_match & ~reject).sum())
    nonmatch = ~reference_match
    nonmatch_count = int(nonmatch.sum())
    nonmatches_rejected = int((nonmatch & reject).sum())
    metrics: dict[str, object] = {
        "rows": rows,
        "supported": supported_count,
        "rejected": rejected,
        "reference_matches": matches,
        "matched_kept": matched_kept,
        "nonmatches": nonmatch_count,
        "nonmatches_rejected": nonmatches_rejected,
        "coverage": supported_count / rows if rows else 0.0,
        "coverage_lower": wilson_lower_bound(supported_count, rows),
        "match_protection_recall": matched_kept / matches if matches else 0.0,
        "match_protection_recall_lower": wilson_lower_bound(matched_kept, matches),
        "nonmatch_rejection_precision": nonmatches_rejected / rejected if rejected else 0.0,
        "nonmatch_rejection_precision_lower": wilson_lower_bound(nonmatches_rejected, rejected),
        "savings": rejected / rows if rows else 0.0,
        "savings_lower": wilson_lower_bound(rejected, rows),
        "nonmatch_recall": nonmatches_rejected / nonmatch_count if nonmatch_count else 0.0,
    }
    metrics["passes_primary_gates"] = all(
        float(metrics[name]) >= cutoff for name, cutoff in PRIMARY_GATES.items()
    )
    return metrics


def build_features(
    *,
    metadata_path: Path,
    geometry_path: Path,
    cohort_manifest_path: Path,
    output_dir: Path,
    feature_calculator: FeatureCalculator = compute_periodic_features,
) -> dict[str, object]:
    """Compute fixed periodic contacts from the OMatG generated x0 archive only."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry": Path(geometry_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role}: {path}")
    hashes = {role: _sha256(path) for role, path in paths.items()}
    cohort = _read_json(paths["cohort_manifest"], "OMatG x0 manifest")
    outputs = cohort.get("outputs_sha256")
    if (
        cohort.get("protocol") != NEXT25_HOLDOUT_PROTOCOL
        or cohort.get("input_role") != "unrelaxed_x0_geometry_only"
        or cohort.get("labels_opened") is not False
        or cohort.get("endpoint_artifacts_opened") is not False
        or cohort.get("relaxed_structures_opened") is not False
        or cohort.get("model_or_proxy_potential_used") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(paths["metadata"].name) != hashes["metadata"]
        or outputs.get(paths["geometry"].name) != hashes["geometry"]
    ):
        raise ValueError("OMatG x0 cohort crossed the no-DFT boundary")
    metadata = pd.read_parquet(paths["metadata"])
    if (
        "material_id" not in metadata
        or metadata.material_id.isna().any()
        or metadata.material_id.duplicated().any()
        or not metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ):
        raise ValueError("invalid OMatG x0 metadata")
    metadata = metadata.sort_values("material_id", kind="stable").reset_index(drop=True)
    ids = metadata.material_id.astype(str).tolist()
    loaded_ids, structures = _load_archive_only(paths["geometry"], ids)
    if loaded_ids != ids:
        raise ValueError("OMatG geometry and metadata identities differ")
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for material_id, atoms in zip(ids, structures, strict=True):
        try:
            values = dict(feature_calculator(atoms))
            error = None
        except Exception as exc:
            values = {}
            error = f"{type(exc).__name__}: {exc}"
        row: dict[str, object] = {"material_id": material_id, "contact_feature_error": error}
        for name in NEXT27_FEATURE_COLUMNS:
            row[name] = values.get(name, np.nan)
        rows.append(row)
    features = pd.DataFrame(rows)
    finite = np.isfinite(features.loc[:, NEXT27_FEATURE_COLUMNS].to_numpy(float)).all(axis=1)
    features["analytic_supported"] = finite & features.contact_feature_error.isna().to_numpy()
    elapsed = time.perf_counter() - started
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / FEATURES_NAME
        features.to_parquet(output_path, index=False)
        source_hashes = {"src/next29_omatg_contact_transport.py": _sha256(Path(__file__).resolve())}
        manifest = {
            "protocol": FEATURE_PROTOCOL,
            "input_role": "unrelaxed_x0_geometry_only",
            "labels_opened": False,
            "endpoint_fields_read": False,
            "relaxed_structures_opened": False,
            "model_or_proxy_potential_used": False,
            "same_composition_candidates_used": False,
            "formula_or_threshold_changed": False,
            "next28_threshold": THRESHOLD,
            "counts": {
                "rows": len(features),
                "supported": int(features.analytic_supported.sum()),
                "feature_errors": int(features.contact_feature_error.notna().sum()),
            },
            "execution": {"wall_time_seconds": elapsed},
            "inputs_sha256": {
                role: {"path": str(path), "sha256": hashes[role]} for role, path in paths.items()
            },
            "outputs_sha256": {FEATURES_NAME: _sha256(output_path)},
            "executed_source_sha256": source_hashes,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for role, path in paths.items():
            if _sha256(path) != hashes[role]:
                raise RuntimeError(f"input {role} changed during feature build")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def evaluate_frames(
    *, predictions: pd.DataFrame, endpoint_and_pauling: pd.DataFrame
) -> dict[str, object]:
    """Evaluate fixed decisions and every Pauling control on exactly the same identities."""

    required_predictions = {"material_id", "analytic_supported", "next28_risk_score", "reject"}
    required_endpoint = {"material_id", "reference_match", "corrected_rmsd"}
    pauling_columns = {
        "pauling_p2": "pauling_p2_decision",
        "pauling_p3": "pauling_p3_decision",
        "pauling_p4": "pauling_p4_decision",
        "pauling_p5": "pauling_p5_decision",
        "pauling_p2_p5_combined": "pauling_p2_p5_decision",
    }
    if not required_predictions.issubset(predictions.columns):
        raise ValueError("NEXT28 predictions are incomplete")
    if not required_endpoint.issubset(endpoint_and_pauling.columns):
        raise ValueError("NEXT25 endpoint table is incomplete")
    missing_pauling = set(pauling_columns.values()) - set(endpoint_and_pauling.columns)
    if missing_pauling:
        raise ValueError(f"Pauling controls lack columns: {sorted(missing_pauling)}")
    for frame, role in ((predictions, "predictions"), (endpoint_and_pauling, "endpoint")):
        if frame.material_id.isna().any() or frame.material_id.duplicated().any():
            raise ValueError(f"{role} identities are invalid")
    if set(predictions.material_id.astype(str)) != set(endpoint_and_pauling.material_id.astype(str)):
        raise ValueError("prediction and endpoint identities differ")
    endpoint_columns = [
        "material_id",
        "reference_match",
        "corrected_rmsd",
        *pauling_columns.values(),
    ]
    joined = predictions.merge(
        endpoint_and_pauling.loc[:, endpoint_columns],
        on="material_id",
        validate="one_to_one",
    )
    joined = joined.sort_values("material_id", kind="stable").reset_index(drop=True)
    matches = joined.reference_match.to_numpy(bool)
    fixed = _decision_metrics(
        supported=joined.analytic_supported.to_numpy(bool),
        reject=joined.reject.to_numpy(bool),
        reference_match=matches,
    )
    pauling: dict[str, dict[str, object]] = {}
    for name, column in pauling_columns.items():
        if column not in joined:
            raise ValueError(f"Pauling controls lack {column}")
        decisions = joined[column].astype(str).to_numpy()
        if not set(decisions) <= {"KEEP", "REJECT", "ABSTAIN"}:
            raise ValueError("Pauling decision vocabulary differs")
        pauling[name] = _decision_metrics(
            supported=decisions != "ABSTAIN",
            reject=decisions == "REJECT",
            reference_match=matches,
        )
    best_safe_pauling = max(
        (
            float(metrics["savings_lower"])
            for metrics in pauling.values()
            if bool(metrics["passes_primary_gates"])
        ),
        default=0.0,
    )
    supported = joined.analytic_supported.to_numpy(bool)
    score = pd.to_numeric(joined.next28_risk_score, errors="coerce").to_numpy(float)
    finite = supported & np.isfinite(score)
    rho: float | None = None
    if finite.sum() >= 3 and np.ptp(score[finite]) > 0:
        value = float(spearmanr(score[finite], joined.loc[finite, "corrected_rmsd"]).statistic)
        rho = value if math.isfinite(value) else None
    return {
        "primary_gates": dict(PRIMARY_GATES),
        "fixed_contact_rule": fixed,
        "pauling_controls": pauling,
        "best_safe_pauling_savings_lower": best_safe_pauling,
        "beyond_pauling_on_this_endpoint": bool(
            fixed["passes_primary_gates"]
            and float(fixed["savings_lower"]) > best_safe_pauling
        ),
        "continuous_diagnostics": {
            "supported_rows": int(finite.sum()),
            "auc_nonmatch": _roc_auc(score[finite], ~matches[finite]) if finite.any() else None,
            "spearman_risk_vs_corrected_rmsd": rho,
        },
        "rows": len(joined),
        "joined": joined,
    }


def evaluate_paths(
    *,
    predictions_path: Path,
    prediction_manifest_path: Path,
    next25_joined_path: Path,
    next25_evaluation_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish an explicitly retrospective cross-domain transport audit."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "predictions": Path(predictions_path).resolve(),
        "prediction_manifest": Path(prediction_manifest_path).resolve(),
        "next25_joined": Path(next25_joined_path).resolve(),
        "next25_evaluation_manifest": Path(next25_evaluation_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role}: {path}")
    hashes = {role: _sha256(path) for role, path in paths.items()}
    prediction_manifest = _read_json(paths["prediction_manifest"], "NEXT28 prediction manifest")
    prediction_outputs = prediction_manifest.get("outputs_sha256")
    if (
        prediction_manifest.get("protocol") != APPLICATION_PROTOCOL
        or prediction_manifest.get("labels_opened") is not False
        or prediction_manifest.get("endpoint_fields_read") is not False
        or prediction_manifest.get("threshold_refit") is not False
        or prediction_manifest.get("threshold") != THRESHOLD
        or not isinstance(prediction_outputs, Mapping)
        or prediction_outputs.get(PREDICTIONS_NAME) != hashes["predictions"]
    ):
        raise ValueError("NEXT28 OMatG predictions are not an immutable label-free application")
    endpoint_manifest = _read_json(paths["next25_evaluation_manifest"], "NEXT25 evaluation manifest")
    endpoint_outputs = endpoint_manifest.get("outputs_sha256")
    if (
        endpoint_manifest.get("protocol") != NEXT25_EVALUATION_PROTOCOL
        or endpoint_manifest.get("mode") != "post_freeze_one_shot_dft_reference_evaluation"
        or not isinstance(endpoint_outputs, Mapping)
        or endpoint_outputs.get(paths["next25_joined"].name) != hashes["next25_joined"]
    ):
        raise ValueError("NEXT25 retrospective endpoint provenance differs")
    payload = evaluate_frames(
        predictions=pd.read_parquet(paths["predictions"]),
        endpoint_and_pauling=pd.read_parquet(paths["next25_joined"]),
    )
    joined = payload.pop("joined")
    assert isinstance(joined, pd.DataFrame)
    result: dict[str, object] = {
        "protocol": EVALUATION_PROTOCOL,
        **payload,
        "next28_formula_changed": False,
        "next28_threshold_refit": False,
        "fresh_blind_evaluation": False,
        "retrospective_reason": "OMatG reference labels were opened before NEXT29 transport",
        "claim_boundary": {
            "endpoint_is_csp_reference_recovery": True,
            "nonmatch_is_thermodynamic_instability": False,
            "convex_hull_stability_established": False,
            "alternate_polymorph_possible": True,
        },
    }
    evaluated_at = datetime.now(timezone.utc).isoformat()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        joined_path = staging / JOINED_NAME
        result_path = staging / RESULT_NAME
        joined.to_parquet(joined_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        source_hashes = {"src/next29_omatg_contact_transport.py": _sha256(Path(__file__).resolve())}
        manifest = {
            "protocol": EVALUATION_PROTOCOL,
            "mode": "historically_exposed_cross_domain_transport_audit",
            "evaluated_at_utc": evaluated_at,
            "fresh_blind_evaluation": False,
            "inputs_sha256": {
                role: {"path": str(path), "sha256": hashes[role]} for role, path in paths.items()
            },
            "outputs_sha256": {JOINED_NAME: _sha256(joined_path), RESULT_NAME: _sha256(result_path)},
            "executed_source_sha256": source_hashes,
            "beyond_pauling_claim_confirmatory": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-features")
    build.add_argument("--metadata", required=True, type=Path)
    build.add_argument("--geometry", required=True, type=Path)
    build.add_argument("--cohort-manifest", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--prediction-manifest", required=True, type=Path)
    evaluate.add_argument("--next25-joined", required=True, type=Path)
    evaluate.add_argument("--next25-evaluation-manifest", required=True, type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "build-features":
        result = build_features(
            metadata_path=args.metadata,
            geometry_path=args.geometry,
            cohort_manifest_path=args.cohort_manifest,
            output_dir=args.output_dir,
        )
    else:
        result = evaluate_paths(
            predictions_path=args.predictions,
            prediction_manifest_path=args.prediction_manifest,
            next25_joined_path=args.next25_joined,
            next25_evaluation_manifest_path=args.next25_evaluation_manifest,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FEATURES_NAME", "build_features", "evaluate_frames", "evaluate_paths"]
