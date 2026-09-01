#!/usr/bin/env python3
"""Apply a frozen next7 protocol once to the historically seen test split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next6_elementa_diagnostics import paired_cluster_bootstrap
from src.next6_elementa_protocol import (
    apply_group_threshold,
    attach_energy_labels,
    evaluate_group_triage,
)
from src.next7_fewstep_protocol import (
    CATALOG_BY_NAME,
    DEVELOPMENT_FREEZE_PROTOCOL,
    EVIDENCE_ROLE,
    FEATURE_PROTOCOL,
    FROZEN_CATALOG,
    TRACKS,
    _atomic_publish_directory_no_replace,
    _read_json_document,
    _sha256_file,
)


EVALUATION_PROTOCOL = "2026-08-01-mattersim-fewstep-retrospective-v1"
OUTPUT_NAMES = (
    "test_predictions.parquet",
    "test_metrics.parquet",
    "paired_bootstrap.parquet",
    "TEST_OPENING.json",
    "MANIFEST.json",
)


def _digest_map(value: object, *, role: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a SHA-256 mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{role} contains an invalid SHA-256")
        result[key] = digest
    return result


def _same_float(actual: object, expected: float) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and np.isclose(value, expected, rtol=0.0, atol=1e-12))


def _validate_rule_record(
    record: object,
    *,
    track_name: str,
    role: str,
) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError(f"frozen {track_name}/{role} rule is missing")
    track = TRACKS[track_name]
    if record.get("protected") != track.protected:
        raise ValueError("frozen rule protected label mismatch")
    if not _same_float(record.get("protected_ev_per_atom"), track.protected_ev_per_atom):
        raise ValueError("frozen rule protected threshold mismatch")
    if record.get("within_group") != track.within_group:
        raise ValueError("frozen rule within-group semantics mismatch")
    if not _same_float(record.get("alpha"), track.alpha):
        raise ValueError("frozen rule alpha mismatch")

    state = record.get("state")
    name = record.get("name")
    if role == "s0_baseline":
        if state != "baseline" or name != "S0":
            raise ValueError("frozen S0 baseline mismatch")
    elif state == "null_keep_all":
        if (
            name != "null_keep_all"
            or record.get("max_step") != 0
            or record.get("cost") != 0
            or record.get("threshold") is not None
            or record.get("threshold_state") != "keep_all"
            or record.get("operator") != "KEEP_ALL"
            or record.get("unsupported_decision") != "KEEP"
        ):
            raise ValueError("invalid frozen keep-all cost/step or decision rule")
        return dict(record)
    elif state != "selected":
        raise ValueError("invalid frozen selected-rule state")

    if not isinstance(name, str) or name not in CATALOG_BY_NAME:
        raise ValueError("frozen rule lies outside the six-formula catalog")
    formula = CATALOG_BY_NAME[name]
    if record.get("max_step") != formula.max_step or record.get("cost") != formula.cost:
        raise ValueError("frozen rule cost/step mismatch")
    if record.get("threshold_state") != "finite":
        raise ValueError("frozen finite rule threshold state mismatch")
    try:
        threshold = float(record.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen rule threshold is invalid") from exc
    if not np.isfinite(threshold):
        raise ValueError("frozen rule threshold must be finite")
    if (
        record.get("operator") != "score > threshold"
        or record.get("unsupported_decision") != "ABSTAIN"
    ):
        raise ValueError("frozen rule decision semantics mismatch")
    return dict(record)


def _validate_frozen_protocol(
    frozen_path: Path,
    *,
    checkpoint_sha256: str,
    feature_inputs: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, dict[str, dict[str, object]]]]:
    frozen = _read_json_document(frozen_path, role="frozen protocol")
    if frozen.get("protocol") != DEVELOPMENT_FREEZE_PROTOCOL:
        raise ValueError("frozen protocol identifier mismatch")
    if frozen.get("state") != "frozen":
        raise ValueError("development protocol is not frozen")
    if frozen.get("evidence_role") != EVIDENCE_ROLE:
        raise ValueError("frozen protocol evidence role mismatch")
    timestamp = frozen.get("frozen_at_utc")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("frozen protocol timestamp is missing")
    if frozen.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("frozen protocol checkpoint mismatch")
    if _digest_map(
        frozen.get("feature_inputs_sha256"), role="frozen feature inputs"
    ) != dict(feature_inputs):
        raise ValueError("frozen protocol feature-input hashes mismatch")

    if frozen.get("catalog") != [item.as_record() for item in FROZEN_CATALOG]:
        raise ValueError("frozen protocol catalog mismatch")
    expected_tracks = {name: item.as_record() for name, item in TRACKS.items()}
    if frozen.get("tracks") != expected_tracks:
        raise ValueError("frozen protocol track definitions mismatch")

    source_dir = Path(__file__).resolve().parent
    expected_code = {
        "next7_mattersim_prerelax.py": _sha256_file(
            source_dir / "next7_mattersim_prerelax.py"
        ),
        "next7_mattersim_features.py": _sha256_file(
            source_dir / "next7_mattersim_features.py"
        ),
    }
    if _digest_map(frozen.get("code_sha256"), role="frozen code hashes") != expected_code:
        raise ValueError("frozen protocol feature-code hashes mismatch")
    if frozen.get("selection_code_sha256") != _sha256_file(
        source_dir / "next7_fewstep_protocol.py"
    ):
        raise ValueError("frozen protocol selection-code hash mismatch")

    rules = frozen.get("rules")
    if not isinstance(rules, dict) or set(rules) != set(TRACKS):
        raise ValueError("frozen protocol must contain exactly two tracks")
    validated: dict[str, dict[str, dict[str, object]]] = {}
    for track_name in TRACKS:
        selected_raw = rules.get(track_name)
        if not isinstance(selected_raw, dict):
            raise ValueError(f"frozen {track_name} rule is missing")
        baseline_raw = selected_raw.get("s0_baseline")
        selected = _validate_rule_record(
            selected_raw, track_name=track_name, role="selected"
        )
        baseline = _validate_rule_record(
            baseline_raw, track_name=track_name, role="s0_baseline"
        )
        validated[track_name] = {
            "selected": selected,
            "s0_baseline": baseline,
        }
    return frozen, validated


def _validate_test_feature_manifest(
    manifest_path: Path,
    *,
    features_path: Path,
    frozen_path: Path,
    checkpoint: Path,
) -> tuple[dict[str, object], dict[str, str], str]:
    manifest = _read_json_document(manifest_path, role="test feature manifest")
    if manifest.get("protocol") != FEATURE_PROTOCOL:
        raise ValueError("test feature manifest protocol mismatch")
    if manifest.get("stages") != ["test"]:
        raise ValueError("feature manifest must contain exactly the test stage")
    if manifest.get("evidence_role") != EVIDENCE_ROLE:
        raise ValueError("test feature evidence role mismatch")

    outputs = _digest_map(
        manifest.get("outputs_sha256"), role="test feature outputs"
    )
    expected_features = outputs.get(features_path.name)
    if expected_features is None or expected_features != _sha256_file(features_path):
        raise ValueError("test feature parquet hash mismatch")

    checkpoint_sha256 = _sha256_file(checkpoint)
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("test feature checkpoint hash mismatch")
    recorded_checkpoint = model.get("checkpoint")
    if not isinstance(recorded_checkpoint, str) or not recorded_checkpoint:
        raise ValueError("test feature checkpoint path is missing")
    checkpoint_name = Path(recorded_checkpoint).name

    frozen_sha256 = _sha256_file(frozen_path)
    frozen_record = manifest.get("frozen_protocol")
    if (
        not isinstance(frozen_record, dict)
        or frozen_record.get("sha256") != frozen_sha256
    ):
        raise ValueError("frozen protocol hash mismatch")
    recorded_frozen_path = frozen_record.get("path")
    if not isinstance(recorded_frozen_path, str) or not recorded_frozen_path:
        raise ValueError("frozen protocol path is missing")

    inputs = _digest_map(manifest.get("inputs_sha256"), role="test feature inputs")
    if inputs.get(checkpoint_name) != checkpoint_sha256:
        raise ValueError("test feature checkpoint input hash mismatch")
    if inputs.get(frozen_path.name) != frozen_sha256:
        raise ValueError("frozen protocol hash mismatch")
    feature_inputs = dict(inputs)
    feature_inputs.pop(checkpoint_name)
    feature_inputs.pop(frozen_path.name)
    if len(feature_inputs) != 4:
        raise ValueError("test manifest must contain exactly four feature-input hashes")
    return manifest, feature_inputs, checkpoint_sha256


def _normalize_test_features(features: pd.DataFrame) -> pd.DataFrame:
    required = {"sid", "rk", "stage", "evidence_role"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"test features are missing columns: {sorted(missing)}")
    data = features.copy()
    if data[["sid", "rk", "stage", "evidence_role"]].isna().any().any():
        raise ValueError("test feature identifiers and roles must be nonmissing")
    data["sid"] = data["sid"].astype(str)
    data["rk"] = data["rk"].astype(str)
    data["stage"] = data["stage"].astype(str)
    if data["sid"].duplicated().any():
        raise ValueError("test feature sid values must be unique")
    if set(data["stage"]) != {"test"}:
        raise ValueError("features must contain the test stage only")
    if set(data["evidence_role"].astype(str)) != {EVIDENCE_ROLE}:
        raise ValueError("test feature evidence role mismatch")
    return data


def _strict_bool(values: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            bool(value) if isinstance(value, (bool, np.bool_)) else False
            for value in values.to_numpy(dtype=object)
        ],
        dtype=bool,
    )


def _prepare_test_scores(features: pd.DataFrame, formula_name: str) -> pd.DataFrame:
    if formula_name not in CATALOG_BY_NAME:
        raise ValueError("test evaluator received a formula outside the frozen catalog")
    formula = CATALOG_BY_NAME[formula_name]
    required = {*formula.energy_source, formula.support_column}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(
            f"test features are missing {formula.name} columns: {sorted(missing)}"
        )
    matrix = np.column_stack(
        [
            pd.to_numeric(features[column], errors="coerce").to_numpy(float)
            for column in formula.energy_source
        ]
    )
    finite = np.isfinite(matrix).all(axis=1)
    energy = np.min(matrix, axis=1)
    preliminary = _strict_bool(features[formula.support_column]) & finite
    support_count = (
        pd.Series(preliminary, index=features.index)
        .groupby(features["rk"], sort=False)
        .transform("sum")
        .to_numpy(int)
    )
    supported = preliminary & (support_count >= 2)
    supported_energy = pd.Series(np.where(supported, energy, np.nan), index=features.index)
    group_min = supported_energy.groupby(features["rk"], sort=False).transform("min")
    score = np.where(supported, energy - group_min.to_numpy(float), np.nan)
    return pd.DataFrame(
        {
            "sid": features["sid"].to_numpy(object),
            "rk": features["rk"].to_numpy(object),
            "stage": features["stage"].to_numpy(object),
            "formula_energy_ev_per_atom": energy,
            "score": score,
            "supported": supported,
        }
    )


def _prepare_test_labels(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    required = {"sid", "rk", "stage", "e_per_atom"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"test labels are missing columns: {sorted(missing)}")
    data = labels.copy()
    if data[["sid", "rk"]].isna().any().any():
        raise ValueError("test label keys must be nonmissing")
    data["sid"] = data["sid"].astype(str)
    data["rk"] = data["rk"].astype(str)
    if data["sid"].duplicated().any():
        raise ValueError("test label sid values must be unique")
    feature_keys = set(zip(features["sid"], features["rk"], strict=True))
    label_keys = set(zip(data["sid"], data["rk"], strict=True))
    if feature_keys != label_keys:
        raise ValueError("feature and label sid/rk keys differ")
    if data["stage"].isna().any() or set(data["stage"].astype(str)) != {"test"}:
        raise ValueError("labels must contain the test stage only")
    expected = features.set_index("sid")["stage"].sort_index()
    observed = pd.Series(
        data["stage"].astype(str).to_numpy(object), index=data["sid"]
    ).sort_index()
    if not observed.equals(expected):
        raise ValueError("feature and label stages differ")
    labelled = attach_energy_labels(data[["sid", "rk", "e_per_atom"]])
    if "material" in data.columns:
        labelled = labelled.merge(
            data[["sid", "material"]], on="sid", how="left", validate="one_to_one"
        )
    return labelled.merge(
        features[["sid", "rk", "stage"]],
        on=["sid", "rk"],
        how="inner",
        validate="one_to_one",
    )


def _observed_cost(manifest: Mapping[str, object]) -> dict[str, int | float]:
    counts = manifest.get("counts")
    execution = manifest.get("execution")
    if not isinstance(counts, dict) or not isinstance(execution, dict):
        raise ValueError("test feature cost accounting is missing")
    try:
        result: dict[str, int | float] = {
            "observed_force_evaluations": int(counts["force_evaluations"]),
            "observed_optimizer_updates": int(counts["optimizer_updates"]),
            "predictor_forward_calls": int(execution["predictor_forward_calls"]),
            "feature_elapsed_seconds": float(execution["total_elapsed_seconds"]),
            "peak_cuda_memory_bytes": int(execution["peak_cuda_memory_bytes"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid test feature cost accounting") from exc
    if any(float(value) < 0 or not np.isfinite(float(value)) for value in result.values()):
        raise ValueError("invalid test feature cost accounting")
    return result


def _evaluate_method(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    track_name: str,
    role: str,
    rule: Mapping[str, object],
    observed_cost: Mapping[str, int | float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    method_id = f"{track_name}:{role}"
    if rule.get("state") == "null_keep_all":
        evaluated = labels.copy()
        evaluated["formula_energy_ev_per_atom"] = np.nan
        evaluated["score"] = np.nan
        evaluated["supported"] = True
        evaluated["decision"] = "KEEP"
    else:
        scores = _prepare_test_scores(features, str(rule["name"]))
        evaluated = scores.merge(
            labels, on=["sid", "rk", "stage"], how="inner", validate="one_to_one"
        )
        if len(evaluated) != len(features):
            raise ValueError("test score and label rows are not aligned")
        evaluated["decision"] = apply_group_threshold(
            evaluated["score"].to_numpy(float),
            evaluated["supported"].to_numpy(bool),
            float(rule["threshold"]),
        )
    evaluated["formula"] = str(rule["name"])
    evaluated["method_id"] = method_id
    evaluated["track"] = track_name
    evaluated["role"] = role
    evaluated["alpha"] = float(rule["alpha"])
    evaluated["threshold"] = (
        float(rule["threshold"]) if rule.get("threshold") is not None else np.inf
    )
    metrics: dict[str, object] = {
        "track": track_name,
        "role": role,
        "method_id": method_id,
        "formula": str(rule["name"]),
        "alpha": float(rule["alpha"]),
        "threshold": (
            float(rule["threshold"]) if rule.get("threshold") is not None else np.inf
        ),
        "threshold_state": str(rule["threshold_state"]),
        "max_step": int(rule["max_step"]),
        "rule_force_evaluations_per_supported_structure": int(rule["cost"]),
        **observed_cost,
        **evaluate_group_triage(evaluated),
    }
    return evaluated, metrics


def _tidy_bootstrap(
    predictions: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for track_name, track in TRACKS.items():
        work = predictions.loc[predictions["track"].eq(track_name)].copy()
        work["formula"] = work["method_id"]
        result = paired_cluster_bootstrap(
            work,
            baseline_formula=f"{track_name}:s0_baseline",
            candidate_formula=f"{track_name}:selected",
            alpha=track.alpha,
            n_resamples=n_resamples,
            seed=seed,
        )
        for metric, values in result["metrics"].items():
            baseline_ci = values["baseline_ci_95"]
            candidate_ci = values["candidate_ci_95"]
            difference_ci = values["difference_ci_95"]
            rows.append(
                {
                    "track": track_name,
                    "metric": metric,
                    "baseline": values["baseline"],
                    "baseline_ci_95_lower": baseline_ci[0],
                    "baseline_ci_95_upper": baseline_ci[1],
                    "candidate": values["candidate"],
                    "candidate_ci_95_lower": candidate_ci[0],
                    "candidate_ci_95_upper": candidate_ci[1],
                    "difference_candidate_minus_baseline": values["difference"],
                    "difference_ci_95_lower": difference_ci[0],
                    "difference_ci_95_upper": difference_ci[1],
                    "n_rows": result["n_rows"],
                    "n_groups": result["n_groups"],
                    "n_resamples": result["n_resamples"],
                    "seed": result["seed"],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["track", "metric"], kind="stable", ignore_index=True
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_staging(staging: Path, manifest: Mapping[str, object]) -> None:
    if {path.name for path in staging.iterdir()} != set(OUTPUT_NAMES):
        raise RuntimeError("retrospective staging artifact set mismatch")
    pd.read_parquet(staging / "test_predictions.parquet")
    pd.read_parquet(staging / "test_metrics.parquet")
    pd.read_parquet(staging / "paired_bootstrap.parquet")
    _read_json_document(staging / "TEST_OPENING.json", role="test opening")
    loaded_manifest = _read_json_document(staging / "MANIFEST.json", role="manifest")
    if loaded_manifest != dict(manifest):
        raise RuntimeError("staged retrospective manifest mismatch")
    hashes = _digest_map(manifest.get("outputs_sha256"), role="output hashes")
    for name, expected in hashes.items():
        if _sha256_file(staging / name) != expected:
            raise RuntimeError(f"staged retrospective output hash mismatch: {name}")


def run_frozen_evaluation(
    features_path: Path,
    labels_path: Path,
    feature_manifest_path: Path,
    frozen_protocol_path: Path,
    output_dir: Path,
    *,
    checkpoint: Path,
    bootstrap_resamples: int = 20_000,
    seed: int = 20260801,
) -> dict[str, object]:
    """Apply fixed decisions once; no test-driven selection is possible here."""

    features_path = Path(features_path)
    labels_path = Path(labels_path)
    feature_manifest_path = Path(feature_manifest_path)
    frozen_protocol_path = Path(frozen_protocol_path)
    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    for path in (
        features_path,
        labels_path,
        feature_manifest_path,
        frozen_protocol_path,
        checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    feature_manifest, feature_inputs, checkpoint_sha256 = (
        _validate_test_feature_manifest(
            feature_manifest_path,
            features_path=features_path,
            frozen_path=frozen_protocol_path,
            checkpoint=checkpoint,
        )
    )
    _, rules = _validate_frozen_protocol(
        frozen_protocol_path,
        checkpoint_sha256=checkpoint_sha256,
        feature_inputs=feature_inputs,
    )
    features = _normalize_test_features(pd.read_parquet(features_path))
    labels = _prepare_test_labels(pd.read_parquet(labels_path), features)
    observed_cost = _observed_cost(feature_manifest)

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for track_name in TRACKS:
        for role in ("selected", "s0_baseline"):
            predictions, metrics = _evaluate_method(
                features,
                labels,
                track_name=track_name,
                role=role,
                rule=rules[track_name][role],
                observed_cost=observed_cost,
            )
            prediction_frames.append(predictions)
            metric_rows.append(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["track", "role", "rk", "sid"], kind="stable", ignore_index=True
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["track", "role"], kind="stable", ignore_index=True
    )
    bootstrap = _tidy_bootstrap(
        predictions, n_resamples=int(bootstrap_resamples), seed=int(seed)
    )

    frozen_sha256 = _sha256_file(frozen_protocol_path)
    opened_at = datetime.now(timezone.utc).isoformat()
    input_hashes = {
        features_path.name: _sha256_file(features_path),
        labels_path.name: _sha256_file(labels_path),
        feature_manifest_path.name: _sha256_file(feature_manifest_path),
        frozen_protocol_path.name: frozen_sha256,
        checkpoint.name: checkpoint_sha256,
    }
    if len(input_hashes) != 5:
        raise ValueError("retrospective input basenames must be unique")
    opening: dict[str, object] = {
        "protocol": EVALUATION_PROTOCOL,
        "opened_at_utc": opened_at,
        "evidence_role": EVIDENCE_ROLE,
        "blind_or_confirmatory": False,
        "test_tuning_permitted": False,
        "test_status": "historically seen by earlier workflows; not a new lockbox",
        "decision_policy": "apply frozen selected and S0 rules once; no formula, step, alpha, or threshold selection",
        "frozen_protocol_sha256": frozen_sha256,
        "test_feature_manifest_sha256": _sha256_file(feature_manifest_path),
        "test_features_sha256": _sha256_file(features_path),
        "test_labels_sha256": _sha256_file(labels_path),
        "n_rows": int(len(features)),
        "n_groups": int(features["rk"].nunique()),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(seed),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    try:
        predictions.to_parquet(staging / "test_predictions.parquet", index=False)
        metrics.to_parquet(staging / "test_metrics.parquet", index=False)
        bootstrap.to_parquet(staging / "paired_bootstrap.parquet", index=False)
        _write_json(staging / "TEST_OPENING.json", opening)
        output_hashes = {
            name: _sha256_file(staging / name)
            for name in OUTPUT_NAMES
            if name != "MANIFEST.json"
        }
        manifest: dict[str, object] = {
            "protocol": EVALUATION_PROTOCOL,
            "state": "retrospective_complete",
            "evidence_role": EVIDENCE_ROLE,
            "inputs_sha256": input_hashes,
            "outputs_sha256": output_hashes,
            "bootstrap": {
                "method": "paired percentile bootstrap over rk composition clusters",
                "n_resamples": int(bootstrap_resamples),
                "seed": int(seed),
            },
            "cost": dict(observed_cost),
        }
        _write_json(staging / "MANIFEST.json", manifest)
        _validate_staging(staging, manifest)
        _atomic_publish_directory_no_replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if staging.exists():
        shutil.rmtree(staging)
    return {
        "protocol": EVALUATION_PROTOCOL,
        "n_rows": int(len(features)),
        "n_groups": int(features["rk"].nunique()),
        "frozen_protocol_sha256": frozen_sha256,
        "outputs_sha256": output_hashes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply a frozen MatterSim few-step protocol retrospectively."
    )
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--frozen-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args(argv)
    summary = run_frozen_evaluation(
        args.features,
        args.labels,
        args.feature_manifest,
        args.frozen_protocol,
        args.output,
        checkpoint=args.checkpoint,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
