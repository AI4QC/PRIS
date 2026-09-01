"""Open WBM labels after NEXT14 gates are sealed and evaluate without refit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next6_wbm_evaluate import apply_frozen_rule
from src.next12_prospective_gates import _compose_phsc_decision
from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file, _strict_json
from src.next14_wbm_acsc_features import PROTOCOL as ACSC_PROTOCOL
from src.next14_wbm_holdout import METADATA_NAME, PROTOCOL as HOLDOUT_PROTOCOL, _publish_directory_no_replace
from src.next14_wbm_pauling import PROTOCOL as PAULING_PROTOCOL


PROTOCOL = "2026-08-02-next14-wbm-acsc-isolated-evaluation-v1"
UPSTREAM_WBM_PROTOCOL = "2026-08-01-dft-pre-screening-design-v1"
RESULT_NAME = "NEXT14_WBM_ACSC_EVALUATION.json"
AGGREGATE_MANIFEST_NAME = "MANIFEST.json"
PRIVATE_JOINED_NAME = "joined_predictions_labels.parquet"
PRIVATE_MANIFEST_NAME = "MANIFEST.json"
DECISIONS = {"KEEP", "REJECT", "ABSTAIN"}
METHOD_COLUMNS: Mapping[str, str] = {
    "pauling_p2_p5": "pauling_p2_p5_decision",
    "phsc": "phsc_decision",
    "phsc_chsc": "phsc_chsc_decision",
    "phsc_chsc_acsc_formal": "phsc_chsc_acsc_formal_decision",
    "phsc_chsc_acsc_nested": "phsc_chsc_acsc_nested_decision",
    "wbm_born_packing": "wbm_born_packing_decision",
}
FROZEN_BOOTSTRAP_REPS = 10_000
FROZEN_BOOTSTRAP_SEED = 20260802
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "metadata": "ace914af28d6d1e82bbdd2a4ca0d7be39dc024fa9a98192c8ce770dfc5c75861",
    "holdout_manifest": "78bfc3aa4876887e9c683ec37571b55d42189e44ffc104e8921f27c5fa3b74db",
    "pauling_features": "d029cabcd5a392865e9d820a8a343fb1e87cc2ec78542fed25d97e3afb6bf34e",
    "pauling_manifest": "f88a5efc5706321ded9b9808a6a782dc6bf60902bca28f0ca739438fc725dc4b",
    "wbm_test_features": "91ac6dc5bda3d9bb27ba390b7f108631b2f4466fae1cc3101f385bd5d69a171f",
    "wbm_manifest": "e08a30ee817986f24b72309e41c2026142205af6a4850dc30ab2529efa47a8cd",
    "frozen_wbm_rule": "0bac84281a810b7cfd52d7636f07630b8b4a55a58cba96b2e95cdf6955f05478",
    "wbm_test_labels": "f2a74cd80f01b9f04d88c038672e77c63ade66e80aa7cc60f34682428c915427",
}


def _validated_table(
    *, path: Path, manifest_path: Path, protocol: str, required: Sequence[str], role: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role=f"{role} manifest")
    if manifest.get("protocol") != protocol:
        raise ValueError(f"{role} protocol differs")
    if manifest.get("labels_opened") is not False or manifest.get("endpoint_artifacts_opened") is not False:
        raise ValueError(f"{role} was not sealed before label opening")
    if role != "holdout" and manifest.get("thresholds_refit") is not False:
        raise ValueError(f"{role} thresholds were refit")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError(f"{role} output hash differs from manifest")
    table = pd.read_parquet(io.BytesIO(data))
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{role} table lacks columns: {sorted(missing)}")
    table = table.loc[:, list(required)].copy()
    table["material_id"] = table["material_id"].astype(str)
    if table["material_id"].isna().any() or table["material_id"].duplicated().any():
        raise ValueError(f"{role} material IDs must be unique")
    return table.sort_values("material_id", kind="stable", ignore_index=True), manifest


def _proportion(successes: int, total: int) -> dict[str, object]:
    if not 0 <= successes <= total:
        raise ValueError("invalid proportion counts")
    if total == 0:
        return {"numerator": successes, "denominator": total, "estimate": None, "wilson_ci95": [None, None]}
    estimate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "numerator": successes,
        "denominator": total,
        "estimate": float(estimate),
        "wilson_ci95": [float(max(0.0, center - half)), float(min(1.0, center + half))],
    }


def _finite_summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "q25": float(np.quantile(finite, 0.25)) if len(finite) else None,
        "q75": float(np.quantile(finite, 0.75)) if len(finite) else None,
    }


def _method_metrics(joined: pd.DataFrame, decision_column: str) -> dict[str, object]:
    decision = joined[decision_column].astype(str).to_numpy(object)
    if not set(decision).issubset(DECISIONS):
        raise ValueError(f"invalid decision in {decision_column}")
    reject = decision == "REJECT"
    coverage = decision != "ABSTAIN"
    stable = joined["stable"].to_numpy(bool)
    energy = joined["e_above_hull_mp2020_corrected_ppd_mp"].to_numpy(float)
    finite_energy = np.isfinite(energy)
    valuable = finite_energy & (energy <= 0.05)
    high = finite_energy & (energy >= 0.20)
    unstable = finite_energy & (energy > 0.0)
    fingerprint = joined["site_stats_fingerprint_init_final_norm_diff"].to_numpy(float)
    return {
        "decision_counts": {value: int((decision == value).sum()) for value in sorted(DECISIONS)},
        "coverage": _proportion(int(coverage.sum()), len(joined)),
        "dft_savings": _proportion(int(reject.sum()), len(joined)),
        "stable_recall": _proportion(int((stable & ~reject).sum()), int(stable.sum())),
        "valuable_recall": _proportion(int((valuable & ~reject).sum()), int(valuable.sum())),
        "high_energy_rejection_recall": _proportion(int((high & reject).sum()), int(high.sum())),
        "reject_precision_unstable": _proportion(int((unstable & reject).sum()), int(reject.sum())),
        "initial_final_fingerprint": {
            "rejected": _finite_summary(fingerprint[reject]),
            "not_rejected": _finite_summary(fingerprint[~reject]),
        },
    }


def _group_aggregates(joined: pd.DataFrame, decision_column: str) -> np.ndarray:
    decision = joined[decision_column].astype(str)
    reject = decision.eq("REJECT")
    stable = joined["stable"].astype(bool)
    energy = pd.to_numeric(joined["e_above_hull_mp2020_corrected_ppd_mp"], errors="coerce")
    high = energy.ge(0.20) & energy.notna()
    unstable = energy.gt(0.0) & energy.notna()
    frame = pd.DataFrame(
        {
            "formula_key": joined["formula_key"].astype(str),
            "n": 1,
            "stable": stable.astype(int),
            "stable_false_reject": (stable & reject).astype(int),
            "high": high.astype(int),
            "high_reject": (high & reject).astype(int),
            "reject": reject.astype(int),
            "unstable_reject": (unstable & reject).astype(int),
        }
    )
    return frame.groupby("formula_key", sort=True).sum().to_numpy(float)


def _metrics_from_sums(values: np.ndarray) -> np.ndarray:
    n, stable, false_reject, high, high_reject, reject, unstable_reject = np.moveaxis(values, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack(
            [
                1.0 - false_reject / stable,
                high_reject / high,
                reject / n,
                unstable_reject / reject,
            ],
            axis=-1,
        )


def _bootstrap_difference(
    joined: pd.DataFrame,
    method_column: str,
    baseline_column: str,
    *,
    reps: int,
    seed: int,
) -> dict[str, object]:
    method = _group_aggregates(joined, method_column)
    baseline = _group_aggregates(joined, baseline_column)
    if method.shape != baseline.shape or reps <= 0:
        raise ValueError("bootstrap groups or repetitions differ")
    point = _metrics_from_sums(method.sum(axis=0)) - _metrics_from_sums(baseline.sum(axis=0))
    rng = np.random.default_rng(seed)
    differences = np.empty((reps, 4), dtype=float)
    batch_size = 256
    groups = len(method)
    for start in range(0, reps, batch_size):
        size = min(batch_size, reps - start)
        indices = rng.integers(0, groups, size=(size, groups))
        method_sums = method[indices].sum(axis=1)
        baseline_sums = baseline[indices].sum(axis=1)
        differences[start : start + size] = _metrics_from_sums(method_sums) - _metrics_from_sums(baseline_sums)
    names = ("stable_recall", "high_energy_rejection_recall", "dft_savings", "reject_precision_unstable")
    def interval(values: np.ndarray) -> list[float | None]:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return [None, None]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    return {
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
        **{
            name: {
                "estimate": float(point[index]) if np.isfinite(point[index]) else None,
                "cluster_bootstrap_ci95": interval(differences[:, index]),
            }
            for index, name in enumerate(names)
        },
    }


def _estimate(metric: Mapping[str, object]) -> float:
    value = metric.get("estimate")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float("nan")


def _complete_superiority(
    method: Mapping[str, object], baseline: Mapping[str, object], comparison: Mapping[str, object]
) -> tuple[bool, dict[str, bool]]:
    stable_ci = method["stable_recall"]["wilson_ci95"]
    high_lower = comparison["high_energy_rejection_recall"]["cluster_bootstrap_ci95"][0]
    savings_lower = comparison["dft_savings"]["cluster_bootstrap_ci95"][0]
    precision_lower = comparison["reject_precision_unstable"]["cluster_bootstrap_ci95"][0]
    clauses = {
        "stable_recall_lower_at_least_0_95": bool(stable_ci[0] is not None and float(stable_ci[0]) >= 0.95),
        "coverage_no_lower_than_pauling": _estimate(method["coverage"]) >= _estimate(baseline["coverage"]),
        "high_energy_recall_difference_lower_above_zero": bool(high_lower is not None and float(high_lower) > 0.0),
        "dft_savings_difference_lower_above_zero": bool(savings_lower is not None and float(savings_lower) > 0.0),
        "reject_precision_difference_lower_at_least_minus_0_02": bool(precision_lower is not None and float(precision_lower) >= -0.02),
    }
    return all(clauses.values()), clauses


def evaluate_wbm_holdout(
    *,
    metadata_path: Path,
    holdout_manifest_path: Path,
    pauling_features_path: Path,
    pauling_manifest_path: Path,
    acsc_features_path: Path,
    acsc_manifest_path: Path,
    wbm_test_features_path: Path,
    wbm_manifest_path: Path,
    frozen_wbm_rule_path: Path,
    wbm_test_labels_path: Path,
    opening_log_path: Path,
    private_output_dir: Path,
    aggregate_output_dir: Path,
    bootstrap_reps: int = FROZEN_BOOTSTRAP_REPS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Validate every sealed method, log opening, then parse WBM DFT labels once."""

    private_target = Path(private_output_dir).resolve()
    aggregate_target = Path(aggregate_output_dir).resolve()
    opening_path = Path(opening_log_path).resolve()
    for target in (private_target, aggregate_target, opening_path):
        if os.path.lexists(target):
            raise FileExistsError(f"refusing existing output/opening: {target}")
    if private_target == aggregate_target:
        raise ValueError("private and aggregate outputs must be separated")
    if type(bootstrap_reps) is not int or bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be a positive exact integer")
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "holdout_manifest": Path(holdout_manifest_path).resolve(),
        "pauling_features": Path(pauling_features_path).resolve(),
        "pauling_manifest": Path(pauling_manifest_path).resolve(),
        "acsc_features": Path(acsc_features_path).resolve(),
        "acsc_manifest": Path(acsc_manifest_path).resolve(),
        "wbm_test_features": Path(wbm_test_features_path).resolve(),
        "wbm_manifest": Path(wbm_manifest_path).resolve(),
        "frozen_wbm_rule": Path(frozen_wbm_rule_path).resolve(),
        "wbm_test_labels": Path(wbm_test_labels_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    preopening_roles = tuple(role for role in paths if role != "wbm_test_labels")
    preopening_hashes = {role: _sha256_file(paths[role]) for role in preopening_roles}
    if require_formal_inputs:
        for role, expected in FROZEN_FORMAL_SHA256.items():
            if role == "wbm_test_labels":
                continue
            if preopening_hashes.get(role) != expected:
                raise ValueError(f"formal NEXT14 evaluator input differs: {role}")
        if bootstrap_reps != FROZEN_BOOTSTRAP_REPS:
            raise ValueError("formal NEXT14 bootstrap repetitions differ")

    metadata, holdout_manifest = _validated_table(
        path=paths["metadata"], manifest_path=paths["holdout_manifest"],
        protocol=HOLDOUT_PROTOCOL, required=("material_id", "rk", "formula", "natoms"), role="holdout",
    )
    pauling, _ = _validated_table(
        path=paths["pauling_features"], manifest_path=paths["pauling_manifest"],
        protocol=PAULING_PROTOCOL, required=("material_id", "pauling_p2_p5_decision"), role="pauling",
    )
    acsc, _ = _validated_table(
        path=paths["acsc_features"], manifest_path=paths["acsc_manifest"],
        protocol=ACSC_PROTOCOL,
        required=("material_id", "phsc_status", "phsc_chsc_decision", "phsc_chsc_acsc_formal_decision", "phsc_chsc_acsc_nested_decision", "nested_three_scale_confirmed"),
        role="acsc",
    )
    expected_ids = set(metadata["material_id"])
    if set(pauling["material_id"]) != expected_ids or set(acsc["material_id"]) != expected_ids:
        raise ValueError("holdout, Pauling, and ACSC material ID sets differ")
    for column in ("pauling_p2_p5_decision",):
        if not set(pauling[column].astype(str)).issubset(DECISIONS):
            raise ValueError("Pauling decision is invalid")
    for column in ("phsc_chsc_decision", "phsc_chsc_acsc_formal_decision", "phsc_chsc_acsc_nested_decision"):
        if not set(acsc[column].astype(str)).issubset(DECISIONS):
            raise ValueError(f"ACSC decision is invalid: {column}")
    acsc["phsc_decision"] = [
        _compose_phsc_decision("KEEP", str(status)) for status in acsc["phsc_status"]
    ]

    wbm_manifest = _strict_json(paths["wbm_manifest"].read_bytes(), role="WBM manifest")
    if wbm_manifest.get("protocol") != UPSTREAM_WBM_PROTOCOL or wbm_manifest.get("input_role") != "unrelaxed_x0_only":
        raise ValueError("WBM protocol differs")
    outputs = wbm_manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(paths["wbm_test_features"].name) != preopening_hashes["wbm_test_features"]:
        raise ValueError("WBM test feature hash differs")
    expected_label_hash = outputs.get(paths["wbm_test_labels"].name) if isinstance(outputs, Mapping) else None
    if type(expected_label_hash) is not str:
        raise ValueError("WBM manifest does not bind the test labels")
    if require_formal_inputs and expected_label_hash != FROZEN_FORMAL_SHA256["wbm_test_labels"]:
        raise ValueError("formal WBM label identity differs")
    wbm_features = pd.read_parquet(paths["wbm_test_features"])
    if wbm_features["material_id"].astype(str).duplicated().any():
        raise ValueError("WBM test feature material IDs must be unique")
    selected_wbm = wbm_features.loc[wbm_features["material_id"].astype(str).isin(expected_ids)].copy()
    if len(selected_wbm) != len(metadata):
        raise ValueError("WBM test features do not cover the frozen holdout")
    frozen_rule = _strict_json(paths["frozen_wbm_rule"].read_bytes(), role="frozen WBM rule")
    if frozen_rule.get("protocol") != UPSTREAM_WBM_PROTOCOL:
        raise ValueError("frozen WBM rule protocol differs")
    baseline_predictions = apply_frozen_rule(selected_wbm, frozen_rule).rename(
        columns={"decision": "wbm_born_packing_decision"}
    )

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next14_wbm_evaluate.py": Path(__file__).resolve(),
        "src/next6_wbm_evaluate.py": repository_root / "src/next6_wbm_evaluate.py",
        "src/next6_wbm_protocol.py": repository_root / "src/next6_wbm_protocol.py",
    }
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    opening = {
        "protocol": PROTOCOL,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "external-source retrospective; WBM test was opened previously by older unrelated workflows",
        "decision_policy": "apply sealed NEXT14 methods once with no sample, threshold, or decision refit",
        "preopening_inputs_sha256": preopening_hashes,
        "expected_wbm_test_labels_sha256": expected_label_hash,
        "executed_evaluator_source_sha256": source_hashes,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": FROZEN_BOOTSTRAP_SEED,
    }
    opening_path.parent.mkdir(parents=True, exist_ok=True)
    with opening_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(opening, allow_nan=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    # First label-byte access in this evaluator occurs only after the durable opening log.
    label_data = paths["wbm_test_labels"].read_bytes()
    actual_label_hash = hashlib.sha256(label_data).hexdigest()
    if actual_label_hash != expected_label_hash:
        raise ValueError("WBM test label hash differs after opening")
    labels_all = pd.read_parquet(io.BytesIO(label_data))
    label_columns = (
        "material_id", "formula", "formula_key", "split", "stage",
        "e_above_hull_mp2020_corrected_ppd_mp",
        "site_stats_fingerprint_init_final_norm_diff", "stable",
    )
    missing = set(label_columns) - set(labels_all.columns)
    if missing:
        raise ValueError(f"WBM labels lack columns: {sorted(missing)}")
    labels = labels_all.loc[labels_all["material_id"].astype(str).isin(expected_ids), list(label_columns)].copy()
    labels["material_id"] = labels["material_id"].astype(str)
    if len(labels) != len(metadata) or labels["material_id"].duplicated().any() or not labels["stage"].eq("test").all():
        raise ValueError("WBM selected label coverage/stage differs")
    energy = pd.to_numeric(labels["e_above_hull_mp2020_corrected_ppd_mp"], errors="coerce").to_numpy(float)
    stable_expected = np.isfinite(energy) & (energy <= 0.0)
    if not np.array_equal(labels["stable"].astype(bool).to_numpy(), stable_expected):
        raise ValueError("WBM stable label differs from frozen hull definition")

    joined = metadata.merge(pauling, on="material_id", validate="one_to_one").merge(
        acsc, on="material_id", validate="one_to_one"
    ).merge(
        baseline_predictions.loc[:, ["material_id", "wbm_born_packing_decision"]],
        on="material_id", validate="one_to_one",
    ).merge(labels, on="material_id", validate="one_to_one", suffixes=("", "_label"))
    if not (joined["rk"].astype(str) == joined["formula_key"].astype(str)).all():
        raise ValueError("WBM frozen reduced compositions differ from labels")
    methods = {
        name: _method_metrics(joined, column) for name, column in METHOD_COLUMNS.items()
    }
    baseline = methods["pauling_p2_p5"]
    comparisons: dict[str, object] = {}
    for index, (name, column) in enumerate(METHOD_COLUMNS.items()):
        if name == "pauling_p2_p5":
            continue
        comparison = _bootstrap_difference(
            joined, column, METHOD_COLUMNS["pauling_p2_p5"],
            reps=bootstrap_reps, seed=FROZEN_BOOTSTRAP_SEED + index,
        )
        complete, clauses = _complete_superiority(methods[name], baseline, comparison)
        comparison["superiority_clauses"] = clauses
        comparison["complete_superiority_over_pauling"] = complete
        comparisons[name] = comparison
    passed = [name for name, value in comparisons.items() if value["complete_superiority_over_pauling"]]
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": opening["evidence_role"],
        "labels_opened_after_all_methods_sealed": True,
        "thresholds_refit": False,
        "counts": {
            "source_label_rows_materialized": len(labels_all),
            "holdout_rows": len(joined),
            "composition_groups": int(joined["formula_key"].nunique()),
            "stable_rows": int(joined["stable"].sum()),
            "valuable_rows": int((np.isfinite(energy) & (energy <= 0.05)).sum()),
            "high_energy_rows": int((np.isfinite(energy) & (energy >= 0.20)).sum()),
        },
        "methods": methods,
        "comparisons_to_pauling": comparisons,
        "complete_superiority_over_pauling_methods": passed,
        "external_source_retrospective_support": bool(passed),
        "scientific_improvement_claim": False,
        "interpretation_guard": "Passing is external-source retrospective support, not a fresh blind lockbox or proof of universal DFT equivalence.",
        "opening_log_sha256": _sha256_file(opening_path),
    }

    input_hashes_after_opening = {
        **preopening_hashes,
        "wbm_test_labels": actual_label_hash,
        "opening_log": _sha256_file(opening_path),
    }
    private_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "identifier_bearing": True,
        "storage_role": "external private prediction/label join",
        "inputs_sha256": input_hashes_after_opening,
        "executed_source_sha256": source_hashes,
    }
    aggregate_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "identifier_bearing": False,
        "labels_opened": True,
        "thresholds_refit": False,
        "inputs_sha256": input_hashes_after_opening,
        "executed_source_sha256": source_hashes,
        "private_output_path": str(private_target),
        "scientific_improvement_claim": False,
    }

    private_target.parent.mkdir(parents=True, exist_ok=True)
    private_staging = Path(tempfile.mkdtemp(prefix=f".{private_target.name}.staging-", dir=private_target.parent))
    try:
        joined_path = private_staging / PRIVATE_JOINED_NAME
        joined.to_parquet(joined_path, index=False)
        private_manifest["outputs_sha256"] = {PRIVATE_JOINED_NAME: _sha256_file(joined_path)}
        (private_staging / PRIVATE_MANIFEST_NAME).write_bytes(_json_bytes(private_manifest))
        _publish_directory_no_replace(private_staging, private_target)
    except Exception:
        shutil.rmtree(private_staging, ignore_errors=True)
        raise

    aggregate_target.parent.mkdir(parents=True, exist_ok=True)
    aggregate_staging = Path(tempfile.mkdtemp(prefix=f".{aggregate_target.name}.staging-", dir=aggregate_target.parent))
    try:
        result_path = aggregate_staging / RESULT_NAME
        result_path.write_bytes(_json_bytes(result))
        aggregate_manifest["outputs_sha256"] = {RESULT_NAME: _sha256_file(result_path)}
        aggregate_manifest["private_outputs_sha256"] = private_manifest["outputs_sha256"]
        (aggregate_staging / AGGREGATE_MANIFEST_NAME).write_bytes(_json_bytes(aggregate_manifest))
        _publish_directory_no_replace(aggregate_staging, aggregate_target)
    except Exception:
        shutil.rmtree(aggregate_staging, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--pauling-features", required=True, type=Path)
    parser.add_argument("--pauling-manifest", required=True, type=Path)
    parser.add_argument("--acsc-features", required=True, type=Path)
    parser.add_argument("--acsc-manifest", required=True, type=Path)
    parser.add_argument("--wbm-test-features", required=True, type=Path)
    parser.add_argument("--wbm-manifest", required=True, type=Path)
    parser.add_argument("--frozen-wbm-rule", required=True, type=Path)
    parser.add_argument("--wbm-test-labels", required=True, type=Path)
    parser.add_argument("--opening-log", required=True, type=Path)
    parser.add_argument("--private-output-dir", required=True, type=Path)
    parser.add_argument("--aggregate-output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    evaluate_wbm_holdout(
        metadata_path=arguments.metadata,
        holdout_manifest_path=arguments.holdout_manifest,
        pauling_features_path=arguments.pauling_features,
        pauling_manifest_path=arguments.pauling_manifest,
        acsc_features_path=arguments.acsc_features,
        acsc_manifest_path=arguments.acsc_manifest,
        wbm_test_features_path=arguments.wbm_test_features,
        wbm_manifest_path=arguments.wbm_manifest,
        frozen_wbm_rule_path=arguments.frozen_wbm_rule,
        wbm_test_labels_path=arguments.wbm_test_labels,
        opening_log_path=arguments.opening_log,
        private_output_dir=arguments.private_output_dir,
        aggregate_output_dir=arguments.aggregate_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
