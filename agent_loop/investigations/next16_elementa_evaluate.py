"""Evaluate the frozen NEXT16 rule against complete-group ELEMENTA DFT endpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file, _strict_json
from src.next14_wbm_evaluate import DECISIONS, _proportion
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next16_elementa_basin_hull import (
    NEXT16_THRESHOLD_EV_PER_ATOM,
    PROTOCOL as FEATURE_PROTOCOL,
    next16_decision,
)


PROTOCOL = "2026-08-02-next16-elementa-basin-hull-retrospective-v1"
UPSTREAM_LABEL_PROTOCOL = "2026-08-01-dft-pre-screening-design-v1"
RESULT_NAME = "NEXT16_ELEMENTA_BASIN_HULL_RETROSPECTIVE.json"
PRIVATE_JOINED_NAME = "joined_elementa_predictions_labels.parquet"
MANIFEST_NAME = "MANIFEST.json"
MINIMUM_TOLERANCE_EV_PER_ATOM = 1.0e-8
VALUABLE_REGRET_EV_PER_ATOM = 0.05
HIGH_REGRET_EV_PER_ATOM = 0.20
FROZEN_BOOTSTRAP_REPS = 10_000
FROZEN_BOOTSTRAP_SEED = 20260816
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "features": "436d3b312009e097866a9383341c75dc2428f6316b461b89f51ca6de9277a46c",
    "features_manifest": "204c5739b66942025a6b15bdb7631794865342ef404ed013064c0c79fd1459f6",
    "labels": "86225b5fea9275113dedb87bd4963ac1bf5bc3d02d478319a993f4285fc48f0f",
    "labels_manifest": "ae3b192f3d28400fffd2eb818e574e60bb9400a4b993d196cc1ba2fcac0ebb99",
}


def _validate_features(
    path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role="NEXT16 feature manifest")
    if manifest.get("protocol") != FEATURE_PROTOCOL:
        raise ValueError("NEXT16 feature protocol differs")
    if manifest.get("thresholds_refit") is not False:
        raise ValueError("NEXT16 threshold was refit")
    if manifest.get("elementa_endpoint_bytes_read_by_execution") is not False:
        raise ValueError("NEXT16 feature execution read ELEMENTA endpoints")
    expected_rule = {
        "formula": "B64 = E_MatterSim_relaxed/N - E_raw_MP_hull(composition)",
        "comparison": ">=",
        "threshold_ev_per_atom": NEXT16_THRESHOLD_EV_PER_ATOM,
        "failure_policy": "ABSTAIN",
        "selection_origin": "post hoc WBM development sweep; frozen before this ELEMENTA execution",
    }
    rule = manifest.get("rule")
    if not isinstance(rule, Mapping) or any(rule.get(key) != value for key, value in expected_rule.items()):
        raise ValueError("NEXT16 sealed rule differs")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError("NEXT16 feature hash differs from manifest")
    table = pd.read_parquet(io.BytesIO(data))
    required = {
        "material_id", "rk", "supported", "error", "capped_at_max_steps",
        "basin_hull_score_ev_per_atom", "basin_hull_decision", "pauling_p2_p5_decision",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"NEXT16 feature table lacks columns: {sorted(missing)}")
    table = table.loc[:, sorted(required)].copy()
    table["material_id"] = table["material_id"].astype(str)
    table["rk"] = table["rk"].astype(str)
    if table["material_id"].isna().any() or table["material_id"].duplicated().any():
        raise ValueError("NEXT16 feature identities must be unique")
    for column in ("basin_hull_decision", "pauling_p2_p5_decision"):
        if not set(table[column].astype(str)).issubset(DECISIONS):
            raise ValueError(f"invalid decision in {column}")
    expected = [
        next16_decision(score, supported=bool(supported))
        for score, supported in zip(
            table["basin_hull_score_ev_per_atom"], table["supported"], strict=True
        )
    ]
    if expected != table["basin_hull_decision"].astype(str).tolist():
        raise ValueError("NEXT16 decision differs from frozen score and threshold")
    return table.sort_values("material_id", kind="stable", ignore_index=True), manifest


def _validate_labels(
    path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role="ELEMENTA label manifest")
    if (
        manifest.get("protocol") != UPSTREAM_LABEL_PROTOCOL
        or manifest.get("input_role") != "unrelaxed_x0_only"
    ):
        raise ValueError("ELEMENTA label source protocol differs")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError("ELEMENTA label hash differs from manifest")
    table = pd.read_parquet(io.BytesIO(data))
    required = {"sid", "rk", "e_per_atom", "final_ionic_step", "final_max_force"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"ELEMENTA label table lacks columns: {sorted(missing)}")
    table = table.loc[:, sorted(required)].rename(
        columns={
            "sid": "material_id",
            "rk": "label_rk",
            "e_per_atom": "dft_energy_ev_per_atom",
        }
    )
    table["material_id"] = table["material_id"].astype(str)
    table["label_rk"] = table["label_rk"].astype(str)
    table["dft_energy_ev_per_atom"] = pd.to_numeric(
        table["dft_energy_ev_per_atom"], errors="coerce"
    )
    if table["material_id"].isna().any() or table["material_id"].duplicated().any():
        raise ValueError("ELEMENTA label identities must be unique")
    if not np.isfinite(table["dft_energy_ev_per_atom"].to_numpy(float)).all():
        raise ValueError("ELEMENTA DFT energies must be finite")
    return table.sort_values("material_id", kind="stable", ignore_index=True), manifest


def _regret_summary(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "n": int(len(finite)),
        "median": float(np.median(finite)) if len(finite) else None,
        "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "maximum": float(finite.max()) if len(finite) else None,
    }


def _method_metrics(joined: pd.DataFrame, decision_column: str) -> dict[str, object]:
    decision = joined[decision_column].astype(str)
    if not set(decision).issubset(DECISIONS):
        raise ValueError(f"invalid decision in {decision_column}")
    reject = decision.eq("REJECT")
    covered = ~decision.eq("ABSTAIN")
    regret = joined["dft_group_regret_ev_per_atom"].to_numpy(float)
    minimum = regret <= MINIMUM_TOLERANCE_EV_PER_ATOM
    valuable = regret <= VALUABLE_REGRET_EV_PER_ATOM
    high = regret >= HIGH_REGRET_EV_PER_ATOM
    above_minimum = regret > MINIMUM_TOLERANCE_EV_PER_ATOM
    grouped = pd.DataFrame(
        {
            "rk": joined["rk"].astype(str),
            "reject": reject.to_numpy(bool),
            "retained_minimum": minimum & ~reject.to_numpy(bool),
        }
    ).groupby("rk", sort=True)
    group_best_retained = grouped["retained_minimum"].any()
    all_rejected = grouped["reject"].all()
    return {
        "decision_counts": {
            value: int(decision.eq(value).sum()) for value in sorted(DECISIONS)
        },
        "coverage": _proportion(int(covered.sum()), len(joined)),
        "dft_savings": _proportion(int(reject.sum()), len(joined)),
        "group_minimum_recall": _proportion(
            int((minimum & ~reject.to_numpy(bool)).sum()), int(minimum.sum())
        ),
        "group_best_retention": _proportion(
            int(group_best_retained.sum()), len(group_best_retained)
        ),
        "valuable_recall": _proportion(
            int((valuable & ~reject.to_numpy(bool)).sum()), int(valuable.sum())
        ),
        "high_energy_rejection_recall": _proportion(
            int((high & reject.to_numpy(bool)).sum()), int(high.sum())
        ),
        "reject_precision_above_minimum": _proportion(
            int((above_minimum & reject.to_numpy(bool)).sum()), int(reject.sum())
        ),
        "all_rejected_groups": int(all_rejected.sum()),
        "retained_regret_ev_per_atom": _regret_summary(regret[~reject.to_numpy(bool)]),
        "rejected_regret_ev_per_atom": _regret_summary(regret[reject.to_numpy(bool)]),
    }


def _group_aggregates(joined: pd.DataFrame, decision_column: str) -> np.ndarray:
    decision = joined[decision_column].astype(str)
    reject = decision.eq("REJECT")
    regret = joined["dft_group_regret_ev_per_atom"].to_numpy(float)
    minimum = regret <= MINIMUM_TOLERANCE_EV_PER_ATOM
    high = regret >= HIGH_REGRET_EV_PER_ATOM
    above_minimum = regret > MINIMUM_TOLERANCE_EV_PER_ATOM
    frame = pd.DataFrame(
        {
            "rk": joined["rk"].astype(str),
            "n": 1,
            "minimum": minimum.astype(int),
            "minimum_false_reject": (minimum & reject.to_numpy(bool)).astype(int),
            "high": high.astype(int),
            "high_reject": (high & reject.to_numpy(bool)).astype(int),
            "reject": reject.astype(int),
            "above_minimum_reject": (above_minimum & reject.to_numpy(bool)).astype(int),
        }
    )
    return frame.groupby("rk", sort=True).sum().to_numpy(float)


def _metrics_from_sums(values: np.ndarray) -> np.ndarray:
    n, minimum, false_reject, high, high_reject, reject, above_minimum_reject = np.moveaxis(
        values, -1, 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack(
            [
                1.0 - false_reject / minimum,
                high_reject / high,
                reject / n,
                above_minimum_reject / reject,
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
    point = _metrics_from_sums(method.sum(axis=0)) - _metrics_from_sums(
        baseline.sum(axis=0)
    )
    rng = np.random.default_rng(seed)
    differences = np.empty((reps, 4), dtype=float)
    groups = len(method)
    for start in range(0, reps, 256):
        size = min(256, reps - start)
        indices = rng.integers(0, groups, size=(size, groups))
        differences[start : start + size] = _metrics_from_sums(
            method[indices].sum(axis=1)
        ) - _metrics_from_sums(baseline[indices].sum(axis=1))

    def interval(values: np.ndarray) -> list[float | None]:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return [None, None]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    names = (
        "group_minimum_recall",
        "high_energy_rejection_recall",
        "dft_savings",
        "reject_precision_above_minimum",
    )
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
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan


def _complete_superiority(
    method: Mapping[str, object], baseline: Mapping[str, object], comparison: Mapping[str, object]
) -> tuple[bool, dict[str, bool]]:
    minimum_ci = method["group_minimum_recall"]["wilson_ci95"]
    high_lower = comparison["high_energy_rejection_recall"]["cluster_bootstrap_ci95"][0]
    savings_lower = comparison["dft_savings"]["cluster_bootstrap_ci95"][0]
    precision_lower = comparison["reject_precision_above_minimum"]["cluster_bootstrap_ci95"][0]
    clauses = {
        "group_minimum_recall_lower_at_least_0_95": bool(
            minimum_ci[0] is not None and float(minimum_ci[0]) >= 0.95
        ),
        "coverage_no_lower_than_pauling": _estimate(method["coverage"]) >= _estimate(
            baseline["coverage"]
        ),
        "high_energy_recall_difference_lower_above_zero": bool(
            high_lower is not None and float(high_lower) > 0.0
        ),
        "dft_savings_difference_lower_above_zero": bool(
            savings_lower is not None and float(savings_lower) > 0.0
        ),
        "reject_precision_difference_lower_at_least_minus_0_02": bool(
            precision_lower is not None and float(precision_lower) >= -0.02
        ),
        "no_composition_group_fully_rejected": int(method["all_rejected_groups"]) == 0,
    }
    return all(clauses.values()), clauses


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    finite = np.isfinite(scores)
    scores, positive = scores[finite], positive[finite]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def _score_diagnostics(joined: pd.DataFrame) -> dict[str, object]:
    score = joined["basin_hull_score_ev_per_atom"].to_numpy(float)
    regret = joined["dft_group_regret_ev_per_atom"].to_numpy(float)
    finite = np.isfinite(score) & np.isfinite(regret)
    correlation = None
    if int(finite.sum()) >= 2 and np.std(score[finite]) > 0 and np.std(regret[finite]) > 0:
        correlation = float(np.corrcoef(score[finite], regret[finite])[0, 1])
    return {
        "finite_pairs": int(finite.sum()),
        "pearson_score_vs_dft_group_regret": correlation,
        "roc_auc_above_group_minimum": _binary_auc(
            score, regret > MINIMUM_TOLERANCE_EV_PER_ATOM
        ),
        "roc_auc_high_regret_ge_0_20": _binary_auc(
            score, regret >= HIGH_REGRET_EV_PER_ATOM
        ),
    }


def evaluate_elementa_retrospective(
    *,
    features_path: Path,
    features_manifest_path: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    private_output_dir: Path,
    aggregate_output_dir: Path,
    bootstrap_reps: int = FROZEN_BOOTSTRAP_REPS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Join sealed x0-only predictions to complete-group DFT endpoint energies."""

    private_target = Path(private_output_dir).resolve()
    aggregate_target = Path(aggregate_output_dir).resolve()
    if private_target == aggregate_target:
        raise ValueError("private and aggregate outputs must be separated")
    for target in (private_target, aggregate_target):
        if os.path.lexists(target):
            raise FileExistsError(f"refusing existing output: {target}")
    if type(bootstrap_reps) is not int or bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be a positive exact integer")
    paths = {
        "features": Path(features_path).resolve(),
        "features_manifest": Path(features_manifest_path).resolve(),
        "labels": Path(labels_path).resolve(),
        "labels_manifest": Path(labels_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs:
        if input_hashes != dict(FROZEN_FORMAL_SHA256):
            raise ValueError("formal NEXT16 retrospective inputs differ")
        if bootstrap_reps != FROZEN_BOOTSTRAP_REPS:
            raise ValueError("formal NEXT16 bootstrap repetitions differ")

    features, _ = _validate_features(paths["features"], paths["features_manifest"])
    all_labels, _ = _validate_labels(paths["labels"], paths["labels_manifest"])
    selected_groups = set(features["rk"])
    labels = all_labels.loc[all_labels["label_rk"].isin(selected_groups)].copy()
    if set(labels["material_id"]) != set(features["material_id"]):
        raise ValueError("NEXT16 features do not contain complete composition groups")
    joined = features.merge(labels, on="material_id", validate="one_to_one")
    if not joined["rk"].eq(joined["label_rk"]).all():
        raise ValueError("NEXT16 feature and ELEMENTA label compositions differ")
    group_minimum = joined.groupby("rk")["dft_energy_ev_per_atom"].transform("min")
    joined["dft_group_regret_ev_per_atom"] = (
        joined["dft_energy_ev_per_atom"] - group_minimum
    )
    regret = joined["dft_group_regret_ev_per_atom"].to_numpy(float)
    if (regret < -MINIMUM_TOLERANCE_EV_PER_ATOM).any():
        raise ValueError("negative ELEMENTA group regret beyond tolerance")
    joined.loc[
        joined["dft_group_regret_ev_per_atom"].abs() <= MINIMUM_TOLERANCE_EV_PER_ATOM,
        "dft_group_regret_ev_per_atom",
    ] = 0.0

    method_columns = {
        "pauling_p2_p5": "pauling_p2_p5_decision",
        "next16_basin_hull": "basin_hull_decision",
    }
    methods = {
        name: _method_metrics(joined, column) for name, column in method_columns.items()
    }
    comparison = _bootstrap_difference(
        joined,
        method_columns["next16_basin_hull"],
        method_columns["pauling_p2_p5"],
        reps=bootstrap_reps,
        seed=FROZEN_BOOTSTRAP_SEED,
    )
    complete, clauses = _complete_superiority(
        methods["next16_basin_hull"], methods["pauling_p2_p5"], comparison
    )
    comparison["superiority_clauses"] = clauses
    comparison["complete_superiority_over_pauling"] = complete
    capped = joined["capped_at_max_steps"].astype(bool).to_numpy()
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": "historical ELEMENTA cross-source retrospective after WBM candidate selection",
        "candidate_selected_on": "WBM",
        "labels_previously_opened": True,
        "fresh_lockbox": False,
        "feature_execution_read_elementa_endpoints": False,
        "thresholds_refit": False,
        "fixed_rule": {
            "formula": "B64 = E_MatterSim_relaxed/N - E_raw_MP_hull(composition)",
            "comparison": ">=",
            "threshold_ev_per_atom": NEXT16_THRESHOLD_EV_PER_ATOM,
        },
        "endpoint_definitions": {
            "group_minimum_tolerance_ev_per_atom": MINIMUM_TOLERANCE_EV_PER_ATOM,
            "valuable_regret_at_most_ev_per_atom": VALUABLE_REGRET_EV_PER_ATOM,
            "high_regret_at_least_ev_per_atom": HIGH_REGRET_EV_PER_ATOM,
        },
        "counts": {
            "rows": len(joined),
            "complete_composition_groups": int(joined["rk"].nunique()),
            "group_minimum_rows": int((joined["dft_group_regret_ev_per_atom"] <= MINIMUM_TOLERANCE_EV_PER_ATOM).sum()),
            "high_regret_rows": int((joined["dft_group_regret_ev_per_atom"] >= HIGH_REGRET_EV_PER_ATOM).sum()),
            "capped_at_max_steps": int(capped.sum()),
            "capped_rejected": int(
                (capped & joined["basin_hull_decision"].astype(str).eq("REJECT").to_numpy()).sum()
            ),
        },
        "methods": methods,
        "comparisons_to_pauling": {"next16_basin_hull": comparison},
        "score_diagnostics": _score_diagnostics(joined),
        "cross_source_retrospective_support": complete,
        "scientific_improvement_claim": False,
        "interpretation_guard": (
            "Passing is cross-source retrospective support only: ELEMENTA labels were historically visible, "
            "the candidate was selected post hoc on WBM, the MP reference contains prior DFT, and this is not "
            "a fresh lockbox or proof of universal DFT equivalence."
        ),
    }

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next16_elementa_evaluate.py": Path(__file__).resolve(),
        "src/next16_elementa_basin_hull.py": repository_root / "src/next16_elementa_basin_hull.py",
        "src/next14_wbm_evaluate.py": repository_root / "src/next14_wbm_evaluate.py",
    }
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    private_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "identifier_bearing": True,
        "storage_role": "external private NEXT16 ELEMENTA prediction/label join",
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
    }
    aggregate_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "identifier_bearing": False,
        "labels_previously_opened": True,
        "fresh_lockbox": False,
        "thresholds_refit": False,
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "private_output_path": str(private_target),
        "scientific_improvement_claim": False,
    }

    private_target.parent.mkdir(parents=True, exist_ok=True)
    private_staging = Path(
        tempfile.mkdtemp(prefix=f".{private_target.name}.staging-", dir=private_target.parent)
    )
    try:
        joined_path = private_staging / PRIVATE_JOINED_NAME
        joined.to_parquet(joined_path, index=False)
        private_manifest["outputs_sha256"] = {PRIVATE_JOINED_NAME: _sha256_file(joined_path)}
        (private_staging / MANIFEST_NAME).write_bytes(_json_bytes(private_manifest))
        _publish_directory_no_replace(private_staging, private_target)
    except Exception:
        shutil.rmtree(private_staging, ignore_errors=True)
        raise

    aggregate_target.parent.mkdir(parents=True, exist_ok=True)
    aggregate_staging = Path(
        tempfile.mkdtemp(prefix=f".{aggregate_target.name}.staging-", dir=aggregate_target.parent)
    )
    try:
        result_path = aggregate_staging / RESULT_NAME
        result_path.write_bytes(_json_bytes(result))
        aggregate_manifest["outputs_sha256"] = {RESULT_NAME: _sha256_file(result_path)}
        aggregate_manifest["private_outputs_sha256"] = private_manifest["outputs_sha256"]
        (aggregate_staging / MANIFEST_NAME).write_bytes(_json_bytes(aggregate_manifest))
        _publish_directory_no_replace(aggregate_staging, aggregate_target)
    except Exception:
        shutil.rmtree(aggregate_staging, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--features-manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--labels-manifest", required=True, type=Path)
    parser.add_argument("--private-output-dir", required=True, type=Path)
    parser.add_argument("--aggregate-output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    evaluate_elementa_retrospective(
        features_path=arguments.features,
        features_manifest_path=arguments.features_manifest,
        labels_path=arguments.labels,
        labels_manifest_path=arguments.labels_manifest,
        private_output_dir=arguments.private_output_dir,
        aggregate_output_dir=arguments.aggregate_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
