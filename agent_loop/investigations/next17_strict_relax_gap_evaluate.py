"""Select and evaluate NEXT17 from a frozen threshold catalog on development labels."""

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
from src.next16_elementa_evaluate import (
    HIGH_REGRET_EV_PER_ATOM,
    MINIMUM_TOLERANCE_EV_PER_ATOM,
    VALUABLE_REGRET_EV_PER_ATOM,
    _validate_labels,
)
from src.next17_strict_relax_gap import PROTOCOL as STRICT_FEATURE_PROTOCOL


PROTOCOL = "2026-08-02-next17-strict-relax-gap-development-v1"
X0_PROTOCOL = "2026-08-01-mattersim-x0-baseline-v1"
CHECKPOINT_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
THRESHOLD_CATALOG = (0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12)
MINIMUM_RECALL_LOWER = 0.95
VALUABLE_RECALL_LOWER = 0.95
PRECISION_LOWER = 0.95
SAVINGS_LOWER = 0.10
HIGH_RECALL_NONINFERIORITY = -0.05
NONTRIVIAL_MEAN_ABS_GAP_CHANGE = 1.0e-6
FROZEN_BOOTSTRAP_REPS = 10_000
FROZEN_BOOTSTRAP_SEED = 20260817
RESULT_NAME = "NEXT17_STRICT_RELAX_GAP_DEVELOPMENT.json"
PRIVATE_JOINED_NAME = "joined_strict_x0_predictions_labels.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "strict_features": "afb86ff81b5bee3159dc96355b9402ee3fef7ac07479b2110ef6d510a306af2a",
    "strict_manifest": "674fc360fdb2f9fbb2c98c1853683d6a390999ac3c84f47776036ced42b80102",
    "x0_features": "7ca60e4b1183e1465e2efde64a4445c313df7c20d52e5ffef53119548c285095",
    "x0_manifest": "4bd43b8074cfd15492d26cf0ee9f7ed9ca6dff1cc11f42f2ccb43a1eccf0eecc",
    "labels": "86225b5fea9275113dedb87bd4963ac1bf5bc3d02d478319a993f4285fc48f0f",
    "labels_manifest": "ae3b192f3d28400fffd2eb818e574e60bb9400a4b993d196cc1ba2fcac0ebb99",
}


def _validate_strict_features(
    path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role="NEXT17 strict manifest")
    if manifest.get("protocol") != STRICT_FEATURE_PROTOCOL:
        raise ValueError("NEXT17 strict protocol differs")
    if (
        manifest.get("elementa_endpoint_bytes_read_by_execution") is not False
        or manifest.get("mp_hull_bytes_read_by_execution") is not False
        or manifest.get("threshold_selected") is not False
    ):
        raise ValueError("NEXT17 strict feature execution was not label-free and threshold-free")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError("NEXT17 strict feature hash differs from manifest")
    table = pd.read_parquet(io.BytesIO(data))
    required = {
        "material_id", "rk", "supported", "strict_group_supported",
        "strict_relative_gap_ev_per_atom", "prediction_steps", "capped_at_max_steps",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"NEXT17 strict features lack columns: {sorted(missing)}")
    table = table.loc[:, sorted(required)].copy()
    table["material_id"] = table["material_id"].astype(str)
    table["rk"] = table["rk"].astype(str)
    if table["material_id"].isna().any() or table["material_id"].duplicated().any():
        raise ValueError("NEXT17 strict material identities must be unique")
    group_supported = table["strict_group_supported"].astype(bool)
    gap = pd.to_numeric(table["strict_relative_gap_ev_per_atom"], errors="coerce")
    if not np.isfinite(gap[group_supported].to_numpy(float)).all() or (
        gap[group_supported] < -1.0e-10
    ).any():
        raise ValueError("NEXT17 strict supported gaps are invalid")
    if gap[~group_supported].notna().any():
        raise ValueError("NEXT17 unsupported group has a finite gap")
    minima = gap[group_supported].groupby(table.loc[group_supported, "rk"]).min()
    if len(minima) and not np.allclose(minima.to_numpy(float), 0.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("NEXT17 strict group gaps are not zero-based")
    return table.sort_values("material_id", kind="stable", ignore_index=True), manifest


def _validate_x0_features(
    path: Path, manifest_path: Path, expected_ids: set[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = path.read_bytes()
    manifest = _strict_json(manifest_path.read_bytes(), role="MatterSim x0 manifest")
    model = manifest.get("model")
    if (
        manifest.get("protocol") != X0_PROTOCOL
        or not isinstance(model, Mapping)
        or model.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise ValueError("MatterSim x0 source contract differs")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError("MatterSim x0 feature hash differs from manifest")
    table = pd.read_parquet(io.BytesIO(data))
    required = {"sid", "rk", "mattersim_feature_ok", "mattersim_energy_per_atom"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"MatterSim x0 features lack columns: {sorted(missing)}")
    table = table.loc[:, sorted(required)].rename(
        columns={"sid": "material_id", "rk": "x0_rk"}
    )
    table["material_id"] = table["material_id"].astype(str)
    table["x0_rk"] = table["x0_rk"].astype(str)
    if table["material_id"].isna().any() or table["material_id"].duplicated().any():
        raise ValueError("MatterSim x0 identities must be unique")
    selected = table.loc[table["material_id"].isin(expected_ids)].copy()
    if set(selected["material_id"]) != expected_ids:
        raise ValueError("MatterSim x0 does not cover the strict cohort")
    ok = selected["mattersim_feature_ok"].astype(bool)
    energy = pd.to_numeric(selected["mattersim_energy_per_atom"], errors="coerce")
    row_supported = ok & np.isfinite(energy.to_numpy(float))
    group_supported = row_supported.groupby(selected["x0_rk"], sort=False).transform("all")
    minimum = energy.where(group_supported).groupby(selected["x0_rk"], sort=False).transform("min")
    selected["x0_group_supported"] = group_supported.astype(bool)
    selected["x0_relative_gap_ev_per_atom"] = (energy - minimum).where(group_supported)
    return selected.sort_values("material_id", kind="stable", ignore_index=True), manifest


def _decisions(gap: pd.Series, supported: pd.Series, threshold: float) -> pd.Series:
    supported_array = supported.astype(bool).to_numpy()
    values = pd.to_numeric(gap, errors="coerce").to_numpy(float)
    if not np.isfinite(values[supported_array]).all():
        raise ValueError("supported decision gap is not finite")
    return pd.Series(
        np.where(~supported_array, "ABSTAIN", np.where(values >= threshold, "REJECT", "KEEP")),
        index=gap.index,
        dtype=object,
    )


def _method_metrics(joined: pd.DataFrame, decisions: pd.Series) -> dict[str, object]:
    decision = decisions.astype(str)
    if not set(decision).issubset(DECISIONS):
        raise ValueError("unknown NEXT17 decision")
    reject = decision.eq("REJECT").to_numpy(bool)
    covered = ~decision.eq("ABSTAIN").to_numpy(bool)
    regret = joined["dft_group_regret_ev_per_atom"].to_numpy(float)
    minimum = regret <= MINIMUM_TOLERANCE_EV_PER_ATOM
    valuable = regret <= VALUABLE_REGRET_EV_PER_ATOM
    high = regret >= HIGH_REGRET_EV_PER_ATOM
    above_minimum = regret > MINIMUM_TOLERANCE_EV_PER_ATOM
    group = pd.DataFrame(
        {"rk": joined["rk"].astype(str), "reject": reject, "covered": covered}
    ).groupby("rk", sort=True)
    fully_rejected = group.apply(
        lambda frame: bool(frame["covered"].all() and frame["reject"].all()),
        include_groups=False,
    )
    return {
        "decision_counts": {
            value: int(decision.eq(value).sum()) for value in sorted(DECISIONS)
        },
        "coverage": _proportion(int(covered.sum()), len(joined)),
        "dft_savings": _proportion(int(reject.sum()), len(joined)),
        "group_minimum_recall": _proportion(
            int((minimum & ~reject).sum()), int(minimum.sum())
        ),
        "valuable_recall": _proportion(
            int((valuable & ~reject).sum()), int(valuable.sum())
        ),
        "high_energy_rejection_recall": _proportion(
            int((high & reject).sum()), int(high.sum())
        ),
        "reject_precision_above_minimum": _proportion(
            int((above_minimum & reject).sum()), int(reject.sum())
        ),
        "all_rejected_groups": int(fully_rejected.sum()),
    }


def _lower(metric: Mapping[str, object]) -> float:
    interval = metric.get("wilson_ci95")
    if not isinstance(interval, Sequence) or not interval or interval[0] is None:
        return math.nan
    return float(interval[0])


def _eligible(metrics: Mapping[str, object]) -> tuple[bool, dict[str, bool]]:
    clauses = {
        "group_minimum_recall_lower_at_least_0_95": _lower(metrics["group_minimum_recall"]) >= MINIMUM_RECALL_LOWER,
        "valuable_recall_lower_at_least_0_95": _lower(metrics["valuable_recall"]) >= VALUABLE_RECALL_LOWER,
        "reject_precision_lower_at_least_0_95": _lower(metrics["reject_precision_above_minimum"]) >= PRECISION_LOWER,
        "dft_savings_lower_at_least_0_10": _lower(metrics["dft_savings"]) >= SAVINGS_LOWER,
        "no_composition_group_fully_rejected": int(metrics["all_rejected_groups"]) == 0,
    }
    return all(clauses.values()), clauses


def _scan_catalog(
    joined: pd.DataFrame, *, gap_column: str, support_column: str
) -> dict[str, dict[str, object]]:
    scan: dict[str, dict[str, object]] = {}
    for threshold in THRESHOLD_CATALOG:
        decisions = _decisions(joined[gap_column], joined[support_column], threshold)
        metrics = _method_metrics(joined, decisions)
        eligible, clauses = _eligible(metrics)
        scan[str(threshold)] = {
            "threshold_ev_per_atom": threshold,
            "eligible": eligible,
            "eligibility_clauses": clauses,
            "metrics": metrics,
        }
    return scan


def _select_threshold(scan: Mapping[str, Mapping[str, object]]) -> float | None:
    eligible: list[tuple[float, float]] = []
    for threshold in THRESHOLD_CATALOG:
        entry = scan.get(str(threshold))
        if not isinstance(entry, Mapping) or entry.get("eligible") is not True:
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        savings = metrics.get("dft_savings")
        estimate = savings.get("estimate") if isinstance(savings, Mapping) else None
        if isinstance(estimate, (int, float)) and not isinstance(estimate, bool):
            eligible.append((float(estimate), threshold))
    return max(eligible)[1] if eligible else None


def _group_aggregates(joined: pd.DataFrame, decision_column: str) -> np.ndarray:
    decision = joined[decision_column].astype(str)
    reject = decision.eq("REJECT")
    covered = ~decision.eq("ABSTAIN")
    regret = joined["dft_group_regret_ev_per_atom"].to_numpy(float)
    minimum = regret <= MINIMUM_TOLERANCE_EV_PER_ATOM
    valuable = regret <= VALUABLE_REGRET_EV_PER_ATOM
    high = regret >= HIGH_REGRET_EV_PER_ATOM
    above = regret > MINIMUM_TOLERANCE_EV_PER_ATOM
    frame = pd.DataFrame(
        {
            "rk": joined["rk"].astype(str),
            "n": 1,
            "minimum": minimum.astype(int),
            "minimum_retained": (minimum & ~reject.to_numpy(bool)).astype(int),
            "valuable": valuable.astype(int),
            "valuable_retained": (valuable & ~reject.to_numpy(bool)).astype(int),
            "high": high.astype(int),
            "high_rejected": (high & reject.to_numpy(bool)).astype(int),
            "reject": reject.astype(int),
            "above_rejected": (above & reject.to_numpy(bool)).astype(int),
            "abstain": (~covered).astype(int),
        }
    )
    return frame.groupby("rk", sort=True).sum().to_numpy(float)


def _metrics_from_sums(values: np.ndarray) -> np.ndarray:
    n, minimum, minimum_retained, valuable, valuable_retained, high, high_rejected, reject, above_rejected, abstain = np.moveaxis(values, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack(
            [
                reject / n,
                minimum_retained / minimum,
                valuable_retained / valuable,
                high_rejected / high,
                above_rejected / reject,
                abstain / n,
            ],
            axis=-1,
        )


def _bootstrap_difference(
    joined: pd.DataFrame, method_column: str, baseline_column: str, *, reps: int
) -> dict[str, object]:
    method = _group_aggregates(joined, method_column)
    baseline = _group_aggregates(joined, baseline_column)
    point = _metrics_from_sums(method.sum(axis=0)) - _metrics_from_sums(
        baseline.sum(axis=0)
    )
    rng = np.random.default_rng(FROZEN_BOOTSTRAP_SEED)
    samples = np.empty((reps, 6), dtype=float)
    groups = len(method)
    for start in range(0, reps, 256):
        size = min(256, reps - start)
        indices = rng.integers(0, groups, size=(size, groups))
        samples[start : start + size] = _metrics_from_sums(
            method[indices].sum(axis=1)
        ) - _metrics_from_sums(baseline[indices].sum(axis=1))

    def interval(values: np.ndarray) -> list[float | None]:
        finite = values[np.isfinite(values)]
        return (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if len(finite)
            else [None, None]
        )

    names = (
        "dft_savings", "group_minimum_recall", "valuable_recall",
        "high_energy_rejection_recall", "reject_precision_above_minimum", "abstention_rate",
    )
    return {
        "bootstrap_reps": reps,
        "bootstrap_seed": FROZEN_BOOTSTRAP_SEED,
        **{
            name: {
                "estimate": float(point[index]) if np.isfinite(point[index]) else None,
                "cluster_bootstrap_ci95": interval(samples[:, index]),
            }
            for index, name in enumerate(names)
        },
    }


def evaluate_strict_relax_development(
    *,
    strict_features_path: Path,
    strict_manifest_path: Path,
    x0_features_path: Path,
    x0_manifest_path: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    private_output_dir: Path,
    aggregate_output_dir: Path,
    bootstrap_reps: int = FROZEN_BOOTSTRAP_REPS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Select strict and x0 thresholds under identical frozen development gates."""

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
        "strict_features": Path(strict_features_path).resolve(),
        "strict_manifest": Path(strict_manifest_path).resolve(),
        "x0_features": Path(x0_features_path).resolve(),
        "x0_manifest": Path(x0_manifest_path).resolve(),
        "labels": Path(labels_path).resolve(),
        "labels_manifest": Path(labels_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs:
        if input_hashes != dict(FROZEN_FORMAL_SHA256):
            raise ValueError("formal NEXT17 development inputs differ")
        if bootstrap_reps != FROZEN_BOOTSTRAP_REPS:
            raise ValueError("formal NEXT17 bootstrap repetitions differ")

    strict, strict_manifest = _validate_strict_features(
        paths["strict_features"], paths["strict_manifest"]
    )
    expected_ids = set(strict["material_id"])
    x0, _ = _validate_x0_features(
        paths["x0_features"], paths["x0_manifest"], expected_ids
    )
    all_labels, _ = _validate_labels(paths["labels"], paths["labels_manifest"])
    selected_groups = set(strict["rk"])
    labels = all_labels.loc[all_labels["label_rk"].isin(selected_groups)].copy()
    if set(labels["material_id"]) != expected_ids:
        raise ValueError("NEXT17 features do not cover complete label groups")
    joined = strict.merge(x0, on="material_id", validate="one_to_one").merge(
        labels, on="material_id", validate="one_to_one"
    )
    if not joined["rk"].eq(joined["x0_rk"]).all() or not joined["rk"].eq(joined["label_rk"]).all():
        raise ValueError("NEXT17 strict, x0, and label compositions differ")
    group_minimum = joined.groupby("rk")["dft_energy_ev_per_atom"].transform("min")
    joined["dft_group_regret_ev_per_atom"] = joined["dft_energy_ev_per_atom"] - group_minimum
    joined.loc[
        joined["dft_group_regret_ev_per_atom"].abs() <= MINIMUM_TOLERANCE_EV_PER_ATOM,
        "dft_group_regret_ev_per_atom",
    ] = 0.0

    strict_scan = _scan_catalog(
        joined,
        gap_column="strict_relative_gap_ev_per_atom",
        support_column="strict_group_supported",
    )
    x0_scan = _scan_catalog(
        joined,
        gap_column="x0_relative_gap_ev_per_atom",
        support_column="x0_group_supported",
    )
    strict_threshold = _select_threshold(strict_scan)
    x0_threshold = _select_threshold(x0_scan)
    comparison: dict[str, object] | None = None
    development_promotion = False
    promotion_clauses: dict[str, bool] = {
        "strict_candidate_eligible": strict_threshold is not None,
        "x0_comparator_eligible": x0_threshold is not None,
        "dft_savings_difference_lower_above_zero": False,
        "high_energy_recall_noninferior_within_0_05": False,
        "within_group_score_changed_nontrivially": False,
    }
    joint = joined["strict_group_supported"].astype(bool) & joined["x0_group_supported"].astype(bool)
    score_change = np.abs(
        joined.loc[joint, "strict_relative_gap_ev_per_atom"].to_numpy(float)
        - joined.loc[joint, "x0_relative_gap_ev_per_atom"].to_numpy(float)
    )
    score_change_summary = {
        "joint_supported_rows": int(joint.sum()),
        "mean_absolute_change_ev_per_atom": float(score_change.mean()) if len(score_change) else None,
        "p95_absolute_change_ev_per_atom": float(np.quantile(score_change, 0.95)) if len(score_change) else None,
        "maximum_absolute_change_ev_per_atom": float(score_change.max()) if len(score_change) else None,
    }
    if strict_threshold is not None and x0_threshold is not None:
        joined["strict_selected_decision"] = _decisions(
            joined["strict_relative_gap_ev_per_atom"], joined["strict_group_supported"], strict_threshold
        )
        joined["x0_selected_decision"] = _decisions(
            joined["x0_relative_gap_ev_per_atom"], joined["x0_group_supported"], x0_threshold
        )
        comparison = _bootstrap_difference(
            joined, "strict_selected_decision", "x0_selected_decision", reps=bootstrap_reps
        )
        savings_lower = comparison["dft_savings"]["cluster_bootstrap_ci95"][0]
        high_lower = comparison["high_energy_rejection_recall"]["cluster_bootstrap_ci95"][0]
        promotion_clauses.update(
            {
                "dft_savings_difference_lower_above_zero": bool(
                    savings_lower is not None and float(savings_lower) > 0.0
                ),
                "high_energy_recall_noninferior_within_0_05": bool(
                    high_lower is not None and float(high_lower) >= HIGH_RECALL_NONINFERIORITY
                ),
                "within_group_score_changed_nontrivially": bool(
                    score_change_summary["mean_absolute_change_ev_per_atom"] is not None
                    and float(score_change_summary["mean_absolute_change_ev_per_atom"])
                    > NONTRIVIAL_MEAN_ABS_GAP_CHANGE
                ),
            }
        )
        development_promotion = all(promotion_clauses.values())
        comparison["promotion_clauses"] = promotion_clauses
        comparison["passes_development_promotion"] = development_promotion

    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "evidence_role": "historically exposed ELEMENTA development selection; not external validation",
        "labels_previously_opened": True,
        "fresh_lockbox": False,
        "threshold_catalog": list(THRESHOLD_CATALOG),
        "selection_policy": "among eligible thresholds maximize DFT savings, then prefer larger threshold",
        "safety_gates": {
            "group_minimum_recall_wilson_lower": MINIMUM_RECALL_LOWER,
            "valuable_recall_wilson_lower": VALUABLE_RECALL_LOWER,
            "reject_precision_wilson_lower": PRECISION_LOWER,
            "dft_savings_wilson_lower": SAVINGS_LOWER,
            "all_rejected_groups": 0,
        },
        "counts": {
            "rows": len(joined),
            "complete_composition_groups": int(joined["rk"].nunique()),
            "high_regret_rows": int((joined["dft_group_regret_ev_per_atom"] >= HIGH_REGRET_EV_PER_ATOM).sum()),
            "strict_capped_at_max_steps": int(joined["capped_at_max_steps"].astype(bool).sum()),
        },
        "strict_relax": {
            "selected_threshold_ev_per_atom": strict_threshold,
            "catalog_scan": strict_scan,
            "execution_wall_time_seconds": strict_manifest.get("execution", {}).get("wall_time_seconds") if isinstance(strict_manifest.get("execution"), Mapping) else None,
        },
        "x0": {
            "selected_threshold_ev_per_atom": x0_threshold,
            "catalog_scan": x0_scan,
        },
        "selected_comparison_strict_minus_x0": comparison,
        "score_change_vs_x0": score_change_summary,
        "development_promotion": development_promotion,
        "promotion_clauses": promotion_clauses,
        "scientific_improvement_claim": False,
        "interpretation_guard": (
            "A passing promotion is a development result on historically exposed ELEMENTA labels. "
            "It freezes a candidate for a later cohort; it is not independent validation or DFT equivalence."
        ),
    }

    repo_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next17_strict_relax_gap_evaluate.py": Path(__file__).resolve(),
        "src/next17_strict_relax_gap.py": repo_root / "src/next17_strict_relax_gap.py",
        "src/next16_elementa_evaluate.py": repo_root / "src/next16_elementa_evaluate.py",
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    private_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "identifier_bearing": True,
        "storage_role": "external private NEXT17 development join",
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
    }
    aggregate_manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "identifier_bearing": False,
        "fresh_lockbox": False,
        "labels_previously_opened": True,
        "inputs_sha256": input_hashes,
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
        (private_staging / MANIFEST_NAME).write_bytes(_json_bytes(private_manifest))
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
        (aggregate_staging / MANIFEST_NAME).write_bytes(_json_bytes(aggregate_manifest))
        _publish_directory_no_replace(aggregate_staging, aggregate_target)
    except Exception:
        shutil.rmtree(aggregate_staging, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-features", required=True, type=Path)
    parser.add_argument("--strict-manifest", required=True, type=Path)
    parser.add_argument("--x0-features", required=True, type=Path)
    parser.add_argument("--x0-manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--labels-manifest", required=True, type=Path)
    parser.add_argument("--private-output-dir", required=True, type=Path)
    parser.add_argument("--aggregate-output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    evaluate_strict_relax_development(
        strict_features_path=args.strict_features,
        strict_manifest_path=args.strict_manifest,
        x0_features_path=args.x0_features,
        x0_manifest_path=args.x0_manifest,
        labels_path=args.labels,
        labels_manifest_path=args.labels_manifest,
        private_output_dir=args.private_output_dir,
        aggregate_output_dir=args.aggregate_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
