#!/usr/bin/env python3
"""Sequential soft-margin anion search for np-next-20260801e."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from better_law_search import (
    _apply_result,
    _candidate_record,
    _evaluate_result,
    law_preliminary_gate,
    pareto_beam,
    selected_rule_coverage,
)
from next4_law_search import (
    _fold_comparisons,
    _minimum_kind,
    _sha256,
    _worst_anion_delta,
    build_next4_candidate_sets,
    load_next4_search_frames,
    paired_robust_strata,
    robust_pareto_beam,
    verify_isolated_manifest,
)

EXPERIMENT_ID = "np-next-20260801e"
FULL_ANION_MARGIN = 0.0025
CELL_MARGIN = 0.01


def soften_paired_floors(
    strict_floors: Mapping[str, float],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    full_anion_margin: float = FULL_ANION_MARGIN,
    cell_margin: float = CELL_MARGIN,
) -> dict[str, float]:
    """Apply the frozen full-anion and anion-by-fold margins."""

    if set(strict_floors) != set(metadata):
        raise ValueError("strict_floors and metadata must have matching keys")
    if full_anion_margin < 0 or cell_margin < 0:
        raise ValueError("soft margins must be non-negative")
    return {
        name: max(
            0.0,
            float(floor)
            - (
                full_anion_margin
                if metadata[name].get("fold") == "all"
                else cell_margin
            ),
        )
        for name, floor in strict_floors.items()
    }


def paired_soft_strata(
    real: pd.DataFrame,
    baseline_mask: Sequence[bool],
    *,
    n_folds: int = 4,
    min_anion_rows: int = 200,
    min_cell_rows: int = 50,
    full_anion_margin: float = FULL_ANION_MARGIN,
    cell_margin: float = CELL_MARGIN,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, dict[str, object]],
]:
    """Build paired strata, including one pooled rare-anion stratum."""

    strata, strict_floors, metadata = paired_robust_strata(
        real,
        baseline_mask,
        n_folds=n_folds,
        min_anion_rows=min_anion_rows,
        min_cell_rows=min_cell_rows,
    )
    baseline = np.asarray(baseline_mask, dtype=bool)
    counts = real["anion"].value_counts(dropna=True)
    rare = sorted(str(anion) for anion, n in counts.items() if n < min_anion_rows)
    values = real["anion"].astype("string").to_numpy(dtype=object)
    rare_mask = np.isin(values, rare)
    if int(rare_mask.sum()) >= min_cell_rows:
        name = "anion:other-anions"
        satisfaction = float(baseline[rare_mask].mean())
        strata[name] = rare_mask
        strict_floors[name] = satisfaction
        metadata[name] = {
            "anion": "other-anions",
            "members": rare,
            "fold": "all",
            "n": int(rare_mask.sum()),
            "baseline_satisfaction": satisfaction,
        }
    floors = soften_paired_floors(
        strict_floors,
        metadata,
        full_anion_margin=full_anion_margin,
        cell_margin=cell_margin,
    )
    return strata, floors, metadata


def joint_feature_coverage_by_stratum(
    frame: pd.DataFrame,
    records: Sequence[Mapping[str, object]],
    strata: Mapping[str, Sequence[bool]],
) -> dict[str, object]:
    """Measure joint finite coverage of every selected target and guard."""

    features = sorted(
        {
            str(value)
            for record in records
            for value in (record["feature"], record.get("guard_feature"))
            if value is not None
        }
    )
    missing = [feature for feature in features if feature not in frame]
    if missing:
        raise ValueError("selected rule feature missing: " + ", ".join(missing))
    finite = np.ones(len(frame), dtype=bool)
    for feature in features:
        finite &= np.isfinite(frame[feature].to_numpy(dtype=float))
    by_stratum: dict[str, float] = {}
    for name, raw_mask in strata.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (len(frame),) or not mask.any():
            raise ValueError(f"invalid coverage stratum: {name}")
        by_stratum[name] = float(finite[mask].mean())
    return {
        "features": features,
        "overall": float(finite.mean()),
        "by_stratum": by_stratum,
        "minimum_stratum": min(by_stratum.values(), default=1.0),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--real-sixfam", type=Path, required=True)
    parser.add_argument("--bad-sixfam", type=Path, required=True)
    parser.add_argument("--real-corrected", type=Path, required=True)
    parser.add_argument("--bad-corrected", type=Path, required=True)
    parser.add_argument("--real-guards", type=Path, required=True)
    parser.add_argument("--bad-guards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.98)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--max-guard-targets", type=int, default=100)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--min-anion-rows", type=int, default=200)
    parser.add_argument("--min-cell-rows", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if not 0 < args.floor < 1:
        raise SystemExit("--floor must lie strictly between zero and one")
    started = time.time()
    isolation_audit = verify_isolated_manifest(args.isolated_dir)
    dr, cr, db, cb = load_next4_search_frames(
        args.isolated_dir,
        args.real_descriptors,
        args.bad_descriptors,
        args.real_sixfam,
        args.bad_sixfam,
        args.real_corrected,
        args.bad_corrected,
        args.real_guards,
        args.bad_guards,
    )
    candidate_sets, candidate_counts = build_next4_candidate_sets(
        dr,
        db,
        min_coverage=args.min_coverage,
        max_guard_targets=args.max_guard_targets,
        guard_min_real_satisfaction=args.floor,
    )
    existing_candidates = candidate_sets["existing_loop"]
    additive_candidates = candidate_sets["additive_corrected_loop"]

    baseline_result = pareto_beam(
        existing_candidates,
        real_size=len(dr),
        bad_size=len(db),
        bad_kinds=db["kind"].to_numpy(),
        satisfaction_floor=args.floor,
        max_rules=args.max_rules,
        width=24,
    )
    baseline_discovery = _evaluate_result(
        dr, db, existing_candidates, baseline_result, use_stored_masks=True
    )
    strata, floors, strata_metadata = paired_soft_strata(
        dr,
        baseline_result.real_mask,
        n_folds=args.n_folds,
        min_anion_rows=args.min_anion_rows,
        min_cell_rows=args.min_cell_rows,
    )
    soft_result = robust_pareto_beam(
        additive_candidates,
        real_size=len(dr),
        bad_size=len(db),
        bad_kinds=db["kind"].to_numpy(),
        satisfaction_floor=args.floor,
        max_rules=args.max_rules,
        width=args.width,
        real_strata=strata,
        stratum_floors=floors,
    )
    soft_discovery = _evaluate_result(
        dr, db, additive_candidates, soft_result, use_stored_masks=True
    )

    baseline_rules = [
        _candidate_record(existing_candidates[index])
        for index in baseline_result.indices
    ]
    selected_rules = [
        _candidate_record(additive_candidates[index])
        for index in soft_result.indices
    ]
    selected_rule_sha256 = hashlib.sha256(
        json.dumps(
            selected_rules,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    baseline_calibration = _evaluate_result(
        cr, cb, existing_candidates, baseline_result, use_stored_masks=False
    )
    soft_calibration = _evaluate_result(
        cr, cb, additive_candidates, soft_result, use_stored_masks=False
    )
    baseline_cal_real = _apply_result(cr, existing_candidates, baseline_result)
    baseline_cal_bad = _apply_result(cb, existing_candidates, baseline_result)
    soft_cal_real = _apply_result(cr, additive_candidates, soft_result)
    soft_cal_bad = _apply_result(cb, additive_candidates, soft_result)

    coverage = selected_rule_coverage(
        pd.concat([dr, cr], ignore_index=True),
        pd.concat([db, cb], ignore_index=True),
        selected_rules,
    )
    discovery_joint_coverage = joint_feature_coverage_by_stratum(
        dr, selected_rules, strata
    )
    discovery_worst = _worst_anion_delta(baseline_discovery, soft_discovery)
    calibration_worst = _worst_anion_delta(
        baseline_calibration, soft_calibration
    )
    fold_comparisons = _fold_comparisons(
        dr,
        db,
        baseline_result.real_mask,
        baseline_result.bad_mask,
        soft_result.real_mask,
        soft_result.bad_mask,
        n_folds=args.n_folds,
    )
    fold_rejection_deltas = [
        float(row["rejection_delta"]) for row in fold_comparisons
    ]
    fold_stability_gate = bool(
        sum(delta >= 0 for delta in fold_rejection_deltas) >= 3
        and min(fold_rejection_deltas, default=-1.0) >= -0.02
    )
    metric_gate = law_preliminary_gate(
        new_descriptor_selected=any(
            str(rule["origin"]).startswith("next4") for rule in selected_rules
        ),
        additive_satisfaction=float(soft_calibration["satisfaction"]),
        base_satisfaction=float(baseline_calibration["satisfaction"]),
        rejection_delta=float(
            soft_calibration["rejection"] - baseline_calibration["rejection"]
        ),
        min_kind_rejection_delta=float(
            _minimum_kind(soft_calibration)
            - _minimum_kind(baseline_calibration)
        ),
        worst_anion_delta=calibration_worst,
        real_coverage=float(coverage["real_min"]),
        bad_coverage=float(coverage["bad_min"]),
        source_tables_materialized_lockbox_rows=False,
    )
    comparison = {
        "discovery_satisfaction_delta": float(
            soft_discovery["satisfaction"] - baseline_discovery["satisfaction"]
        ),
        "discovery_rejection_delta": float(
            soft_discovery["rejection"] - baseline_discovery["rejection"]
        ),
        "discovery_minimum_kind_delta": float(
            _minimum_kind(soft_discovery) - _minimum_kind(baseline_discovery)
        ),
        "discovery_worst_shared_anion_delta": discovery_worst,
        "calibration_satisfaction_delta": float(
            soft_calibration["satisfaction"]
            - baseline_calibration["satisfaction"]
        ),
        "calibration_rejection_delta": float(
            soft_calibration["rejection"] - baseline_calibration["rejection"]
        ),
        "calibration_minimum_kind_delta": float(
            _minimum_kind(soft_calibration)
            - _minimum_kind(baseline_calibration)
        ),
        "calibration_worst_shared_anion_delta": calibration_worst,
        "selected_rule_coverage": coverage,
        "discovery_joint_coverage": discovery_joint_coverage,
        "calibration_metric_gate": metric_gate,
        "discovery_fold_stability_gate": fold_stability_gate,
        "pre_loko_gate": bool(
            metric_gate
            and fold_stability_gate
            and discovery_joint_coverage["minimum_stratum"] >= 0.90
        ),
        "status": "pending true LOKO and all-295 unknown-fails-closed falsification",
    }
    report = {
        "protocol": {
            "experiment": EXPERIMENT_ID,
            "design": "docs/plans/2026-08-01-soft-margin-anion-search.md",
            "sequential_adaptive": True,
            "selection_split": "discovery only",
            "calibration_role": "historical diagnostic; adaptively reused",
            "floor": args.floor,
            "baseline_width": 24,
            "robust_width": args.width,
            "max_rules": args.max_rules,
            "full_anion_margin": FULL_ANION_MARGIN,
            "anion_by_fold_margin": CELL_MARGIN,
            "ranking": "minimum perturbation-kind rejection, then pooled rejection",
            "missing_feature_offline_semantics": "pass/abstain; falsification is unknown-fails-closed",
            "lockbox_access": False,
        },
        "isolation_audit": isolation_audit,
        "counts": {
            "discovery_real": len(dr),
            "discovery_bad": len(db),
            "calibration_real": len(cr),
            "calibration_bad": len(cb),
            "robust_strata": len(strata),
            **candidate_counts,
        },
        "robust_strata": {
            name: {
                **strata_metadata[name],
                "soft_floor": float(floors[name]),
                "margin": float(
                    FULL_ANION_MARGIN
                    if strata_metadata[name].get("fold") == "all"
                    else CELL_MARGIN
                ),
                "candidate_satisfaction": float(soft_result.real_mask[mask].mean()),
                "candidate_minus_baseline": float(
                    soft_result.real_mask[mask].mean()
                    - float(strata_metadata[name]["baseline_satisfaction"])
                ),
                "candidate_minus_soft_floor": float(
                    soft_result.real_mask[mask].mean() - floors[name]
                ),
            }
            for name, mask in strata.items()
        },
        "frontiers": {
            "existing_loop": {
                str(args.floor): {
                    "rules": baseline_rules,
                    "discovery": baseline_discovery,
                    "calibration": baseline_calibration,
                }
            },
            "additive_corrected_soft_margin_loop": {
                str(args.floor): {
                    "rules": selected_rules,
                    "rule_sha256": selected_rule_sha256,
                    "discovery": soft_discovery,
                    "calibration": soft_calibration,
                }
            },
        },
        "discovery_fold_comparisons": fold_comparisons,
        "calibration_masks_summary": {
            "baseline_real_pass": int(baseline_cal_real.sum()),
            "baseline_bad_pass": int(baseline_cal_bad.sum()),
            "candidate_real_pass": int(soft_cal_real.sum()),
            "candidate_bad_pass": int(soft_cal_bad.sum()),
        },
        "comparison": comparison,
        "comparison_variant": "additive_corrected_soft_margin_loop",
        "provenance": {
            "runtime_seconds": time.time() - started,
            "input_sha256": {
                "real_descriptors": _sha256(args.real_descriptors),
                "bad_descriptors": _sha256(args.bad_descriptors),
                "real_sixfam": _sha256(args.real_sixfam),
                "bad_sixfam": _sha256(args.bad_sixfam),
                "real_corrected": _sha256(args.real_corrected),
                "bad_corrected": _sha256(args.bad_corrected),
                "real_guards": _sha256(args.real_guards),
                "bad_guards": _sha256(args.bad_guards),
            },
            "implementation_sha256": _sha256(Path(__file__)),
            "dependency_sha256": {
                "next4_law_search": _sha256(
                    Path(__file__).with_name("next4_law_search.py")
                ),
                "better_law_search": _sha256(
                    Path(__file__).with_name("better_law_search.py")
                ),
            },
            "design_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "docs/plans/2026-08-01-soft-margin-anion-search.md"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"selected {len(selected_rules)} rules; calibration sat delta "
        f"{comparison['calibration_satisfaction_delta']:+.4f}, rejection "
        f"{comparison['calibration_rejection_delta']:+.4f}, minimum-kind "
        f"{comparison['calibration_minimum_kind_delta']:+.4f}, anion "
        f"{comparison['calibration_worst_shared_anion_delta']:+.4f}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
