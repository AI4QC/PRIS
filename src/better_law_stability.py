#!/usr/bin/env python3
"""Leave-one-perturbation-kind-out stability diagnostics for new law searches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from better_law_search import (
    BeamResult,
    LawCandidate,
    _candidate_record,
    _evaluate_result,
    _hash_file,
    _load_search_frames,
    build_candidate_sets,
    leave_one_kind_out_frames,
    pareto_beam,
)


CandidateBuilder = Callable[
    ...,
    tuple[dict[str, list[LawCandidate]], dict[str, int]],
]


def _check_no_lockbox(frames: Sequence[tuple[str, pd.DataFrame]]) -> None:
    for name, frame in frames:
        for split_column in ("split", "psplit"):
            if split_column not in frame:
                continue
            split = frame[split_column]
            if split.isna().any() or split.eq("lockbox").any():
                raise ValueError(f"{name} contains lockbox or unknown rows")


def _held_metrics(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    candidates: Sequence[LawCandidate],
    result: BeamResult,
) -> dict[str, object] | None:
    if bad.empty:
        return None
    return _evaluate_result(
        real,
        bad,
        candidates,
        result,
        use_stored_masks=False,
    )


def _macro(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run_leave_one_kind_out(
    discovery_real: pd.DataFrame,
    calibration_real: pd.DataFrame,
    discovery_bad: pd.DataFrame,
    calibration_bad: pd.DataFrame,
    *,
    floor: float,
    min_coverage: float,
    width: int,
    max_rules: int,
    max_guard_targets: int,
    p1_vocabulary: str = "frozen",
    paired_anion_guard: bool = True,
    reference_by_kind: Mapping[str, Mapping[str, float]] | None = None,
    candidate_builder: CandidateBuilder = build_candidate_sets,
) -> dict[str, object]:
    """Refit while withholding each discovery perturbation kind in turn."""

    if not 0 < floor <= 1:
        raise ValueError("floor must lie in (0, 1]")
    _check_no_lockbox(
        (
            ("discovery_real", discovery_real),
            ("calibration_real", calibration_real),
            ("discovery_bad", discovery_bad),
            ("calibration_bad", calibration_bad),
        )
    )
    folds = leave_one_kind_out_frames(discovery_bad, calibration_bad)
    additive_name = (
        "additive_bvloc_anion_guarded_loop"
        if paired_anion_guard
        else "additive_bvloc_loop"
    )
    source_access_audit = dict(
        discovery_real.attrs.get(
            "source_access_audit",
            {
                "source_tables_materialized_all_splits": False,
                "lockbox_access": False,
                "lockbox_rows_in_fit_or_evaluation": False,
                "materialized_lockbox_real_rows": 0,
                "materialized_lockbox_bad_rows": 0,
            },
        )
    )
    report: dict[str, object] = {
        "protocol": {
            "diagnostic": "leave-one-perturbation-kind-out refit",
            "selection_split": "discovery only",
            "calibration_role": (
                "historical diagnostic; never used for candidate or threshold selection"
            ),
            **source_access_audit,
            "floor": floor,
            "min_coverage": min_coverage,
            "paired_anion_guard": paired_anion_guard,
            "p1_vocabulary": p1_vocabulary,
            "held_kind_role": "evaluation only",
        },
        "comparison_variant": additive_name,
        "folds": {},
    }

    for held_kind, training_bad, held_discovery, held_calibration in folds:
        pools, candidate_counts = candidate_builder(
            discovery_real,
            training_bad,
            min_coverage=min_coverage,
            max_guard_targets=max_guard_targets,
            guard_min_real_satisfaction=floor,
            p1_vocabulary=p1_vocabulary,
        )
        existing_candidates = pools["existing_loop"]
        additive_candidates = pools["additive_bvloc_loop"]
        existing_result = pareto_beam(
            existing_candidates,
            real_size=len(discovery_real),
            bad_size=len(training_bad),
            bad_kinds=training_bad["kind"].to_numpy(),
            satisfaction_floor=floor,
            max_rules=max_rules,
            width=width,
        )

        real_strata: dict[str, np.ndarray] = {}
        stratum_floors: dict[str, float] = {}
        if paired_anion_guard and "anion" in discovery_real:
            counts = discovery_real["anion"].value_counts()
            real_strata = {
                str(anion): discovery_real["anion"].eq(anion).to_numpy()
                for anion in counts[counts >= 200].index
            }
            stratum_floors = {
                anion: max(
                    float(existing_result.real_mask[stratum].mean()) - 0.01,
                    0.0,
                )
                for anion, stratum in real_strata.items()
            }
        additive_result = pareto_beam(
            additive_candidates,
            real_size=len(discovery_real),
            bad_size=len(training_bad),
            bad_kinds=training_bad["kind"].to_numpy(),
            satisfaction_floor=floor,
            max_rules=max_rules,
            width=width,
            real_strata=real_strata,
            stratum_floors=stratum_floors,
        )

        variants = {}
        for variant, candidates, result in (
            ("existing_loop", existing_candidates, existing_result),
            (additive_name, additive_candidates, additive_result),
        ):
            variants[variant] = {
                "rules": [
                    _candidate_record(candidates[index])
                    for index in result.indices
                ],
                "training": _evaluate_result(
                    discovery_real,
                    training_bad,
                    candidates,
                    result,
                    use_stored_masks=True,
                ),
                "held_discovery": _held_metrics(
                    discovery_real,
                    held_discovery,
                    candidates,
                    result,
                ),
                "held_calibration": _held_metrics(
                    calibration_real,
                    held_calibration,
                    candidates,
                    result,
                ),
            }
        report["folds"][held_kind] = {
            "training_kinds": sorted(
                str(kind) for kind in training_bad["kind"].unique()
            ),
            "training_bad_count": int(len(training_bad)),
            "held_discovery_count": int(len(held_discovery)),
            "held_calibration_count": int(len(held_calibration)),
            "candidate_counts": candidate_counts,
            "paired_discovery_anion_floors": stratum_floors,
            "variants": variants,
        }

    existing_held = []
    additive_held = []
    existing_calibration = []
    additive_calibration = []
    additive_minus_existing = {}
    stability_shifts: dict[str, dict[str, float]] = {
        "existing_loop": {},
        additive_name: {},
    }
    references = dict(reference_by_kind or {})
    for held_kind, fold in report["folds"].items():
        variants = fold["variants"]
        existing_value = float(
            variants["existing_loop"]["held_discovery"]["rejection"]
        )
        additive_value = float(
            variants[additive_name]["held_discovery"]["rejection"]
        )
        existing_held.append(existing_value)
        additive_held.append(additive_value)
        additive_minus_existing[held_kind] = additive_value - existing_value
        for variant, value in (
            ("existing_loop", existing_value),
            (additive_name, additive_value),
        ):
            if variant in references and held_kind in references[variant]:
                stability_shifts[variant][held_kind] = (
                    value - float(references[variant][held_kind])
                )
        old_cal = variants["existing_loop"]["held_calibration"]
        new_cal = variants[additive_name]["held_calibration"]
        if old_cal is not None:
            existing_calibration.append(float(old_cal["rejection"]))
        if new_cal is not None:
            additive_calibration.append(float(new_cal["rejection"]))

    stability_summary = {}
    for variant, shifts in stability_shifts.items():
        if shifts:
            stability_summary[variant] = {
                "signed_change_by_held_kind": shifts,
                "signed_mean": float(np.mean(list(shifts.values()))),
                "mean_absolute_change": float(
                    np.mean(np.abs(list(shifts.values())))
                ),
                "reference": "full-fit discovery by-kind rejection",
            }
    report["summary"] = {
        "macro_held_discovery_rejection": {
            "existing_loop": _macro(existing_held),
            additive_name: _macro(additive_held),
        },
        "macro_held_calibration_rejection": {
            "existing_loop": _macro(existing_calibration),
            additive_name: _macro(additive_calibration),
        },
        "additive_minus_existing_held_discovery_by_kind": (
            additive_minus_existing
        ),
        "additive_minus_existing_held_discovery_macro": float(
            np.mean(list(additive_minus_existing.values()))
        ),
        "stability_against_full_fit": stability_summary,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refit the additive PRIS law loop while withholding each "
            "S1-S5 perturbation kind."
        )
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.98)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--max-guard-targets", type=int, default=100)
    parser.add_argument(
        "--p1-vocabulary",
        choices=("frozen", "expanded"),
        default="frozen",
    )
    parser.add_argument("--no-paired-anion-guard", action="store_true")
    return parser


def _reference_by_kind(
    path: Path | None,
    *,
    floor: float,
    additive_name: str,
) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    source = json.loads(path.read_text(encoding="utf-8"))
    key = str(floor)
    return {
        "existing_loop": source["frontiers"]["existing_loop"][key]["discovery"][
            "by_kind"
        ],
        additive_name: source["frontiers"][additive_name][key]["discovery"][
            "by_kind"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    started = time.time()
    discovery_real, calibration_real, discovery_bad, calibration_bad = (
        _load_search_frames(
            args.features_dir,
            args.real_descriptors,
            args.bad_descriptors,
        )
    )
    paired = not args.no_paired_anion_guard
    additive_name = (
        "additive_bvloc_anion_guarded_loop"
        if paired
        else "additive_bvloc_loop"
    )
    report = run_leave_one_kind_out(
        discovery_real,
        calibration_real,
        discovery_bad,
        calibration_bad,
        floor=args.floor,
        min_coverage=args.min_coverage,
        width=args.width,
        max_rules=args.max_rules,
        max_guard_targets=args.max_guard_targets,
        p1_vocabulary=args.p1_vocabulary,
        paired_anion_guard=paired,
        reference_by_kind=_reference_by_kind(
            args.reference_report,
            floor=args.floor,
            additive_name=additive_name,
        ),
    )
    report["provenance"] = {
        "runtime_seconds": time.time() - started,
        "input_sha256": {
            "real_descriptors": _hash_file(args.real_descriptors),
            "bad_descriptors": _hash_file(args.bad_descriptors),
            "reference_report": (
                None
                if args.reference_report is None
                else _hash_file(args.reference_report)
            ),
        },
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
