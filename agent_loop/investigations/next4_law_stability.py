#!/usr/bin/env python3
"""True leave-one-perturbation-kind-out refits for np-next-20260801d."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from better_law_search import (
    _apply_result,
    _candidate_record,
    evaluate_masks,
    pareto_beam,
)
from next4_law_search import (
    build_next4_candidate_sets,
    load_next4_search_frames,
    paired_robust_strata,
    robust_pareto_beam,
    verify_isolated_manifest,
)


def iter_true_loko(
    discovery_bad: pd.DataFrame,
    calibration_bad: pd.DataFrame,
) -> Iterable[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield training and held frames without exposing the held kind to fitting."""

    for held in sorted(str(value) for value in discovery_bad["kind"].dropna().unique()):
        training = discovery_bad.loc[~discovery_bad["kind"].eq(held)].reset_index(
            drop=True
        )
        held_discovery = discovery_bad.loc[
            discovery_bad["kind"].eq(held)
        ].reset_index(drop=True)
        held_calibration = calibration_bad.loc[
            calibration_bad["kind"].eq(held)
        ].reset_index(drop=True)
        yield held, training, held_discovery, held_calibration


def signed_delta_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Report signed and absolute behavior without cancellation."""

    array = np.asarray(values, dtype=float)
    if not len(array):
        return {
            "signed_mean": 0.0,
            "mean_absolute": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "positive": 0,
            "negative": 0,
            "zero": 0,
        }
    return {
        "signed_mean": float(np.mean(array)),
        "mean_absolute": float(np.mean(np.abs(array))),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive": int((array > 0).sum()),
        "negative": int((array < 0).sum()),
        "zero": int((array == 0).sum()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _held_metrics(real, bad, candidates, result) -> dict[str, object] | None:
    if bad.empty:
        return None
    real_mask = _apply_result(real, candidates, result)
    bad_mask = _apply_result(bad, candidates, result)
    return evaluate_masks(
        real_mask=real_mask,
        bad_mask=bad_mask,
        bad_groups=bad["parent"].to_numpy(),
        bad_kinds=bad["kind"].to_numpy(),
    )


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
    parser.add_argument("--reference-report", type=Path, required=True)
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
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    floor_key = str(args.floor)
    full_existing = reference["frontiers"]["existing_loop"][floor_key][
        "discovery"
    ]["by_kind"]
    full_candidate = reference["frontiers"][
        reference["comparison_variant"]
    ][floor_key]["discovery"]["by_kind"]

    folds: dict[str, object] = {}
    discovery_deltas: list[float] = []
    calibration_deltas: list[float] = []
    fullfit_changes: list[float] = []
    for held, training_bad, held_discovery, held_calibration in iter_true_loko(db, cb):
        candidate_sets, counts = build_next4_candidate_sets(
            dr,
            training_bad,
            min_coverage=args.min_coverage,
            max_guard_targets=args.max_guard_targets,
            guard_min_real_satisfaction=args.floor,
        )
        existing_candidates = candidate_sets["existing_loop"]
        additive_candidates = candidate_sets["additive_corrected_loop"]
        baseline = pareto_beam(
            existing_candidates,
            real_size=len(dr),
            bad_size=len(training_bad),
            bad_kinds=training_bad["kind"].to_numpy(),
            satisfaction_floor=args.floor,
            max_rules=args.max_rules,
            width=24,
        )
        strata, floors, _metadata = paired_robust_strata(
            dr,
            baseline.real_mask,
            n_folds=args.n_folds,
            min_anion_rows=args.min_anion_rows,
            min_cell_rows=args.min_cell_rows,
        )
        candidate = robust_pareto_beam(
            additive_candidates,
            real_size=len(dr),
            bad_size=len(training_bad),
            bad_kinds=training_bad["kind"].to_numpy(),
            satisfaction_floor=args.floor,
            max_rules=args.max_rules,
            width=args.width,
            real_strata=strata,
            stratum_floors=floors,
        )
        base_discovery = _held_metrics(
            dr, held_discovery, existing_candidates, baseline
        )
        candidate_discovery = _held_metrics(
            dr, held_discovery, additive_candidates, candidate
        )
        base_calibration = _held_metrics(
            cr, held_calibration, existing_candidates, baseline
        )
        candidate_calibration = _held_metrics(
            cr, held_calibration, additive_candidates, candidate
        )
        if base_discovery is None or candidate_discovery is None:
            raise RuntimeError(f"held discovery kind unexpectedly empty: {held}")
        discovery_delta = float(
            candidate_discovery["rejection"] - base_discovery["rejection"]
        )
        discovery_deltas.append(discovery_delta)
        fullfit_change = float(
            candidate_discovery["rejection"] - float(full_candidate[held])
        )
        fullfit_changes.append(fullfit_change)
        calibration_delta = None
        if base_calibration is not None and candidate_calibration is not None:
            calibration_delta = float(
                candidate_calibration["rejection"] - base_calibration["rejection"]
            )
            calibration_deltas.append(calibration_delta)
        rules = [
            _candidate_record(additive_candidates[index])
            for index in candidate.indices
        ]
        rule_sha = hashlib.sha256(
            json.dumps(
                rules,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        folds[held] = {
            "training_kinds": sorted(set(training_bad["kind"]).difference({held})),
            "training_bad_count": len(training_bad),
            "held_discovery_count": len(held_discovery),
            "held_calibration_count": len(held_calibration),
            "robust_strata": len(strata),
            "candidate_counts": counts,
            "rules": rules,
            "rule_sha256": rule_sha,
            "existing_held_discovery": base_discovery,
            "candidate_held_discovery": candidate_discovery,
            "candidate_minus_existing_held_discovery_rejection": discovery_delta,
            "candidate_held_minus_fullfit_held_discovery_rejection": fullfit_change,
            "existing_fullfit_held_discovery_rejection": float(full_existing[held]),
            "candidate_fullfit_held_discovery_rejection": float(full_candidate[held]),
            "existing_held_calibration": base_calibration,
            "candidate_held_calibration": candidate_calibration,
            "candidate_minus_existing_held_calibration_rejection": calibration_delta,
        }
        print(
            f"held {held}: discovery delta {discovery_delta:+.4f}; "
            f"calibration delta "
            f"{calibration_delta if calibration_delta is not None else float('nan'):+.4f}; "
            f"full-fit change {fullfit_change:+.4f}",
            flush=True,
        )

    report = {
        "protocol": {
            "experiment": "np-next-20260801d",
            "analysis": "true leave-one-perturbation-kind-out refit",
            "held_kind_exposed_to_search": False,
            "floor": args.floor,
            "robust_width": args.width,
            "lockbox_access": False,
        },
        "isolation_audit": isolation_audit,
        "folds": folds,
        "summary": {
            "candidate_minus_existing_held_discovery_rejection": signed_delta_summary(
                discovery_deltas
            ),
            "candidate_minus_existing_held_calibration_rejection": signed_delta_summary(
                calibration_deltas
            ),
            "candidate_held_minus_fullfit_held_discovery_rejection": signed_delta_summary(
                fullfit_changes
            ),
        },
        "provenance": {
            "runtime_seconds": time.time() - started,
            "reference_report_sha256": _sha256(args.reference_report),
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
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
