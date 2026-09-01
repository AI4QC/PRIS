#!/usr/bin/env python3
"""np-next-20260801 grouped nested sparse formula loop on isolated splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from better_formula_search import (  # noqa: E402
    _json_ready,
    _nested_search,
    outer_fold_direction_gate,
)
from next_law_search import is_frozen_next_search_feature  # noqa: E402

NEW_FAMILY_PREFIXES = ("bvloc", "p2vor_", "p3haw_", "p5hop_")
DROP = {"source_id", "rk", "e_hull", "split", "anion", "sid", "parent", "kind"}


def next_eligible_features(
    discovery: pd.DataFrame,
    *,
    include_new: bool,
    min_coverage: float,
) -> list[str]:
    """Feature eligibility mirroring the existing loop with the new families."""

    features = []
    for column in discovery:
        if column in DROP or discovery[column].dtype.kind != "f":
            continue
        is_new = column.startswith(NEW_FAMILY_PREFIXES)
        if is_new and not include_new:
            continue
        if is_new and not is_frozen_next_search_feature(column):
            continue
        values = discovery[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.mean() < min_coverage:
            continue
        if np.nanstd(values) <= 1e-12:
            continue
        features.append(column)
    return sorted(features)


def load_isolated_formula_frame(
    isolated_dir: Path,
    real_descriptors: Path,
) -> pd.DataFrame:
    frame = pd.read_parquet(isolated_dir / "formula_rank.parquet")
    extra = pd.read_parquet(real_descriptors).drop(columns=["split"], errors="ignore")
    if extra["source_id"].duplicated().any():
        raise ValueError("real descriptor source_id is not unique")
    before = len(frame)
    frame = frame.merge(extra, on="source_id", how="left", validate="one_to_one")
    if len(frame) != before:
        raise AssertionError("descriptor merge changed formula row count")
    if frame["split"].isna().any() or frame["split"].eq("lockbox").any():
        raise ValueError("formula frame contains lockbox or unknown rows")
    frame.attrs["source_access_audit"] = {
        "source_tables_materialized_all_splits": False,
        "lockbox_access": False,
        "lockbox_rows_in_fit_or_evaluation": False,
        "materialized_lockbox_real_rank_rows": 0,
        "reason": (
            "formula loop loads the physically isolated ranking table written "
            "by next_isolate.py; no monolithic split-bearing table is read "
            "downstream of that audited builder"
        ),
    }
    return frame


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--term-grid", type=int, nargs="+", default=[3, 5, 7])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if max(args.term_grid) > 7:
        raise SystemExit("formula complexity is frozen at at most seven terms")
    started = time.time()
    frame = load_isolated_formula_frame(args.isolated_dir, args.real_descriptors)
    source_access_audit = dict(frame.attrs["source_access_audit"])
    discovery = frame[frame["split"].eq("discovery")].reset_index(drop=True)
    calibration = frame[frame["split"].eq("calibration")].reset_index(drop=True)
    print(
        f"discovery {len(discovery):,}/{discovery.rk.nunique():,} groups; "
        f"calibration {len(calibration):,}/{calibration.rk.nunique():,} groups; "
        "physically isolated table: zero lockbox rows materialized",
        flush=True,
    )
    existing_features = next_eligible_features(
        discovery, include_new=False, min_coverage=args.min_coverage
    )
    additive_features = next_eligible_features(
        discovery, include_new=True, min_coverage=args.min_coverage
    )
    print(
        f"eligible features existing={len(existing_features)}, "
        f"additive={len(additive_features)}",
        flush=True,
    )
    variants = {}
    for name, features in (
        ("existing_formula_loop", existing_features),
        ("additive_p235_formula_loop", additive_features),
    ):
        print(f"nested search: {name}", flush=True)
        variants[name] = _nested_search(
            discovery,
            calibration,
            features,
            c_grid=args.c_grid,
            term_grid=args.term_grid,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
            min_gap=args.min_gap,
            min_feature_coverage=args.min_coverage,
        )
        print(
            f"  outer={variants[name]['outer_group_equal_mean']:.4f}; "
            f"calibration="
            f"{variants[name]['calibration_metrics']['group_equal_accuracy']:.4f}",
            flush=True,
        )

    existing = variants["existing_formula_loop"]
    additive = variants["additive_p235_formula_loop"]
    fold_deltas = [
        additive["outer_folds"][index]["metrics"]["group_equal_accuracy"]
        - existing["outer_folds"][index]["metrics"]["group_equal_accuracy"]
        for index in range(args.outer_folds)
    ]
    final_new_features = [
        term["feature"]
        for term in additive["final_formula"]["terms"]
        if term["feature"].startswith(NEW_FAMILY_PREFIXES)
    ]
    commitment_targets = ("1.00", "0.30", "0.10")
    outer_commitment = {}
    for target in commitment_targets:
        existing_scores = [
            float(row["outer_abstention"][target]["group_equal_accuracy"])
            for row in existing["outer_folds"]
        ]
        additive_scores = [
            float(row["outer_abstention"][target]["group_equal_accuracy"])
            for row in additive["outer_folds"]
        ]
        deltas = [
            new_value - old_value
            for old_value, new_value in zip(existing_scores, additive_scores)
        ]
        outer_commitment[target] = {
            "existing_group_equal_mean": float(np.mean(existing_scores)),
            "additive_group_equal_mean": float(np.mean(additive_scores)),
            "group_equal_accuracy_delta": float(np.mean(deltas)),
            "fold_deltas": deltas,
            "fold_direction_gate": outer_fold_direction_gate(deltas),
            "existing_realized_coverage_mean": float(
                np.mean(
                    [
                        row["outer_abstention"][target]["coverage"]
                        for row in existing["outer_folds"]
                    ]
                )
            ),
            "additive_realized_coverage_mean": float(
                np.mean(
                    [
                        row["outer_abstention"][target]["coverage"]
                        for row in additive["outer_folds"]
                    ]
                )
            ),
        }
    qualifying_targets = [
        target
        for target, metrics in outer_commitment.items()
        if metrics["group_equal_accuracy_delta"] >= 0.02
        and metrics["fold_direction_gate"]
    ]
    comparison = {
        "outer_group_equal_accuracy_delta": (
            additive["outer_group_equal_mean"] - existing["outer_group_equal_mean"]
        ),
        "calibration_group_equal_accuracy_delta": (
            additive["calibration_metrics"]["group_equal_accuracy"]
            - existing["calibration_metrics"]["group_equal_accuracy"]
        ),
        "outer_fold_deltas": fold_deltas,
        "outer_positive_folds": int(sum(delta > 0 for delta in fold_deltas)),
        "new_features_in_final_formula": final_new_features,
        "outer_fixed_commitment": outer_commitment,
        "qualifying_commitment_targets": qualifying_targets,
    }
    comparison["metric_gate_without_source_access_condition"] = bool(
        final_new_features and qualifying_targets
    )
    comparison["preliminary_gate"] = bool(
        comparison["metric_gate_without_source_access_condition"]
        and not (
            source_access_audit["source_tables_materialized_all_splits"]
            and source_access_audit["materialized_lockbox_real_rank_rows"] > 0
        )
    )
    comparison["status"] = (
        "pending temporal/source holdout; historical calibration is diagnostic"
    )
    report = {
        "protocol": {
            "experiment": "np-next-20260801",
            "selection_data": "discovery only",
            "outer_cv": "deterministic reduced-formula grouped",
            "inner_cv": "deterministic reduced-formula grouped",
            "pair_weighting": "each reduced-formula group sums to one",
            "antisymmetric_double_write": True,
            "max_terms": 7,
            "p1_vocabulary": "frozen",
            "new_vocabulary": "frozen P1 (fallback parameters) + P2 + P3 + P5",
            "abstention_threshold_source": (
                "numeric thresholds fit on grouped inner out-of-fold pair scores"
            ),
            "calibration_role": "historical diagnostic; previously adaptively reused",
            **source_access_audit,
            "min_gap_eV_per_atom": args.min_gap,
        },
        "counts": {
            "discovery_structures": len(discovery),
            "discovery_groups": int(discovery.rk.nunique()),
            "calibration_structures": len(calibration),
            "calibration_groups": int(calibration.rk.nunique()),
            "existing_features": len(existing_features),
            "additive_features": len(additive_features),
            "new_eligible_features": len(
                set(additive_features).difference(existing_features)
            ),
            "min_coverage": args.min_coverage,
        },
        "variants": variants,
        "comparison": comparison,
        "provenance": {
            "runtime_seconds": time.time() - started,
            "descriptor_sha256": _hash_file(args.real_descriptors),
            "descriptor_metadata_sha256": _hash_file(
                args.real_descriptors.with_suffix(
                    args.real_descriptors.suffix + ".meta.json"
                )
            ),
            "implementation_sha256": _hash_file(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
