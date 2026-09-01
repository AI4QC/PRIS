#!/usr/bin/env python3
"""np-next-20260801 law loop on physically isolated splits.

Reuses the existing loop machinery from ``better_law_search`` unchanged;
only the data loading (isolated tables) and the frozen P2/P3/P5 vocabulary
are new.  Emits the same report schema so ``law_falsification.py`` can run
on the output unmodified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from better_law_search import (  # noqa: E402
    LawCandidate,
    _apply_result,
    _candidate_record,
    _deduplicate,
    _evaluate_result,
    _fold_diagnostics,
    _historical_metrics,
    build_band_candidates,
    build_guarded_candidates,
    build_one_sided_candidates,
    law_preliminary_gate,
    pareto_beam,
    selected_rule_coverage,
)
from better_search import is_frozen_p1_search_feature  # noqa: E402

P2_METRICS = ("sa_effective_cn", "sa_like_fraction", "sa_max_fraction")
P2_AGGREGATES = ("mean", "q95", "max")
P3_METRICS = (
    "nnls_relres",
    "minnorm_relres",
    "pauling_gap",
    "rank_deficiency",
    "unbonded_charged_fraction",
    "site_relres_q95",
    "site_relres_max",
)

FROZEN_P235_SEARCH_FEATURES = frozenset(
    [
        f"p2vor_{charge}_{metric}_{aggregate}"
        for charge in ("cat", "an")
        for metric in P2_METRICS
        for aggregate in P2_AGGREGATES
    ]
    + [f"p3haw_{metric}" for metric in P3_METRICS]
    + [
        f"p5hop_{charge}_econ_strict_{aggregate}"
        for charge in ("cat", "an")
        for aggregate in ("mean", "max")
    ]
    + [
        f"p5hop_{charge}_econ_delta_{aggregate}"
        for charge in ("cat", "an")
        for aggregate in ("mean", "max")
    ]
    + [
        f"p5hop_{charge}_mefir_rel_{aggregate}"
        for charge in ("cat", "an")
        for aggregate in ("mean", "min", "max")
    ]
    + [
        f"p5hop_{charge}_mefir_delta_{aggregate}"
        for charge in ("cat", "an")
        for aggregate in ("mean", "max")
    ]
)

NEW_FAMILY_PREFIXES = ("bvloc", "p2vor_", "p3haw_", "p5hop_")

ADDITIVE_VARIANT = "additive_p235_loop"
ADDITIVE_GUARDED_VARIANT = "additive_p235_anion_guarded_loop"


def is_frozen_next_search_feature(column: str) -> bool:
    """Membership in the frozen np-next-20260801 searchable vocabulary."""

    return is_frozen_p1_search_feature(column) or (
        column in FROZEN_P235_SEARCH_FEATURES
    )


def build_next_candidate_sets(
    real: pd.DataFrame,
    bad: pd.DataFrame,
    *,
    min_coverage: float,
    max_guard_targets: int,
    guard_min_real_satisfaction: float = 0.95,
    p1_vocabulary: str = "frozen",
) -> tuple[dict[str, list[LawCandidate]], dict[str, int]]:
    """Existing pool plus the frozen P1/P2/P3/P5 additive pool."""

    if p1_vocabulary != "frozen":
        raise ValueError(
            "np-next-20260801 froze exactly one vocabulary; no expanded branch"
        )
    alphas = (
        0.0003,
        0.0007,
        0.0015,
        0.003,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.12,
    )
    old_features = [
        column
        for column in real
        if column in bad
        and real[column].dtype.kind == "f"
        and not column.startswith(NEW_FAMILY_PREFIXES)
    ]
    new_features = [
        column
        for column in real
        if column in bad and is_frozen_next_search_feature(column)
    ]
    old_candidates = build_one_sided_candidates(
        real,
        bad,
        old_features,
        alphas=alphas,
        min_coverage=min_coverage,
        min_rejection=0.01,
        min_real_satisfaction=0.88,
        origin="existing",
    )
    new_one_sided = build_one_sided_candidates(
        real,
        bad,
        new_features,
        alphas=alphas,
        min_coverage=min_coverage,
        min_rejection=0.01,
        min_real_satisfaction=0.88,
        origin="next-p235",
    )
    new_bands = build_band_candidates(
        real,
        bad,
        new_features,
        central_coverages=(0.99, 0.98, 0.95, 0.90),
        min_coverage=min_coverage,
        min_rejection=0.01,
        min_real_satisfaction=0.88,
        origin="next-p235",
    )
    loose_targets = build_one_sided_candidates(
        real,
        bad,
        new_features,
        alphas=(0.15, 0.20, 0.25, 0.30),
        min_coverage=min_coverage,
        min_rejection=0.02,
        min_real_satisfaction=0.50,
        origin="next-p235",
    )
    guarded = build_guarded_candidates(
        real,
        bad,
        [*new_one_sided, *new_bands, *loose_targets],
        guard_columns=(
            "mean_cn_cat",
            "z_cat_max",
            "cn_an_mean",
            "n_el",
            "cat_an_ratio",
            "fi",
            "dchi",
        ),
        guard_quantiles=(0.25, 0.5, 0.75),
        min_real_satisfaction=guard_min_real_satisfaction,
        min_rejection=0.02,
        max_targets=max_guard_targets,
    )
    existing = _deduplicate(old_candidates)
    additive = _deduplicate([*existing, *new_one_sided, *new_bands, *guarded])
    counts = {
        "old_features": len(old_features),
        "new_features_eligible": len(new_features),
        "old_candidates": len(existing),
        "new_one_sided_candidates": len(new_one_sided),
        "new_band_candidates": len(new_bands),
        "new_guarded_candidates": len(guarded),
        "combined_candidates": len(additive),
    }
    return {
        "existing_loop": existing,
        "additive_bvloc_loop": additive,  # key name required by the LOKO driver
    }, counts


def load_isolated_search_frames(
    isolated_dir: Path,
    real_descriptors: Path,
    bad_descriptors: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load isolated law tables and add descriptors without changing rows."""

    real = pd.read_parquet(isolated_dir / "law_real.parquet")
    bad = pd.read_parquet(isolated_dir / "law_bad.parquet")
    # Fail closed: the isolated tables must be physically free of lockbox or
    # unknown-split rows before any filtering happens.
    for name, frame, column in (
        ("law_real", real, "split"),
        ("law_bad", bad, "psplit"),
    ):
        if frame[column].isna().any() or frame[column].eq("lockbox").any():
            raise ValueError(
                f"{name} contains lockbox or unknown rows; the isolated "
                "tables are corrupted or were not built by next_isolate.py"
            )
    real_extra = pd.read_parquet(real_descriptors).drop(
        columns=["split"], errors="ignore"
    )
    bad_extra = pd.read_parquet(bad_descriptors).drop(
        columns=["kind", "parent", "split"], errors="ignore"
    )
    if real_extra["source_id"].duplicated().any():
        raise ValueError("real descriptor source_id is not unique")
    if bad_extra["sid"].duplicated().any():
        raise ValueError("bad descriptor sid is not unique")
    real_count, bad_count = len(real), len(bad)
    real = real.merge(real_extra, on="source_id", how="left", validate="one_to_one")
    bad = bad.merge(bad_extra, on="sid", how="left", validate="one_to_one")
    if len(real) != real_count or len(bad) != bad_count:
        raise AssertionError("descriptor merge changed the isolated row set")

    discovery_real = real[real["split"].eq("discovery")].reset_index(drop=True)
    calibration_real = real[real["split"].eq("calibration")].reset_index(drop=True)
    discovery_bad = bad[bad["psplit"].eq("discovery")].reset_index(drop=True)
    calibration_bad = bad[bad["psplit"].eq("calibration")].reset_index(drop=True)
    for name, frame, column in (
        ("discovery real", discovery_real, "split"),
        ("calibration real", calibration_real, "split"),
        ("discovery bad", discovery_bad, "psplit"),
        ("calibration bad", calibration_bad, "psplit"),
    ):
        if frame.empty:
            raise ValueError(f"{name} is empty")
        if frame[column].isna().any() or frame[column].eq("lockbox").any():
            raise ValueError(f"{name} contains lockbox or unknown rows")
    audit = {
        "source_tables_materialized_all_splits": False,
        "lockbox_access": False,
        "lockbox_rows_in_fit_or_evaluation": False,
        "materialized_lockbox_real_rows": 0,
        "materialized_lockbox_bad_rows": 0,
        "reason": (
            "search loads physically isolated discovery/calibration tables "
            "written by next_isolate.py; no monolithic split-bearing table "
            "is read downstream of that audited builder"
        ),
    }
    for frame in (discovery_real, calibration_real, discovery_bad, calibration_bad):
        frame.attrs["source_access_audit"] = audit
    return discovery_real, calibration_real, discovery_bad, calibration_bad


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
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.99, 0.98, 0.95])
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--max-guard-targets", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if any(not 0 < floor < 1 for floor in args.floors):
        raise SystemExit("all --floors must lie strictly between 0 and 1")

    started = time.time()
    dr, cr, db, cb = load_isolated_search_frames(
        args.isolated_dir,
        args.real_descriptors,
        args.bad_descriptors,
    )
    source_access_audit = dict(dr.attrs["source_access_audit"])
    print(
        f"discovery real/bad {len(dr):,}/{len(db):,}; "
        f"calibration real/bad {len(cr):,}/{len(cb):,}; "
        "physically isolated tables: zero lockbox rows materialized",
        flush=True,
    )
    candidate_sets, candidate_counts = build_next_candidate_sets(
        dr,
        db,
        min_coverage=args.min_coverage,
        max_guard_targets=args.max_guard_targets,
        guard_min_real_satisfaction=min(args.floors),
    )
    old_candidates = candidate_sets["existing_loop"]
    additive_candidates = candidate_sets["additive_bvloc_loop"]
    print(
        f"candidates existing={candidate_counts['old_candidates']:,}; "
        f"new one-sided={candidate_counts['new_one_sided_candidates']:,}, "
        f"bands={candidate_counts['new_band_candidates']:,}, "
        f"guarded={candidate_counts['new_guarded_candidates']:,}; "
        f"combined={candidate_counts['combined_candidates']:,}",
        flush=True,
    )

    report: dict[str, object] = {
        "protocol": {
            "experiment": "np-next-20260801",
            "selection_split": "discovery only",
            "calibration_role": "historical diagnostic; previously adaptively reused",
            **source_access_audit,
            "missing_feature_offline_semantics": "pass/abstain",
            "bad_weighting": ["pooled", "parent-group-equal", "per-kind"],
            "candidate_threshold_source": "real discovery quantiles only",
            "floors": args.floors,
            "min_coverage": args.min_coverage,
            "p1_vocabulary": "frozen",
            "new_vocabulary": "frozen P1 (fallback parameters) + P2 + P3 + P5 (61 columns)",
        },
        "counts": {
            "discovery_real": len(dr),
            "discovery_bad": len(db),
            "calibration_real": len(cr),
            "calibration_bad": len(cb),
            **candidate_counts,
        },
        "historical": {
            "discovery": _historical_metrics(dr, db),
            "calibration": _historical_metrics(cr, cb),
        },
        "frontiers": {},
    }
    variant_results: dict[str, dict[str, object]] = {}
    for variant, candidates in (
        ("existing_loop", old_candidates),
        (ADDITIVE_VARIANT, additive_candidates),
    ):
        variant_results[variant] = {}
        report["frontiers"][variant] = {}
        for floor in args.floors:
            print(f"search {variant} floor={floor:.3f}", flush=True)
            result = pareto_beam(
                candidates,
                real_size=len(dr),
                bad_size=len(db),
                bad_kinds=db["kind"].to_numpy(),
                satisfaction_floor=floor,
                max_rules=args.max_rules,
                width=args.width,
            )
            discovery = _evaluate_result(dr, db, candidates, result, use_stored_masks=True)
            calibration = _evaluate_result(
                cr, cb, candidates, result, use_stored_masks=False
            )
            calibration_real_mask = _apply_result(cr, candidates, result)
            calibration_bad_mask = _apply_result(cb, candidates, result)
            report["frontiers"][variant][str(floor)] = {
                "rules": [
                    _candidate_record(candidates[index]) for index in result.indices
                ],
                "discovery": discovery,
                "calibration": calibration,
                "discovery_fold_diagnostic": _fold_diagnostics(
                    dr, db, result.real_mask, result.bad_mask
                ),
                "calibration_fold_diagnostic": _fold_diagnostics(
                    cr, cb, calibration_real_mask, calibration_bad_mask
                ),
            }
            variant_results[variant][str(floor)] = (candidates, result)
            print(
                f"  discovery sat/rej={discovery['satisfaction']:.4f}/"
                f"{discovery['rejection']:.4f}; calibration="
                f"{calibration['satisfaction']:.4f}/{calibration['rejection']:.4f}; "
                f"N={len(result.indices)}",
                flush=True,
            )

    # Paired-anion-guard variant on the additive pool.
    variant = ADDITIVE_GUARDED_VARIANT
    candidates = additive_candidates
    variant_results[variant] = {}
    report["frontiers"][variant] = {}
    anion_counts = dr["anion"].value_counts()
    strata = {
        str(anion): dr["anion"].eq(anion).to_numpy()
        for anion in anion_counts[anion_counts >= 200].index
    }
    for floor in args.floors:
        key = str(floor)
        base_result = variant_results["existing_loop"][key][1]
        paired_floors = {
            anion: max(float(base_result.real_mask[mask].mean()) - 0.01, 0.0)
            for anion, mask in strata.items()
        }
        print(
            f"search {variant} floor={floor:.3f} "
            f"with {len(strata)} paired anion constraints",
            flush=True,
        )
        result = pareto_beam(
            candidates,
            real_size=len(dr),
            bad_size=len(db),
            bad_kinds=db["kind"].to_numpy(),
            satisfaction_floor=floor,
            max_rules=args.max_rules,
            width=args.width,
            real_strata=strata,
            stratum_floors=paired_floors,
        )
        discovery = _evaluate_result(dr, db, candidates, result, use_stored_masks=True)
        calibration = _evaluate_result(
            cr, cb, candidates, result, use_stored_masks=False
        )
        calibration_real_mask = _apply_result(cr, candidates, result)
        calibration_bad_mask = _apply_result(cb, candidates, result)
        report["frontiers"][variant][key] = {
            "rules": [
                _candidate_record(candidates[index]) for index in result.indices
            ],
            "discovery": discovery,
            "calibration": calibration,
            "paired_discovery_anion_floors": paired_floors,
            "discovery_fold_diagnostic": _fold_diagnostics(
                dr, db, result.real_mask, result.bad_mask
            ),
            "calibration_fold_diagnostic": _fold_diagnostics(
                cr, cb, calibration_real_mask, calibration_bad_mask
            ),
        }
        variant_results[variant][key] = (candidates, result)
        print(
            f"  discovery sat/rej={discovery['satisfaction']:.4f}/"
            f"{discovery['rejection']:.4f}; calibration="
            f"{calibration['satisfaction']:.4f}/{calibration['rejection']:.4f}; "
            f"N={len(result.indices)}",
            flush=True,
        )

    comparison_variant = ADDITIVE_GUARDED_VARIANT
    comparisons = {}
    for floor in args.floors:
        key = str(floor)
        base = report["frontiers"]["existing_loop"][key]["calibration"]
        additive = report["frontiers"][comparison_variant][key]["calibration"]
        base_min = min(base["by_kind"].values())
        additive_min = min(additive["by_kind"].values())
        shared_anions = set(base["by_anion"]).intersection(additive["by_anion"])
        worst_anion_delta = min(
            (
                additive["by_anion"][anion]["satisfaction"]
                - base["by_anion"][anion]["satisfaction"]
                for anion in shared_anions
            ),
            default=0.0,
        )
        selected_rules = report["frontiers"][comparison_variant][key]["rules"]
        new_selected = any(rule["origin"] == "next-p235" for rule in selected_rules)
        coverage = selected_rule_coverage(
            pd.concat([dr, cr], ignore_index=True),
            pd.concat([db, cb], ignore_index=True),
            selected_rules,
        )
        metric_gate = law_preliminary_gate(
            new_descriptor_selected=new_selected,
            additive_satisfaction=additive["satisfaction"],
            base_satisfaction=base["satisfaction"],
            rejection_delta=additive["rejection"] - base["rejection"],
            min_kind_rejection_delta=additive_min - base_min,
            worst_anion_delta=worst_anion_delta,
            real_coverage=float(coverage["real_min"]),
            bad_coverage=float(coverage["bad_min"]),
            source_tables_materialized_lockbox_rows=False,
        )
        comparisons[key] = {
            "calibration_satisfaction_delta": (
                additive["satisfaction"] - base["satisfaction"]
            ),
            "calibration_rejection_delta": additive["rejection"] - base["rejection"],
            "calibration_group_equal_rejection_delta": (
                additive["group_equal_rejection"] - base["group_equal_rejection"]
            ),
            "calibration_min_kind_rejection_delta": additive_min - base_min,
            "worst_shared_anion_satisfaction_delta": worst_anion_delta,
            "new_descriptor_selected": new_selected,
            "selected_rule_coverage": coverage,
            "metric_gate_without_source_access_condition": metric_gate,
            "preliminary_gate": law_preliminary_gate(
                new_descriptor_selected=new_selected,
                additive_satisfaction=additive["satisfaction"],
                base_satisfaction=base["satisfaction"],
                rejection_delta=additive["rejection"] - base["rejection"],
                min_kind_rejection_delta=additive_min - base_min,
                worst_anion_delta=worst_anion_delta,
                real_coverage=float(coverage["real_min"]),
                bad_coverage=float(coverage["bad_min"]),
                source_tables_materialized_lockbox_rows=bool(
                    source_access_audit["source_tables_materialized_all_splits"]
                    and (
                        source_access_audit["materialized_lockbox_real_rows"] > 0
                        or source_access_audit["materialized_lockbox_bad_rows"] > 0
                    )
                ),
            ),
            "status": "pending false-positive and perturbation-lineage falsification",
        }
    report["comparisons"] = comparisons
    report["comparison_variant"] = comparison_variant
    report["provenance"] = {
        "runtime_seconds": time.time() - started,
        "input_sha256": {
            "law_real": _hash_file(args.isolated_dir / "law_real.parquet"),
            "law_bad": _hash_file(args.isolated_dir / "law_bad.parquet"),
            "real_descriptors": _hash_file(args.real_descriptors),
            "bad_descriptors": _hash_file(args.bad_descriptors),
            "real_descriptor_metadata": _hash_file(
                args.real_descriptors.with_suffix(
                    args.real_descriptors.suffix + ".meta.json"
                )
            ),
            "bad_descriptor_metadata": _hash_file(
                args.bad_descriptors.with_suffix(
                    args.bad_descriptors.suffix + ".meta.json"
                )
            ),
        },
        "implementation_sha256": _hash_file(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
