#!/usr/bin/env python3
"""Pre-registered targeted guard analysis for np-next-20260802 (design §4.4).

Not a search: takes the beam-selected 0.98 additive set from
np-next-20260801/2 and evaluates fixed, named variants in which the
unguarded ``p2vor_an_sa_like_fraction_max`` rule is (a) guarded by
``z_an_abs <= 2`` (non-nitride/phosphide chemistry only), (b) guarded by
high ``fi``, or (c) dropped.  All thresholds are the ones already fitted
on real discovery quantiles; nothing is refitted.
"""

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

from better_law_search import (  # noqa: E402
    LawCandidate,
    evaluate_masks,
)
from next2_law_search import load_next2_search_frames  # noqa: E402

TARGET_FEATURE = "p2vor_an_sa_like_fraction_max"
FLOOR = "0.98"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(real, bad, mask_real, mask_bad):
    out = evaluate_masks(
        real_mask=mask_real,
        bad_mask=mask_bad,
        bad_groups=bad["parent"].to_numpy(),
        bad_kinds=bad["kind"].to_numpy(),
    )
    by_anion = {}
    for anion, subset in real.groupby("anion", dropna=True):
        if len(subset) < 100:
            continue
        by_anion[str(anion)] = {
            "n": int(len(subset)),
            "satisfaction": float(mask_real[subset.index.to_numpy()].mean()),
        }
    out["by_anion"] = by_anion
    return out


def _gate_against(base_cal, add_cal, worst_anion_delta, coverage_min):
    base_min = min(base_cal["by_kind"].values())
    add_min = min(add_cal["by_kind"].values())
    return {
        "satisfaction_delta": add_cal["satisfaction"] - base_cal["satisfaction"],
        "rejection_delta": add_cal["rejection"] - base_cal["rejection"],
        "min_kind_delta": add_min - base_min,
        "worst_anion_delta": worst_anion_delta,
        "headline_or_clause": bool(
            add_cal["rejection"] - base_cal["rejection"] >= 0.02
            or add_min - base_min >= 0.03
        ),
        "satisfaction_clause": bool(
            add_cal["satisfaction"] >= base_cal["satisfaction"] - 0.005
        ),
        "anion_clause": bool(worst_anion_delta >= -0.01),
        "coverage_clause": bool(coverage_min >= 0.90),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--real-guards", type=Path, required=True)
    parser.add_argument("--bad-guards", type=Path, required=True)
    parser.add_argument("--law-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    started = time.time()
    dr, cr, db, cb = load_next2_search_frames(
        args.isolated_dir,
        args.real_descriptors,
        args.bad_descriptors,
        args.real_guards,
        args.bad_guards,
    )
    source = json.loads(args.law_report.read_text(encoding="utf-8"))
    base_rules = source["frontiers"]["existing_loop"][FLOOR]["rules"]
    additive_rules = source["frontiers"]["additive_next2_loop"][FLOOR]["rules"]

    target = [
        rule for rule in additive_rules if rule["feature"] == TARGET_FEATURE
    ]
    if len(target) != 1:
        raise SystemExit(
            f"expected exactly one {TARGET_FEATURE} rule in the 0.98 additive set"
        )
    target = target[0]
    threshold = float(target["thresholds"][0])
    fi_guard_threshold = float(np.quantile(dr["fi"].to_numpy(dtype=float), 0.75))

    def as_candidate(rule):
        return LawCandidate(
            description=str(rule["description"]),
            feature=str(rule["feature"]),
            family=str(rule["family"]),
            origin=str(rule["origin"]),
            side=str(rule["side"]),
            thresholds=tuple(float(value) for value in rule["thresholds"]),
            real_mask=np.empty(0, dtype=bool),
            bad_mask=np.empty(0, dtype=bool),
            real_coverage=float(rule.get("real_coverage", float("nan"))),
            bad_coverage=float(rule.get("bad_coverage", float("nan"))),
            guard_feature=rule.get("guard_feature"),
            guard_side=rule.get("guard_side"),
            guard_threshold=rule.get("guard_threshold"),
        )

    def guarded_copy(rule, guard_feature, guard_side, guard_threshold):
        return LawCandidate(
            description=(
                f"if {guard_feature} {guard_side} {guard_threshold:.8g} "
                f"then ({rule['description']})"
            ),
            feature=rule["feature"],
            family=rule["family"],
            origin=rule["origin"],
            side=rule["side"],
            thresholds=rule["thresholds"],
            real_mask=np.empty(0, dtype=bool),
            bad_mask=np.empty(0, dtype=bool),
            real_coverage=rule["real_coverage"],
            bad_coverage=rule["bad_coverage"],
            guard_feature=guard_feature,
            guard_side=guard_side,
            guard_threshold=guard_threshold,
        )

    others = [
        as_candidate(rule)
        for rule in additive_rules
        if rule["feature"] != TARGET_FEATURE
    ]
    variants = {
        "V0_as_discovered": [*others, as_candidate(target)],
        "V1_zan_le2_guard": [
            *others,
            guarded_copy(target, "z_an_abs", "lo", 2.0),
        ],
        "V2_fi_q75_guard": [
            *others,
            guarded_copy(target, "fi", "hi", fi_guard_threshold),
        ],
        "V3_rule_dropped": [*others],
    }

    from better_law_search import apply_candidate

    def combined_mask(frame, candidates):
        mask = np.ones(len(frame), dtype=bool)
        for candidate in candidates:
            mask &= apply_candidate(frame, candidate)
        return mask

    base_candidates = [as_candidate(rule) for rule in base_rules]
    results = {"existing_loop": {}, "variants": {}}
    for split_name, real, bad in (("discovery", dr, db), ("calibration", cr, cb)):
        results["existing_loop"][split_name] = _metrics(
            real,
            bad,
            combined_mask(real, base_candidates),
            combined_mask(bad, base_candidates),
        )
        for name, candidates in variants.items():
            results["variants"].setdefault(name, {})[split_name] = _metrics(
                real,
                bad,
                combined_mask(real, candidates),
                combined_mask(bad, candidates),
            )

    comparisons = {}
    for name in variants:
        base_cal = results["existing_loop"]["calibration"]
        add_cal = results["variants"][name]["calibration"]
        shared = set(base_cal["by_anion"]).intersection(add_cal["by_anion"])
        worst = min(
            (
                add_cal["by_anion"][anion]["satisfaction"]
                - base_cal["by_anion"][anion]["satisfaction"]
                for anion in shared
            ),
            default=0.0,
        )
        comparisons[name] = {
            "gate_arithmetic_vs_existing_loop": _gate_against(
                base_cal, add_cal, worst, coverage_min=0.9746
            ),
            "calibration": add_cal,
            "discovery": results["variants"][name]["discovery"],
        }
    report = {
        "protocol": {
            "experiment": "np-next-20260802",
            "analysis": "pre-registered targeted guard replacement (design §4.4)",
            "search_performed": False,
            "floor": FLOOR,
            "target_rule": target["description"],
            "fi_guard_threshold_q75": fi_guard_threshold,
            "lockbox_access": False,
        },
        "existing_loop": results["existing_loop"],
        "comparisons": comparisons,
        "provenance": {
            "runtime_seconds": time.time() - started,
            "law_report_sha256": _hash_file(args.law_report),
            "implementation_sha256": _hash_file(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out}", flush=True)
    for name, comparison in comparisons.items():
        gate = comparison["gate_arithmetic_vs_existing_loop"]
        print(
            f"{name}: sat {gate['satisfaction_delta']:+.4f} "
            f"rej {gate['rejection_delta']:+.4f} "
            f"min-kind {gate['min_kind_delta']:+.4f} "
            f"worst-anion {gate['worst_anion_delta']:+.4f} "
            f"headline={gate['headline_or_clause']} anion={gate['anion_clause']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
