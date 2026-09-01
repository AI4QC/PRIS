#!/usr/bin/env python3
"""Diagnosed guard variants for np-next-20260802 (post-freeze diagnostics).

Evaluates the beam-selected 0.98 additive set with the unguarded
``p2vor_an_sa_like_fraction_max`` rule replaced by a guarded form given on
the command line.  Created after inspecting the round's failure modes, so
every variant here is a post-freeze diagnostic and hypothesis generator,
not a gate-passing claim of this round.
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

from better_law_search import LawCandidate, apply_candidate, evaluate_masks  # noqa: E402
from next2_law_search import load_next2_search_frames  # noqa: E402

TARGET_FEATURE = "p2vor_an_sa_like_fraction_max"
FLOOR = "0.98"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_candidate(rule, guard=None):
    guard_feature = guard_side = None
    guard_threshold = None
    if guard is not None:
        guard_feature, guard_side, guard_threshold = guard
    elif rule.get("guard_feature") is not None:
        guard_feature = rule["guard_feature"]
        guard_side = rule["guard_side"]
        guard_threshold = rule["guard_threshold"]
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
        guard_feature=guard_feature,
        guard_side=guard_side,
        guard_threshold=guard_threshold,
    )


def _combined(frame, candidates):
    mask = np.ones(len(frame), dtype=bool)
    for candidate in candidates:
        mask &= apply_candidate(frame, candidate)
    return mask


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--real-guards", type=Path, required=True)
    parser.add_argument("--bad-guards", type=Path, required=True)
    parser.add_argument("--law-report", type=Path, required=True)
    parser.add_argument("--guard-feature", required=True)
    parser.add_argument("--guard-side", choices=("hi", "lo"), required=True)
    parser.add_argument("--guard-threshold", type=float, required=True)
    parser.add_argument("--label", required=True)
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
        raise SystemExit(f"expected one {TARGET_FEATURE} rule at floor {FLOOR}")
    target = target[0]

    guard = (args.guard_feature, args.guard_side, float(args.guard_threshold))
    others = [
        _as_candidate(rule)
        for rule in additive_rules
        if rule["feature"] != TARGET_FEATURE
    ]
    variant = [*others, _as_candidate(target, guard=guard)]
    baseline = [_as_candidate(rule) for rule in base_rules]

    report = {
        "protocol": {
            "experiment": "np-next-20260802",
            "analysis": "post-freeze diagnosed guard variant",
            "search_performed": False,
            "floor": FLOOR,
            "target_rule": target["description"],
            "guard": {
                "feature": args.guard_feature,
                "side": args.guard_side,
                "threshold": float(args.guard_threshold),
            },
            "label": args.label,
            "lockbox_access": False,
        },
        "splits": {},
    }
    for split_name, real, bad in (("discovery", dr, db), ("calibration", cr, cb)):
        base_metrics = _metrics(
            real, bad, _combined(real, baseline), _combined(bad, baseline)
        )
        variant_metrics = _metrics(
            real, bad, _combined(real, variant), _combined(bad, variant)
        )
        shared = set(base_metrics["by_anion"]).intersection(
            variant_metrics["by_anion"]
        )
        worst = min(
            (
                variant_metrics["by_anion"][anion]["satisfaction"]
                - base_metrics["by_anion"][anion]["satisfaction"]
                for anion in shared
            ),
            default=0.0,
        )
        base_min = min(base_metrics["by_kind"].values())
        variant_min = min(variant_metrics["by_kind"].values())
        report["splits"][split_name] = {
            "baseline": base_metrics,
            "variant": variant_metrics,
            "gate_arithmetic_vs_existing_loop": {
                "satisfaction_delta": (
                    variant_metrics["satisfaction"] - base_metrics["satisfaction"]
                ),
                "rejection_delta": (
                    variant_metrics["rejection"] - base_metrics["rejection"]
                ),
                "min_kind_delta": variant_min - base_min,
                "worst_anion_delta": worst,
                "headline_or_clause": bool(
                    variant_metrics["rejection"] - base_metrics["rejection"] >= 0.02
                    or variant_min - base_min >= 0.03
                ),
                "satisfaction_clause": bool(
                    variant_metrics["satisfaction"]
                    >= base_metrics["satisfaction"] - 0.005
                ),
                "anion_clause": bool(worst >= -0.01),
            },
        }
    report["provenance"] = {
        "runtime_seconds": time.time() - started,
        "law_report_sha256": _hash_file(args.law_report),
        "implementation_sha256": _hash_file(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    gate = report["splits"]["calibration"]["gate_arithmetic_vs_existing_loop"]
    print(
        f"{args.label} @calibration: sat {gate['satisfaction_delta']:+.4f} "
        f"rej {gate['rejection_delta']:+.4f} "
        f"min-kind {gate['min_kind_delta']:+.4f} "
        f"worst-anion {gate['worst_anion_delta']:+.4f}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
