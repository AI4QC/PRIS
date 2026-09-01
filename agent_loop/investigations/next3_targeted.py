#!/usr/bin/env python3
"""Targeted guard-repair variants for the np-next-20260801c 0.98 set.

Same protocol as np-next-20260802: take the beam-selected additive set and
evaluate fixed named variants in which the unguarded
``p2vor_an_sa_like_fraction_max`` rule is guarded by (W1) integer anion
valence, (W2) polyanion-free structure, or (W3) both.  Nothing is refitted.
W1-W3 on discovery/calibration are pre-registered targeted analyses; the fp
evaluation reuses the merged false-positive frame with unknown failing
closed.
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
from law_falsification import apply_law_set_unknown_fails  # noqa: E402
from next3_law_search import load_next3_search_frames  # noqa: E402

TARGET_FEATURE = "p2vor_an_sa_like_fraction_max"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_candidate(rule, extra_guards=()):
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


def _guarded_copy(rule, guard_feature, guard_side, guard_threshold):
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


def _gate(base, variant):
    shared = set(base["by_anion"]).intersection(variant["by_anion"])
    worst = min(
        (
            variant["by_anion"][anion]["satisfaction"]
            - base["by_anion"][anion]["satisfaction"]
            for anion in shared
        ),
        default=0.0,
    )
    base_min = min(base["by_kind"].values())
    variant_min = min(variant["by_kind"].values())
    return {
        "satisfaction_delta": variant["satisfaction"] - base["satisfaction"],
        "rejection_delta": variant["rejection"] - base["rejection"],
        "min_kind_delta": variant_min - base_min,
        "worst_anion_delta": worst,
        "headline_or_clause": bool(
            variant["rejection"] - base["rejection"] >= 0.02
            or variant_min - base_min >= 0.03
        ),
        "satisfaction_clause": bool(
            variant["satisfaction"] >= base["satisfaction"] - 0.005
        ),
        "anion_clause": bool(worst >= -0.01),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--real-sixfam", type=Path, required=True)
    parser.add_argument("--bad-sixfam", type=Path, required=True)
    parser.add_argument("--real-guards", type=Path, required=True)
    parser.add_argument("--bad-guards", type=Path, required=True)
    parser.add_argument("--law-report", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--fp-p235", type=Path, required=True)
    parser.add_argument("--fp-guards", type=Path, required=True)
    parser.add_argument("--fp-sixfam", type=Path, required=True)
    parser.add_argument("--variant-name", default="additive_sixfam_loop")
    parser.add_argument("--floor", default="0.98")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    started = time.time()
    dr, cr, db, cb = load_next3_search_frames(
        args.isolated_dir,
        args.real_descriptors,
        args.bad_descriptors,
        args.real_sixfam,
        args.bad_sixfam,
        args.real_guards,
        args.bad_guards,
    )
    source = json.loads(args.law_report.read_text(encoding="utf-8"))
    base_rules = source["frontiers"]["existing_loop"][args.floor]["rules"]
    additive_rules = source["frontiers"][args.variant_name][args.floor]["rules"]
    target = [rule for rule in additive_rules if rule["feature"] == TARGET_FEATURE]
    if len(target) != 1:
        raise SystemExit(
            f"expected one {TARGET_FEATURE} rule in {args.variant_name}@{args.floor}"
        )
    target = target[0]
    others = [
        _as_candidate(rule)
        for rule in additive_rules
        if rule["feature"] != TARGET_FEATURE
    ]
    baseline = [_as_candidate(rule) for rule in base_rules]
    variants = {
        "W0_as_discovered": [*others, _as_candidate(target)],
        "W1_zan_ge1": [*others, _guarded_copy(target, "z_an_abs", "hi", 0.99)],
        "W2_polyanion_free": [
            *others,
            _guarded_copy(target, "p7poly_an_contact_min", "hi", 1.3),
        ],
        "W3_both_guards": [
            *others,
            _guarded_copy(target, "z_an_abs", "hi", 0.99),
            _guarded_copy(target, "p7poly_an_contact_min", "hi", 1.3),
        ],
    }

    report = {
        "protocol": {
            "experiment": "np-next-20260801c",
            "analysis": "targeted guard-repair variants of the 0.98 additive set",
            "search_performed": False,
            "floor": args.floor,
            "source_variant": args.variant_name,
            "target_rule": target["description"],
            "lockbox_access": False,
        },
        "splits": {},
        "false_positive": {},
    }
    for split_name, real, bad in (("discovery", dr, db), ("calibration", cr, cb)):
        base_metrics = _metrics(
            real, bad, _combined(real, baseline), _combined(bad, baseline)
        )
        split_block = {"baseline": base_metrics, "variants": {}}
        for name, candidates in variants.items():
            metrics = _metrics(
                real,
                bad,
                _combined(real, candidates),
                _combined(bad, candidates),
            )
            split_block["variants"][name] = {
                "metrics": metrics,
                "gate_arithmetic_vs_existing_loop": _gate(base_metrics, metrics),
            }
        report["splits"][split_name] = split_block

    # False-positive evaluation with the merged fp frame.
    fp_source = pd.read_parquet(args.features_dir / "false_positive.parquet")
    p235 = pd.read_parquet(args.fp_p235).drop(columns=["split"], errors="ignore")
    fp_guards = pd.read_parquet(args.fp_guards)
    fp_sixfam = pd.read_parquet(args.fp_sixfam).drop(columns=["split"], errors="ignore")
    fp_frame = (
        fp_source.merge(p235, on="sid", how="left", validate="one_to_one")
        .merge(fp_guards, on="sid", how="left", validate="one_to_one")
        .merge(fp_sixfam, on="sid", how="left", validate="one_to_one")
    )

    def fp_eval(candidates):
        rules = []
        for candidate in candidates:
            rules.append(
                {
                    "description": candidate.description,
                    "feature": candidate.feature,
                    "family": candidate.family,
                    "origin": candidate.origin,
                    "side": candidate.side,
                    "thresholds": list(candidate.thresholds),
                    "real_coverage": candidate.real_coverage,
                    "bad_coverage": candidate.bad_coverage,
                    "guard_feature": candidate.guard_feature,
                    "guard_side": candidate.guard_side,
                    "guard_threshold": candidate.guard_threshold,
                }
            )
        passed, known = apply_law_set_unknown_fails(fp_frame, rules)
        return {
            "pass_rate": float(passed.mean()),
            "known_rate": float(known.mean()),
            "unknown_count": int((~known).sum()),
        }

    fp_base = fp_eval(baseline)
    for name, candidates in variants.items():
        result = fp_eval(candidates)
        result["pass_delta_vs_existing"] = float(
            result["pass_rate"] - fp_base["pass_rate"]
        )
        report["false_positive"][name] = result
    report["false_positive"]["existing_loop"] = fp_base

    report["provenance"] = {
        "runtime_seconds": time.time() - started,
        "law_report_sha256": _hash_file(args.law_report),
        "implementation_sha256": _hash_file(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    for name in variants:
        gate = report["splits"]["calibration"]["variants"][name][
            "gate_arithmetic_vs_existing_loop"
        ]
        fp = report["false_positive"][name]
        print(
            f"{name}: sat {gate['satisfaction_delta']:+.4f} "
            f"rej {gate['rejection_delta']:+.4f} "
            f"minkind {gate['min_kind_delta']:+.4f} "
            f"anion {gate['worst_anion_delta']:+.4f} "
            f"| headline={gate['headline_or_clause']} anion={gate['anion_clause']} "
            f"| fp {fp['pass_delta_vs_existing']:+.4f}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
