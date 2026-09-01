#!/usr/bin/env python3
"""Second-order diagnosed variant W4 for np-next-20260801c.

W2 repaired the nitride cost of the p2vor rule but left P/Se/I/Br costs
traced to the two bond-valence-mismatch rules driven by the post-freeze
parameter fallback (exact-fraction means: P 0.29, Te 0.33, N 0.51, Se 0.53,
Br/I 0.78 vs O/F/Cl >= 0.94).  W4 scopes each bond-valence rule to
structures with ``bvloc_parameter_exact_fraction >= 0.9`` (the parameter
reliability domain) in addition to its original guard.  Post-freeze
diagnostic, same evaluation protocol as next3_targeted.py.
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
from next3_targeted import _gate, _metrics  # noqa: E402

EXACT_GUARD = ("bvloc_parameter_exact_fraction", "hi", 0.9)
POLY_GUARD = ("p7poly_an_contact_min", "hi", 1.3)
TARGET_P2 = "p2vor_an_sa_like_fraction_max"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_candidate(rule):
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
    baseline = [_as_candidate(rule) for rule in base_rules]

    w2_rules = []
    for rule in additive_rules:
        candidate = _as_candidate(rule)
        if rule["feature"] == TARGET_P2:
            candidate = _guarded_copy(rule, *POLY_GUARD)
        w2_rules.append(candidate)

    w4_rules = []
    for rule in additive_rules:
        candidate = _as_candidate(rule)
        if rule["feature"] == TARGET_P2:
            candidate = _guarded_copy(rule, *POLY_GUARD)
        elif rule["feature"].startswith("bvloc_"):
            # AND of the exact-parameter guard with the original guard:
            # keep the original guarded copy and add the exact-guarded copy.
            w4_rules.append(_guarded_copy(rule, *EXACT_GUARD))
        w4_rules.append(candidate)

    variants = {
        "W2_polyanion_free": w2_rules,
        "W4_exact_params": w4_rules,
    }
    report = {
        "protocol": {
            "experiment": "np-next-20260801c",
            "analysis": "second-order diagnosed guard variants (post-freeze)",
            "search_performed": False,
            "floor": args.floor,
            "exact_guard": list(EXACT_GUARD),
            "poly_guard": list(POLY_GUARD),
            "lockbox_access": False,
        },
        "splits": {},
    }
    for split_name, real, bad in (("discovery", dr, db), ("calibration", cr, cb)):
        base_metrics = _metrics(
            real, bad, _combined(real, baseline), _combined(bad, baseline)
        )
        block = {"baseline": base_metrics, "variants": {}}
        for name, candidates in variants.items():
            metrics = _metrics(
                real, bad, _combined(real, candidates), _combined(bad, candidates)
            )
            block["variants"][name] = {
                "metrics": metrics,
                "gate_arithmetic_vs_existing_loop": _gate(base_metrics, metrics),
            }
        report["splits"][split_name] = block
    report["provenance"] = {
        "runtime_seconds": time.time() - started,
        "law_report_sha256": _hash_file(args.law_report),
        "implementation_sha256": _hash_file(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    for name in variants:
        for split_name in ("discovery", "calibration"):
            gate = report["splits"][split_name]["variants"][name][
                "gate_arithmetic_vs_existing_loop"
            ]
            print(
                f"{name} @{split_name}: sat {gate['satisfaction_delta']:+.4f} "
                f"rej {gate['rejection_delta']:+.4f} "
                f"minkind {gate['min_kind_delta']:+.4f} "
                f"anion {gate['worst_anion_delta']:+.4f} "
                f"headline={gate['headline_or_clause']} anion={gate['anion_clause']}",
                flush=True,
            )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
