#!/usr/bin/env python3
"""Corrected AND-of-guards variants for np-next-20260801c.

Supersedes next3_targeted_w4.py, whose construction appended two
independently-guarded copies of the same rule:  (not g1 or T) and
(not g2 or T)  ==  if (g1 OR g2) then T,  the logical opposite of the
intended conjunction.  Here compound rules are evaluated with explicit
masks:  if (g1 AND g2) then T.

Variants:
- W3c: p2vor rule under (integer anion valence AND polyanion-free);
- W4c: p2vor rule under polyanion-free, and each bond-valence rule under
  (original guard AND bvloc_parameter_exact_fraction >= 0.9);
- W2: the single-guard reference re-evaluated by the same mask code.
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

from better_law_search import apply_candidate, evaluate_masks  # noqa: E402
from next3_law_search import load_next3_search_frames  # noqa: E402
from next3_targeted import _gate, _metrics  # noqa: E402

TARGET_P2 = "p2vor_an_sa_like_fraction_max"
EXACT_GUARD = ("bvloc_parameter_exact_fraction", "hi", 0.9)
ZAN_GUARD = ("z_an_abs", "hi", 0.99)
POLY_GUARD = ("p7poly_an_contact_min", "hi", 1.3)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one_sided_mask(values: np.ndarray, side: str, threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    comparison = values <= threshold if side == "hi" else values >= threshold
    return (~finite) | comparison


def _guard_mask(values: np.ndarray, side: str, threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    comparison = values > threshold if side == "hi" else values <= threshold
    return finite & comparison


def _rule_mask(frame: pd.DataFrame, rule: dict) -> np.ndarray:
    """if (all guards) then target, with fail-closed guard semantics."""

    guards = list(rule.get("guards", []))
    target = _one_sided_mask(
        frame[rule["feature"]].to_numpy(dtype=float),
        rule["side"],
        float(rule["thresholds"][0]),
    )
    if rule["side"] == "band":
        lower, upper = (float(v) for v in rule["thresholds"])
        values = frame[rule["feature"]].to_numpy(dtype=float)
        finite = np.isfinite(values)
        target = (~finite) | ((values >= lower) & (values <= upper))
    if not guards:
        return target
    guard_all = np.ones(len(frame), dtype=bool)
    for guard_feature, guard_side, guard_threshold in guards:
        guard_all &= _guard_mask(
            frame[guard_feature].to_numpy(dtype=float),
            guard_side,
            float(guard_threshold),
        )
    return (~guard_all) | target


def _set_mask(frame: pd.DataFrame, rules: Sequence[dict]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for rule in rules:
        mask &= _rule_mask(frame, rule)
    return mask


def _serialize(rule: dict) -> dict:
    if rule.get("guard_feature") is None:
        return {
            "feature": rule["feature"],
            "side": rule["side"],
            "thresholds": rule["thresholds"],
            "guards": [],
        }
    return {
        "feature": rule["feature"],
        "side": rule["side"],
        "thresholds": rule["thresholds"],
        "guards": [
            (rule["guard_feature"], rule["guard_side"], rule["guard_threshold"])
        ],
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
    base_rules = [_serialize(rule) for rule in source["frontiers"]["existing_loop"][args.floor]["rules"]]
    additive = source["frontiers"][args.variant_name][args.floor]["rules"]

    def build(mode: str) -> list[dict]:
        rules = []
        for rule in additive:
            serialized = _serialize(rule)
            if mode == "W2" and rule["feature"] == TARGET_P2:
                serialized["guards"] = [POLY_GUARD]
            elif mode == "W3c" and rule["feature"] == TARGET_P2:
                serialized["guards"] = [ZAN_GUARD, POLY_GUARD]
            elif mode == "W4c":
                if rule["feature"] == TARGET_P2:
                    serialized["guards"] = [POLY_GUARD]
                elif rule["feature"].startswith("bvloc_"):
                    serialized["guards"] = [
                        *serialized["guards"],
                        EXACT_GUARD,
                    ]
            rules.append(serialized)
        return rules

    variants = {
        "W2_single_guard": build("W2"),
        "W3c_and_zan_poly": build("W3c"),
        "W4c_and_exact_params": build("W4c"),
    }
    report = {
        "protocol": {
            "experiment": "np-next-20260801c",
            "analysis": "corrected AND-of-guards diagnosed variants (post-freeze)",
            "supersedes": "next3_targeted_w4.py (OR-guard bug)",
            "search_performed": False,
            "floor": args.floor,
            "lockbox_access": False,
        },
        "splits": {},
    }
    for split_name, real, bad in (("discovery", dr, db), ("calibration", cr, cb)):
        base_metrics = _metrics(
            real, bad, _set_mask(real, base_rules), _set_mask(bad, base_rules)
        )
        block = {"baseline": base_metrics, "variants": {}}
        for name, rules in variants.items():
            metrics = _metrics(
                real, bad, _set_mask(real, rules), _set_mask(bad, rules)
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
