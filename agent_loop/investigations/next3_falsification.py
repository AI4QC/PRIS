#!/usr/bin/env python3
"""295 DFT-relaxed falsification for the np-next-20260801c law report.

Merges the three false-positive descriptor caches (P1-P5 from
np-next-20260801, valence guards from np-next-20260802, six families from
np-next-20260801c) and evaluates every frontier in the law report with
unknown-fails-closed semantics.
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

from law_falsification import apply_law_set_unknown_fails  # noqa: E402


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--law-report", type=Path, required=True)
    parser.add_argument("--fp-p235", type=Path, required=True)
    parser.add_argument("--fp-guards", type=Path, required=True)
    parser.add_argument("--fp-sixfam", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    started = time.time()
    source = pd.read_parquet(args.features_dir / "false_positive.parquet")
    p235 = pd.read_parquet(args.fp_p235).drop(columns=["split"], errors="ignore")
    guards = pd.read_parquet(args.fp_guards)
    sixfam = pd.read_parquet(args.fp_sixfam).drop(columns=["split"], errors="ignore")
    for name, frame in (("p235", p235), ("guards", guards), ("sixfam", sixfam)):
        if frame["sid"].duplicated().any():
            raise SystemExit(f"duplicate sid in {name} descriptors")
    frame = (
        source.merge(p235, on="sid", how="left", validate="one_to_one")
        .merge(guards, on="sid", how="left", validate="one_to_one")
        .merge(sixfam, on="sid", how="left", validate="one_to_one")
    )
    if len(frame) != len(source):
        raise SystemExit("descriptor merges changed the audit row set")

    report = json.loads(args.law_report.read_text(encoding="utf-8"))
    evaluations = {}
    for variant, frontier in report["frontiers"].items():
        evaluations[variant] = {}
        for floor, entry in frontier.items():
            rules = entry["rules"]
            required = sorted(
                {
                    str(value)
                    for rule in rules
                    for value in (rule["feature"], rule.get("guard_feature"))
                    if value is not None
                }
            )
            missing_columns = [
                feature for feature in required if feature not in frame.columns
            ]
            if missing_columns:
                evaluations[variant][floor] = {
                    "error": "missing features: " + ", ".join(missing_columns)
                }
                continue
            passed, known = apply_law_set_unknown_fails(frame, rules)
            evaluations[variant][floor] = {
                "pass_rate": float(passed.mean()),
                "rejection_rate": float((~passed).mean()),
                "known_rate": float(known.mean()),
                "unknown_count": int((~known).sum()),
                "n": int(len(passed)),
                "calibration_real_satisfaction": float(
                    entry["calibration"]["satisfaction"]
                ),
                "pass_minus_calibration_satisfaction": float(
                    passed.mean() - entry["calibration"]["satisfaction"]
                ),
                "selected_feature_coverage": {
                    feature: float(
                        np.isfinite(frame[feature].to_numpy(dtype=float)).mean()
                    )
                    for feature in required
                },
            }
    comparison_variant = report["comparison_variant"]
    comparisons = {}
    for floor, prior in report["comparisons"].items():
        existing = evaluations.get("existing_loop", {}).get(floor, {})
        additive = evaluations.get(comparison_variant, {}).get(floor, {})
        if "error" in existing or "error" in additive:
            comparisons[floor] = {"error": "missing features"}
            continue
        pass_delta = additive["pass_rate"] - existing["pass_rate"]
        comparisons[floor] = {
            "comparison_variant": comparison_variant,
            "false_positive_pass_delta_vs_existing_loop": float(pass_delta),
            "passes_false_positive_gate": bool(pass_delta >= -0.03),
            "prior_preliminary_gate": bool(prior["preliminary_gate"]),
            "combined_gate_before_unseen_source_holdout": bool(
                prior["preliminary_gate"] and pass_delta >= -0.03
            ),
        }
    output = {
        "protocol": {
            "experiment": "np-next-20260801c",
            "population": "frozen 295 DFT-relaxed LeMat/ELEMENTA candidates",
            "threshold_refit": False,
            "missing_feature_semantics": "unknown fails closed",
            "identifier_output": False,
            "lockbox_access": False,
            "gate": "additive pass rate may not fall >0.03 below paired existing loop",
        },
        "n_false_positive_structures": int(len(frame)),
        "frontiers": evaluations,
        "comparisons": comparisons,
        "provenance": {
            "runtime_seconds": time.time() - started,
            "law_report_sha256": _hash_file(args.law_report),
            "fp_p235_sha256": _hash_file(args.fp_p235),
            "fp_guards_sha256": _hash_file(args.fp_guards),
            "fp_sixfam_sha256": _hash_file(args.fp_sixfam),
            "implementation_sha256": _hash_file(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    for floor, comparison in comparisons.items():
        print(
            f"floor {floor}: pass delta "
            f"{comparison.get('false_positive_pass_delta_vs_existing_loop', float('nan')):+.4f}; "
            f"gate={comparison.get('passes_false_positive_gate')}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
