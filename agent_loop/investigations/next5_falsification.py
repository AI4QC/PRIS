#!/usr/bin/env python3
"""All-295 unknown-fails-closed audit for np-next-20260801e."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import pandas as pd

from next4_falsification import evaluate_rule_set
from next4_law_search import _sha256
from next5_law_search import EXPERIMENT_ID


def development_gate(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    prior_gate: bool,
) -> dict[str, object]:
    """Combine the frozen all-295 checks with pre-LOKO development gates."""

    pass_delta = float(candidate["pass_rate"]) - float(baseline["pass_rate"])
    coverage = float(candidate["joint_required_feature_coverage"])
    coverage_gate = coverage >= 0.90
    pass_gate = pass_delta >= -0.03 - 1e-12
    return {
        "all_295_pass_delta_vs_existing_loop": pass_delta,
        "joint_required_feature_coverage": coverage,
        "passes_joint_coverage_gate": bool(coverage_gate),
        "passes_all_295_false_positive_gate": bool(pass_gate),
        "prior_soft_margin_pre_loko_gate": bool(prior_gate),
        "combined_development_gate_before_loko": bool(
            coverage_gate and pass_gate and prior_gate
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--law-report", type=Path, required=True)
    parser.add_argument("--fp-p235", type=Path, required=True)
    parser.add_argument("--fp-guards", type=Path, required=True)
    parser.add_argument("--fp-sixfam", type=Path, required=True)
    parser.add_argument("--fp-corrected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    started = time.time()
    source = pd.read_parquet(args.features_dir / "false_positive.parquet")
    merged = source.copy()
    input_paths = (
        ("p235", args.fp_p235),
        ("guards", args.fp_guards),
        ("sixfam", args.fp_sixfam),
        ("corrected", args.fp_corrected),
    )
    for name, path in input_paths:
        frame = pd.read_parquet(path).drop(
            columns=["split", "next4_valence_source"], errors="ignore"
        )
        if frame["sid"].duplicated().any():
            raise SystemExit(f"duplicate sid in {name} false-positive descriptors")
        overlapping = [
            column for column in frame if column != "sid" and column in merged
        ]
        if overlapping:
            frame = frame.drop(columns=overlapping)
        before = len(merged)
        merged = merged.merge(frame, on="sid", how="left", validate="one_to_one")
        if len(merged) != before:
            raise SystemExit(f"{name} merge changed the all-295 audit row set")

    law = json.loads(args.law_report.read_text(encoding="utf-8"))
    if law.get("protocol", {}).get("experiment") != EXPERIMENT_ID:
        raise SystemExit("law report experiment is incompatible with next5 audit")
    evaluations: dict[str, dict[str, object]] = {}
    for variant, frontier in law["frontiers"].items():
        evaluations[variant] = {
            floor: evaluate_rule_set(merged, entry["rules"])
            for floor, entry in frontier.items()
        }

    comparison_variant = str(law["comparison_variant"])
    prior_gate = bool(law["comparison"].get("pre_loko_gate", False))
    comparisons = {
        floor: {
            "comparison_variant": comparison_variant,
            **development_gate(
                evaluations["existing_loop"][floor],
                evaluations[comparison_variant][floor],
                prior_gate=prior_gate,
            ),
        }
        for floor in law["frontiers"]["existing_loop"]
    }
    output = {
        "protocol": {
            "experiment": EXPERIMENT_ID,
            "population": "frozen 295 DFT-relaxed LeMat/ELEMENTA candidates",
            "n_population": int(len(merged)),
            "threshold_refit": False,
            "rule_selection": False,
            "missing_feature_semantics": "unknown fails closed on all 295",
            "known_only_role": "sensitivity analysis only",
            "joint_required_feature_coverage_gate": 0.90,
            "identifier_output": False,
            "lockbox_access": False,
        },
        "frontiers": evaluations,
        "comparisons": comparisons,
        "provenance": {
            "runtime_seconds": time.time() - started,
            "law_report_sha256": _sha256(args.law_report),
            "fp_p235_sha256": _sha256(args.fp_p235),
            "fp_guards_sha256": _sha256(args.fp_guards),
            "fp_sixfam_sha256": _sha256(args.fp_sixfam),
            "fp_corrected_sha256": _sha256(args.fp_corrected),
            "implementation_sha256": _sha256(Path(__file__)),
            "dependency_sha256": _sha256(
                Path(__file__).with_name("next4_falsification.py")
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    for floor, comparison in comparisons.items():
        print(
            f"floor {floor}: all-295 pass delta "
            f"{comparison['all_295_pass_delta_vs_existing_loop']:+.4f}; "
            f"joint coverage {comparison['joint_required_feature_coverage']:.4f}; "
            f"gate={comparison['combined_development_gate_before_loko']}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
