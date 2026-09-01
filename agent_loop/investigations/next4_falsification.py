#!/usr/bin/env python3
"""All-295 unknown-fails-closed audit for np-next-20260801d rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd

from law_falsification import apply_law_set_unknown_fails


def evaluate_rule_set(
    frame: pd.DataFrame,
    rules: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Evaluate on the full frame; known-only is reported as sensitivity only."""

    required = sorted(
        {
            str(value)
            for rule in rules
            for value in (rule["feature"], rule.get("guard_feature"))
            if value is not None
        }
    )
    passed, known = apply_law_set_unknown_fails(frame, rules)
    coverage = {
        feature: float(np.isfinite(frame[feature].to_numpy(dtype=float)).mean())
        for feature in required
    }
    return {
        "n": int(len(frame)),
        "pass_rate": float(passed.mean()),
        "rejection_rate": float((~passed).mean()),
        "known_rate": float(known.mean()),
        "unknown_count": int((~known).sum()),
        "known_only_pass_rate": (
            float(passed[known].mean()) if known.any() else None
        ),
        "joint_required_feature_coverage": float(known.mean()),
        "required_feature_coverage": coverage,
    }


def _sha256(path: Path) -> str:
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
    parser.add_argument("--fp-corrected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    started = time.time()
    source = pd.read_parquet(args.features_dir / "false_positive.parquet")
    inputs = []
    for name, path in (
        ("p235", args.fp_p235),
        ("guards", args.fp_guards),
        ("sixfam", args.fp_sixfam),
        ("corrected", args.fp_corrected),
    ):
        frame = pd.read_parquet(path).drop(
            columns=["split", "next4_valence_source"], errors="ignore"
        )
        if frame["sid"].duplicated().any():
            raise SystemExit(f"duplicate sid in {name} false-positive descriptors")
        inputs.append((name, frame))
    merged = source.copy()
    for name, frame in inputs:
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
    evaluations: dict[str, dict[str, object]] = {}
    for variant, frontier in law["frontiers"].items():
        evaluations[variant] = {}
        for floor, entry in frontier.items():
            evaluations[variant][floor] = evaluate_rule_set(merged, entry["rules"])

    comparison_variant = str(law["comparison_variant"])
    comparisons = {}
    for floor in law["frontiers"]["existing_loop"]:
        baseline = evaluations["existing_loop"][floor]
        candidate = evaluations[comparison_variant][floor]
        pass_delta = float(candidate["pass_rate"] - baseline["pass_rate"])
        coverage_gate = bool(candidate["joint_required_feature_coverage"] >= 0.90)
        pass_gate = bool(pass_delta >= -0.03)
        prior_gate = bool(law["comparison"]["calibration_metric_gate"])
        comparisons[floor] = {
            "comparison_variant": comparison_variant,
            "all_295_pass_delta_vs_existing_loop": pass_delta,
            "joint_required_feature_coverage": candidate[
                "joint_required_feature_coverage"
            ],
            "passes_joint_coverage_gate": coverage_gate,
            "passes_all_295_false_positive_gate": pass_gate,
            "prior_calibration_metric_gate": prior_gate,
            "combined_development_gate_before_loko": bool(
                coverage_gate and pass_gate and prior_gate
            ),
        }
    output = {
        "protocol": {
            "experiment": "np-next-20260801d",
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
