#!/usr/bin/env python3
"""Post-hoc semantic-rule recurrence diagnostic across next5 LOKO refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from better_law_search import apply_serialized_law_set, evaluate_masks
from next4_law_search import (
    _minimum_kind,
    _sha256,
    _worst_anion_delta,
    load_next4_search_frames,
    verify_isolated_manifest,
)
from next5_law_search import EXPERIMENT_ID
from next5_law_stability import validate_input_domain_audit


def semantic_rule_key(record: Mapping[str, object]) -> tuple[object, ...]:
    """Canonicalize rule semantics while ignoring sample-specific coverage."""

    return (
        str(record["feature"]),
        str(record["family"]),
        str(record["side"]),
        tuple(float(value) for value in record["thresholds"]),
        record.get("guard_feature"),
        record.get("guard_side"),
        (
            None
            if record.get("guard_threshold") is None
            else float(record["guard_threshold"])
        ),
    )


def recurrent_semantic_rules(
    folds: Mapping[str, Mapping[str, object]],
    *,
    min_count: int = 2,
) -> list[dict[str, object]]:
    """Count a semantic identity no more than once per held-kind refit."""

    occurrences: dict[tuple[object, ...], dict[str, object]] = {}
    for held, entry in folds.items():
        seen: set[tuple[object, ...]] = set()
        for raw_rule in entry["rules"]:
            rule = dict(raw_rule)
            key = semantic_rule_key(rule)
            if key in seen:
                continue
            seen.add(key)
            bucket = occurrences.setdefault(
                key, {"rule": rule, "held_kinds": []}
            )
            bucket["held_kinds"].append(str(held))
    retained = []
    for key, bucket in occurrences.items():
        held_kinds = sorted(set(bucket["held_kinds"]))
        if len(held_kinds) < min_count:
            continue
        retained.append(
            {
                "semantic_key": {
                    "feature": key[0],
                    "family": key[1],
                    "side": key[2],
                    "thresholds": list(key[3]),
                    "guard_feature": key[4],
                    "guard_side": key[5],
                    "guard_threshold": key[6],
                },
                "rule": bucket["rule"],
                "held_kinds": held_kinds,
                "count": len(held_kinds),
            }
        )
    retained.sort(
        key=lambda entry: (
            -int(entry["count"]),
            json.dumps(entry["semantic_key"], sort_keys=True),
        )
    )
    maximum = max((int(entry["count"]) for entry in retained), default=0)
    for entry in retained:
        entry["is_maximum_recurrence"] = int(entry["count"]) == maximum
    return retained


def _evaluate(frame_real, frame_bad, rules: Sequence[Mapping[str, object]]):
    real_mask = apply_serialized_law_set(frame_real, rules)
    bad_mask = apply_serialized_law_set(frame_bad, rules)
    metrics = evaluate_masks(
        real_mask=real_mask,
        bad_mask=bad_mask,
        bad_groups=frame_bad["parent"].to_numpy(),
        bad_kinds=frame_bad["kind"].to_numpy(),
    )
    anions = frame_real["anion"].astype("string").to_numpy(dtype=object)
    metrics["by_anion"] = {
        str(anion): {
            "n": int((anions == anion).sum()),
            "satisfaction": float(real_mask[anions == anion].mean()),
        }
        for anion in sorted(set(anions.tolist()), key=str)
        if anion is not None and str(anion) != "<NA>"
    }
    return metrics


def _comparison(base: Mapping[str, object], candidate: Mapping[str, object]):
    return {
        "satisfaction_delta": float(
            candidate["satisfaction"] - base["satisfaction"]
        ),
        "rejection_delta": float(candidate["rejection"] - base["rejection"]),
        "minimum_kind_delta": float(
            _minimum_kind(candidate) - _minimum_kind(base)
        ),
        "worst_shared_anion_delta": _worst_anion_delta(base, candidate),
        "by_kind_delta": {
            kind: float(candidate["by_kind"][kind] - value)
            for kind, value in base["by_kind"].items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--real-sixfam", type=Path, required=True)
    parser.add_argument("--bad-sixfam", type=Path, required=True)
    parser.add_argument("--real-corrected", type=Path, required=True)
    parser.add_argument("--bad-corrected", type=Path, required=True)
    parser.add_argument("--real-guards", type=Path, required=True)
    parser.add_argument("--bad-guards", type=Path, required=True)
    parser.add_argument("--input-domain-audit", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--loko-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    input_paths = {
        "real_descriptors": args.real_descriptors,
        "bad_descriptors": args.bad_descriptors,
        "real_sixfam": args.real_sixfam,
        "bad_sixfam": args.bad_sixfam,
        "real_corrected": args.real_corrected,
        "bad_corrected": args.bad_corrected,
        "real_guards": args.real_guards,
        "bad_guards": args.bad_guards,
    }
    input_sha256 = {name: _sha256(path) for name, path in input_paths.items()}
    input_audit = json.loads(
        args.input_domain_audit.read_text(encoding="utf-8")
    )
    validate_input_domain_audit(input_audit, input_sha256)
    isolation_audit = verify_isolated_manifest(args.isolated_dir)
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    loko = json.loads(args.loko_report.read_text(encoding="utf-8"))
    if reference.get("protocol", {}).get("experiment") != EXPERIMENT_ID:
        raise SystemExit("reference experiment is incompatible")
    if loko.get("protocol", {}).get("experiment") != EXPERIMENT_ID:
        raise SystemExit("LOKO experiment is incompatible")
    if loko.get("provenance", {}).get("reference_report_sha256") != _sha256(
        args.reference_report
    ):
        raise SystemExit("LOKO report references a different law report")
    if reference.get("provenance", {}).get("input_sha256") != input_sha256:
        raise SystemExit("reference report input hashes are incompatible")

    recurrence = recurrent_semantic_rules(loko["folds"], min_count=2)
    consensus_rules = [
        entry["rule"] for entry in recurrence if entry["is_maximum_recurrence"]
    ]
    floor_key = str(reference["protocol"]["floor"])
    baseline_rules = reference["frontiers"]["existing_loop"][floor_key]["rules"]
    dr, cr, db, cb = load_next4_search_frames(
        args.isolated_dir,
        args.real_descriptors,
        args.bad_descriptors,
        args.real_sixfam,
        args.bad_sixfam,
        args.real_corrected,
        args.bad_corrected,
        args.real_guards,
        args.bad_guards,
    )
    split_results = {}
    for split, real, bad in (
        ("discovery", dr, db),
        ("calibration", cr, cb),
    ):
        baseline = _evaluate(real, bad, baseline_rules)
        candidate = _evaluate(real, bad, [*baseline_rules, *consensus_rules])
        split_results[split] = {
            "baseline": baseline,
            "consensus": candidate,
            "comparison": _comparison(baseline, candidate),
        }
    output = {
        "protocol": {
            "experiment": "np-next-20260801e-posthoc-consensus",
            "source_experiment": EXPERIMENT_ID,
            "post_hoc": True,
            "selection": "semantic rule identities with maximum recurrence across true LOKO refits",
            "minimum_reported_recurrence": 2,
            "threshold_refit": False,
            "calibration_role": "adaptively reused diagnostic",
            "lockbox_access": False,
        },
        "isolation_audit": isolation_audit,
        "recurring_rules": recurrence,
        "consensus_rules": consensus_rules,
        "results": split_results,
        "interpretation_gate": {
            "confirmed_new_law": False,
            "reason": "post-hoc recurrence diagnostic has no independent validation split",
        },
        "provenance": {
            "reference_report_sha256": _sha256(args.reference_report),
            "loko_report_sha256": _sha256(args.loko_report),
            "input_domain_audit_sha256": _sha256(args.input_domain_audit),
            "input_sha256": input_sha256,
            "implementation_sha256": _sha256(Path(__file__)),
            "design_sha256": _sha256(
                Path(__file__).resolve().parents[1]
                / "docs/plans/2026-08-01-loko-consensus-diagnostic.md"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    calibration = split_results["calibration"]["comparison"]
    print(
        f"consensus rules={len(consensus_rules)}; calibration rejection "
        f"{calibration['rejection_delta']:+.4f}; worst anion "
        f"{calibration['worst_shared_anion_delta']:+.4f}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
