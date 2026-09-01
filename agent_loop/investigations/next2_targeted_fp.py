#!/usr/bin/env python3
"""DFT-relaxed false-positive evaluation for the diagnosed V4 variant.

Applies the existing-loop 0.98 set, the as-discovered additive set (V0),
and the integer-valence-guarded variant (V4) to the frozen 295 DFT-relaxed
candidates with unknown-fails-closed semantics.  Guard columns come from
``guards_fp.parquet``; 62/295 compositions lack a guard value and therefore
fail closed whenever a guarded rule is evaluated — a conservative bias
disclosed in the report.
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

from better_law_search import apply_serialized_law_set  # noqa: E402
from law_falsification import apply_law_set_unknown_fails  # noqa: E402

FLOOR = "0.98"
TARGET_FEATURE = "p2vor_an_sa_like_fraction_max"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--fp-descriptors", type=Path, required=True)
    parser.add_argument("--fp-guards", type=Path, required=True)
    parser.add_argument("--law-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    started = time.time()
    source_fp = pd.read_parquet(args.features_dir / "false_positive.parquet")
    descriptors = pd.read_parquet(args.fp_descriptors).drop(
        columns=["split"], errors="ignore"
    )
    guards = pd.read_parquet(args.fp_guards)
    frame = source_fp.merge(
        descriptors, on="sid", how="left", validate="one_to_one"
    ).merge(guards, on="sid", how="left", validate="one_to_one")
    if len(frame) != len(source_fp):
        raise SystemExit("descriptor/guard merge changed the audit row set")

    report = json.loads(args.law_report.read_text(encoding="utf-8"))
    base_rules = report["frontiers"]["existing_loop"][FLOOR]["rules"]
    v0_rules = report["frontiers"]["additive_next2_loop"][FLOOR]["rules"]
    v4_rules = []
    for rule in v0_rules:
        record = dict(rule)
        if record["feature"] == TARGET_FEATURE:
            record["guard_feature"] = "z_an_abs"
            record["guard_side"] = "hi"
            record["guard_threshold"] = 0.99
            record["description"] = (
                f"if z_an_abs > 0.99 then ({record['description']})"
            )
        v4_rules.append(record)

    evaluations = {}
    for name, rules in (
        ("existing_loop", base_rules),
        ("V0_as_discovered", v0_rules),
        ("V4_integer_valence_only", v4_rules),
    ):
        passed, known = apply_law_set_unknown_fails(frame, rules)
        evaluations[name] = {
            "pass_rate": float(passed.mean()),
            "rejection_rate": float((~passed).mean()),
            "known_rate": float(known.mean()),
            "unknown_count": int((~known).sum()),
            "n": int(len(passed)),
        }
    output = {
        "protocol": {
            "experiment": "np-next-20260802",
            "population": "frozen 295 DFT-relaxed LeMat/ELEMENTA candidates",
            "threshold_refit": False,
            "missing_feature_semantics": "unknown fails closed",
            "identifier_output": False,
            "lockbox_access": False,
            "guard_coverage_note": (
                "62/295 audit compositions have no guard value (exotic "
                "composition-level valence ambiguity) and fail closed on "
                "guarded rules"
            ),
        },
        "evaluations": evaluations,
        "deltas_vs_existing": {
            name: float(
                evaluations[name]["pass_rate"]
                - evaluations["existing_loop"]["pass_rate"]
            )
            for name in ("V0_as_discovered", "V4_integer_valence_only")
        },
        "provenance": {
            "runtime_seconds": time.time() - started,
            "law_report_sha256": _hash_file(args.law_report),
            "implementation_sha256": _hash_file(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    for name, record in evaluations.items():
        print(
            f"{name}: pass {record['pass_rate']:.4f} "
            f"(known {record['known_rate']:.3f}, unknown {record['unknown_count']})",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
