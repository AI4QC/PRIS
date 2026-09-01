#!/usr/bin/env python3
"""LOKO stability for the np-next-20260802 guard-extended law loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from better_law_stability import run_leave_one_kind_out  # noqa: E402
from next_law_stability import _rename_variant  # noqa: E402
from next2_law_search import (  # noqa: E402
    ADDITIVE_GUARDED_VARIANT,
    build_next2_candidate_sets,
    load_next2_search_frames,
)

LEGACY_ADDITIVE_NAME = "additive_bvloc_anion_guarded_loop"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_by_kind(
    path: Path | None,
    *,
    floor: float,
) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    source = json.loads(path.read_text(encoding="utf-8"))
    key = str(floor)
    return {
        "existing_loop": source["frontiers"]["existing_loop"][key]["discovery"][
            "by_kind"
        ],
        LEGACY_ADDITIVE_NAME: source["frontiers"][ADDITIVE_GUARDED_VARIANT][key][
            "discovery"
        ]["by_kind"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--real-descriptors", type=Path, required=True)
    parser.add_argument("--bad-descriptors", type=Path, required=True)
    parser.add_argument("--real-guards", type=Path, required=True)
    parser.add_argument("--bad-guards", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.98)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--max-rules", type=int, default=12)
    parser.add_argument("--max-guard-targets", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    report = run_leave_one_kind_out(
        dr,
        cr,
        db,
        cb,
        floor=args.floor,
        min_coverage=args.min_coverage,
        width=args.width,
        max_rules=args.max_rules,
        max_guard_targets=args.max_guard_targets,
        p1_vocabulary="frozen",
        paired_anion_guard=True,
        reference_by_kind=_reference_by_kind(
            args.reference_report,
            floor=args.floor,
        ),
        candidate_builder=build_next2_candidate_sets,
    )
    _rename_variant(report, LEGACY_ADDITIVE_NAME, ADDITIVE_GUARDED_VARIANT)
    report["protocol"]["experiment"] = "np-next-20260802"
    report["provenance"] = {
        "runtime_seconds": time.time() - started,
        "input_sha256": {
            "real_descriptors": _hash_file(args.real_descriptors),
            "bad_descriptors": _hash_file(args.bad_descriptors),
            "real_guards": _hash_file(args.real_guards),
            "bad_guards": _hash_file(args.bad_guards),
            "reference_report": (
                None
                if args.reference_report is None
                else _hash_file(args.reference_report)
            ),
        },
        "implementation_sha256": _hash_file(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
