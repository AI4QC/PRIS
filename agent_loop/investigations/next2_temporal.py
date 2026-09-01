#!/usr/bin/env python3
"""Post-2019 COD temporal holdout evaluation for np-next-20260802.

The cod_delta table is the post-2019 COD increment that matdata (and hence
the whole analysis set) does not contain — a genuinely untouched source.
This script featurizes those structures with the exact definitions used by
the search pipeline (P1/P2/P3/P5 via ``next_features``; Shannon/ECoN/MEFIR
via ``phys_law``/``elec_feat``/``geom_feat``; Pauling criteria via
``discriminate.criteria``; guards via ``next2_guards``), builds the same
S1-S5 perturbation lineage, and applies the frozen rule sets exactly once.

Modes: ``real`` and ``bad`` featurize; ``eval`` applies the frozen sets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MAX_SITES = 80


def _temporal_features(structure, valences) -> dict[str, float]:
    """All rule-required quantities for one holdout structure."""

    from discriminate import criteria
    from elec_feat import elec_feats
    from geom_feat import geom_feats
    from next_features import next_local_features
    from next2_guards import guard_features_from_structure
    from phys_law import phys_feats

    out: dict[str, float] = {}
    p235, _failures = next_local_features(structure, valences)
    out.update(p235)
    out.update(guard_features_from_structure(structure))
    for family in (phys_feats, elec_feats, geom_feats):
        try:
            result = family(structure, valences)
            if result:
                out.update(result)
        except Exception:
            pass
    try:
        crit = criteria(structure, valences)
        if crit:
            out.update(crit)
    except Exception:
        pass
    out["z_cat_max"] = float(max(v for v in valences if v > 0))
    return out


def _parse_and_check(cif_text: str):
    from pymatgen.core import Structure

    structure = Structure.from_str(cif_text, fmt="cif")
    if len(structure) > MAX_SITES or len(structure) < 2:
        return None, None
    from discriminate import guess_oxi

    valences, ok = guess_oxi(structure)
    if not ok:
        return None, None
    return structure, valences


def _real_worker(record: Mapping[str, object]):
    failures: Counter[str] = Counter()
    try:
        structure, valences = _parse_and_check(str(record["cif"]))
        if structure is None:
            failures["parse_or_valence"] += 1
            return None, failures
        out = _temporal_features(structure, valences)
        out["sid"] = record["sid"]
        out["split"] = "temporal_holdout"
        return out, failures
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
        return None, failures


def _bad_worker(record: Mapping[str, object]):
    from make_negatives import perturb, swapped_val
    from phys_law import seed_of

    failures: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    try:
        structure, valences = _parse_and_check(str(record["cif"]))
        if structure is None:
            failures["parse_or_valence"] += 1
            return rows, failures
        rng = np.random.default_rng(seed_of(str(record["sid"])))
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            changed = perturb(structure, kind, rng, valences)
            if changed is None:
                continue
            try:
                out = _temporal_features(changed, swapped_val(changed, valences))
                out.update(
                    sid=f"{record['sid']}_{kind}",
                    kind=kind,
                    parent=record["sid"],
                    split="temporal_holdout",
                )
                rows.append(out)
            except Exception as exc:
                failures[f"{kind}:{type(exc).__name__}"] += 1
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
    return rows, failures


def _load_holdout_records(features_dir: Path) -> list[dict[str, object]]:
    frame = pd.read_parquet(
        features_dir / "cod_delta.parquet",
        columns=["file", "year", "formula", "n_anion_kinds", "has_H", "cif_text"],
    )
    frame = frame[
        (frame["n_anion_kinds"] == 1) & (~frame["has_H"].astype(bool))
    ].reset_index(drop=True)
    return [
        {"sid": f"cod-{int(row.file)}", "cif": row.cif_text}
        for row in frame.itertuples(index=False)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_featurize(args) -> int:
    records = _load_holdout_records(args.features_dir)
    if args.limit:
        records = records[: args.limit]
    print(f"{args.mode}: {len(records):,} holdout records", flush=True)
    worker = _real_worker if args.mode == "real" else _bad_worker
    single = args.mode == "real"
    rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (payload, counter) in enumerate(
            executor.map(worker, records, chunksize=4), start=1
        ):
            failures.update(counter)
            if single:
                if payload is not None:
                    rows.append(payload)
            else:
                rows.extend(payload)
            if index % 200 == 0:
                print(f"  {index:,}/{len(records):,} -> {len(rows):,}", flush=True)
    if not rows:
        raise SystemExit("no descriptor rows were produced")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.out, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "np-next-20260802",
        "mode": args.mode,
        "population": "post-2019 COD increment absent from matdata (cod_delta)",
        "n_input_records": len(records),
        "n_output_rows": len(frame),
        "failure_counts": dict(sorted(failures.items())),
        "lockbox_access": False,
        "input_sha256": _sha256(args.features_dir / "cod_delta.parquet"),
        "implementation_sha256": _sha256(Path(__file__)),
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(frame):,} rows)", flush=True)
    print(f"failure counts: {dict(sorted(failures.items()))}", flush=True)
    return 0


def _rule_records():
    """Frozen rule sets, copied verbatim from the np-next-20260802 reports."""

    source = json.loads(
        Path(
            "$PRIS_LAW_TABLES/law_next2.json"
        ).read_text(encoding="utf-8")
    )
    floor = "0.98"
    existing = source["frontiers"]["existing_loop"][floor]["rules"]
    v0 = source["frontiers"]["additive_next2_loop"][floor]["rules"]
    v4 = []
    for rule in v0:
        record = dict(rule)
        if record["feature"] == "p2vor_an_sa_like_fraction_max":
            record["guard_feature"] = "z_an_abs"
            record["guard_side"] = "hi"
            record["guard_threshold"] = 0.99
            record["description"] = (
                f"if z_an_abs > 0.99 then ({record['description']})"
            )
        v4.append(record)
    return existing, v0, v4


def _run_eval(args) -> int:
    from better_law_search import apply_serialized_law_set, evaluate_masks

    real = pd.read_parquet(args.real_descriptors)
    bad = pd.read_parquet(args.bad_descriptors)
    existing, v0, v4 = _rule_records()
    evaluations = {}
    for name, rules in (
        ("existing_loop", existing),
        ("V0_as_discovered", v0),
        ("V4_integer_valence_only", v4),
    ):
        mask_real = apply_serialized_law_set(real, rules)
        mask_bad = apply_serialized_law_set(bad, rules)
        metrics = evaluate_masks(
            real_mask=mask_real,
            bad_mask=mask_bad,
            bad_groups=bad["parent"].to_numpy(),
            bad_kinds=bad["kind"].to_numpy(),
        )
        evaluations[name] = metrics
    report = {
        "protocol": {
            "experiment": "np-next-20260802",
            "population": (
                "post-2019 COD temporal holdout; disjoint from the analysis "
                "set by construction; never used to fit or select anything"
            ),
            "evaluation": "single frozen application of pre-registered rule sets",
            "missing_feature_offline_semantics": "pass/abstain",
            "lockbox_access": False,
        },
        "counts": {"real": len(real), "bad": len(bad)},
        "evaluations": evaluations,
        "deltas_vs_existing": {
            name: {
                "satisfaction": (
                    evaluations[name]["satisfaction"]
                    - evaluations["existing_loop"]["satisfaction"]
                ),
                "rejection": (
                    evaluations[name]["rejection"]
                    - evaluations["existing_loop"]["rejection"]
                ),
                "min_kind": (
                    min(evaluations[name]["by_kind"].values())
                    - min(evaluations["existing_loop"]["by_kind"].values())
                ),
            }
            for name in ("V0_as_discovered", "V4_integer_valence_only")
        },
        "provenance": {
            "real_descriptors_sha256": _sha256(args.real_descriptors),
            "bad_descriptors_sha256": _sha256(args.bad_descriptors),
            "implementation_sha256": _sha256(Path(__file__)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    for name, metrics in evaluations.items():
        print(
            f"{name}: satisfaction {metrics['satisfaction']:.4f} "
            f"exclusion {metrics['rejection']:.4f} "
            f"by-kind { {k: round(v, 3) for k, v in metrics['by_kind'].items()} }",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("real", "bad", "eval"))
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--real-descriptors", type=Path)
    parser.add_argument("--bad-descriptors", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    if args.mode == "eval":
        if args.real_descriptors is None or args.bad_descriptors is None:
            raise SystemExit("eval mode requires --real-descriptors and --bad-descriptors")
        return _run_eval(args)
    return _run_featurize(args)


if __name__ == "__main__":
    raise SystemExit(main())
