#!/usr/bin/env python3
"""Lockbox confirmation for the frozen V4 rule set (np-next-20260802).

Runs exactly once under the authorised opening recorded in
``lockbox/openings.log`` (opening_index=1).  Featurizes the lockbox real
structures and their S1-S5 perturbation lineage with the same definitions
as the search pipeline, then applies the frozen rule sets a single time.

The lockbox rows are read here — that is the purpose of the authorised
opening — and every aggregate emitted is identifier-free.
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

LOCKDIR = Path("$PRIS_ARCHIVE/lockbox")
MAX_SITES = 80


def _load_lockbox_sids(opening_index: int = 1) -> list[str]:
    path = LOCKDIR / f"lockbox_sids_opening{opening_index}.txt"
    if not path.exists():
        raise SystemExit(
            f"missing {path}; run seal_lockbox.py --open first (authorised)"
        )
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _records_real(features_dir: Path, sids: list[str]) -> list[dict[str, object]]:
    provenance = pd.read_parquet(
        features_dir / "provenance.parquet",
        columns=["source_id", "blob_offset", "blob_length", "n_elements", "n_sites"],
    )
    frame = provenance[provenance["source_id"].isin(set(sids))]
    frame = frame[
        (frame["n_elements"] >= 2) & (frame["n_sites"] <= MAX_SITES)
    ].reset_index(drop=True)
    return [
        {
            "sid": row.source_id,
            "off": int(row.blob_offset),
            "ln": int(row.blob_length),
        }
        for row in frame.itertuples(index=False)
    ]


def _records_bad(features_dir: Path, sids: list[str]) -> list[dict[str, object]]:
    """The exact S1-S5 lineage of the lockbox parents in phys_bad."""

    bad = pd.read_parquet(
        features_dir / "phys_bad.parquet", columns=["sid", "kind", "parent"]
    )
    bad = bad[bad["parent"].isin(set(sids))]
    parents = (
        bad.groupby("parent", sort=False)["kind"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .reset_index()
        .rename(columns={"kind": "kinds"})
    )
    provenance = pd.read_parquet(
        features_dir / "provenance.parquet",
        columns=["source_id", "blob_offset", "blob_length", "n_elements", "n_sites"],
    ).rename(columns={"source_id": "parent"})
    parents = parents.merge(
        provenance, on="parent", how="inner", validate="one_to_one", sort=False
    )
    parents = parents[
        (parents["n_elements"] >= 2) & (parents["n_sites"] <= MAX_SITES)
    ].reset_index(drop=True)
    return [
        {
            "sid": row.parent,
            "off": int(row.blob_offset),
            "ln": int(row.blob_length),
            "kinds": str(row.kinds),
        }
        for row in parents.itertuples(index=False)
    ]


def _real_worker(record: Mapping[str, object]):
    from pymatgen.core import Structure

    from discriminate import guess_oxi, read_blob_cif
    from next2_temporal import _temporal_features

    failures: Counter[str] = Counter()
    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])), fmt="cif"
        )
        valences, ok = guess_oxi(structure)
        if not ok:
            failures["valence:guess_oxi"] += 1
            return None, failures
        out = _temporal_features(structure, valences)
        out["sid"] = record["sid"]
        out["split"] = "lockbox"
        return out, failures
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
        return None, failures


def _bad_worker(record: Mapping[str, object]):
    from pymatgen.core import Structure

    from discriminate import guess_oxi, read_blob_cif
    from make_negatives import perturb, swapped_val
    from next2_temporal import _temporal_features
    from phys_law import seed_of

    failures: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])), fmt="cif"
        )
        valences, ok = guess_oxi(structure)
        if not ok:
            failures["valence:guess_oxi"] += 1
            return rows, failures
        rng = np.random.default_rng(seed_of(str(record["sid"])))
        wanted = set(str(record["kinds"]).split(","))
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            changed = perturb(structure, kind, rng, valences)
            if changed is None or kind not in wanted:
                continue
            try:
                out = _temporal_features(changed, swapped_val(changed, valences))
                out.update(
                    sid=f"{record['sid']}_{kind}",
                    kind=kind,
                    parent=record["sid"],
                    split="lockbox",
                )
                rows.append(out)
            except Exception as exc:
                failures[f"{kind}:{type(exc).__name__}"] += 1
    except Exception as exc:
        failures[f"structure:{type(exc).__name__}"] += 1
    return rows, failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_featurize(args, sids: list[str]) -> int:
    records = (
        _records_real(args.features_dir, sids)
        if args.mode == "real"
        else _records_bad(args.features_dir, sids)
    )
    if args.limit:
        records = records[: args.limit]
    print(f"{args.mode}: {len(records):,} lockbox records", flush=True)
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
            if index % 500 == 0:
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
        "population": "sealed lockbox split, authorised opening_index=1",
        "n_input_records": len(records),
        "n_output_rows": len(frame),
        "failure_counts": dict(sorted(failures.items())),
        "lockbox_access": True,
        "lockbox_opening_index": 1,
        "implementation_sha256": _sha256(Path(__file__)),
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(frame):,} rows)", flush=True)
    print(f"failure counts: {dict(sorted(failures.items()))}", flush=True)
    return 0


def _run_eval(args) -> int:
    from better_law_search import apply_serialized_law_set, evaluate_masks
    from next2_temporal import _rule_records

    real = pd.read_parquet(args.real_descriptors)
    bad = pd.read_parquet(args.bad_descriptors)
    existing, v0, v4 = _rule_records()
    evaluations = {}
    for name, rules in (
        ("existing_loop", existing),
        ("V0_as_discovered", v0),
        ("V4_integer_valence_only", v4),
    ):
        metrics = evaluate_masks(
            real_mask=apply_serialized_law_set(real, rules),
            bad_mask=apply_serialized_law_set(bad, rules),
            bad_groups=bad["parent"].to_numpy(),
            bad_kinds=bad["kind"].to_numpy(),
        )
        evaluations[name] = metrics
    report = {
        "protocol": {
            "experiment": "np-next-20260802",
            "population": (
                "sealed lockbox split (5,748 structures; 15% of the analysis "
                "set), evaluated once under authorised opening_index=1"
            ),
            "evaluation": "single frozen application of pre-registered rule sets",
            "missing_feature_offline_semantics": "pass/abstain",
            "lockbox_access": True,
            "lockbox_opening_index": 1,
            "v4_ruleset_sha256": (
                "ef79bda693b7ddcd290028f2a45bb2071649fc07a38b92803c5f28d30cb1d0b2"
            ),
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
            raise SystemExit(
                "eval mode requires --real-descriptors and --bad-descriptors"
            )
        return _run_eval(args)
    sids = _load_lockbox_sids()
    return _run_featurize(args, sids)


if __name__ == "__main__":
    raise SystemExit(main())
