#!/usr/bin/env python3
"""Composition-only valence-confidence guard columns for np-next-20260802.

Computes the three frozen guard columns — ``z_an_abs``, ``oxi_n_guesses``,
``oxi_unique`` — from the blob CIFs of the physically isolated records only
(no monolithic split-bearing table is read).  Perturbations preserve
composition, so bad rows inherit their parent's values at merge time.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GARD_COLUMNS = ("z_an_abs", "oxi_n_guesses", "oxi_unique")
MAX_GUESSES = 20


def guard_features_from_structure(structure) -> dict[str, float]:
    """Frozen guard columns from one structure's composition only."""

    from pymatgen.core import Composition

    composition = Composition(structure.composition.reduced_formula)
    try:
        guesses = composition.oxi_state_guesses(max_sites=-10)
    except Exception:
        guesses = []
    if not guesses:
        return {}
    assignment = guesses[0]
    negatives = {element: value for element, value in assignment.items() if value < 0}
    if len(negatives) != 1:
        return {}
    z_an = abs(float(next(iter(negatives.values()))))
    return {
        "z_an_abs": z_an,
        "oxi_n_guesses": float(min(len(guesses), MAX_GUESSES)),
        "oxi_unique": float(len(guesses) == 1),
    }


def _worker(record: Mapping[str, object]):
    from pymatgen.core import Structure

    from discriminate import read_blob_cif

    try:
        structure = Structure.from_str(
            read_blob_cif(int(record["off"]), int(record["ln"])),
            fmt="cif",
        )
        out = guard_features_from_structure(structure)
        out[record["key"]] = record["id"]
        return out if len(out) == len(GARD_COLUMNS) + 1 else None
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(
    isolated_dir: Path,
    mode: str,
    *,
    features_dir: Path | None,
    materials_database: Path | None,
) -> tuple[str, list[dict[str, object]]]:
    if mode == "real":
        frame = pd.read_parquet(isolated_dir / "records_real.parquet")
        return "source_id", [
            {
                "id": row.source_id,
                "off": int(row.blob_offset),
                "ln": int(row.blob_length),
                "split": str(row.split),
            }
            for row in frame.itertuples(index=False)
        ]
    if mode == "bad":
        frame = pd.read_parquet(isolated_dir / "records_bad.parquet")
        return "parent", [
            {
                "id": row.parent,
                "off": int(row.blob_offset),
                "ln": int(row.blob_length),
                "split": str(row.split),
            }
            for row in frame.itertuples(index=False)
        ]
    if mode == "false-positive":
        if materials_database is None or features_dir is None:
            raise SystemExit(
                "false-positive mode requires --materials-database and --features-dir"
            )
        audit_ids = (
            pd.read_parquet(features_dir / "false_positive.parquet", columns=["sid"])[
                "sid"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
        resolved: dict[str, tuple[int, int]] = {}
        connection = sqlite3.connect(f"file:{materials_database}?mode=ro", uri=True)
        try:
            for start in range(0, len(audit_ids), 500):
                batch = audit_ids[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                query = (
                    "SELECT material_id, blob_offset, blob_length FROM materials "
                    f"WHERE material_id IN ({placeholders})"
                )
                for material_id, offset, length in connection.execute(query, batch):
                    resolved[str(material_id)] = (int(offset), int(length))
        finally:
            connection.close()
        return "sid", [
            {
                "id": sid,
                "off": resolved[sid][0],
                "ln": resolved[sid][1],
                "split": "false_positive_audit",
            }
            for sid in audit_ids
            if sid in resolved
        ]
    raise SystemExit(f"unknown mode: {mode}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("real", "bad", "false-positive"))
    parser.add_argument("--isolated-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--materials-database", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    key, records = _records(
        args.isolated_dir,
        args.mode,
        features_dir=args.features_dir,
        materials_database=args.materials_database,
    )
    print(f"{args.mode}: resolved {len(records):,} records", flush=True)
    for record in records:
        record["key"] = key

    rows: list[dict[str, object]] = []
    n_failed = 0
    if args.workers == 1:
        results = map(_worker, records)
        for result in results:
            if result is not None:
                rows.append(result)
            else:
                n_failed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(_worker, records, chunksize=16):
                if result is not None:
                    rows.append(result)
                else:
                    n_failed += 1
    if not rows:
        raise SystemExit("no guard rows were produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame[key].duplicated().any():
        raise SystemExit(f"duplicate guard keys for {key}")
    frame.to_parquet(args.out, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "np-next-20260802",
        "mode": args.mode,
        "key": key,
        "guard_columns": list(GARD_COLUMNS),
        "max_guesses_cap": MAX_GUESSES,
        "n_input_records": len(records),
        "n_output_rows": len(frame),
        "n_failed": n_failed,
        "lockbox_access": False,
        "source_access_note": (
            "guard rows are computed from blob CIFs of the physically isolated "
            "records; no monolithic split-bearing table is read"
            if args.mode != "false-positive"
            else "false-positive audit inputs contain no experimental lockbox rows"
        ),
        "implementation_sha256": _sha256(Path(__file__)),
    }
    metadata_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.out} ({len(frame):,} rows, {n_failed} failed)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
