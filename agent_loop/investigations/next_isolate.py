#!/usr/bin/env python3
"""Build physically isolated discovery/calibration source tables.

This is the single audited exception that scans the monolithic feature
parquets: it calls the existing loaders once, filters to the permitted
splits, asserts that no lockbox or unknown-split row survives, and writes
isolated tables to an external cache directory.  Every downstream step of
experiment np-next-20260801 loads only the isolated tables, so no later
code path can materialize a lockbox row.

Nothing in the repository or the feature store is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPECTED_COUNTS = {
    "law_real": {"discovery": 12632, "calibration": 5297},
    "law_bad": {"discovery": 8590, "calibration": 3612},
    "formula_rank": {"discovery": 3268, "calibration": 1348},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_clean(frame: pd.DataFrame, column: str, name: str) -> None:
    if column not in frame:
        raise ValueError(f"{name}: missing split column {column}")
    split = frame[column]
    if split.isna().any():
        raise ValueError(f"{name}: {int(split.isna().sum())} rows have unknown split")
    if split.eq("lockbox").any():
        raise ValueError(
            f"{name}: {int(split.eq('lockbox').sum())} lockbox rows survived"
        )
    observed = set(split.astype(str).unique())
    if not observed.issubset({"discovery", "calibration"}):
        raise ValueError(f"{name}: unexpected split labels {sorted(observed)}")


def _records_tables(
    features_dir: Path,
    law_real: pd.DataFrame,
    law_bad: pd.DataFrame,
    *,
    max_sites: int = 80,
) -> dict[str, pd.DataFrame]:
    """Blob-resolution records for the permitted rows only.

    The provenance scan is part of this builder's audited monolithic
    exception; the written tables contain only discovery/calibration rows.
    """
    provenance = pd.read_parquet(
        features_dir / "provenance.parquet",
        columns=["source_id", "blob_offset", "blob_length", "n_elements", "n_sites"],
    )
    records_real = law_real[["source_id", "split"]].merge(
        provenance, on="source_id", how="inner", validate="one_to_one", sort=False
    )
    records_real = records_real[
        (records_real["n_elements"] >= 2) & (records_real["n_sites"] <= max_sites)
    ].reset_index(drop=True)

    parents = (
        law_bad.groupby(["parent", "psplit"], sort=False)["kind"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .reset_index()
        .rename(columns={"psplit": "split", "kind": "kinds"})
    )
    records_bad = parents.merge(
        provenance.rename(columns={"source_id": "parent"}),
        on="parent",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    records_bad = records_bad[
        (records_bad["n_elements"] >= 2) & (records_bad["n_sites"] <= max_sites)
    ].reset_index(drop=True)
    return {"records_real": records_real, "records_bad": records_bad}


def build_isolated_tables(features_dir: Path, out_dir: Path) -> dict[str, object]:
    """Run the legacy loaders once and write lockbox-free tables."""
    import formula2
    import rules_final

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "law_real": out_dir / "law_real.parquet",
        "law_bad": out_dir / "law_bad.parquet",
        "formula_rank": out_dir / "formula_rank.parquet",
        "records_real": out_dir / "records_real.parquet",
        "records_bad": out_dir / "records_bad.parquet",
    }
    for path in outputs.values():
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")

    previous_rules = rules_final.F
    previous_formula = formula2.F
    rules_final.F = str(features_dir) + os.sep
    formula2.F = str(features_dir) + os.sep
    try:
        real, bad = rules_final.load(phys=True)
        rank = formula2.load(phys=True)
    finally:
        rules_final.F = previous_rules
        formula2.F = previous_formula

    law_real = real[real["split"].isin(("discovery", "calibration"))].reset_index(
        drop=True
    )
    law_bad = bad[bad["psplit"].isin(("discovery", "calibration"))].reset_index(
        drop=True
    )
    formula_rank = rank[rank["split"].isin(("discovery", "calibration"))].reset_index(
        drop=True
    )

    frames = {
        "law_real": (law_real, "split"),
        "law_bad": (law_bad, "psplit"),
        "formula_rank": (formula_rank, "split"),
    }
    for name, (frame, column) in frames.items():
        _assert_clean(frame, column, name)

    records = _records_tables(features_dir, law_real, law_bad)
    for name, frame in records.items():
        _assert_clean(frame, "split", name)
    frames = {**frames, **{k: (v, "split") for k, v in records.items()}}

    manifest: dict[str, object] = {
        "experiment": "np-next-20260801",
        "purpose": (
            "physically isolated discovery/calibration source tables; the "
            "single audited monolithic-scan exception of this round"
        ),
        "features_dir": str(features_dir),
        "tables": {},
        "input_sha256": {},
    }
    for name, (frame, column) in frames.items():
        path = outputs[name]
        frame.to_parquet(path, index=False)
        counts = {
            str(key): int(value)
            for key, value in frame[column].value_counts().sort_index().items()
        }
        if name in EXPECTED_COUNTS:
            expected = EXPECTED_COUNTS[name]
            if counts != expected:
                raise SystemExit(
                    f"{name}: row counts {counts} do not match the frozen split "
                    f"sizes {expected}; refusing to certify the isolated table"
                )
        manifest["tables"][name] = {
            "path": str(path),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "split_counts": counts,
            "sha256": _sha256(path),
        }

    for name in (
        "real_all.parquet",
        "phys_real.parquet",
        "phys_bad.parquet",
        "elec_real.parquet",
        "elec_bad.parquet",
        "geom_real.parquet",
        "geom_bad.parquet",
        "t0_guard.parquet",
        "splits.parquet",
        "real_rank.parquet",
        "provenance.parquet",
    ):
        path = features_dir / name
        if path.exists():
            manifest["input_sha256"][name] = _sha256(path)

    # Post-write verification: re-read from disk and re-assert purity.
    for name, (_, column) in frames.items():
        reread = pd.read_parquet(outputs[name], columns=[column])
        _assert_clean(reread, column, f"{name} (re-read)")

    manifest_path = out_dir / "isolated_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: v["split_counts"] for k, v in manifest["tables"].items()}))
    print(f"wrote isolated tables to {out_dir}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_isolated_tables(args.features_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
