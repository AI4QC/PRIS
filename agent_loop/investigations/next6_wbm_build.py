#!/usr/bin/env python3
"""Build label-isolated WBM x0 feature artifacts without reading relaxed geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.next6_wbm_features import geometry_features, parse_extxyz
from src.next6_wbm_protocol import (
    formula_split,
    reduced_formula_key,
    stable_from_wbm_hull,
    wbm_stage,
)


DEFAULT_SUMMARY = Path(
    "<other-repo>/data/raw/acquired/matbench-discovery/"
    "2023-12-13-wbm-summary.csv.gz"
)
DEFAULT_INITIAL_ZIP = Path(
    "<other-repo>/data/raw/acquired/matbench-discovery/"
    "2024-08-04-wbm-initial-atoms.extxyz.zip"
)


def sha256_file(path: Path) -> str:
    """Hash file contents in bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_wbm_inventory(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter the official unique-prototype population and freeze labels/splits."""

    required = {
        "material_id",
        "formula",
        "unique_prototype",
        "e_above_hull_mp2020_corrected_ppd_mp",
        "site_stats_fingerprint_init_final_norm_diff",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"missing WBM summary columns: {missing}")
    inventory = raw.loc[raw["unique_prototype"].fillna(False).astype(bool)].copy()
    if inventory["material_id"].duplicated().any():
        duplicate = inventory.loc[inventory["material_id"].duplicated(), "material_id"].iloc[0]
        raise ValueError(f"duplicate material_id: {duplicate}")
    inventory["formula_key"] = inventory["formula"].map(reduced_formula_key)
    inventory["split"] = inventory["formula_key"].map(formula_split)
    inventory["stage"] = inventory["formula_key"].map(wbm_stage)
    inventory["stable"] = stable_from_wbm_hull(
        inventory["e_above_hull_mp2020_corrected_ppd_mp"]
    )
    return inventory.sort_values("material_id", kind="stable").reset_index(drop=True)


def _feature_one(item: tuple[str, str]) -> dict[str, object]:
    material_id, text = item
    frame = parse_extxyz(text)
    if frame.material_id != material_id:
        raise ValueError(
            f"zip member ID mismatch: expected {material_id}, found {frame.material_id}"
        )
    return {"material_id": material_id, **geometry_features(frame)}


def _zip_records(
    archive: zipfile.ZipFile,
    material_ids: list[str],
) -> Iterator[tuple[str, str]]:
    names = set(archive.namelist())
    for material_id in material_ids:
        member = f"{material_id}.extxyz"
        if member not in names:
            raise FileNotFoundError(f"initial WBM frame missing from zip: {member}")
        yield material_id, archive.read(member).decode("utf-8")


def _write_split_features(
    rows: Iterator[dict[str, object]],
    split_by_id: dict[str, str],
    output_dir: Path,
    *,
    batch_size: int = 5_000,
) -> None:
    stages = ("formula_selection", "threshold_calibration", "test")
    buffers: dict[str, list[dict[str, object]]] = {stage: [] for stage in stages}
    writers: dict[str, pq.ParquetWriter] = {}

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        table = pa.Table.from_pylist(buffers[split])
        path = output_dir / f"{split}_x0_features.parquet"
        if split not in writers:
            writers[split] = pq.ParquetWriter(path, table.schema, compression="zstd")
        writers[split].write_table(table)
        buffers[split].clear()

    try:
        for row in rows:
            split = split_by_id[str(row["material_id"])]
            buffers[split].append(row)
            if len(buffers[split]) >= batch_size:
                flush(split)
        for split in buffers:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()

    for split in buffers:
        path = output_dir / f"{split}_x0_features.parquet"
        if not path.exists():
            pd.DataFrame({"material_id": pd.Series(dtype="str")}).to_parquet(
                path, index=False
            )


def build_wbm_artifacts(
    *,
    summary_path: Path,
    initial_zip: Path,
    output_dir: Path,
    workers: int,
    limit: int | None = None,
) -> dict[str, object]:
    """Create physically separate calibration/test labels and x0 feature tables."""

    summary_path = Path(summary_path)
    initial_zip = Path(initial_zip)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = prepare_wbm_inventory(pd.read_csv(summary_path))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        inventory = inventory.head(limit).copy()

    label_columns = [
        column
        for column in (
            "material_id",
            "formula",
            "formula_key",
            "split",
            "stage",
            "e_above_hull_wbm",
            "e_above_hull_mp2020_corrected_ppd_mp",
            "site_stats_fingerprint_init_final_norm_diff",
            "stable",
        )
        if column in inventory.columns
    ]
    stages = ("formula_selection", "threshold_calibration", "test")
    for stage in stages:
        inventory.loc[inventory["stage"] == stage, label_columns].to_parquet(
            output_dir / f"{stage}_labels.parquet", index=False
        )

    material_ids = inventory["material_id"].astype(str).tolist()
    split_by_id = dict(zip(inventory["material_id"].astype(str), inventory["stage"]))
    with zipfile.ZipFile(initial_zip) as archive:
        records = _zip_records(archive, material_ids)
        if workers == 1:
            feature_rows = map(_feature_one, records)
            _write_split_features(feature_rows, split_by_id, output_dir)
        else:
            if workers <= 0:
                raise ValueError("workers must be positive")
            with mp.get_context("spawn").Pool(workers) as pool:
                feature_rows = pool.imap(_feature_one, records, chunksize=128)
                _write_split_features(feature_rows, split_by_id, output_dir)

    counts = inventory["stage"].value_counts().to_dict()
    counts_out = {
        "formula_selection": int(counts.get("formula_selection", 0)),
        "threshold_calibration": int(counts.get("threshold_calibration", 0)),
        "test": int(counts.get("test", 0)),
        "total": int(len(inventory)),
    }
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.glob("*.parquet"))
    }
    manifest: dict[str, object] = {
        "protocol": "2026-08-01-dft-pre-screening-design-v1",
        "input_role": "unrelaxed_x0_only",
        "counts": counts_out,
        "inputs": {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "initial_zip": str(initial_zip),
            "initial_zip_sha256": sha256_file(initial_zip),
        },
        "outputs_sha256": output_hashes,
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--initial-zip", type=Path, default=DEFAULT_INITIAL_ZIP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = build_wbm_artifacts(
        summary_path=args.summary,
        initial_zip=args.initial_zip,
        output_dir=args.output,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
