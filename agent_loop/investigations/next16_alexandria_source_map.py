"""Reconstruct Alexandria benchmark source cohorts from final-database locations."""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Sequence

import pandas as pd

from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace


PROTOCOL = "2026-08-02-next16-alexandria-benchmark-source-map-v1"
OUTPUT_NAME = "alexandria_benchmark_sources.parquet"
MANIFEST_NAME = "MANIFEST.json"
KNOWN_OFFICIAL_ORPHAN_IDS = frozenset({"agm002135948"})


def classify_location(location: object) -> str:
    """Reduce an Alexandria location to its workflow/source family."""

    if type(location) is not str or not location.strip():
        return "unknown"
    parts = location.strip().split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return "unknown"
    root = parts[0].lower()
    family = parts[1].split("_", 1)[0].lower()
    if root == "orbital":
        root = "orbital"
    return f"{root}/{family}"


def _load_benchmark_ids(path: Path) -> tuple[set[str], int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "# mat_id":
        raise ValueError("Alexandria benchmark header differs")
    values = [line.strip() for line in lines[1:] if line.strip()]
    if not values or any(not value for value in values):
        raise ValueError("Alexandria benchmark IDs must be nonempty")
    return set(values), len(values)


def _scan_shard(job: tuple[Path, set[str]]) -> tuple[dict[str, object], list[tuple[str, str]]]:
    path, wanted = job
    digest = _sha256_file(path)
    with bz2.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"Alexandria shard lacks entries: {path}")
    rows: list[tuple[str, str]] = []
    for entry in entries:
        data = entry.get("data") if isinstance(entry, dict) else None
        if not isinstance(data, dict):
            continue
        material_id = data.get("mat_id")
        if material_id not in wanted:
            continue
        location = data.get("location")
        if type(location) is not str or not location:
            raise ValueError(f"benchmark entry lacks location: {material_id}")
        rows.append((str(material_id), location))
    identity = {"name": path.name, "bytes": path.stat().st_size, "sha256": digest}
    return identity, rows


def _scan_all(
    paths: Iterable[Path], wanted: set[str], *, workers: int
) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    jobs = [(path, wanted) for path in paths]
    identities: list[dict[str, object]] = []
    rows: list[tuple[str, str]] = []
    if workers == 1:
        results = map(_scan_shard, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_scan_shard, jobs)
    try:
        for identity, part in results:
            identities.append(identity)
            rows.extend(part)
    finally:
        if workers != 1:
            executor.shutdown()
    return identities, rows


def build_source_map(
    *, benchmark_ids_path: Path, database_dir: Path, output_dir: Path, workers: int = 6
) -> dict[str, object]:
    """Map every official benchmark ID without publishing an endpoint label."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive exact integer")
    benchmark_path = Path(benchmark_ids_path).resolve()
    database = Path(database_dir).resolve()
    if not benchmark_path.is_file() or not database.is_dir():
        raise FileNotFoundError("benchmark file or Alexandria database directory is missing")
    wanted, benchmark_rows = _load_benchmark_ids(benchmark_path)
    shards = sorted(database.glob("alexandria_*.json.bz2"))
    if not shards:
        raise FileNotFoundError("no Alexandria database shards found")
    identities, raw_rows = _scan_all(shards, wanted, workers=workers)

    locations: dict[str, str] = {}
    for material_id, location in raw_rows:
        previous = locations.setdefault(material_id, location)
        if previous != location:
            raise ValueError(f"benchmark material has conflicting locations: {material_id}")
    missing = wanted - set(locations)
    unexpected_missing = missing - KNOWN_OFFICIAL_ORPHAN_IDS
    if unexpected_missing:
        raise ValueError(
            f"missing {len(unexpected_missing)} benchmark IDs from Alexandria database"
        )
    table = pd.DataFrame(
        [
            {
                "material_id": material_id,
                "source_cohort": classify_location(location),
                "location": location,
            }
            for material_id, location in sorted(locations.items())
        ]
    )
    counts = Counter(table["source_cohort"].astype(str))
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "endpoint_bytes_read_by_source_mapper": True,
        "endpoint_labels_emitted": False,
        "selection_fields_emitted": ["material_id", "source_cohort", "location"],
        "inputs": {
            "benchmark_ids": {
                "path": str(benchmark_path),
                "bytes": benchmark_path.stat().st_size,
                "sha256": _sha256_file(benchmark_path),
            },
            "database_dir": str(database),
            "database_shards": sorted(identities, key=lambda value: str(value["name"])),
        },
        "counts": {
            "benchmark_rows": benchmark_rows,
            "benchmark_unique_ids": len(wanted),
            "benchmark_duplicate_rows": benchmark_rows - len(wanted),
            "official_orphan_ids": sorted(missing),
            "mapped_ids": len(table),
            "source_cohorts": dict(sorted(counts.items())),
            "m3gnet_rng": int(counts.get("m3gnet/rng", 0)),
        },
        "executed_source_sha256": {
            "src/next16_alexandria_source_map.py": _sha256_file(Path(__file__).resolve())
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / OUTPUT_NAME
        table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-ids", required=True, type=Path)
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args(argv)
    build_source_map(
        benchmark_ids_path=arguments.benchmark_ids,
        database_dir=arguments.database_dir,
        output_dir=arguments.output_dir,
        workers=arguments.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
