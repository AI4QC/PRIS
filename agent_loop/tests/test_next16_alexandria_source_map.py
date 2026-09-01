"""Contracts for reconstructing official Alexandria benchmark provenance."""

from __future__ import annotations

import bz2
import json
from pathlib import Path

import pandas as pd
import pytest


def _write_shard(path: Path, entries: list[dict]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"entries": entries}, stream)


def _entry(material_id: str, location: str) -> dict:
    return {"data": {"mat_id": material_id, "location": location, "e_above_hull": 9.9}}


def test_location_classifier_preserves_the_benchmark_source_family() -> None:
    from src.next16_alexandria_source_map import classify_location

    assert classify_location("m3gnet/rng_004/O/foo") == "m3gnet/rng"
    assert classify_location("m3gnet/alignn_018/Sc/foo") == "m3gnet/alignn"
    assert classify_location("orbital/alignn_002/Ca/foo") == "orbital/alignn"
    assert classify_location("cgat_prot/ABC2_prot10_spg123/runs/foo") == "cgat_prot/abc2"
    assert classify_location("") == "unknown"


def test_source_map_is_identifier_complete_and_does_not_publish_labels(tmp_path: Path) -> None:
    from src import next16_alexandria_source_map as module

    benchmark = tmp_path / "benchmarks_pbe.csv"
    benchmark.write_text("# mat_id\na1\na2\na2\na3\n", encoding="utf-8")
    database = tmp_path / "database"
    database.mkdir()
    _write_shard(
        database / "alexandria_00000.json.bz2",
        [
            _entry("a1", "m3gnet/rng_001/O/a1"),
            _entry("not-wanted", "m3gnet/alignn_001/O/x"),
            _entry("a2", "orbital/alignn_002/Ca/a2"),
        ],
    )
    _write_shard(
        database / "alexandria_00001.json.bz2",
        [_entry("a2", "orbital/alignn_002/Ca/a2"), _entry("a3", "m3gnet/rng_004/N/a3")],
    )
    output = tmp_path / "output"
    manifest = module.build_source_map(
        benchmark_ids_path=benchmark,
        database_dir=database,
        output_dir=output,
        workers=1,
    )
    table = pd.read_parquet(output / module.OUTPUT_NAME)
    assert table.to_dict("records") == [
        {"material_id": "a1", "source_cohort": "m3gnet/rng", "location": "m3gnet/rng_001/O/a1"},
        {"material_id": "a2", "source_cohort": "orbital/alignn", "location": "orbital/alignn_002/Ca/a2"},
        {"material_id": "a3", "source_cohort": "m3gnet/rng", "location": "m3gnet/rng_004/N/a3"},
    ]
    assert manifest["counts"]["m3gnet_rng"] == 2
    assert manifest["counts"]["benchmark_rows"] == 4
    assert manifest["counts"]["benchmark_unique_ids"] == 3
    assert manifest["counts"]["benchmark_duplicate_rows"] == 1
    assert manifest["endpoint_bytes_read_by_source_mapper"] is True
    assert "e_above_hull" not in table.columns
    assert b"9.9" not in (output / module.OUTPUT_NAME).read_bytes()
    with pytest.raises(FileExistsError):
        module.build_source_map(
            benchmark_ids_path=benchmark,
            database_dir=database,
            output_dir=output,
            workers=1,
        )


def test_source_map_refuses_missing_or_conflicting_benchmark_identity(tmp_path: Path) -> None:
    from src.next16_alexandria_source_map import build_source_map

    benchmark = tmp_path / "benchmarks_pbe.csv"
    benchmark.write_text("# mat_id\na1\na2\n", encoding="utf-8")
    database = tmp_path / "database"
    database.mkdir()
    _write_shard(database / "alexandria_00000.json.bz2", [_entry("a1", "m3gnet/rng_001/O/a1")])
    with pytest.raises(ValueError, match="missing 1 benchmark"):
        build_source_map(
            benchmark_ids_path=benchmark,
            database_dir=database,
            output_dir=tmp_path / "missing",
            workers=1,
        )


def test_source_map_records_the_single_known_official_orphan(tmp_path: Path) -> None:
    from src import next16_alexandria_source_map as module

    orphan = next(iter(module.KNOWN_OFFICIAL_ORPHAN_IDS))
    benchmark = tmp_path / "benchmarks_pbe.csv"
    benchmark.write_text(f"# mat_id\na1\n{orphan}\n", encoding="utf-8")
    database = tmp_path / "database"
    database.mkdir()
    _write_shard(database / "alexandria_00000.json.bz2", [_entry("a1", "m3gnet/rng_001/O/a1")])
    manifest = module.build_source_map(
        benchmark_ids_path=benchmark,
        database_dir=database,
        output_dir=tmp_path / "output",
        workers=1,
    )
    assert manifest["counts"]["official_orphan_ids"] == [orphan]
    assert manifest["counts"]["mapped_ids"] == 1

    _write_shard(database / "alexandria_00001.json.bz2", [_entry("a1", "orbital/alignn_001/O/a1"), _entry("a2", "m3gnet/rng_002/O/a2")])
    with pytest.raises(ValueError, match="conflicting locations"):
        module.build_source_map(
            benchmark_ids_path=benchmark,
            database_dir=database,
            output_dir=tmp_path / "conflict",
            workers=1,
        )
