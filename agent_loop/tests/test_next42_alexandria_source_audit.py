"""Contracts for the NEXT42 raw-x0 Alexandria source qualification audit."""

from __future__ import annotations

import bz2
import json
from pathlib import Path

import pandas as pd
import pytest


def _write_path_shard(path: Path, material_ids: list[str]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as stream:
        json.dump({material_id: [{"steps": []}] for material_id in material_ids}, stream)


def _write_database_shard(path: Path, rows: list[tuple[str, str]]) -> None:
    payload = {
        "entries": [
            {
                "data": {
                    "mat_id": material_id,
                    "location": location,
                    "e_above_hull": 9876.54321,
                    "energy_total": -1234.5,
                }
            }
            for material_id, location in rows
        ]
    }
    with bz2.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def test_source_qualification_is_conservative_and_explicit() -> None:
    from src.next42_alexandria_source_audit import source_qualification

    assert source_qualification("cgat_comp/ternaries", official_benchmark=False) == (
        True,
        "eligible_round1_raw_x0",
    )
    assert source_qualification("cgat_comp/binaries", official_benchmark=False) == (
        True,
        "eligible_round1_raw_x0",
    )
    assert source_qualification("cgat_comp/quaternaries", official_benchmark=False) == (
        False,
        "documented_mlip_prerelaxation",
    )
    assert source_qualification("cgat_comp_2/ternaries", official_benchmark=False) == (
        False,
        "documented_mlip_prerelaxation",
    )
    assert source_qualification("orbital/alignn", official_benchmark=False) == (
        False,
        "documented_mlip_prerelaxation",
    )
    assert source_qualification("extra/batch-3000", official_benchmark=False) == (
        False,
        "unverified_raw_x0_provenance",
    )
    assert source_qualification("cgat_comp/ternaries", official_benchmark=True) == (
        False,
        "official_benchmark_identity",
    )


def test_source_audit_emits_only_identity_provenance_and_qualification(tmp_path: Path) -> None:
    from src import next42_alexandria_source_audit as module

    shard0 = tmp_path / "pbe_0000.json.bz2"
    shard1 = tmp_path / "pbe_0001.json.bz2"
    _write_path_shard(shard0, ["raw-ter", "round2", "orb"])
    _write_path_shard(shard1, ["raw-bin", "unknown", "benchmark"])
    benchmark = tmp_path / "benchmarks_pbe.csv"
    benchmark.write_text("# mat_id\nbenchmark\nnot-in-path\n", encoding="utf-8")
    database = tmp_path / "database"
    database.mkdir()
    _write_database_shard(
        database / "alexandria_00000.json.bz2",
        [
            ("raw-ter", "cgat_comp/ternaries/ABC/runs/raw-ter"),
            ("round2", "cgat_comp/quaternaries/ABCD/runs/round2"),
            ("orb", "orbital/alignn_001/O/orb"),
        ],
    )
    _write_database_shard(
        database / "alexandria_00001.json.bz2",
        [
            ("raw-bin", "cgat_comp/binaries/AB/runs/raw-bin"),
            ("unknown", "extra/batch-3000/Mg/unknown"),
            ("benchmark", "cgat_comp/ternaries/ABC/runs/benchmark"),
            ("not-in-path", "cgat_comp/ternaries/ABC/runs/not-in-path"),
        ],
    )
    output = tmp_path / "audit"
    manifest = module.build_source_audit(
        shard_0000_path=shard0,
        shard_0001_path=shard1,
        benchmark_ids_path=benchmark,
        database_dir=database,
        output_dir=output,
        workers=1,
        require_formal_inputs=False,
        expected_path_rows=6,
    )
    table = pd.read_parquet(output / module.OUTPUT_NAME)
    assert table.material_id.tolist() == [
        "benchmark",
        "orb",
        "raw-bin",
        "raw-ter",
        "round2",
        "unknown",
    ]
    assert table.loc[table.raw_x0_eligible, "material_id"].tolist() == [
        "raw-bin",
        "raw-ter",
    ]
    assert manifest["counts"]["path_rows"] == 6
    assert manifest["counts"]["official_benchmark_overlap"] == 1
    assert manifest["counts"]["raw_x0_eligible"] == 2
    assert manifest["scientific_labels_emitted"] is False
    assert manifest["source_qualification_frozen_before_endpoint_evaluation"] is True
    assert set(table.columns) == {
        "material_id",
        "source_family",
        "location",
        "official_benchmark",
        "raw_x0_eligible",
        "qualification_reason",
    }
    payload = (output / module.OUTPUT_NAME).read_bytes()
    assert b"9876.54321" not in payload
    assert b"-1234.5" not in payload
    with pytest.raises(FileExistsError):
        module.build_source_audit(
            shard_0000_path=shard0,
            shard_0001_path=shard1,
            benchmark_ids_path=benchmark,
            database_dir=database,
            output_dir=output,
            workers=1,
            require_formal_inputs=False,
            expected_path_rows=6,
        )


def test_source_audit_fails_closed_on_missing_or_duplicate_identity(tmp_path: Path) -> None:
    from src.next42_alexandria_source_audit import build_source_audit

    shard0 = tmp_path / "pbe_0000.json.bz2"
    shard1 = tmp_path / "pbe_0001.json.bz2"
    _write_path_shard(shard0, ["a", "b"])
    _write_path_shard(shard1, ["c"])
    benchmark = tmp_path / "benchmarks_pbe.csv"
    benchmark.write_text("# mat_id\n", encoding="utf-8")
    database = tmp_path / "database"
    database.mkdir()
    _write_database_shard(
        database / "alexandria_00000.json.bz2",
        [("a", "cgat_comp/ternaries/x"), ("b", "cgat_comp/ternaries/y")],
    )
    with pytest.raises(ValueError, match="missing 1 path identities"):
        build_source_audit(
            shard_0000_path=shard0,
            shard_0001_path=shard1,
            benchmark_ids_path=benchmark,
            database_dir=database,
            output_dir=tmp_path / "missing",
            workers=1,
            require_formal_inputs=False,
            expected_path_rows=3,
        )

    _write_database_shard(
        database / "alexandria_00001.json.bz2",
        [("a", "orbital/alignn/x"), ("c", "cgat_comp/binaries/z")],
    )
    with pytest.raises(ValueError, match="conflicting locations"):
        build_source_audit(
            shard_0000_path=shard0,
            shard_0001_path=shard1,
            benchmark_ids_path=benchmark,
            database_dir=database,
            output_dir=tmp_path / "conflict",
            workers=1,
            require_formal_inputs=False,
            expected_path_rows=3,
        )
