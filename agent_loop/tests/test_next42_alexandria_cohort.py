"""Contracts for freezing geometry-only raw-x0 NEXT42 Alexandria rows."""

from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
from pymatgen.core import Lattice, Structure
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calculation(structure: Structure, marker: float) -> list[dict]:
    final = structure.copy()
    final.translate_sites([0], [0.05, 0.0, 0.0], frac_coords=False)
    return [
        {
            "steps": [
                {
                    "structure": structure.as_dict(),
                    "energy": marker,
                    "forces": [[9.0, 9.0, 9.0]] * len(structure),
                    "stress": [[8.0] * 3] * 3,
                },
                {
                    "structure": final.as_dict(),
                    "energy": marker + 1.0,
                    "forces": [[0.0, 0.0, 0.0]] * len(structure),
                    "stress": [[0.0] * 3] * 3,
                },
            ]
        }
    ]


def _write_shard(path: Path, rows: dict[str, list[dict]]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as stream:
        json.dump(rows, stream, sort_keys=True)


def _source_audit(
    tmp_path: Path, shard0: Path, shard1: Path, *, include_missing: bool = False
) -> tuple[Path, Path]:
    from src.next42_alexandria_source_audit import OUTPUT_NAME, PROTOCOL

    table = pd.DataFrame(
        [
            {
                "material_id": "a",
                "source_family": "cgat_comp/ternaries",
                "location": "cgat_comp/ternaries/a",
                "official_benchmark": False,
                "raw_x0_eligible": True,
                "qualification_reason": "eligible_round1_raw_x0",
            },
            {
                "material_id": "b",
                "source_family": "orbital/alignn",
                "location": "orbital/alignn/b",
                "official_benchmark": False,
                "raw_x0_eligible": False,
                "qualification_reason": "documented_mlip_prerelaxation",
            },
            {
                "material_id": "c",
                "source_family": "cgat_comp/binaries",
                "location": "cgat_comp/binaries/c",
                "official_benchmark": False,
                "raw_x0_eligible": True,
                "qualification_reason": "eligible_round1_raw_x0",
            },
        ]
    )
    if include_missing:
        table.loc[len(table)] = {
            "material_id": "missing",
            "source_family": "cgat_comp/ternaries",
            "location": "cgat_comp/ternaries/missing",
            "official_benchmark": False,
            "raw_x0_eligible": True,
            "qualification_reason": "eligible_round1_raw_x0",
        }
    audit = tmp_path / ("source-missing" if include_missing else "source")
    audit.mkdir()
    table_path = audit / OUTPUT_NAME
    table.to_parquet(table_path, index=False)
    manifest = {
        "protocol": PROTOCOL,
        "scientific_labels_emitted": False,
        "source_qualification_frozen_before_endpoint_evaluation": True,
        "trajectory_endpoint_values_accessed_for_qualification": False,
        "raw_x0_source_families": ["cgat_comp/binaries", "cgat_comp/ternaries"],
        "inputs": {
            "fixed_sha256": {
                "pbe_0000": _sha(shard0),
                "pbe_0001": _sha(shard1),
                "benchmarks_pbe": "test-only",
            }
        },
        "outputs_sha256": {OUTPUT_NAME: _sha(table_path)},
    }
    manifest_path = audit / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return table_path, manifest_path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    lattice = Lattice.cubic(5.0)
    a = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    b = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    c = Structure(lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    shard0 = tmp_path / "pbe_0000.json.bz2"
    shard1 = tmp_path / "pbe_0001.json.bz2"
    _write_shard(shard0, {"a": _calculation(a, 12345.25), "b": _calculation(b, 23456.25)})
    _write_shard(shard1, {"c": _calculation(c, 34567.25)})
    return shard0, shard1


def test_cohort_freezes_every_eligible_raw_x0_and_strips_endpoints(tmp_path: Path) -> None:
    from src import next42_alexandria_cohort as module

    shard0, shard1 = _inputs(tmp_path)
    table, audit_manifest = _source_audit(tmp_path, shard0, shard1)
    target = tmp_path / "cohort"
    manifest = module.build_next42_cohort(
        shard_0000_path=shard0,
        shard_0001_path=shard1,
        source_table_path=table,
        source_manifest_path=audit_manifest,
        output_dir=target,
    )
    metadata = pd.read_parquet(target / module.COHORT_NAME)
    assert metadata.material_id.tolist() == ["a", "c"]
    assert metadata.source_family.tolist() == [
        "cgat_comp/ternaries",
        "cgat_comp/binaries",
    ]
    assert metadata.input_role.eq("raw_pre_dft_pre_mlip_x0_geometry_only").all()
    assert manifest["counts"]["selected_rows"] == 2
    assert manifest["dft_values_read"] is False
    assert manifest["mlip_prerelaxation_used"] is False
    assert manifest["later_geometry_accessed"] is False
    with zipfile.ZipFile(target / module.GEOMETRY_NAME) as archive:
        assert archive.namelist() == ["a.extxyz", "c.extxyz"]
        payload = b"".join(archive.read(name) for name in archive.namelist())
    for forbidden in (b"12345.25", b"34567.25", b"energy", b"forces", b"stress"):
        assert forbidden not in payload
    with pytest.raises(FileExistsError):
        module.build_next42_cohort(
            shard_0000_path=shard0,
            shard_0001_path=shard1,
            source_table_path=table,
            source_manifest_path=audit_manifest,
            output_dir=target,
        )


def test_cohort_fails_closed_when_an_eligible_identity_is_missing(tmp_path: Path) -> None:
    from src.next42_alexandria_cohort import build_next42_cohort

    shard0, shard1 = _inputs(tmp_path)
    table, audit_manifest = _source_audit(
        tmp_path, shard0, shard1, include_missing=True
    )
    with pytest.raises(ValueError, match="eligible identities are missing"):
        build_next42_cohort(
            shard_0000_path=shard0,
            shard_0001_path=shard1,
            source_table_path=table,
            source_manifest_path=audit_manifest,
            output_dir=tmp_path / "missing",
        )


def test_cohort_cli_cannot_accept_labels_sampling_or_extra_shards() -> None:
    from src.next42_alexandria_cohort import main

    for forbidden in ("--labels", "--energy", "--forces", "--sample", "--extra-shard"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
