from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src.next83_scigen_source_audit import AUDIT_NAME, MANIFEST_NAME as AUDIT_MANIFEST
from src.next83_scigen_source_audit import audit_scigen_source
from src.next84_scigen_geometry_lockbox import (
    GEOMETRY_NAMES,
    MANIFEST_NAME,
    METADATA_NAME,
    PARTITIONS,
    assign_partition,
    build_scigen_geometry_lockbox,
)


def _poscar(symbols: str, counts: str) -> bytes:
    natoms = sum(int(value) for value in counts.split())
    coords = "\n".join(
        f"{(index * 0.23) % 1:.6f} {(index * 0.31) % 1:.6f} {(index * 0.41) % 1:.6f}"
        for index in range(natoms)
    )
    return (
        "test\n1.0\n6 0 0\n0 6 0\n0 0 6\n"
        f"{symbols}\n{counts}\nDirect\n{coords}\n"
    ).encode("utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    archive = tmp_path / "source.zip"
    rows = {
        "tri_000_00001": _poscar("Na Cl", "1 1"),
        "hon_000_00002": _poscar("Na Cl", "2 2"),
        "kag_000_00003": _poscar("Li F", "1 1"),
    }
    with zipfile.ZipFile(archive, "w") as zf:
        for material_id, payload in rows.items():
            zf.writestr(f"03_scigen_materials_relaxed/{material_id}/POSCAR", payload)
            zf.writestr(
                f"03_scigen_materials_relaxed/{material_id}/CONTCAR",
                b"this relaxed payload must never be parsed",
            )
        zf.writestr(
            "03_scigen_materials_relaxed/output.dat",
            b"this endpoint payload must never be parsed",
        )
        zf.writestr("03_scigen_materials_relaxed/si_table_tri.csv", b"labels")
    metadata = tmp_path / "figshare.json"
    import hashlib

    metadata.write_text(
        json.dumps(
            {
                "id": 26082733,
                "doi": "10.6084/m9.figshare.26082733.v3",
                "license": {"name": "CC BY 4.0"},
                "files": [
                    {
                        "id": 57245942,
                        "name": "03_scigen_materials_relaxed.zip",
                        "size": archive.stat().st_size,
                        "computed_md5": hashlib.md5(archive.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    design = tmp_path / "design.md"
    design.write_text("frozen\n", encoding="utf-8")
    audit_dir = tmp_path / "audit"
    audit_scigen_source(
        source_archive_path=archive,
        figshare_metadata_path=metadata,
        design_path=design,
        output_dir=audit_dir,
        expected_rows=3,
        expected_supplementary_tables=("si_table_tri.csv",),
        require_formal_inputs=False,
    )
    return archive, audit_dir / AUDIT_NAME, audit_dir / AUDIT_MANIFEST, design


def test_formula_group_partition_is_deterministic() -> None:
    assert assign_partition("NaCl") == assign_partition("NaCl")
    assert assign_partition("Na2Cl2") in PARTITIONS
    with pytest.raises(ValueError):
        assign_partition("")


def test_build_reads_only_poscar_and_physically_splits_geometry(tmp_path: Path) -> None:
    archive, audit, audit_manifest, design = _inputs(tmp_path)
    target = tmp_path / "cohort"
    manifest = build_scigen_geometry_lockbox(
        source_archive_path=archive,
        source_audit_path=audit,
        source_audit_manifest_path=audit_manifest,
        design_path=design,
        output_dir=target,
        require_formal_inputs=False,
    )

    table = pd.read_parquet(target / METADATA_NAME)
    assert len(table) == 3
    assert not table["material_id"].duplicated().any()
    assert set(table["partition_role"]) <= set(PARTITIONS)
    assert table.groupby("reduced_formula")["partition_role"].nunique().max() == 1
    assert set(table["input_role"]) == {"raw_generated_pre_dft_unrelaxed_x0"}
    assert not any("energy" in name.lower() for name in table.columns)
    assert manifest["labels_opened"] is False
    assert manifest["relaxed_structures_opened"] is False
    assert manifest["endpoint_payloads_opened"] is False

    archived_ids: set[str] = set()
    for role, filename in GEOMETRY_NAMES.items():
        with zipfile.ZipFile(target / filename) as zf:
            names = zf.namelist()
            assert names == sorted(names)
            for name in names:
                payload = zf.read(name)
                assert b"energy" not in payload.lower()
                assert b"forces" not in payload.lower()
                archived_ids.add(Path(name).stem)
                expected_role = table.set_index("material_id").loc[Path(name).stem, "partition_role"]
                assert expected_role == role
    assert archived_ids == set(table["material_id"])
    assert (target / MANIFEST_NAME).is_file()


def test_build_is_no_replace_and_rejects_audit_hash_change(tmp_path: Path) -> None:
    archive, audit, audit_manifest, design = _inputs(tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        build_scigen_geometry_lockbox(
            source_archive_path=archive,
            source_audit_path=audit,
            source_audit_manifest_path=audit_manifest,
            design_path=design,
            output_dir=target,
            require_formal_inputs=False,
        )

    payload = json.loads(audit_manifest.read_text(encoding="utf-8"))
    payload["outputs_sha256"][AUDIT_NAME] = "0" * 64
    audit_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="audit output hash"):
        build_scigen_geometry_lockbox(
            source_archive_path=archive,
            source_audit_path=audit,
            source_audit_manifest_path=audit_manifest,
            design_path=design,
            output_dir=tmp_path / "bad",
            require_formal_inputs=False,
        )
