from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from src.next83_scigen_source_audit import (
    AUDIT_NAME,
    MANIFEST_NAME,
    audit_scigen_source,
)


def _source(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for material_id in ("tri_000_00001", "hon_000_00002"):
            zf.writestr(f"03_scigen_materials_relaxed/{material_id}/POSCAR", b"x0")
            zf.writestr(f"03_scigen_materials_relaxed/{material_id}/CONTCAR", b"xf")
        zf.writestr("03_scigen_materials_relaxed/output.dat", b"endpoint")
        zf.writestr("03_scigen_materials_relaxed/si_table_tri.csv", b"table")
    metadata = tmp_path / "figshare.json"
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
    design.write_text("frozen design\n", encoding="utf-8")
    return archive, metadata, design


def test_audit_reads_central_directory_without_endpoint_payload(tmp_path: Path) -> None:
    archive, metadata, design = _source(tmp_path)
    target = tmp_path / "audit"
    manifest = audit_scigen_source(
        source_archive_path=archive,
        figshare_metadata_path=metadata,
        design_path=design,
        output_dir=target,
        expected_rows=2,
        expected_supplementary_tables=("si_table_tri.csv",),
        require_formal_inputs=False,
        prior_label_free_poscar_probe_count=1,
    )

    audit = json.loads((target / AUDIT_NAME).read_text(encoding="utf-8"))
    assert audit["counts"] == {
        "contcar_members": 2,
        "output_table_members": 1,
        "poscar_members": 2,
        "supplementary_table_members": 1,
        "unique_material_ids": 2,
        "zip_members": 6,
    }
    assert audit["endpoint_payloads_opened"] is False
    assert audit["relaxed_structures_opened"] is False
    assert audit["prior_label_free_poscar_probe_count"] == 1
    assert manifest["labels_opened"] is False
    assert (target / MANIFEST_NAME).is_file()


def test_audit_rejects_identity_or_member_pair_mismatch(tmp_path: Path) -> None:
    archive, metadata, design = _source(tmp_path)
    with pytest.raises(ValueError, match="source identity"):
        audit_scigen_source(
            source_archive_path=archive,
            figshare_metadata_path=metadata,
            design_path=design,
            output_dir=tmp_path / "bad-hash",
            expected_rows=2,
            expected_supplementary_tables=("si_table_tri.csv",),
            expected_sha256="0" * 64,
            require_formal_inputs=True,
        )

    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(broken, "w") as zf:
        zf.writestr("03_scigen_materials_relaxed/tri_000_00001/POSCAR", b"x0")
        zf.writestr("03_scigen_materials_relaxed/output.dat", b"endpoint")
        zf.writestr("03_scigen_materials_relaxed/si_table_tri.csv", b"table")
    broken_meta = tmp_path / "broken.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["files"][0]["size"] = broken.stat().st_size
    payload["files"][0]["computed_md5"] = hashlib.md5(broken.read_bytes()).hexdigest()
    broken_meta.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="POSCAR/CONTCAR"):
        audit_scigen_source(
            source_archive_path=broken,
            figshare_metadata_path=broken_meta,
            design_path=design,
            output_dir=tmp_path / "bad-members",
            expected_rows=1,
            expected_supplementary_tables=("si_table_tri.csv",),
            require_formal_inputs=False,
        )


def test_audit_is_no_replace(tmp_path: Path) -> None:
    archive, metadata, design = _source(tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        audit_scigen_source(
            source_archive_path=archive,
            figshare_metadata_path=metadata,
            design_path=design,
            output_dir=target,
            expected_rows=2,
            expected_supplementary_tables=("si_table_tri.csv",),
            require_formal_inputs=False,
        )
