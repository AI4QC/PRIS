from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src.next43_analytic_feature_bank import CANDIDATE_FEATURE_NAMES as NEXT43_FEATURES
from src.next44_rich_analytic_features import CANDIDATE_FEATURE_NAMES as NEXT44_FEATURES
from src.next80_periodic_repulsive_load_resolvability import PRLR_FEATURE_NAMES
from src.next83_scigen_source_audit import AUDIT_NAME, MANIFEST_NAME as AUDIT_MANIFEST
from src.next83_scigen_source_audit import audit_scigen_source
from src.next84_scigen_geometry_lockbox import MANIFEST_NAME as COHORT_MANIFEST
from src.next84_scigen_geometry_lockbox import build_scigen_geometry_lockbox
from src.next85_scigen_label_free_features import (
    FEATURE_NAMES,
    MANIFEST_NAME,
    build_scigen_label_free_features,
    compute_scigen_feature_row,
)


def _poscar(symbols: str, counts: str, shift: float) -> bytes:
    natoms = sum(int(value) for value in counts.split())
    coords = "\n".join(
        f"{(shift + index * 0.23) % 1:.6f} {(index * 0.31) % 1:.6f} {(index * 0.41) % 1:.6f}"
        for index in range(natoms)
    )
    return (
        "test\n1.0\n6 0 0\n0 6 0\n0 0 6\n"
        f"{symbols}\n{counts}\nDirect\n{coords}\n"
    ).encode("utf-8")


def _cohort(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "source.zip"
    rows = {
        "tri_000_00001": _poscar("Na Cl", "1 1", 0.0),
        "hon_000_00002": _poscar("Li F", "1 1", 0.1),
        "kag_000_00003": _poscar("Mg O", "1 1", 0.2),
    }
    with zipfile.ZipFile(archive, "w") as zf:
        for material_id, payload in rows.items():
            zf.writestr(f"03_scigen_materials_relaxed/{material_id}/POSCAR", payload)
            zf.writestr(f"03_scigen_materials_relaxed/{material_id}/CONTCAR", b"forbidden")
        zf.writestr("03_scigen_materials_relaxed/output.dat", b"forbidden")
        zf.writestr("03_scigen_materials_relaxed/si_table_tri.csv", b"forbidden")
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
    cohort_dir = tmp_path / "cohort"
    build_scigen_geometry_lockbox(
        source_archive_path=archive,
        source_audit_path=audit_dir / AUDIT_NAME,
        source_audit_manifest_path=audit_dir / AUDIT_MANIFEST,
        design_path=design,
        output_dir=cohort_dir,
        require_formal_inputs=False,
    )
    return cohort_dir, design


def test_compute_row_contains_all_frozen_families() -> None:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp import Poscar

    structure = Poscar.from_str(_poscar("Na Cl", "1 1", 0.0).decode("utf-8")).structure
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = None
    atoms.info.clear()
    for name in list(atoms.arrays):
        if name not in {"numbers", "positions"}:
            del atoms.arrays[name]
    row = compute_scigen_feature_row(atoms)
    assert set(NEXT43_FEATURES) <= set(row)
    assert set(NEXT44_FEATURES) <= set(row)
    assert set(PRLR_FEATURE_NAMES) <= set(row)
    assert "pauling_p2_p5_decision" in row


def test_build_freezes_all_partitions_without_endpoint_inputs(tmp_path: Path) -> None:
    cohort, design = _cohort(tmp_path)
    target = tmp_path / "features"
    manifest = build_scigen_label_free_features(
        cohort_dir=cohort,
        design_path=design,
        output_dir=target,
        workers=1,
        require_formal_inputs=False,
    )
    frames = [pd.read_parquet(target / FEATURE_NAMES[role]) for role in FEATURE_NAMES]
    table = pd.concat(frames, ignore_index=True)
    assert len(table) == 3
    assert not table["material_id"].duplicated().any()
    assert set(NEXT43_FEATURES) <= set(table.columns)
    assert set(NEXT44_FEATURES) <= set(table.columns)
    assert set(PRLR_FEATURE_NAMES) <= set(table.columns)
    assert "pauling_p2_p5_decision" in table
    assert not any(name in table for name in ("E_start", "E_final", "F_max", "d_latt", "d_xyz"))
    assert manifest["labels_opened"] is False
    assert manifest["relaxed_structures_opened"] is False
    assert manifest["learned_energy_force_stress_proxy_used"] is False
    assert (target / MANIFEST_NAME).is_file()


def test_build_is_no_replace_and_cli_has_no_endpoint_argument(tmp_path: Path) -> None:
    cohort, design = _cohort(tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        build_scigen_label_free_features(
            cohort_dir=cohort,
            design_path=design,
            output_dir=target,
            workers=1,
            require_formal_inputs=False,
        )
    source = Path("src/next85_scigen_label_free_features.py").read_text(encoding="utf-8")
    assert "--endpoint" not in source
    assert "--contcar" not in source
