"""Contracts for the NEXT43 raw-x0 analytic descriptor bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
import pytest

from src.next11_geometry_only_frames import _canonical_frame


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_cohort(tmp_path: Path) -> tuple[Path, Path, Path]:
    from src.next42_alexandria_cohort import COHORT_NAME, GEOMETRY_NAME, INPUT_ROLE, PROTOCOL

    root = tmp_path / "cohort"
    root.mkdir()
    structures = {
        "alex-a": Structure(
            Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "alex-b": Structure(
            Lattice.cubic(4.6), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
    }
    metadata = pd.DataFrame(
        [
            {
                "material_id": material_id,
                "source_family": "cgat_comp/binaries",
                "source_shard": "pbe_0000",
                "formula": structure.composition.formula.replace(" ", ""),
                "reduced_formula": structure.composition.reduced_formula,
                "natoms": len(structure),
                "input_role": INPUT_ROLE,
            }
            for material_id, structure in structures.items()
        ]
    )
    metadata_path = root / COHORT_NAME
    metadata.to_parquet(metadata_path, index=False)
    geometry_path = root / GEOMETRY_NAME
    with zipfile.ZipFile(geometry_path, "w") as archive:
        for material_id, structure in structures.items():
            info = zipfile.ZipInfo(
                f"{material_id}.extxyz", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, _canonical_frame(structure.to_ase_atoms()))
    manifest = {
        "protocol": PROTOCOL,
        "input_role": INPUT_ROLE,
        "later_geometry_accessed": False,
        "dft_values_read": False,
        "mlip_prerelaxation_used": False,
        "physical_relaxation_executed": False,
        "selection": {"endpoint_fields_used": False, "sampled": False},
        "outputs_sha256": {
            COHORT_NAME: _sha(metadata_path),
            GEOMETRY_NAME: _sha(geometry_path),
        },
    }
    manifest_path = root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return metadata_path, geometry_path, manifest_path


def test_single_x0_row_composes_only_analytic_families() -> None:
    from src.next43_analytic_feature_bank import (
        CANDIDATE_FEATURE_NAMES,
        FAMILY_NAMES,
        compute_analytic_feature_row,
    )

    atoms = Structure(
        Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).to_ase_atoms()
    row = compute_analytic_feature_row(atoms)
    assert set(CANDIDATE_FEATURE_NAMES).issubset(row)
    assert all(f"{family}_supported" in row for family in FAMILY_NAMES)
    assert row["contact_supported"] is True
    assert row["symmetry_supported"] is True
    assert row["steric_supported"] is True
    assert np.isfinite(row["cov_q05"])
    assert not any(
        token in name.lower()
        for name in row
        for token in ("dft", "energy", "force", "relax", "mattersim")
    )


def test_builder_reads_only_sealed_geometry_and_publishes_hashed_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next43_analytic_feature_bank as module

    metadata, geometry, cohort_manifest = _sealed_cohort(tmp_path)

    def cheap_row(_atoms):
        row = {name: np.nan for name in module.CANDIDATE_FEATURE_NAMES}
        row["cov_q05"] = 0.9
        for family in module.FAMILY_NAMES:
            row[f"{family}_supported"] = family == "contact"
            row[f"{family}_failure"] = None if family == "contact" else "test unsupported"
        row["valence_assignment_policy"] = "test"
        return row

    monkeypatch.setattr(module, "compute_analytic_feature_row", cheap_row)
    target = tmp_path / "features"
    manifest = module.build_feature_bank(
        metadata_path=metadata,
        geometry_path=geometry,
        cohort_manifest_path=cohort_manifest,
        output_dir=target,
    )
    table = pd.read_parquet(target / module.FEATURE_NAME)
    assert table.material_id.tolist() == ["alex-a", "alex-b"]
    assert table.cov_q05.tolist() == [0.9, 0.9]
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert manifest["dft_values_used"] is False
    assert manifest["mlip_or_model_potential_used"] is False
    assert manifest["physical_relaxation_executed"] is False
    assert manifest["outputs_sha256"][module.FEATURE_NAME] == _sha(
        target / module.FEATURE_NAME
    )
    with pytest.raises(FileExistsError):
        module.build_feature_bank(
            metadata_path=metadata,
            geometry_path=geometry,
            cohort_manifest_path=cohort_manifest,
            output_dir=target,
        )


def test_builder_fails_closed_if_cohort_boundary_changes(tmp_path: Path) -> None:
    from src.next43_analytic_feature_bank import build_feature_bank

    metadata, geometry, cohort_manifest = _sealed_cohort(tmp_path)
    document = json.loads(cohort_manifest.read_text())
    document["dft_values_read"] = True
    cohort_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry-only boundary"):
        build_feature_bank(
            metadata_path=metadata,
            geometry_path=geometry,
            cohort_manifest_path=cohort_manifest,
            output_dir=tmp_path / "blocked",
        )


def test_feature_bank_cli_has_no_endpoint_or_model_arguments() -> None:
    from src.next43_analytic_feature_bank import main

    for forbidden in ("--labels", "--endpoint", "--forces", "--energy", "--model"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
