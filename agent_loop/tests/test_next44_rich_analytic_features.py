"""Contracts for additive NEXT44 rich raw-x0 descriptors."""

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


def _cohort(tmp_path: Path) -> tuple[Path, Path, Path]:
    from src.next42_alexandria_cohort import COHORT_NAME, GEOMETRY_NAME, INPUT_ROLE, PROTOCOL

    root = tmp_path / "cohort"
    root.mkdir()
    structures = {
        "a": Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        "b": Structure(Lattice.cubic(4.6), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    }
    metadata = pd.DataFrame(
        [
            {
                "material_id": key,
                "source_family": "cgat_comp/binaries",
                "source_shard": "pbe_0000",
                "formula": value.composition.formula.replace(" ", ""),
                "reduced_formula": value.composition.reduced_formula,
                "natoms": len(value),
                "input_role": INPUT_ROLE,
            }
            for key, value in structures.items()
        ]
    )
    metadata_path = root / COHORT_NAME
    metadata.to_parquet(metadata_path, index=False)
    geometry_path = root / GEOMETRY_NAME
    with zipfile.ZipFile(geometry_path, "w") as archive:
        for key, value in structures.items():
            info = zipfile.ZipInfo(f"{key}.extxyz", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, _canonical_frame(value.to_ase_atoms()))
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


def test_rich_single_x0_row_adds_full_and_geometry_families() -> None:
    from src.next44_rich_analytic_features import (
        CANDIDATE_FEATURE_NAMES,
        FAMILY_NAMES,
        compute_rich_feature_row,
    )

    atoms = Structure(
        Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).to_ase_atoms()
    row = compute_rich_feature_row(atoms)
    assert set(CANDIDATE_FEATURE_NAMES).issubset(row)
    assert all(f"{name}_supported" in row for name in FAMILY_NAMES)
    assert row["cell_composition_supported"] is True
    assert row["extended_contact_supported"] is True
    assert np.isfinite(row["geom_volume_pa"])
    assert np.isfinite(row["cov_coord110_mean"])
    assert "sivr_stiffness_min" in row
    assert "nm_point_reduced" in row
    assert "scbv_effective_cn_min" in row


def test_rich_builder_is_label_free_hashed_and_nonreplacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next44_rich_analytic_features as module

    metadata, geometry, cohort_manifest = _cohort(tmp_path)

    def cheap(_atoms):
        row = {name: np.nan for name in module.CANDIDATE_FEATURE_NAMES}
        row["geom_volume_pa"] = 20.0
        for family in module.FAMILY_NAMES:
            row[f"{family}_supported"] = family == "cell_composition"
            row[f"{family}_failure"] = None if family == "cell_composition" else "test"
        return row

    monkeypatch.setattr(module, "compute_rich_feature_row", cheap)
    target = tmp_path / "rich"
    manifest = module.build_rich_feature_bank(
        metadata_path=metadata,
        geometry_path=geometry,
        cohort_manifest_path=cohort_manifest,
        output_dir=target,
    )
    table = pd.read_parquet(target / module.FEATURE_NAME)
    assert table.material_id.tolist() == ["a", "b"]
    assert table.geom_volume_pa.tolist() == [20.0, 20.0]
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert manifest["dft_values_used"] is False
    assert manifest["outputs_sha256"][module.FEATURE_NAME] == _sha(target / module.FEATURE_NAME)
    with pytest.raises(FileExistsError):
        module.build_rich_feature_bank(
            metadata_path=metadata,
            geometry_path=geometry,
            cohort_manifest_path=cohort_manifest,
            output_dir=target,
        )


def test_rich_cli_cannot_accept_labels_endpoints_or_models() -> None:
    from src.next44_rich_analytic_features import main

    for forbidden in ("--labels", "--endpoint", "--energy", "--forces", "--model"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
