from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src.next13d_acsc_dft_pairs import _sha256_file
from src.next84_scigen_geometry_lockbox import (
    MANIFEST_NAME as COHORT_MANIFEST_NAME,
    METADATA_NAME,
    PROTOCOL as COHORT_PROTOCOL,
)
from src.next86_scigen_endpoint_router import (
    ENDPOINT_NAME,
    MANIFEST_NAME,
    route_scigen_endpoints,
)
from src.next86_scigen_term_catalogue import (
    CATALOGUE_NAME,
    MANIFEST_NAME as TERM_MANIFEST_NAME,
    PROTOCOL as TERM_PROTOCOL,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source.zip"
    header = (
        "# structure E_start/atom E_final/atom E_diff/atom E_conv "
        "F_max N_step d_latt d_xyz\n"
    )
    lines = [
        "tri_000_00001 -1 -2 -1 0.0001 0.005 10 0.5 0.2",
        "tsq_000_00002 -1 -2 -1 0.0001 0.020 20 2.0 0.5",
        "trh_000_00003 -1 -2 -1 0.0001 0.001 30 0.3 0.2",
    ]
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr(
            "03_scigen_materials_relaxed/output.dat",
            (header + "----------------------------------------\n" + "\n".join(lines) + "\n").encode("utf-8"),
        )
        zf.writestr("03_scigen_materials_relaxed/tri_000_00001/CONTCAR", b"forbidden")
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    metadata = pd.DataFrame(
        {
            "material_id": ["tri_000_00001", "tsq_000_00002", "trh_000_00003"],
            "lattice_class": ["tri", "tsq", "trh"],
            "partition_role": [
                "discovery",
                "internal_validation",
                "internal_replication",
            ],
        }
    )
    metadata.to_parquet(cohort / METADATA_NAME, index=False)
    (cohort / COHORT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "protocol": COHORT_PROTOCOL,
                "labels_opened": False,
                "endpoint_payloads_opened": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": {METADATA_NAME: _sha256_file(cohort / METADATA_NAME)},
            }
        ),
        encoding="utf-8",
    )
    terms = tmp_path / "terms"
    terms.mkdir()
    (terms / CATALOGUE_NAME).write_text(
        json.dumps({"protocol": TERM_PROTOCOL, "labels_opened": False}), encoding="utf-8"
    )
    (terms / TERM_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "protocol": TERM_PROTOCOL,
                "labels_opened": False,
                "endpoint_payloads_opened": False,
                "outputs_sha256": {CATALOGUE_NAME: _sha256_file(terms / CATALOGUE_NAME)},
            }
        ),
        encoding="utf-8",
    )
    design = tmp_path / "design.md"
    design.write_text("frozen\n", encoding="utf-8")
    return source, cohort, terms, design


def test_router_physically_separates_without_energy_columns(tmp_path: Path) -> None:
    source, cohort, terms, design = _inputs(tmp_path)
    targets = {
        "discovery": tmp_path / "discovery",
        "internal_validation": tmp_path / "validation",
        "internal_replication": tmp_path / "replication",
    }
    manifests = route_scigen_endpoints(
        source_archive_path=source,
        cohort_dir=cohort,
        term_catalogue_dir=terms,
        design_path=design,
        output_dirs=targets,
        require_formal_inputs=False,
    )
    for role, target in targets.items():
        table = pd.read_parquet(target / ENDPOINT_NAME)
        assert len(table) == 1
        assert set(table["partition_role"]) == {role}
        assert not any(name.startswith("energy") or name.startswith("E_") for name in table)
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["endpoint_values_summarized_or_inspected"] is False
        assert manifest["relaxed_structures_opened"] is False
        assert manifests[role]["partition_role"] == role
    discovery = pd.read_parquet(targets["discovery"] / ENDPOINT_NAME).iloc[0]
    validation = pd.read_parquet(targets["internal_validation"] / ENDPOINT_NAME).iloc[0]
    assert discovery["protected"]
    assert validation["severe"]
    assert validation["distortion_ratio"] == pytest.approx(2.0)


def test_router_is_no_replace_and_requires_exact_identity(tmp_path: Path) -> None:
    source, cohort, terms, design = _inputs(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        route_scigen_endpoints(
            source_archive_path=source,
            cohort_dir=cohort,
            term_catalogue_dir=terms,
            design_path=design,
            output_dirs={
                "discovery": existing,
                "internal_validation": tmp_path / "v",
                "internal_replication": tmp_path / "r",
            },
            require_formal_inputs=False,
        )

    metadata = pd.read_parquet(cohort / METADATA_NAME)
    metadata.loc[0, "material_id"] = "tri_000_99999"
    metadata.to_parquet(cohort / METADATA_NAME, index=False)
    manifest = json.loads((cohort / COHORT_MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["outputs_sha256"][METADATA_NAME] = _sha256_file(cohort / METADATA_NAME)
    (cohort / COHORT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        route_scigen_endpoints(
            source_archive_path=source,
            cohort_dir=cohort,
            term_catalogue_dir=terms,
            design_path=design,
            output_dirs={
                "discovery": tmp_path / "d2",
                "internal_validation": tmp_path / "v2",
                "internal_replication": tmp_path / "r2",
            },
            require_formal_inputs=False,
        )
