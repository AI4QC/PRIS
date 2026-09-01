from __future__ import annotations

import json
import math
import hashlib
import inspect
from pathlib import Path
import zipfile

from ase import Atoms
import pandas as pd
from pymatgen.core import Lattice, Structure

from src.next11_geometry_only_frames import _canonical_frame
from src.next84_scigen_geometry_lockbox import (
    GEOMETRY_NAMES as SCIGEN_GEOMETRY_NAMES,
    MANIFEST_NAME as SCIGEN_MANIFEST_NAME,
    METADATA_NAME as SCIGEN_METADATA_NAME,
    PROTOCOL as SCIGEN_PROTOCOL,
)
from src.next93b_wyformer_blind_lockbox import (
    GEOMETRY_NAMES as WYFORMER_GEOMETRY_NAMES,
    INPUT_ROLE as WYFORMER_INPUT_ROLE,
    MANIFEST_NAME as WYFORMER_MANIFEST_NAME,
    METADATA_NAME as WYFORMER_METADATA_NAME,
    PROTOCOL as WYFORMER_PROTOCOL,
)
from src.next102_cross_source_dobvr_features import (
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    MANIFEST_NAME,
    PROTOCOL,
    build_cross_source_discovery_dobvr_features,
    compute_dobvr_feature_row,
)


def test_next102_schema_is_additive_and_has_no_endpoint_quantities() -> None:
    assert PROTOCOL == "2026-08-04-next102-cross-source-discovery-dobvr-v1"
    assert FEATURE_COLUMNS
    forbidden = (
        "energy",
        "force",
        "stress",
        "relax",
        "dft",
        "model",
        "proxy",
        "endpoint",
    )
    assert not any(
        token in name.lower() for name in FEATURE_COLUMNS for token in forbidden
    )


def test_row_computes_both_frozen_families_from_one_raw_structure() -> None:
    structure = Structure(
        Lattice.cubic(4.2),
        ["Cs", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    row = compute_dobvr_feature_row(structure, graph_mode="voronoi")

    assert tuple(row) == FEATURE_COLUMNS
    assert row["dobvr_supported"] is True
    assert row["dobvrb_supported"] is True
    assert row["dobvr_failure"] is None
    assert row["dobvrb_failure"] is None
    assert len(str(row["dobvr_catalogue_sha256"])) == 64
    assert len(str(row["dobvrb_catalogue_sha256"])) == 64
    assert json.loads(str(row["dobvr_best_assignment_json"]))
    assert json.loads(str(row["dobvrb_best_assignment_json"]))
    numeric = [
        value
        for name, value in row.items()
        if name.startswith(("dobvr_", "dobvrb_"))
        and name not in {
            "dobvr_supported",
            "dobvr_failure",
            "dobvr_catalogue_sha256",
            "dobvr_pymatgen_version",
            "dobvr_best_assignment_json",
            "dobvrb_supported",
            "dobvrb_failure",
            "dobvrb_catalogue_sha256",
            "dobvrb_pymatgen_version",
            "dobvrb_best_assignment_json",
        }
    ]
    assert all(math.isfinite(float(value)) for value in numeric)


def test_row_abstains_with_nan_features_for_single_element_structure() -> None:
    structure = Structure(
        Lattice.cubic(3.6),
        ["Cu"],
        [[0.0, 0.0, 0.0]],
    )

    row = compute_dobvr_feature_row(structure, graph_mode="voronoi")

    assert row["dobvr_supported"] is False
    assert row["dobvrb_supported"] is False
    assert "neutral" in str(row["dobvr_failure"]).lower()
    assert "neutral" in str(row["dobvrb_failure"]).lower()
    for name in FEATURE_COLUMNS:
        if name.startswith("dobvr_") and name not in {
            "dobvr_supported",
            "dobvr_failure",
            "dobvr_catalogue_sha256",
            "dobvr_pymatgen_version",
            "dobvr_best_assignment_json",
        }:
            assert math.isnan(float(row[name]))
        if name.startswith("dobvrb_") and name not in {
            "dobvrb_supported",
            "dobvrb_failure",
            "dobvrb_catalogue_sha256",
            "dobvrb_pymatgen_version",
            "dobvrb_best_assignment_json",
        }:
            assert math.isnan(float(row[name]))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_cohorts(root: Path) -> tuple[Path, Path]:
    scigen = root / "scigen"
    scigen.mkdir()
    scigen_id = "tri_000_00001"
    atoms = Atoms(
        ["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[5.5, 5.5, 5.5],
        pbc=True,
    )
    scigen_geometry = scigen / SCIGEN_GEOMETRY_NAMES["discovery"]
    with zipfile.ZipFile(scigen_geometry, "w") as archive:
        archive.writestr(f"{scigen_id}.extxyz", _canonical_frame(atoms))
    for role in ("internal_validation", "internal_replication"):
        (scigen / SCIGEN_GEOMETRY_NAMES[role]).write_bytes(
            b"NEXT102 must never parse this non-discovery payload"
        )
    scigen_metadata = pd.DataFrame(
        [
            {
                "material_id": scigen_id,
                "lattice_class": "tri",
                "reduced_formula": "NaCl",
                "chemical_system": "Cl-Na",
                "natoms": 2,
                "partition_role": "discovery",
                "input_role": "raw_generated_pre_dft_unrelaxed_x0",
            }
        ]
    )
    scigen_metadata.to_parquet(scigen / SCIGEN_METADATA_NAME, index=False)
    scigen_outputs = {
        SCIGEN_METADATA_NAME: _sha(scigen / SCIGEN_METADATA_NAME),
        SCIGEN_GEOMETRY_NAMES["discovery"]: _sha(scigen_geometry),
    }
    (scigen / SCIGEN_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "protocol": SCIGEN_PROTOCOL,
                "labels_opened": False,
                "endpoint_payloads_opened": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": scigen_outputs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    wyformer = root / "wyformer"
    wyformer.mkdir()
    wyformer_id = "WYF-test-00001"
    structure = Structure(
        Lattice.cubic(4.2),
        ["Cs", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    wyformer_geometry = wyformer / WYFORMER_GEOMETRY_NAMES["discovery"]
    pd.DataFrame(
        {
            "material_id": [wyformer_id],
            "structure_json": [json.dumps(structure.as_dict(), sort_keys=True)],
        }
    ).to_parquet(wyformer_geometry, index=False)
    for role in ("internal_validation", "internal_replication"):
        (wyformer / WYFORMER_GEOMETRY_NAMES[role]).write_bytes(
            b"NEXT102 must never parse this non-discovery payload"
        )
    pd.DataFrame(
        [
            {
                "material_id": wyformer_id,
                "reduced_formula": "CsCl",
                "chemical_system": "Cl-Cs",
                "natoms": 2,
                "generated_space_group": 221,
                "crystal_system": "cubic",
                "partition_role": "discovery",
                "input_role": WYFORMER_INPUT_ROLE,
            }
        ]
    ).to_parquet(wyformer / WYFORMER_METADATA_NAME, index=False)
    wyformer_outputs = {
        WYFORMER_METADATA_NAME: _sha(wyformer / WYFORMER_METADATA_NAME),
        WYFORMER_GEOMETRY_NAMES["discovery"]: _sha(wyformer_geometry),
    }
    (wyformer / WYFORMER_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "protocol": WYFORMER_PROTOCOL,
                "labels_opened_by_feature_builder": False,
                "discovery_endpoint_opened": False,
                "validation_endpoint_opened": False,
                "replication_endpoint_opened": False,
                "relaxed_structures_published": False,
                "learned_proxy_execution_input": False,
                "outputs_sha256": wyformer_outputs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return scigen, wyformer


def test_builder_reads_only_discovery_and_has_no_endpoint_argument(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(
        build_cross_source_discovery_dobvr_features
    ).parameters
    assert not any(
        token in name for name in parameters for token in ("endpoint", "validation", "replication")
    )
    scigen, wyformer = _synthetic_cohorts(tmp_path)
    design = tmp_path / "design.md"
    amendment = tmp_path / "amendment.md"
    design.write_text("frozen design\n", encoding="utf-8")
    amendment.write_text("frozen amendment\n", encoding="utf-8")
    output = tmp_path / "features"

    manifest = build_cross_source_discovery_dobvr_features(
        scigen_cohort_dir=scigen,
        wyformer_cohort_dir=wyformer,
        design_path=design,
        amendment_path=amendment,
        output_dir=output,
        workers=1,
        require_formal_inputs=False,
    )

    assert manifest["partitions_read"] == ["discovery"]
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_payloads_opened"] is False
    assert manifest["dft_values_used_by_features"] is False
    assert manifest["learned_energy_force_stress_proxy_used"] is False
    assert (output / MANIFEST_NAME).is_file()
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(output / FEATURE_NAMES[source])
        assert len(table) == 1
        assert set(FEATURE_COLUMNS) <= set(table.columns)
        assert not any(
            token in name.lower()
            for name in table.columns
            for token in ("dft", "energy", "endpoint")
        )
