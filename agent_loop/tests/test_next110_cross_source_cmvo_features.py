from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import zipfile

from ase import Atoms
import pandas as pd
from pymatgen.core import Lattice, Structure
import pytest

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
from src.next110_cross_source_cmvo_features import (
    EXPECTED_INPUT_SHA256,
    FEATURE_COLUMNS,
    FEATURE_FILES,
    MANIFEST_NAME,
    NUMERIC_FEATURE_NAMES,
    PROTOCOL,
    build_cross_source_discovery_cmvo_features,
    compute_cmvo_feature_row,
)
import src.next110_cross_source_cmvo_features as next110


def test_formal_scigen_discovery_identity_matches_the_immutable_cohort() -> None:
    assert EXPECTED_INPUT_SHA256["scigen_geometry_discovery"] == (
        "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575"
    )


def test_next110_schema_is_additive_and_has_no_endpoint_quantities() -> None:
    assert PROTOCOL == "2026-08-08-next110-cross-source-discovery-cmvo-v1"
    assert NUMERIC_FEATURE_NAMES
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


def test_row_computes_core_and_expanded_certificates_independently() -> None:
    structure = Structure(
        Lattice.cubic(4.2),
        ["Na", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    row = compute_cmvo_feature_row(structure, graph_mode="voronoi")

    assert tuple(row) == FEATURE_COLUMNS
    assert row["cmvo_core_supported"] is True
    assert row["cmvo_expanded_supported"] is True
    assert row["cmvo_core_failure"] is None
    assert row["cmvo_expanded_failure"] is None
    assert row["cmvo_core_global_balance_gap"] > 0.0
    assert row["cmvo_expanded_global_balance_gap"] == pytest.approx(0.0)
    assert len(str(row["cmvo_core_catalogue_sha256"])) == 64
    assert len(str(row["cmvo_expanded_catalogue_sha256"])) == 64
    assert all(math.isfinite(float(row[name])) for name in NUMERIC_FEATURE_NAMES)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_cohorts(root: Path) -> tuple[Path, Path]:
    scigen = root / "scigen"
    scigen.mkdir()
    scigen_id = "tri_000_00001"
    atoms = Atoms(
        ["Na", "O"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[4.2, 4.2, 4.2],
        pbc=True,
    )
    discovery_zip = scigen / SCIGEN_GEOMETRY_NAMES["discovery"]
    with zipfile.ZipFile(discovery_zip, "w") as archive:
        archive.writestr(f"{scigen_id}.extxyz", _canonical_frame(atoms))
    for role in ("internal_validation", "internal_replication"):
        (scigen / SCIGEN_GEOMETRY_NAMES[role]).write_bytes(
            b"NEXT110 must never parse this non-discovery payload"
        )
    pd.DataFrame(
        [{
            "material_id": scigen_id,
            "lattice_class": "tri",
            "reduced_formula": "NaO",
            "chemical_system": "Na-O",
            "natoms": 2,
            "partition_role": "discovery",
            "input_role": "raw_generated_pre_dft_unrelaxed_x0",
        }]
    ).to_parquet(scigen / SCIGEN_METADATA_NAME, index=False)
    (scigen / SCIGEN_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "protocol": SCIGEN_PROTOCOL,
                "labels_opened": False,
                "endpoint_payloads_opened": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": {
                    SCIGEN_METADATA_NAME: _sha(scigen / SCIGEN_METADATA_NAME),
                    SCIGEN_GEOMETRY_NAMES["discovery"]: _sha(discovery_zip),
                },
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
    discovery_parquet = wyformer / WYFORMER_GEOMETRY_NAMES["discovery"]
    pd.DataFrame(
        {
            "material_id": [wyformer_id],
            "structure_json": [json.dumps(structure.as_dict(), sort_keys=True)],
        }
    ).to_parquet(discovery_parquet, index=False)
    for role in ("internal_validation", "internal_replication"):
        (wyformer / WYFORMER_GEOMETRY_NAMES[role]).write_bytes(
            b"NEXT110 must never parse this non-discovery payload"
        )
    pd.DataFrame(
        [{
            "material_id": wyformer_id,
            "reduced_formula": "CsCl",
            "chemical_system": "Cl-Cs",
            "natoms": 2,
            "generated_space_group": 221,
            "crystal_system": "cubic",
            "partition_role": "discovery",
            "input_role": WYFORMER_INPUT_ROLE,
        }]
    ).to_parquet(wyformer / WYFORMER_METADATA_NAME, index=False)
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
                "outputs_sha256": {
                    WYFORMER_METADATA_NAME: _sha(wyformer / WYFORMER_METADATA_NAME),
                    WYFORMER_GEOMETRY_NAMES["discovery"]: _sha(discovery_parquet),
                },
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
        build_cross_source_discovery_cmvo_features
    ).parameters
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "validation", "replication")
    )
    scigen, wyformer = _synthetic_cohorts(tmp_path)
    design = tmp_path / "design.md"
    design.write_text("frozen NEXT109 design\n", encoding="utf-8")
    output = tmp_path / "features"

    manifest = build_cross_source_discovery_cmvo_features(
        scigen_cohort_dir=scigen,
        wyformer_cohort_dir=wyformer,
        design_path=design,
        output_dir=output,
        workers=1,
        require_formal_inputs=False,
    )

    assert manifest["partitions_read"] == ["discovery"]
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_payloads_opened"] is False
    assert manifest["validation_geometry_opened"] is False
    assert manifest["replication_geometry_opened"] is False
    assert manifest["dft_values_used_by_features"] is False
    assert manifest["learned_energy_force_stress_proxy_used"] is False
    assert set(manifest["solver_thread_environment"]) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
    assert (output / MANIFEST_NAME).is_file()
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(output / FEATURE_FILES[source])
        assert len(table) == 1
        assert set(FEATURE_COLUMNS) <= set(table.columns)


def test_formal_solver_thread_environment_is_single_thread_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    assert next110._solver_thread_environment(require_formal_inputs=True) == expected

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="single-thread solver environment"):
        next110._solver_thread_environment(require_formal_inputs=True)
