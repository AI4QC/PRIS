"""Contracts for the NEXT25 OMatG generated-x0 sanitizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
from ase.io import write
import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[Atoms]]:
    from src.next25_omatg_compositions import PROTOCOL as COMPOSITION_PROTOCOL
    from src.next25_omatg_run import PROTOCOL as RUN_PROTOCOL

    composition_dir = tmp_path / "compositions"
    composition_dir.mkdir()
    cohort = composition_dir / "composition_cohort.parquet"
    pd.DataFrame(
        {
            "material_id": ["next25-test-0000", "next25-test-0001"],
            "source_split": ["test", "test"],
            "source_index": [3, 8],
            "formula": ["Li2O", "NaCl"],
            "reduced_formula": ["Li2O", "NaCl"],
            "atomic_numbers_json": ["[3,3,8]", "[11,17]"],
            "natoms": [3, 2],
            "selection_key": ["1" * 64, "2" * 64],
            "selection_rank": [0, 1],
            "input_role": ["composition_only", "composition_only"],
        }
    ).to_parquet(cohort, index=False)
    composition_lmdb = composition_dir / "compositions_only.lmdb"
    composition_lmdb.write_bytes(b"dummy")
    composition_manifest = composition_dir / "MANIFEST.json"
    composition_manifest.write_text(
        json.dumps(
            {
                "protocol": COMPOSITION_PROTOCOL,
                "input_role": "composition_only",
                "reference_geometry_fields_accessed": False,
                "property_label_fields_accessed": False,
                "labels_opened": False,
                "counts": {"selected_rows": 2, "selected_atoms": 5},
                "outputs_sha256": {
                    cohort.name: _sha(cohort),
                    composition_lmdb.name: _sha(composition_lmdb),
                },
            }
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    generated = run_dir / "generated.xyz"
    frames = [
        Atoms(
            [3, 3, 8],
            positions=[[0.12, 0.23, 0.34], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
            cell=[[5.1, 0.1, 0.2], [0.0, 5.2, 0.3], [0.0, 0.0, 5.3]],
            pbc=True,
        ),
        Atoms(
            [11, 17],
            positions=[[0.4, 0.5, 0.6], [2.4, 2.5, 2.6]],
            cell=[[6.1, 0.2, 0.0], [0.0, 6.2, 0.1], [0.0, 0.0, 6.3]],
            pbc=True,
        ),
    ]
    write(generated, frames, format="extxyz")
    run_manifest = run_dir / "MANIFEST.json"
    run_manifest.write_text(
        json.dumps(
            {
                "protocol": RUN_PROTOCOL,
                "input_role": "composition_only",
                "output_role": "raw_unrelaxed_generator_x0",
                "all_generator_outputs_retained": True,
                "post_generation_validity_filter_used": False,
                "reference_geometry_fields_accessed": False,
                "property_label_fields_accessed": False,
                "dft_or_relaxed_structures_accessed": False,
                "energy_or_force_model_used": False,
                "physical_relaxation_used": False,
                "runtime_config_contains_reference_paths": False,
                "counts": {"composition_rows": 2, "generated_frames": 2},
                "inputs_sha256": {
                    "composition_cohort": {
                        "path": str(cohort.resolve()),
                        "sha256": _sha(cohort),
                    },
                    "composition_manifest": {
                        "path": str(composition_manifest.resolve()),
                        "sha256": _sha(composition_manifest),
                    },
                },
                "outputs_sha256": {generated.name: _sha(generated)},
            }
        ),
        encoding="utf-8",
    )
    return cohort, composition_manifest, generated, run_manifest, frames


def test_sanitizer_preserves_every_frame_and_projects_geometry_only(tmp_path: Path) -> None:
    from src import next25_omatg_holdout as module
    from src.next11_geometry_only_frames import _load_archive_only

    cohort, composition_manifest, generated, run_manifest, frames = _source(tmp_path)
    output = tmp_path / "holdout"
    result = module.freeze_omatg_x0(
        composition_cohort_path=cohort,
        composition_manifest_path=composition_manifest,
        generated_xyz_path=generated,
        generation_manifest_path=run_manifest,
        output_dir=output,
        require_formal_inputs=False,
    )

    metadata = pd.read_parquet(output / module.METADATA_NAME)
    assert metadata.columns.tolist() == [
        "material_id",
        "rk",
        "formula",
        "natoms",
        "input_role",
    ]
    assert metadata.material_id.tolist() == ["next25-test-0000", "next25-test-0001"]
    assert metadata.rk.eq("omatg_mp20_csp_linear_ode").all()
    assert metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    ids, frozen = _load_archive_only(
        output / module.GEOMETRY_NAME, tuple(metadata.material_id)
    )
    assert ids == metadata.material_id.tolist()
    for original, sanitized in zip(frames, frozen, strict=True):
        assert sanitized.get_atomic_numbers().tolist() == original.get_atomic_numbers().tolist()
        np.testing.assert_array_equal(sanitized.positions, original.positions)
        np.testing.assert_array_equal(sanitized.cell.array, original.cell.array)

    assert result["all_generator_outputs_retained"] is True
    assert result["coordinates_or_cell_modified"] is False
    assert result["labels_opened"] is False
    assert result["endpoint_artifacts_opened"] is False
    assert result["model_or_proxy_potential_used"] is False
    assert result["counts"] == {"rows": 2, "frames": 2, "atoms": 5}
    with zipfile.ZipFile(output / module.GEOMETRY_NAME) as archive:
        payload = b"\n".join(archive.read(name) for name in archive.namelist())
        for forbidden in (b"energy", b"label", b"endpoint", b"dft", b"SECRET"):
            assert forbidden not in payload


def test_sanitizer_accepts_declared_lmdb_byte_key_order_and_canonicalizes_it(
    tmp_path: Path,
) -> None:
    from src import next25_omatg_holdout as module
    from src.next11_geometry_only_frames import _load_archive_only

    cohort_path, composition_manifest_path, generated, run_manifest_path, _ = _source(
        tmp_path
    )
    ids = [f"next25-test-{index:04d}" for index in range(12)]
    frames = [
        Atoms(
            [index + 1],
            positions=[[index / 10, 0.0, 0.0]],
            cell=np.eye(3) * (5.0 + index / 10),
            pbc=True,
        )
        for index in range(12)
    ]
    cohort = pd.DataFrame(
        {
            "material_id": ids,
            "source_split": "test",
            "source_index": range(12),
            "formula": [frame.get_chemical_formula() for frame in frames],
            "reduced_formula": [frame.get_chemical_formula() for frame in frames],
            "atomic_numbers_json": [json.dumps(frame.numbers.tolist()) for frame in frames],
            "natoms": 1,
            "selection_key": [str(index) * 64 for index in range(12)],
            "selection_rank": range(12),
            "input_role": "composition_only",
        }
    )
    cohort.to_parquet(cohort_path, index=False)
    composition_manifest = json.loads(composition_manifest_path.read_text())
    composition_manifest["counts"] = {"selected_rows": 12, "selected_atoms": 12}
    composition_manifest["outputs_sha256"][cohort_path.name] = _sha(cohort_path)
    composition_manifest_path.write_text(json.dumps(composition_manifest), encoding="utf-8")

    raw_order = sorted(range(12), key=lambda index: str(index).encode("ascii"))
    write(generated, [frames[index] for index in raw_order], format="extxyz")
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["counts"] = {"composition_rows": 12, "generated_frames": 12}
    run_manifest["inputs_sha256"]["composition_cohort"]["sha256"] = _sha(cohort_path)
    run_manifest["inputs_sha256"]["composition_manifest"]["sha256"] = _sha(
        composition_manifest_path
    )
    run_manifest["outputs_sha256"][generated.name] = _sha(generated)
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")

    output = tmp_path / "lexicographic-holdout"
    result = module.freeze_omatg_x0(
        composition_cohort_path=cohort_path,
        composition_manifest_path=composition_manifest_path,
        generated_xyz_path=generated,
        generation_manifest_path=run_manifest_path,
        output_dir=output,
        require_formal_inputs=False,
    )
    loaded_ids, loaded_frames = _load_archive_only(
        output / module.GEOMETRY_NAME, tuple(ids)
    )
    assert loaded_ids == ids
    for expected, actual in zip(frames, loaded_frames, strict=True):
        assert actual.numbers.tolist() == expected.numbers.tolist()
        np.testing.assert_array_equal(actual.positions, expected.positions)
        np.testing.assert_array_equal(actual.cell.array, expected.cell.array)
    assert result["generator_frame_order"] == "ascending_lmdb_decimal_byte_key"
    assert result["canonical_output_order"] == "ascending_selection_rank"


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("reference", "boundary"),
        ("filter", "boundary"),
        ("hash", "hash"),
        ("order", "composition"),
        ("metadata", "metadata"),
    ],
)
def test_sanitizer_rejects_boundary_hash_composition_or_metadata_drift(
    tmp_path: Path, mutation: str, match: str
) -> None:
    from src.next25_omatg_holdout import freeze_omatg_x0

    cohort, composition_manifest, generated, run_manifest, frames = _source(tmp_path)
    manifest = json.loads(run_manifest.read_text())
    if mutation == "reference":
        manifest["reference_geometry_fields_accessed"] = True
    elif mutation == "filter":
        manifest["post_generation_validity_filter_used"] = True
    elif mutation == "hash":
        manifest["outputs_sha256"][generated.name] = "0" * 64
    elif mutation == "order":
        write(generated, list(reversed(frames)), format="extxyz")
        manifest["outputs_sha256"][generated.name] = _sha(generated)
    elif mutation == "metadata":
        tainted = [frame.copy() for frame in frames]
        tainted[0].info["energy"] = -10.0
        write(generated, tainted, format="extxyz")
        manifest["outputs_sha256"][generated.name] = _sha(generated)
    run_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        freeze_omatg_x0(
            composition_cohort_path=cohort,
            composition_manifest_path=composition_manifest,
            generated_xyz_path=generated,
            generation_manifest_path=run_manifest,
            output_dir=tmp_path / "out",
            require_formal_inputs=False,
        )


def test_sanitizer_refuses_overwrite_and_endpoint_cli(tmp_path: Path) -> None:
    from src.next25_omatg_holdout import freeze_omatg_x0, main

    cohort, composition_manifest, generated, run_manifest, _frames = _source(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        freeze_omatg_x0(
            composition_cohort_path=cohort,
            composition_manifest_path=composition_manifest,
            generated_xyz_path=generated,
            generation_manifest_path=run_manifest,
            output_dir=output,
            require_formal_inputs=False,
        )
    for forbidden in ("--labels", "--endpoint", "--reference", "--relaxed"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
