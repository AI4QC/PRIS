from __future__ import annotations

from zipfile import ZipFile

import pytest
from pymatgen.core import Lattice, Structure

from experiments.property_design_20260821.run_mattergen import (
    build_generation_manifest,
    read_generated_archive,
)


def _diamond() -> Structure:
    return Structure(
        Lattice.cubic(3.57),
        ["C", "C"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def test_read_generated_archive_parses_and_hashes_each_cif(tmp_path):
    archive = tmp_path / "generated_crystals_cif.zip"
    cif_text = _diamond().to(fmt="cif")
    with ZipFile(archive, "w") as handle:
        handle.writestr("gen_0.cif", cif_text)
        handle.writestr("gen_1.cif", cif_text)

    records = read_generated_archive(archive)

    assert len(records) == 2
    assert [record["archive_member"] for record in records] == ["gen_0.cif", "gen_1.cif"]
    assert all(record["formula"] == "C" for record in records)
    assert all(record["num_sites"] == 2 for record in records)
    assert all(len(record["structure_sha256"]) == 64 for record in records)


def test_generation_manifest_records_independent_property_condition_and_provenance(tmp_path):
    checkpoint = tmp_path / "last.ckpt"
    config = tmp_path / "config.yaml"
    archive = tmp_path / "generated_crystals_cif.zip"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("model: test\n")
    with ZipFile(archive, "w") as handle:
        handle.writestr("gen_0.cif", _diamond().to(fmt="cif"))

    manifest = build_generation_manifest(
        archive_path=archive,
        checkpoint_path=checkpoint,
        config_path=config,
        model_repo_commit="abc123",
        requested_count=1,
        batch_size=1,
        num_batches=1,
        target_bulk_modulus_gpa=400.0,
        guidance_factor=2.0,
        seed=20260821,
        wall_seconds=12.5,
    )

    assert manifest["generator"]["name"] == "MatterGen"
    assert manifest["generator"]["model_repo_commit"] == "abc123"
    assert manifest["condition"] == {"ml_bulk_modulus": 400.0}
    assert manifest["sampling"]["guidance_factor"] == 2.0
    assert manifest["sampling"]["seed"] == 20260821
    assert manifest["outputs"]["generated_count"] == 1
    assert len(manifest["model"]["checkpoint_sha256"]) == 64
    assert len(manifest["outputs"]["archive_sha256"]) == 64


def test_generation_manifest_fails_if_archive_count_does_not_match_request(tmp_path):
    checkpoint = tmp_path / "last.ckpt"
    config = tmp_path / "config.yaml"
    archive = tmp_path / "generated_crystals_cif.zip"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("model: test\n")
    with ZipFile(archive, "w") as handle:
        handle.writestr("gen_0.cif", _diamond().to(fmt="cif"))

    with pytest.raises(ValueError, match="requested 2 structures but parsed 1"):
        build_generation_manifest(
            archive_path=archive,
            checkpoint_path=checkpoint,
            config_path=config,
            model_repo_commit="abc123",
            requested_count=2,
            batch_size=1,
            num_batches=2,
            target_bulk_modulus_gpa=400.0,
            guidance_factor=2.0,
            seed=20260821,
            wall_seconds=12.5,
        )
