from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from pymatgen.core import Lattice, Structure

from experiments.property_design_20260821.aggregate_mattergen import (
    AGGREGATE_ARCHIVE_NAME,
    AGGREGATE_MANIFEST_NAME,
    aggregate_mattergen_shards,
)
from experiments.property_design_20260821.run_mattergen import (
    build_generation_manifest,
)


def _structure(lattice_parameter: float, species: str = "C") -> Structure:
    return Structure(
        Lattice.cubic(lattice_parameter),
        [species, species],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _write_shard(
    root: Path,
    name: str,
    structures: list[Structure],
    *,
    checkpoint: Path,
    config: Path,
    guidance_factor: float = 2.0,
    seed: int = 1,
) -> Path:
    shard_dir = root / name
    shard_dir.mkdir()
    archive = shard_dir / "generated_crystals_cif.zip"
    with ZipFile(archive, "w") as handle:
        for index, structure in enumerate(structures):
            handle.writestr(f"gen_{index}.cif", structure.to(fmt="cif"))
    manifest = build_generation_manifest(
        archive_path=archive,
        checkpoint_path=checkpoint,
        config_path=config,
        model_repo_commit="mattergen-commit",
        requested_count=len(structures),
        batch_size=len(structures),
        num_batches=1,
        target_bulk_modulus_gpa=400.0,
        guidance_factor=guidance_factor,
        seed=seed,
        wall_seconds=1.0,
    )
    (shard_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return shard_dir


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "last.ckpt"
    config = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"fixed-checkpoint")
    config.write_text("model: fixed\n")
    return checkpoint, config


def test_aggregate_succeeds_and_is_byte_deterministic(tmp_path, model_files):
    checkpoint, config = model_files
    shard_b = _write_shard(
        tmp_path,
        "shard_b",
        [_structure(3.60, "C"), _structure(3.70, "Si")],
        checkpoint=checkpoint,
        config=config,
        seed=2,
    )
    shard_a = _write_shard(
        tmp_path,
        "shard_a",
        [_structure(3.40, "Ge"), _structure(3.50, "O")],
        checkpoint=checkpoint,
        config=config,
        seed=1,
    )

    first = aggregate_mattergen_shards(
        [shard_b, shard_a], tmp_path / "aggregate_a", min_unique_structures=4
    )
    second = aggregate_mattergen_shards(
        [shard_a, shard_b], tmp_path / "aggregate_b", min_unique_structures=4
    )

    assert first == second
    assert first["source"] == "mattergen"
    assert first["counts"] == {
        "input_shards": 2,
        "declared_structures": 4,
        "parsed_structures": 4,
        "exact_hash_unique_structures": 4,
        "unique_structures": 4,
        "exact_hash_duplicates": 0,
        "matcher_equivalent_duplicates": 0,
    }
    assert len(first["protocol_sha256"]) == 64
    assert len(first["outputs"]["archive_sha256"]) == 64
    assert len(first["inputs"]) == 2
    assert all(len(item["manifest_sha256"]) == 64 for item in first["inputs"])
    assert all(len(item["archive_sha256"]) == 64 for item in first["inputs"])
    assert [item["source_seed"] for item in first["inputs"]] == [1, 2]
    candidates = first["outputs"]["candidates"]
    assert [candidate["source_seed"] for candidate in candidates] == [1, 1, 2, 2]
    assert all(
        len(candidate["source_manifest_sha256"]) == 64
        for candidate in candidates
    )
    source_manifest_hashes = {
        item["shard_dir"]: item["manifest_sha256"] for item in first["inputs"]
    }
    assert all(
        candidate["source_manifest_sha256"]
        == source_manifest_hashes[candidate["source_shard"]]
        for candidate in candidates
    )
    assert (
        (tmp_path / "aggregate_a" / AGGREGATE_ARCHIVE_NAME).read_bytes()
        == (tmp_path / "aggregate_b" / AGGREGATE_ARCHIVE_NAME).read_bytes()
    )
    assert (
        (tmp_path / "aggregate_a" / AGGREGATE_MANIFEST_NAME).read_bytes()
        == (tmp_path / "aggregate_b" / AGGREGATE_MANIFEST_NAME).read_bytes()
    )
    with ZipFile(tmp_path / "aggregate_a" / AGGREGATE_ARCHIVE_NAME) as handle:
        assert handle.namelist() == [
            "crystal_000000.cif",
            "crystal_000001.cif",
            "crystal_000002.cif",
            "crystal_000003.cif",
        ]


def test_aggregate_deduplicates_exact_hashes_and_records_both_sources(
    tmp_path, model_files
):
    checkpoint, config = model_files
    repeated = _structure(3.57)
    shard_a = _write_shard(
        tmp_path,
        "shard_a",
        [repeated],
        checkpoint=checkpoint,
        config=config,
        seed=1,
    )
    shard_b = _write_shard(
        tmp_path,
        "shard_b",
        [repeated],
        checkpoint=checkpoint,
        config=config,
        seed=2,
    )

    manifest = aggregate_mattergen_shards(
        [shard_b, shard_a], tmp_path / "aggregate", min_unique_structures=1
    )

    assert manifest["counts"]["parsed_structures"] == 2
    assert manifest["counts"]["exact_hash_unique_structures"] == 1
    assert manifest["counts"]["unique_structures"] == 1
    assert manifest["counts"]["exact_hash_duplicates"] == 1
    assert manifest["counts"]["matcher_equivalent_duplicates"] == 0
    excluded = manifest["deduplication"]["exact_hash"]["excluded"]
    assert len(excluded) == 1
    assert excluded[0]["excluded"]["source_shard"].endswith("shard_b")
    assert excluded[0]["excluded"]["source_member"] == "gen_0.cif"
    assert excluded[0]["representative"]["source_shard"].endswith("shard_a")
    assert excluded[0]["representative"]["source_member"] == "gen_0.cif"
    assert excluded[0]["representative"]["aggregate_member"] == (
        "crystal_000000.cif"
    )


def test_aggregate_deduplicates_structurematcher_equivalents_by_reduced_composition(
    tmp_path, model_files
):
    checkpoint, config = model_files
    primitive = _structure(3.57)
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])
    shard = _write_shard(
        tmp_path,
        "shard",
        [primitive, supercell],
        checkpoint=checkpoint,
        config=config,
        seed=1,
    )

    manifest = aggregate_mattergen_shards(
        [shard], tmp_path / "aggregate", min_unique_structures=1
    )

    assert manifest["counts"]["exact_hash_unique_structures"] == 2
    assert manifest["counts"]["unique_structures"] == 1
    assert manifest["counts"]["exact_hash_duplicates"] == 0
    assert manifest["counts"]["matcher_equivalent_duplicates"] == 1
    matcher = manifest["deduplication"]["structure_matcher"]
    assert matcher["grouping"] == "reduced_composition"
    assert matcher["ltol"] == 0.2
    assert matcher["stol"] == 0.3
    assert matcher["angle_tol"] == 5.0
    assert len(matcher["excluded"]) == 1
    duplicate = matcher["excluded"][0]["excluded"]
    assert duplicate["source_shard"] == str(shard.resolve())
    assert duplicate["source_member"] == "gen_1.cif"
    assert duplicate["formula"] == "C"
    assert duplicate["num_sites"] == 4
    assert len(duplicate["cif_sha256"]) == 64
    assert len(duplicate["structure_sha256"]) == 64
    representative = matcher["excluded"][0]["representative"]
    assert representative["aggregate_member"] == "crystal_000000.cif"
    assert representative["source_shard"] == str(shard.resolve())
    assert representative["source_member"] == "gen_0.cif"
    assert representative["formula"] == "C"
    assert representative["num_sites"] == 2
    assert representative["cif_sha256"] == manifest["outputs"]["candidates"][0][
        "cif_sha256"
    ]
    assert representative["structure_sha256"] == manifest["outputs"][
        "candidates"
    ][0]["structure_sha256"]


def test_aggregate_applies_minimum_after_matcher_deduplication(tmp_path, model_files):
    checkpoint, config = model_files
    primitive = _structure(3.57)
    shifted = primitive.copy()
    shifted.translate_sites(
        range(len(shifted)), [0.1, 0.1, 0.1], to_unit_cell=True
    )
    shard = _write_shard(
        tmp_path,
        "shard",
        [primitive, shifted],
        checkpoint=checkpoint,
        config=config,
        seed=1,
    )

    with pytest.raises(
        ValueError, match="requires at least 2 unique structures; found 1"
    ):
        aggregate_mattergen_shards(
            [shard], tmp_path / "aggregate", min_unique_structures=2
        )


def test_aggregate_uses_natural_shard_order_for_first_representative(
    tmp_path, model_files
):
    checkpoint, config = model_files
    repeated = _structure(3.57)
    shard_10 = _write_shard(
        tmp_path,
        "shard_10",
        [repeated],
        checkpoint=checkpoint,
        config=config,
        seed=10,
    )
    shard_2 = _write_shard(
        tmp_path,
        "shard_2",
        [repeated],
        checkpoint=checkpoint,
        config=config,
        seed=2,
    )

    manifest = aggregate_mattergen_shards(
        [shard_10, shard_2], tmp_path / "aggregate", min_unique_structures=1
    )

    assert manifest["outputs"]["candidates"][0]["source_shard"].endswith(
        "shard_2"
    )
    assert manifest["deduplication"]["exact_hash"]["excluded"][0][
        "excluded"
    ]["source_shard"].endswith("shard_10")


def test_aggregate_rejects_protocol_mismatch(tmp_path, model_files):
    checkpoint, config = model_files
    shard_a = _write_shard(
        tmp_path,
        "shard_a",
        [_structure(3.50)],
        checkpoint=checkpoint,
        config=config,
        guidance_factor=2.0,
        seed=1,
    )
    shard_b = _write_shard(
        tmp_path,
        "shard_b",
        [_structure(3.60)],
        checkpoint=checkpoint,
        config=config,
        guidance_factor=3.0,
        seed=2,
    )

    with pytest.raises(ValueError, match="protocol mismatch.*guidance_factor"):
        aggregate_mattergen_shards(
            [shard_a, shard_b], tmp_path / "aggregate", min_unique_structures=2
        )


def test_aggregate_rejects_shard_declared_count_mismatch(tmp_path, model_files):
    checkpoint, config = model_files
    shard = _write_shard(
        tmp_path,
        "shard",
        [_structure(3.50)],
        checkpoint=checkpoint,
        config=config,
        seed=1,
    )
    manifest_path = shard / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["generated_count"] = 2
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="declares inconsistent structure counts"):
        aggregate_mattergen_shards(
            [shard], tmp_path / "aggregate", min_unique_structures=1
        )


def test_aggregate_requires_1024_unique_structures_by_default(tmp_path, model_files):
    checkpoint, config = model_files
    shard = _write_shard(
        tmp_path,
        "shard",
        [_structure(3.50)],
        checkpoint=checkpoint,
        config=config,
        seed=1,
    )

    with pytest.raises(ValueError, match="requires at least 1024 unique structures"):
        aggregate_mattergen_shards([shard], tmp_path / "aggregate")
