from __future__ import annotations

import hashlib
import errno
import json
from pathlib import Path
import zipfile

from ase import Atoms
import pandas as pd
import pytest
import src.next19_feature_build as feature_builder

from src.next11_geometry_only_frames import _canonical_frame
from src.next19_feature_build import (
    FEATURE_NAME,
    MANIFEST_NAME,
    build_feature_batch,
    validate_geometry_metadata,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_archive(path: Path, frames: dict[str, Atoms]) -> None:
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for sid in sorted(frames):
            info = zipfile.ZipInfo(f"{sid}.extxyz", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _canonical_frame(frames[sid]))


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive = tmp_path / "geometry_only_frames.zip"
    frames = {
        "s1": Atoms(
            "CsCl",
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            cell=[4.2, 4.2, 4.2],
            pbc=True,
        ),
        "s2": Atoms(
            "NaCl",
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            cell=[3.5, 3.5, 3.5],
            pbc=True,
        ),
    }
    _write_archive(archive, frames)
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ["s2", "s1"],
            "rk": ["NaCl", "ClCs"],
            "formula": ["NaCl", "CsCl"],
            "natoms": [2, 2],
            "input_role": ["unrelaxed_x0_geometry_only"] * 2,
        }
    ).to_parquet(metadata, index=False)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "unit-fixture",
                "input_role": "unrelaxed_x0_geometry_only",
                "endpoint_fields_accessed": False,
                "scientific_improvement_claim": False,
                "outputs_sha256": {
                    "geometry_only_frames.zip": _sha256(archive)
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return archive, manifest, metadata


def test_metadata_contract_rejects_endpoint_like_columns() -> None:
    table = pd.DataFrame(
        {
            "material_id": ["s1"],
            "rk": ["NaCl"],
            "formula": ["NaCl"],
            "natoms": [2],
            "input_role": ["unrelaxed_x0_geometry_only"],
            "dft_energy": [-1.0],
        }
    )

    with pytest.raises(ValueError, match="forbidden metadata column"):
        validate_geometry_metadata(table)


def test_builder_writes_deterministic_geometry_only_feature_contract(
    tmp_path: Path,
) -> None:
    archive, manifest, metadata = _fixture_inputs(tmp_path)
    output = tmp_path / "features"

    build_feature_batch(
        archive_path=archive,
        source_manifest_path=manifest,
        metadata_path=metadata,
        source_role="unit",
        output_dir=output,
        graph_modes=("crystalnn",),
        alphas=(0.0, 2.0),
    )

    feature_path = output / FEATURE_NAME
    manifest_path = output / MANIFEST_NAME
    assert feature_path.is_file()
    assert manifest_path.is_file()
    features = pd.read_parquet(feature_path)
    assert features["material_id"].tolist() == ["s1", "s2"]
    assert len(features) == 2
    assert set(features["valence_policy"]) == {"integer_oxidation_state"}
    assert "crystalnn_a0__vt_overload" in features
    assert "crystalnn_a2__vt_reallocation" in features
    assert not any(
        token in column.lower()
        for column in features
        for token in ("energy", "force", "stress", "relax", "mattersim", "dft")
    )
    sealed = json.loads(manifest_path.read_text())
    assert sealed["counts"]["rows"] == 2
    assert sealed["endpoint_fields_read"] is False
    assert sealed["outputs_sha256"][FEATURE_NAME] == _sha256(feature_path)

    with pytest.raises(FileExistsError):
        build_feature_batch(
            archive_path=archive,
            source_manifest_path=manifest,
            metadata_path=metadata,
            source_role="unit",
            output_dir=output,
            graph_modes=("crystalnn",),
            alphas=(0.0, 2.0),
        )


def test_builder_rejects_source_archive_hash_mismatch(tmp_path: Path) -> None:
    archive, manifest, metadata = _fixture_inputs(tmp_path)
    source = json.loads(manifest.read_text())
    source["outputs_sha256"]["geometry_only_frames.zip"] = "0" * 64
    manifest.write_text(json.dumps(source) + "\n")

    with pytest.raises(ValueError, match="archive hash mismatch"):
        build_feature_batch(
            archive_path=archive,
            source_manifest_path=manifest,
            metadata_path=metadata,
            source_role="unit",
            output_dir=tmp_path / "features",
            graph_modes=("crystalnn",),
            alphas=(0.0,),
        )


def test_builder_accepts_legacy_next14_wbm_x0_manifest_contract(
    tmp_path: Path,
) -> None:
    archive, manifest, metadata = _fixture_inputs(tmp_path)
    source = json.loads(manifest.read_text())
    source.pop("input_role")
    source.update(
        {
            "protocol": "2026-08-02-next14-wbm-acsc-label-free-holdout-v1",
            "mode": "external_source_label_free_small_cell_holdout",
            "endpoint_artifacts_opened": False,
            "labels_opened": False,
            "relaxed_structures_opened": False,
            "production_protocol_eligible": True,
        }
    )
    manifest.write_text(json.dumps(source) + "\n")

    output = tmp_path / "features"
    build_feature_batch(
        archive_path=archive,
        source_manifest_path=manifest,
        metadata_path=metadata,
        source_role="legacy-wbm",
        output_dir=output,
        graph_modes=("crystalnn",),
        alphas=(0.0,),
    )

    assert (output / FEATURE_NAME).is_file()


def test_builder_falls_back_when_filesystem_rejects_rename_noreplace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manifest, metadata = _fixture_inputs(tmp_path)

    def unsupported(_source: Path, target: Path) -> None:
        raise OSError(errno.EINVAL, "unsupported rename flag", str(target))

    monkeypatch.setattr(
        feature_builder, "_atomic_publish_directory_no_replace", unsupported
    )
    output = tmp_path / "features"
    build_feature_batch(
        archive_path=archive,
        source_manifest_path=manifest,
        metadata_path=metadata,
        source_role="unit-fallback",
        output_dir=output,
        graph_modes=("crystalnn",),
        alphas=(0.0,),
    )

    assert (output / FEATURE_NAME).is_file()
    assert not (tmp_path / ".features.publish.lock").exists()
