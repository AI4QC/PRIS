from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
import pandas as pd
import pytest

from src.next11_geometry_only_frames import _canonical_frame
from src.next21_feature_build import FEATURE_NAME, MANIFEST_NAME, build_feature_batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive_path = tmp_path / "geometry_only_frames.zip"
    atoms = Atoms(
        "CsCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[4.2, 4.2, 4.2],
        pbc=True,
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo("cscl.extxyz", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, _canonical_frame(atoms))
    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ["cscl"],
            "rk": ["Cl1|Cs1"],
            "formula": ["CsCl"],
            "natoms": [2],
            "input_role": ["unrelaxed_x0_geometry_only"],
        }
    ).to_parquet(metadata_path, index=False)
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": "unit-fixture",
                "input_role": "unrelaxed_x0_geometry_only",
                "endpoint_fields_accessed": False,
                "scientific_improvement_claim": False,
                "outputs_sha256": {
                    "geometry_only_frames.zip": _sha256(archive_path)
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return archive_path, manifest_path, metadata_path


def test_builder_seals_independent_normalized_madelung_features(tmp_path: Path) -> None:
    archive, manifest, metadata = _inputs(tmp_path)
    output = tmp_path / "features"
    build_feature_batch(
        archive_path=archive,
        source_manifest_path=manifest,
        metadata_path=metadata,
        source_role="unit",
        output_dir=output,
    )
    feature_path = output / FEATURE_NAME
    frame = pd.read_parquet(feature_path)
    assert frame["material_id"].tolist() == ["cscl"]
    assert frame["nm_supported"].tolist() == [True]
    assert frame.loc[0, "nm_total_reduced"] < 0.0
    assert not any(
        token in column.lower()
        for column in frame
        for token in ("energy", "force", "stress", "relax", "mattersim", "dft")
    )
    sealed = json.loads((output / MANIFEST_NAME).read_text())
    assert sealed["counts"]["supported"] == 1
    assert sealed["endpoint_fields_read"] is False
    assert sealed["outputs_sha256"][FEATURE_NAME] == _sha256(feature_path)

    with pytest.raises(FileExistsError):
        build_feature_batch(
            archive_path=archive,
            source_manifest_path=manifest,
            metadata_path=metadata,
            source_role="unit",
            output_dir=output,
        )
