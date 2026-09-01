from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
import pandas as pd
import pytest

import src.next20_feature_build as feature_builder
from src.next11_geometry_only_frames import _canonical_frame
from src.next20_feature_build import FEATURE_NAME, MANIFEST_NAME, build_feature_batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive_path = tmp_path / "geometry_only_frames.zip"
    frames = {
        "nacl": Atoms(
            "NaCl",
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            cell=[5.64, 5.64, 5.64],
            pbc=True,
        ),
        "cscl": Atoms(
            "CsCl",
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            cell=[4.2, 4.2, 4.2],
            pbc=True,
        ),
    }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for sid in sorted(frames):
            info = zipfile.ZipInfo(f"{sid}.extxyz", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _canonical_frame(frames[sid]))
    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ["nacl", "cscl"],
            "rk": ["Cl1|Na1", "Cl1|Cs1"],
            "formula": ["NaCl", "CsCl"],
            "natoms": [2, 2],
            "input_role": ["unrelaxed_x0_geometry_only"] * 2,
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


def test_builder_reuses_each_neighbor_graph_across_weight_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manifest, metadata = _fixture(tmp_path)
    real_builder = feature_builder.build_periodic_edge_geometry
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(feature_builder, "build_periodic_edge_geometry", counted)
    output = tmp_path / "features"
    build_feature_batch(
        archive_path=archive,
        source_manifest_path=manifest,
        metadata_path=metadata,
        source_role="unit",
        output_dir=output,
        graph_modes=("voronoi",),
        charge_weight_exponents=(0.0, 0.5),
    )
    assert calls == 2

    feature_path = output / FEATURE_NAME
    features = pd.read_parquet(feature_path)
    assert features["material_id"].tolist() == ["cscl", "nacl"]
    assert features["voronoi_q0__supported"].all()
    assert features["voronoi_q05__supported"].all()
    assert "voronoi_q05__sivr_negative_mode_fraction" in features
    assert not any(
        token in column.lower()
        for column in features
        for token in ("energy", "force", "stress", "relax", "mattersim", "dft")
    )
    sealed = json.loads((output / MANIFEST_NAME).read_text())
    assert sealed["counts"]["rows"] == 2
    assert sealed["endpoint_fields_read"] is False
    assert sealed["model_or_proxy_potential_used"] is False
    assert sealed["outputs_sha256"][FEATURE_NAME] == _sha256(feature_path)

    with pytest.raises(FileExistsError):
        build_feature_batch(
            archive_path=archive,
            source_manifest_path=manifest,
            metadata_path=metadata,
            source_role="unit",
            output_dir=output,
            graph_modes=("voronoi",),
            charge_weight_exponents=(0.0, 0.5),
        )
