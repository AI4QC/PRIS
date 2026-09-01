"""Contracts for an additive, label-free ELEMENTA group holdout."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(symbols: str) -> str:
    atoms = Atoms(symbols, positions=[[float(i), 0.0, 0.0] for i in range(len(Atoms(symbols)))], cell=[8, 8, 8], pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1.0, forces=np.zeros((len(atoms), 3)), stress=np.zeros(6)
    )
    stream = io.StringIO()
    write(stream, atoms, format="extxyz")
    return stream.getvalue()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    frames = tmp_path / "elementa_initial_frames.zip"
    with zipfile.ZipFile(frames, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for group, formula in [("Li2|O1", "Li2O"), ("Na1|Cl1", "NaCl"), ("K1|Br1", "KBr")]:
            for index in range(2):
                sid = f"{formula}-{index}"
                rows.append(
                    {
                        "sid": sid,
                        "rk": group,
                        "material": f"{formula}_{index:02d}",
                        "input_role": "unrelaxed_x0_only",
                    }
                )
                archive.writestr(f"{sid}.extxyz", _frame(formula))
    features = tmp_path / "elementa_x0_features.parquet"
    pd.DataFrame(rows).to_parquet(features, index=False)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-01-dft-pre-screening-design-v1",
                "input_role": "unrelaxed_x0_only",
                "outputs_sha256": {features.name: _sha(features), frames.name: _sha(frames)},
            }
        ),
        encoding="utf-8",
    )
    return features, frames, manifest


def test_group_holdout_is_complete_deterministic_and_label_free(tmp_path: Path) -> None:
    from src import next16_elementa_holdout as module

    features, frames, upstream_manifest = _inputs(tmp_path)
    output = tmp_path / "holdout"
    result = module.build_elementa_holdout(
        features_path=features,
        frames_zip_path=frames,
        upstream_manifest_path=upstream_manifest,
        output_dir=output,
        group_count=2,
        require_formal_inputs=False,
    )
    metadata = pd.read_parquet(output / module.METADATA_NAME)
    assert metadata.rk.nunique() == 2
    assert len(metadata) == 4
    assert metadata.groupby("rk").size().eq(2).all()
    assert set(metadata.input_role) == {"unrelaxed_x0_geometry_only"}
    assert result["endpoint_bytes_read_by_execution"] is False
    assert result["labels_previously_opened_elsewhere"] is True
    with zipfile.ZipFile(output / module.GEOMETRY_NAME) as archive:
        assert sorted(archive.namelist()) == sorted(f"{sid}.extxyz" for sid in metadata.material_id)
    from src.next11_geometry_only_frames import _load_archive_only

    loaded_sids, structures = _load_archive_only(
        output / module.GEOMETRY_NAME, tuple(metadata.material_id.astype(str))
    )
    assert loaded_sids == metadata.material_id.astype(str).tolist()
    assert all(atoms.calc is None and not atoms.info and set(atoms.arrays) == {"numbers", "positions"} for atoms in structures)
    with pytest.raises(FileExistsError):
        module.build_elementa_holdout(
            features_path=features,
            frames_zip_path=frames,
            upstream_manifest_path=upstream_manifest,
            output_dir=output,
            group_count=2,
            require_formal_inputs=False,
        )


def test_group_holdout_rejects_partial_or_mismatched_geometry_archive(tmp_path: Path) -> None:
    from src.next16_elementa_holdout import build_elementa_holdout

    features, frames, upstream_manifest = _inputs(tmp_path)
    with zipfile.ZipFile(frames, "a") as archive:
        archive.writestr("extra.extxyz", _frame("Li"))
    manifest = json.loads(upstream_manifest.read_text(encoding="utf-8"))
    manifest["outputs_sha256"][frames.name] = _sha(frames)
    upstream_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="member set"):
        build_elementa_holdout(
            features_path=features,
            frames_zip_path=frames,
            upstream_manifest_path=upstream_manifest,
            output_dir=tmp_path / "bad",
            group_count=2,
            require_formal_inputs=False,
        )


def test_group_holdout_cli_cannot_read_labels_or_choose_rows() -> None:
    from src.next16_elementa_holdout import main

    for forbidden in ("--labels", "--sids", "--threshold", "--exclude"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
