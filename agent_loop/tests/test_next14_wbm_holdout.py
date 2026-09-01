"""Contracts for label-free deterministic WBM holdout geometry freezing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(material_id: str, symbols: list[str], *, marker: str) -> str:
    lines = [
        str(len(symbols)),
        f'Lattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 material_id={material_id} energy=-99 marker={marker} pbc="T T T"',
    ]
    lines.extend(f"{symbol} {0.2 + i} {0.3 + i} {0.4 + i}" for i, symbol in enumerate(symbols))
    return "\n".join(lines) + "\n"


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    ids = ["wbm-a", "wbm-b", "wbm-c", "wbm-d", "wbm-big"]
    features = tmp_path / "test_x0_features.parquet"
    pd.DataFrame({"material_id": ids, "feature_ok": [True] * len(ids)}).to_parquet(features, index=False)
    archive = tmp_path / "initial.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("wbm-a.extxyz", _frame("wbm-a", ["Li", "O"], marker="SECRET-A"))
        handle.writestr("wbm-b.extxyz", _frame("wbm-b", ["Na", "Cl"], marker="SECRET-B"))
        handle.writestr("wbm-c.extxyz", _frame("wbm-c", ["K", "Br"], marker="SECRET-C"))
        handle.writestr("wbm-d.extxyz", _frame("wbm-d", ["Mg", "O", "O"], marker="SECRET-D"))
        handle.writestr("wbm-big.extxyz", _frame("wbm-big", ["C"] * 5, marker="SECRET-BIG"))
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-01-dft-pre-screening-design-v1",
                "input_role": "unrelaxed_x0_only",
                "outputs_sha256": {features.name: _sha(features)},
                "inputs": {"initial_zip_sha256": _sha(archive)},
            }
        ),
        encoding="utf-8",
    )
    return features, archive, manifest


def test_holdout_is_hash_selected_size_bounded_and_metadata_free(tmp_path: Path) -> None:
    from src import next14_wbm_holdout as module

    features, archive, manifest = _inputs(tmp_path)
    output = tmp_path / "holdout"
    result = module.freeze_wbm_holdout(
        test_features_path=features,
        wbm_manifest_path=manifest,
        initial_zip_path=archive,
        output_dir=output,
        sample_size=2,
        min_atoms=2,
        max_atoms=4,
        require_formal_inputs=False,
    )
    metadata = pd.read_parquet(output / module.METADATA_NAME)
    expected = sorted(
        ["wbm-a", "wbm-b", "wbm-c", "wbm-d"],
        key=lambda value: module.selection_key(value),
    )[:2]
    assert metadata.material_id.tolist() == sorted(expected)
    assert metadata.natoms.between(2, 4).all()
    assert result["counts"] == {
        "source_test_rows": 5,
        "eligible_rows": 4,
        "selected_rows": 2,
        "total_atoms": int(metadata.natoms.sum()),
    }
    assert result["labels_opened"] is False
    assert result["relaxed_structures_opened"] is False
    with zipfile.ZipFile(output / module.GEOMETRY_NAME) as frozen:
        assert frozen.namelist() == [f"{material_id}.extxyz" for material_id in sorted(expected)]
        payload = b"\n".join(frozen.read(name) for name in frozen.namelist())
        assert b"energy" not in payload and b"marker" not in payload and b"SECRET" not in payload
        assert b'Properties="species:S:1:pos:R:3"' in payload


def test_holdout_refuses_insufficient_rows_overwrite_and_label_cli(tmp_path: Path) -> None:
    from src.next14_wbm_holdout import freeze_wbm_holdout, main

    features, archive, manifest = _inputs(tmp_path)
    with pytest.raises(ValueError, match="eligible"):
        freeze_wbm_holdout(
            test_features_path=features,
            wbm_manifest_path=manifest,
            initial_zip_path=archive,
            output_dir=tmp_path / "too-many",
            sample_size=5,
            min_atoms=2,
            max_atoms=4,
            require_formal_inputs=False,
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        freeze_wbm_holdout(
            test_features_path=features,
            wbm_manifest_path=manifest,
            initial_zip_path=archive,
            output_dir=existing,
            sample_size=2,
            require_formal_inputs=False,
        )
    for forbidden in ("--labels", "--summary", "--relaxed-zip"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2


def test_external_filesystem_publication_fallback_reserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next14_wbm_holdout as module

    features, archive, manifest = _inputs(tmp_path)

    def unsupported(_source: Path, _target: Path) -> None:
        raise OSError(22, "atomic no-replace unsupported")

    monkeypatch.setattr(module, "_atomic_publish_directory_no_replace", unsupported)
    output = tmp_path / "fallback"
    module.freeze_wbm_holdout(
        test_features_path=features,
        wbm_manifest_path=manifest,
        initial_zip_path=archive,
        output_dir=output,
        sample_size=2,
        min_atoms=2,
        max_atoms=4,
        require_formal_inputs=False,
    )
    assert (output / module.MANIFEST_NAME).is_file()
    assert not any(path.name.startswith(".fallback.staging-") for path in tmp_path.iterdir())
