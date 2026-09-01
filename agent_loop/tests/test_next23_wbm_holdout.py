"""Contracts for a development-disjoint, label-free NEXT23 WBM cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(material_id: str, symbols: list[str], *, secret: str) -> str:
    lines = [
        str(len(symbols)),
        (
            'Lattice="5 0 0 0 5 0 0 0 5" '
            f"Properties=species:S:1:pos:R:3 material_id={material_id} "
            f'energy=-99 private={secret} pbc="T T T"'
        ),
    ]
    lines.extend(
        f"{symbol} {0.25 + index} {0.35 + index} {0.45 + index}"
        for index, symbol in enumerate(symbols)
    )
    return "\n".join(lines) + "\n"


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    ids = [f"wbm-{letter}" for letter in "abcdefgh"] + ["wbm-too-big"]
    features = tmp_path / "test_x0_features.parquet"
    pd.DataFrame({"material_id": ids, "feature_ok": [True] * len(ids)}).to_parquet(
        features, index=False
    )
    archive = tmp_path / "initial.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for index, material_id in enumerate(ids[:-1]):
            symbols = ["Li", "O"] if index % 2 == 0 else ["Na", "Cl", "Cl"]
            handle.writestr(
                f"{material_id}.extxyz",
                _frame(material_id, symbols, secret=f"SECRET-{index}"),
            )
        handle.writestr(
            "wbm-too-big.extxyz",
            _frame("wbm-too-big", ["C"] * 6, secret="SECRET-BIG"),
        )
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
    return features, archive, manifest, ids


def test_holdout_excludes_development_ids_before_salted_selection(tmp_path: Path) -> None:
    from src import next23_wbm_holdout as module

    features, archive, manifest, ids = _inputs(tmp_path)
    exclusion = tmp_path / "development_metadata.parquet"
    excluded = ["wbm-b", "wbm-f"]
    pd.DataFrame(
        {
            "material_id": excluded,
            "rk": ["NaCl2", "NaCl2"],
            "natoms": [3, 3],
            "input_role": ["unrelaxed_x0_geometry_only"] * 2,
        }
    ).to_parquet(exclusion, index=False)
    output = tmp_path / "blind"

    result = module.freeze_disjoint_wbm_holdout(
        test_features_path=features,
        wbm_manifest_path=manifest,
        initial_zip_path=archive,
        exclusion_metadata_path=exclusion,
        output_dir=output,
        sample_size=3,
        min_atoms=2,
        max_atoms=4,
        require_formal_inputs=False,
    )

    eligible = [value for value in ids[:-1] if value not in excluded]
    expected = sorted(eligible, key=module.selection_key)[:3]
    metadata = pd.read_parquet(output / module.METADATA_NAME)
    assert metadata.material_id.tolist() == sorted(expected)
    assert not set(metadata.material_id) & set(excluded)
    assert result["counts"] == {
        "source_test_rows": 9,
        "source_exclusion_rows": 2,
        "size_eligible_rows": 8,
        "excluded_size_eligible_rows": 2,
        "selection_eligible_rows": 6,
        "selected_rows": 3,
        "total_atoms": int(metadata.natoms.sum()),
    }
    assert result["labels_opened"] is False
    assert result["relaxed_structures_opened"] is False
    assert result["input_role"] == "unrelaxed_x0_geometry_only"
    assert result["selection"]["ranking"].startswith("ascending SHA-256")
    assert result["selection"]["excluded_material_ids_sha256"] == hashlib.sha256(
        "\n".join(sorted(excluded)).encode("utf-8") + b"\n"
    ).hexdigest()
    assert result["inputs_sha256"]["exclusion_metadata"]["sha256"] == _sha(exclusion)

    with zipfile.ZipFile(output / module.GEOMETRY_NAME) as frozen:
        assert frozen.namelist() == [f"{value}.extxyz" for value in sorted(expected)]
        payload = b"\n".join(frozen.read(name) for name in frozen.namelist())
        assert b"energy" not in payload
        assert b"private" not in payload
        assert b"SECRET" not in payload


def test_holdout_rejects_label_bearing_or_invalid_exclusion_metadata(tmp_path: Path) -> None:
    from src.next23_wbm_holdout import freeze_disjoint_wbm_holdout

    features, archive, manifest, _ids = _inputs(tmp_path)
    cases = [
        pd.DataFrame({"material_id": ["wbm-a"], "target": [1]}),
        pd.DataFrame({"material_id": ["wbm-a"], "dft_energy": [-1.0]}),
        pd.DataFrame({"material_id": ["wbm-a", "wbm-a"]}),
        pd.DataFrame({"rk": ["LiO"]}),
    ]
    for index, frame in enumerate(cases):
        exclusion = tmp_path / f"invalid-{index}.parquet"
        frame.to_parquet(exclusion, index=False)
        with pytest.raises(ValueError):
            freeze_disjoint_wbm_holdout(
                test_features_path=features,
                wbm_manifest_path=manifest,
                initial_zip_path=archive,
                exclusion_metadata_path=exclusion,
                output_dir=tmp_path / f"out-{index}",
                sample_size=2,
                min_atoms=2,
                max_atoms=4,
                require_formal_inputs=False,
            )


def test_holdout_refuses_overlap_shortfall_overwrite_and_label_cli(tmp_path: Path) -> None:
    from src.next23_wbm_holdout import freeze_disjoint_wbm_holdout, main

    features, archive, manifest, ids = _inputs(tmp_path)
    exclusion = tmp_path / "exclude-all.parquet"
    pd.DataFrame({"material_id": ids[:-1]}).to_parquet(exclusion, index=False)
    with pytest.raises(ValueError, match="eligible"):
        freeze_disjoint_wbm_holdout(
            test_features_path=features,
            wbm_manifest_path=manifest,
            initial_zip_path=archive,
            exclusion_metadata_path=exclusion,
            output_dir=tmp_path / "shortfall",
            sample_size=1,
            min_atoms=2,
            max_atoms=4,
            require_formal_inputs=False,
        )

    empty_exclusion = tmp_path / "empty.parquet"
    pd.DataFrame({"material_id": pd.Series(dtype=str)}).to_parquet(
        empty_exclusion, index=False
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        freeze_disjoint_wbm_holdout(
            test_features_path=features,
            wbm_manifest_path=manifest,
            initial_zip_path=archive,
            exclusion_metadata_path=empty_exclusion,
            output_dir=existing,
            sample_size=2,
            require_formal_inputs=False,
        )
    for forbidden in ("--labels", "--summary", "--relaxed-zip", "--endpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
