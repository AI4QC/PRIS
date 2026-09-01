"""Contracts for the additive NEXT24 SSAGEN geometry-only sanitizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame(symbols: list[str], *, shift: float = 0.0) -> bytes:
    lines = [
        str(len(symbols)),
        (
            'Lattice="5.1 0.2 0.1 0 6.2 0.3 0 0 7.3" '
            'Properties=species:S:1:pos:R:3 pbc="T T T"'
        ),
    ]
    lines.extend(
        f"{symbol} {0.2 + index + shift:.12f} {0.3 + index:.12f} {0.4 + index:.12f}"
        for index, symbol in enumerate(symbols)
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _source(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, bytes]]:
    from src.next12_prospective_cohort import PROTOCOL

    payloads = {
        "ssagen-test-a0000": _frame(["Li", "O"]),
        "ssagen-test-a0001": _frame(["Na", "Cl", "Cl"], shift=0.05),
        "ssagen-test-a0002": _frame(["S", "O", "O"], shift=-0.03),
    }
    archive = tmp_path / "source_frames.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for sid, payload in payloads.items():
            handle.writestr(f"frames/{sid}.extxyz", payload)
    cohort = tmp_path / "cohort.parquet"
    rows = []
    formulas = ["LiO", "NaCl2", "SO2"]
    for index, ((sid, payload), formula) in enumerate(zip(payloads.items(), formulas)):
        rows.append(
            {
                "attempt_index": index,
                "sid": sid,
                "generator": "SSAGEN-CIVAE-Transformer-500",
                "generation_status": "generated",
                "natoms": int(payload.decode().splitlines()[0]),
                "formula": formula,
                "geometry_sha256": hashlib.sha256(payload).hexdigest(),
                "archive_member": f"frames/{sid}.extxyz",
                "latent_sha256": hashlib.sha256(f"latent-{index}".encode()).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_parquet(cohort, index=False)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "mode": "prospective_x0_geometry_freeze",
                "all_attempts_retained": True,
                "labels_opened": False,
                "energy_or_force_models_called": False,
                "production_protocol_eligible": True,
                "scientific_improvement_claim": False,
                "counts": {
                    "attempts": 3,
                    "generated": 3,
                    "failed": 0,
                    "archive_frames": 3,
                    "total_atoms": 8,
                },
                "outputs_sha256": {
                    "cohort.parquet": _sha(cohort),
                    "geometry_only_frames.zip": _sha(archive),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return cohort, archive, manifest, payloads


def test_sanitizer_projects_and_preserves_every_generated_x0(tmp_path: Path) -> None:
    from src import next24_ssagen_holdout as module
    from src.next11_geometry_only_frames import _load_archive_only, _parse_frame

    cohort, archive, manifest, payloads = _source(tmp_path)
    output = tmp_path / "next24"
    result = module.freeze_ssagen_x0(
        cohort_path=cohort,
        frames_zip_path=archive,
        source_manifest_path=manifest,
        output_dir=output,
        require_formal_inputs=False,
    )

    assert {path.name for path in output.iterdir()} == {
        module.METADATA_NAME,
        module.GEOMETRY_NAME,
        module.MANIFEST_NAME,
    }
    metadata = pd.read_parquet(output / module.METADATA_NAME)
    assert metadata.columns.tolist() == [
        "material_id",
        "rk",
        "formula",
        "natoms",
        "input_role",
    ]
    assert metadata.material_id.tolist() == sorted(payloads)
    assert metadata.rk.eq("ssagen_civae_transformer_500").all()
    assert metadata.input_role.eq("unrelaxed_x0_geometry_only").all()

    ids, canonical = _load_archive_only(
        output / module.GEOMETRY_NAME, tuple(metadata.material_id)
    )
    assert ids == metadata.material_id.tolist()
    for sid, atoms in zip(ids, canonical, strict=True):
        original = _parse_frame(payloads[sid], strict_output=True).atoms
        assert atoms.get_chemical_symbols() == original.get_chemical_symbols()
        np.testing.assert_array_equal(atoms.positions, original.positions)
        np.testing.assert_array_equal(atoms.cell.array, original.cell.array)

    assert result["protocol"] == module.PROTOCOL
    assert result["input_role"] == "unrelaxed_x0_geometry_only"
    assert result["labels_opened"] is False
    assert result["endpoint_artifacts_opened"] is False
    assert result["relaxed_structures_opened"] is False
    assert result["model_or_proxy_potential_used"] is False
    assert result["coordinates_or_cell_modified"] is False
    assert result["all_generated_attempts_retained"] is True
    assert result["counts"] == {"rows": 3, "frames": 3, "atoms": 8}
    assert result["inputs_sha256"] == {
        "cohort": {"path": str(cohort.resolve()), "sha256": _sha(cohort)},
        "geometry_only_frames": {
            "path": str(archive.resolve()),
            "sha256": _sha(archive),
        },
        "source_manifest": {
            "path": str(manifest.resolve()),
            "sha256": _sha(manifest),
        },
    }
    assert result["outputs_sha256"] == {
        module.METADATA_NAME: _sha(output / module.METADATA_NAME),
        module.GEOMETRY_NAME: _sha(output / module.GEOMETRY_NAME),
    }


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("labels", "boundary"),
        ("proxy", "boundary"),
        ("failed", "generated"),
        ("frame_hash", "frame hash"),
        ("duplicate", "unique"),
        ("formula", "formula"),
    ],
)
def test_sanitizer_rejects_boundary_or_identity_drift(
    tmp_path: Path, mutation: str, match: str
) -> None:
    from src.next24_ssagen_holdout import freeze_ssagen_x0

    cohort, archive, manifest, _payloads = _source(tmp_path)
    source_manifest = json.loads(manifest.read_text())
    table = pd.read_parquet(cohort)
    if mutation == "labels":
        source_manifest["labels_opened"] = True
    elif mutation == "proxy":
        source_manifest["energy_or_force_models_called"] = True
    elif mutation == "failed":
        table.loc[1, "generation_status"] = "failed"
    elif mutation == "frame_hash":
        table.loc[1, "geometry_sha256"] = "0" * 64
    elif mutation == "duplicate":
        table.loc[1, "sid"] = table.loc[0, "sid"]
    elif mutation == "formula":
        table.loc[1, "formula"] = "KBr"
    if mutation in {"failed", "frame_hash", "duplicate", "formula"}:
        table.to_parquet(cohort, index=False)
        source_manifest["outputs_sha256"]["cohort.parquet"] = _sha(cohort)
    manifest.write_text(json.dumps(source_manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        freeze_ssagen_x0(
            cohort_path=cohort,
            frames_zip_path=archive,
            source_manifest_path=manifest,
            output_dir=tmp_path / "out",
            require_formal_inputs=False,
        )


def test_sanitizer_refuses_overwrite_and_endpoint_cli(tmp_path: Path) -> None:
    from src.next24_ssagen_holdout import freeze_ssagen_x0, main

    cohort, archive, manifest, _payloads = _source(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        freeze_ssagen_x0(
            cohort_path=cohort,
            frames_zip_path=archive,
            source_manifest_path=manifest,
            output_dir=output,
            require_formal_inputs=False,
        )
    assert marker.read_text() == "keep"

    for forbidden in ("--labels", "--endpoint", "--relaxed", "--energy"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2

