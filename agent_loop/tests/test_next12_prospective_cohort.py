"""Contract tests for the NEXT12 prospective geometry-only cohort freezer."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read


class _FakeRealAdapter:
    model_class = "fake.RealCIVAE"
    model_parameter_count = 12345
    latent_dim = 4
    device = "cpu"

    def generate(self, latent: np.ndarray, attempt_index: int) -> Atoms:
        if attempt_index == 1:
            raise RuntimeError("fixed generation failure")
        symbol = "H2" if attempt_index % 2 == 0 else "He2"
        shift = float(np.tanh(latent[0]) * 0.05)
        return Atoms(
            symbol,
            scaled_positions=[[0.1 + shift, 0.2, 0.3], [0.6, 0.7 - shift, 0.8]],
            cell=[[7.0, 0.0, 0.0], [0.2, 8.0, 0.0], [0.1, 0.3, 9.0]],
            pbc=True,
        )


class _FakeMockAdapter(_FakeRealAdapter):
    model_class = "sasgen.eval_utils.MockCDVAE"


class _FakeChdirAdapter(_FakeRealAdapter):
    def __init__(self, destination: Path) -> None:
        self._destination = destination

    def generate(self, latent: np.ndarray, attempt_index: int) -> Atoms:
        os.chdir(self._destination)
        return super().generate(latent, attempt_index)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for name, payload in (
        ("model.ckpt", b"checkpoint"),
        ("lattice_scaler.pt", b"lattice"),
        ("prop_scaler.pt", b"property"),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        result[name] = path
    return result


def test_freezer_retains_every_attempt_and_exports_geometry_only(tmp_path: Path) -> None:
    from src import next12_prospective_cohort as module

    inputs = _inputs(tmp_path)
    output_dir = tmp_path / "prospective"
    manifest = module.freeze_prospective_cohort(
        checkpoint_path=inputs["model.ckpt"],
        lattice_scaler_path=inputs["lattice_scaler.pt"],
        prop_scaler_path=inputs["prop_scaler.pt"],
        output_dir=output_dir,
        adapter=_FakeRealAdapter(),
        attempt_count=4,
        seed=20260802,
    )

    assert {path.name for path in output_dir.iterdir()} == {
        module.COHORT_NAME,
        module.ARCHIVE_NAME,
        module.MANIFEST_NAME,
    }
    table = pd.read_parquet(output_dir / module.COHORT_NAME)
    assert list(table.columns) == list(module.COHORT_COLUMNS)
    assert table["attempt_index"].tolist() == [0, 1, 2, 3]
    assert table["generation_status"].tolist() == [
        "generated",
        "failed",
        "generated",
        "generated",
    ]
    assert table["sid"].is_unique
    assert table.loc[1, "error"] == "RuntimeError: fixed generation failure"
    assert table.loc[1, "geometry_sha256"] is None
    assert table["latent_sha256"].map(lambda value: len(value) == 64).all()

    with zipfile.ZipFile(output_dir / module.ARCHIVE_NAME) as archive:
        assert archive.namelist() == [
            f"frames/{table.loc[index, 'sid']}.extxyz" for index in (0, 2, 3)
        ]
        for member in archive.namelist():
            payload = archive.read(member)
            text = payload.decode("utf-8")
            lowered = text.lower()
            assert "energy=" not in lowered
            assert "forces" not in lowered
            assert "stress" not in lowered
            assert "endpoint" not in lowered
            atoms = read(
                io.StringIO(text),
                format="extxyz",
                index=0,
                parallel=False,
                do_not_split_by_at_sign=True,
            )
            assert set(atoms.arrays) == {"numbers", "positions"}
            assert set(atoms.info) == set()

    loaded = json.loads((output_dir / module.MANIFEST_NAME).read_text("utf-8"))
    assert loaded == manifest
    assert manifest["protocol"] == module.PROTOCOL
    assert manifest["labels_opened"] is False
    assert manifest["energy_or_force_models_called"] is False
    assert manifest["all_attempts_retained"] is True
    assert manifest["generation"] == {
        "adapter_mode": "injected_test_double",
        "model_class": "fake.RealCIVAE",
        "model_parameter_count": 12345,
        "latent_dim": 4,
        "device": "cpu",
        "seed": 20260802,
        "attempt_count": 4,
    }
    assert manifest["counts"] == {
        "attempts": 4,
        "generated": 3,
        "failed": 1,
        "archive_frames": 3,
        "total_atoms": 6,
    }
    assert manifest["production_protocol_eligible"] is False
    assert manifest["inputs_sha256"] == {
        "checkpoint": {
            "path": str(inputs["model.ckpt"].resolve()),
            "sha256": _sha256(inputs["model.ckpt"]),
        },
        "lattice_scaler": {
            "path": str(inputs["lattice_scaler.pt"].resolve()),
            "sha256": _sha256(inputs["lattice_scaler.pt"]),
        },
        "prop_scaler": {
            "path": str(inputs["prop_scaler.pt"].resolve()),
            "sha256": _sha256(inputs["prop_scaler.pt"]),
        },
    }
    assert manifest["outputs_sha256"] == {
        module.COHORT_NAME: _sha256(output_dir / module.COHORT_NAME),
        module.ARCHIVE_NAME: _sha256(output_dir / module.ARCHIVE_NAME),
    }
    assert not list(tmp_path.glob(".prospective.staging-*"))


def test_same_seed_is_byte_reproducible_except_manifest_paths(tmp_path: Path) -> None:
    from src.next12_prospective_cohort import (
        ARCHIVE_NAME,
        COHORT_NAME,
        freeze_prospective_cohort,
    )

    inputs = _inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    kwargs = dict(
        checkpoint_path=inputs["model.ckpt"],
        lattice_scaler_path=inputs["lattice_scaler.pt"],
        prop_scaler_path=inputs["prop_scaler.pt"],
        adapter=_FakeRealAdapter(),
        attempt_count=4,
        seed=17,
    )
    freeze_prospective_cohort(output_dir=first, **kwargs)
    freeze_prospective_cohort(output_dir=second, **kwargs)
    assert (first / ARCHIVE_NAME).read_bytes() == (second / ARCHIVE_NAME).read_bytes()
    pd.testing.assert_frame_equal(
        pd.read_parquet(first / COHORT_NAME),
        pd.read_parquet(second / COHORT_NAME),
    )


def test_mock_model_is_rejected_before_any_output(tmp_path: Path) -> None:
    from src.next12_prospective_cohort import freeze_prospective_cohort

    inputs = _inputs(tmp_path)
    output_dir = tmp_path / "mock"
    with pytest.raises(RuntimeError, match="Mock"):
        freeze_prospective_cohort(
            checkpoint_path=inputs["model.ckpt"],
            lattice_scaler_path=inputs["lattice_scaler.pt"],
            prop_scaler_path=inputs["prop_scaler.pt"],
            output_dir=output_dir,
            adapter=_FakeMockAdapter(),
            attempt_count=2,
        )
    assert not output_dir.exists()


def test_freezer_never_overwrites_existing_output(tmp_path: Path) -> None:
    from src.next12_prospective_cohort import freeze_prospective_cohort

    inputs = _inputs(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        freeze_prospective_cohort(
            checkpoint_path=inputs["model.ckpt"],
            lattice_scaler_path=inputs["lattice_scaler.pt"],
            prop_scaler_path=inputs["prop_scaler.pt"],
            output_dir=output_dir,
            adapter=_FakeRealAdapter(),
            attempt_count=2,
        )
    assert sentinel.read_text("utf-8") == "keep"


def test_relative_output_is_bound_before_generator_changes_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.next12_prospective_cohort import freeze_prospective_cohort

    inputs = _inputs(tmp_path)
    initial = tmp_path / "initial"
    changed = tmp_path / "changed"
    initial.mkdir()
    changed.mkdir()
    monkeypatch.chdir(initial)
    freeze_prospective_cohort(
        checkpoint_path=inputs["model.ckpt"],
        lattice_scaler_path=inputs["lattice_scaler.pt"],
        prop_scaler_path=inputs["prop_scaler.pt"],
        output_dir=Path("prospective"),
        adapter=_FakeChdirAdapter(changed),
        attempt_count=1,
        seed=20260802,
    )
    assert (initial / "prospective" / "MANIFEST.json").is_file()
    assert not (changed / "prospective").exists()


def test_cli_exposes_no_label_energy_or_filter_argument() -> None:
    from src.next12_prospective_cohort import main

    for forbidden in ("--labels", "--energy", "--filter", "--endpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
