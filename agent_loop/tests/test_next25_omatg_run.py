"""Contracts for the isolated NEXT25 OMatG generation runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from ase import Atoms
from ase.io import write
import pandas as pd
import pytest
import yaml


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    from src.next25_omatg_compositions import PROTOCOL

    composition_dir = tmp_path / "compositions"
    composition_dir.mkdir(parents=True)
    cohort = composition_dir / "composition_cohort.parquet"
    pd.DataFrame(
        {
            "material_id": ["next25-test-0000", "next25-test-0001"],
            "source_split": ["test", "test"],
            "source_index": [3, 8],
            "formula": ["Li2O", "NaCl"],
            "reduced_formula": ["Li2O", "NaCl"],
            "atomic_numbers_json": ["[3,3,8]", "[11,17]"],
            "natoms": [3, 2],
            "selection_key": ["1" * 64, "2" * 64],
            "selection_rank": [0, 1],
            "input_role": ["composition_only", "composition_only"],
        }
    ).to_parquet(cohort, index=False)
    lmdb_path = composition_dir / "compositions_only.lmdb"
    lmdb_path.write_bytes(b"dummy-lmdb")
    manifest = composition_dir / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "input_role": "composition_only",
                "reference_geometry_fields_accessed": False,
                "property_label_fields_accessed": False,
                "labels_opened": False,
                "counts": {"selected_rows": 2, "selected_atoms": 5},
                "outputs_sha256": {
                    cohort.name: _sha(cohort),
                    lmdb_path.name: _sha(lmdb_path),
                },
            }
        ),
        encoding="utf-8",
    )
    official_config = tmp_path / "train.yaml"
    official_config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "si": {"init_args": {"integration_time_steps": 210}},
                    "generation_xyz_filename": None,
                },
                "data": {
                    "train_dataset": {
                        "class_path": "omg.datamodule.StructureDataset",
                        "init_args": {"file_path": "data/mp_20/train.lmdb"},
                    },
                    "val_dataset": {
                        "class_path": "omg.datamodule.StructureDataset",
                        "init_args": {"file_path": "data/mp_20/val.lmdb"},
                    },
                    "pred_dataset": {
                        "class_path": "omg.datamodule.StructureDataset",
                        "init_args": {"file_path": "data/mp_20/test.lmdb"},
                    },
                    "batch_size": 128,
                    "num_workers": 4,
                    "pin_memory": True,
                    "persistent_workers": True,
                },
                "trainer": {"callbacks": [{"class_path": "SecretCallback"}]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"official-checkpoint")
    runtime_python_real = tmp_path / "python-real"
    runtime_python_real.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_python_real.chmod(0o755)
    runtime_python = tmp_path / "python"
    runtime_python.symlink_to(runtime_python_real.name)
    source_dir = tmp_path / "OMatG"
    (source_dir / "omg").mkdir(parents=True)
    (source_dir / "omg" / "__init__.py").write_text("", encoding="utf-8")
    return {
        "composition_dir": composition_dir,
        "official_config": official_config,
        "checkpoint": checkpoint,
        "runtime_python": runtime_python,
        "source_dir": source_dir,
    }


def test_runtime_config_and_command_remove_every_reference_dataset_path(
    tmp_path: Path,
) -> None:
    from src.next25_omatg_run import build_predict_command, write_safe_runtime_config

    paths = _inputs(tmp_path)
    safe = tmp_path / "runtime.yaml"
    composition_lmdb = paths["composition_dir"] / "compositions_only.lmdb"
    write_safe_runtime_config(
        official_config_path=paths["official_config"],
        composition_lmdb_path=composition_lmdb,
        output_path=tmp_path / "generated.xyz",
        runtime_config_path=safe,
        batch_size=2,
    )
    text = safe.read_text(encoding="utf-8")
    assert "data/mp_20" not in text
    assert "SecretCallback" not in text
    config = yaml.safe_load(text)
    for split in ("train_dataset", "val_dataset", "pred_dataset"):
        assert config["data"][split]["init_args"]["file_path"] == str(
            composition_lmdb.resolve()
        )
        assert config["data"][split]["init_args"]["lazy_storage"] is True
    assert config["data"]["batch_size"] == 2
    assert config["data"]["num_workers"] == 0
    assert config["data"]["pin_memory"] is False
    assert config["data"]["persistent_workers"] is False
    assert config["trainer"]["accelerator"] == "cpu"
    assert config["trainer"]["devices"] == 1
    assert config["trainer"]["enable_checkpointing"] is False

    command = build_predict_command(
        runtime_python=paths["runtime_python"],
        runtime_config_path=safe,
        checkpoint_path=paths["checkpoint"],
        seed=250803,
    )
    joined = " ".join(command)
    assert command[:4] == [
        str(paths["runtime_python"].absolute()),
        "-m",
        "omg.main",
        "predict",
    ]
    assert "data/mp_20" not in joined
    assert "test.lmdb" not in joined
    assert "--seed_everything=250803" in command


def test_runner_seals_full_unfiltered_output_and_safe_provenance(tmp_path: Path) -> None:
    from src import next25_omatg_run as module

    paths = _inputs(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_runner(command, **kwargs):
        calls.append({"command": command, **kwargs})
        config_arg = next(value for value in command if value.startswith("--config="))
        config = yaml.safe_load(Path(config_arg.split("=", 1)[1]).read_text())
        output = Path(kwargs["cwd"]) / config["model"]["generation_xyz_filename"]
        frames = [
            Atoms("Li2O", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[5, 5, 5], pbc=True),
            Atoms("NaCl", positions=[[0, 0, 0], [1, 1, 1]], cell=[6, 6, 6], pbc=True),
        ]
        write(output, frames, format="extxyz")
        write(output.with_stem(output.stem + "_init"), frames, format="extxyz")
        (Path(kwargs["cwd"]) / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    output = tmp_path / "generation"
    result = module.run_omatg_generation(
        composition_dir=paths["composition_dir"],
        official_config_path=paths["official_config"],
        checkpoint_path=paths["checkpoint"],
        runtime_python=paths["runtime_python"],
        omatg_source_dir=paths["source_dir"],
        output_dir=output,
        batch_size=2,
        seed=250803,
        require_formal_inputs=False,
        runner=fake_runner,
    )

    assert len(calls) == 1
    assert calls[0]["cwd"] != paths["source_dir"]
    assert calls[0]["env"]["PYTHONPATH"] == str(paths["source_dir"].resolve())
    assert {path.name for path in output.iterdir()} == {
        module.GENERATED_NAME,
        module.INITIAL_NAME,
        module.RUNTIME_CONFIG_NAME,
        module.RESOLVED_CONFIG_NAME,
        module.STDOUT_NAME,
        module.STDERR_NAME,
        module.MANIFEST_NAME,
    }
    assert result["counts"] == {"composition_rows": 2, "generated_frames": 2}
    assert result["all_generator_outputs_retained"] is True
    assert result["reference_geometry_fields_accessed"] is False
    assert result["dft_or_relaxed_structures_accessed"] is False
    assert result["energy_or_force_model_used"] is False
    assert result["physical_relaxation_used"] is False
    assert result["seed"] == 250803
    assert result["accelerator"] == "cpu"
    assert result["cpu_threads"] == 8
    assert result["integration_time_steps"] == 210
    assert result["runtime_config_contains_reference_paths"] is False
    assert result["outputs_sha256"][module.GENERATED_NAME] == _sha(
        output / module.GENERATED_NAME
    )
    assert "data/mp_20" not in (output / module.RESOLVED_CONFIG_NAME).read_text()


def test_runner_refuses_overwrite_bad_source_manifest_and_nonzero_exit(
    tmp_path: Path,
) -> None:
    from src.next25_omatg_run import run_omatg_generation

    paths = _inputs(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        run_omatg_generation(
            composition_dir=paths["composition_dir"],
            official_config_path=paths["official_config"],
            checkpoint_path=paths["checkpoint"],
            runtime_python=paths["runtime_python"],
            omatg_source_dir=paths["source_dir"],
            output_dir=existing,
            require_formal_inputs=False,
        )

    source_manifest = paths["composition_dir"] / "MANIFEST.json"
    bad = json.loads(source_manifest.read_text())
    bad["labels_opened"] = True
    source_manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary"):
        run_omatg_generation(
            composition_dir=paths["composition_dir"],
            official_config_path=paths["official_config"],
            checkpoint_path=paths["checkpoint"],
            runtime_python=paths["runtime_python"],
            omatg_source_dir=paths["source_dir"],
            output_dir=tmp_path / "bad-boundary",
            require_formal_inputs=False,
        )

    paths = _inputs(tmp_path / "again")

    def failed_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="boom")

    with pytest.raises(RuntimeError, match="exit code 7.*boom"):
        run_omatg_generation(
            composition_dir=paths["composition_dir"],
            official_config_path=paths["official_config"],
            checkpoint_path=paths["checkpoint"],
            runtime_python=paths["runtime_python"],
            omatg_source_dir=paths["source_dir"],
            output_dir=tmp_path / "failed-run",
            require_formal_inputs=False,
            runner=failed_runner,
        )
    assert not (tmp_path / "failed-run").exists()
