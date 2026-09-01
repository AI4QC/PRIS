#!/usr/bin/env python3
"""Run official OMatG against the frozen NEXT25 composition-only cohort."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from ase.io import read
import pandas as pd
import yaml

from src.next19_feature_build import _publish_directory_no_replace
from src.next25_omatg_compositions import (
    COHORT_NAME,
    COMPOSITIONS_LMDB_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    PROTOCOL as SOURCE_PROTOCOL,
)


PROTOCOL = "2026-08-03-next25-omatg-linear-ode-generate-v1"
GENERATED_NAME = "generated.xyz"
INITIAL_NAME = "generated_init.xyz"
RUNTIME_CONFIG_NAME = "runtime_config.yaml"
LIGHTNING_CONFIG_NAME = "config.yaml"
RESOLVED_CONFIG_NAME = "resolved_config.yaml"
STDOUT_NAME = "stdout.log"
STDERR_NAME = "stderr.log"
MANIFEST_NAME = "MANIFEST.json"
FORMAL_SEED = 250803
FORMAL_BATCH_SIZE = 8
FORMAL_ACCELERATOR = "cpu"
FORMAL_CPU_THREADS = 8
FORMAL_COMPOSITION_MANIFEST_SHA256 = (
    "d1177218f58b1393a48238b130571a1836adb16357dd9804b90af51fdf813ad3"
)
FORMAL_CONFIG_SHA256 = (
    "989ae8e5edb66c6a71466f7b9a3199e697d70a89a2b022010675f3da9435a4f2"
)
FORMAL_CHECKPOINT_SHA256 = (
    "94646125a726145a0d016ba3e1c5eedabb02f4a9dc341b36b4cfcdf7d2ca204a"
)
FORMAL_OMATG_GIT_COMMIT = "fcb9ba2c2cfd70505b0f142a5b3c44944d78e7f0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid composition source manifest") from exc
    if not isinstance(value, dict):
        raise ValueError("composition source manifest must be an object")
    return value


def _load_source(composition_dir: Path) -> tuple[pd.DataFrame, dict[str, object], dict[str, str]]:
    paths = {
        "cohort": composition_dir / COHORT_NAME,
        "compositions_lmdb": composition_dir / COMPOSITIONS_LMDB_NAME,
        "manifest": composition_dir / SOURCE_MANIFEST_NAME,
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"composition {role} is not a file: {path}")
    hashes = {role: _sha256(path) for role, path in paths.items()}
    manifest = _strict_json(paths["manifest"])
    if (
        manifest.get("protocol") != SOURCE_PROTOCOL
        or manifest.get("input_role") != "composition_only"
        or manifest.get("reference_geometry_fields_accessed") is not False
        or manifest.get("property_label_fields_accessed") is not False
        or manifest.get("labels_opened") is not False
    ):
        raise ValueError("composition source crossed the label-free boundary")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or (
        outputs.get(COHORT_NAME) != hashes["cohort"]
        or outputs.get(COMPOSITIONS_LMDB_NAME) != hashes["compositions_lmdb"]
    ):
        raise ValueError("composition source output hashes differ")
    try:
        cohort = pd.read_parquet(
            paths["cohort"],
            columns=[
                "material_id",
                "formula",
                "atomic_numbers_json",
                "natoms",
                "selection_rank",
                "input_role",
            ],
        )
    except Exception as exc:
        raise ValueError("invalid composition cohort") from exc
    if (
        cohort.empty
        or cohort["material_id"].isna().any()
        or cohort["material_id"].duplicated().any()
        or cohort["selection_rank"].tolist() != list(range(len(cohort)))
        or not cohort["input_role"].eq("composition_only").all()
    ):
        raise ValueError("composition cohort identities or ordering differ")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("selected_rows") != len(cohort)
        or counts.get("selected_atoms") != int(cohort["natoms"].sum())
    ):
        raise ValueError("composition source counts differ")
    return cohort, manifest, hashes


def write_safe_runtime_config(
    *,
    official_config_path: Path,
    composition_lmdb_path: Path,
    output_path: Path,
    runtime_config_path: Path,
    batch_size: int,
    accelerator: str = FORMAL_ACCELERATOR,
) -> int:
    """Rewrite only runtime/data plumbing; remove every reference-data path."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive exact integer")
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("accelerator must be cpu or gpu")
    try:
        config = yaml.safe_load(Path(official_config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("invalid official OMatG configuration") from exc
    if not isinstance(config, dict):
        raise ValueError("official OMatG configuration must be a mapping")
    try:
        integration_steps = int(
            config["model"]["si"]["init_args"]["integration_time_steps"]
        )
        data = config["data"]
        trainer = config["trainer"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("official OMatG configuration lacks required fields") from exc
    if integration_steps != 210 or not isinstance(data, dict) or not isinstance(trainer, dict):
        raise ValueError("unexpected official OMatG generation configuration")
    safe_lmdb = str(Path(composition_lmdb_path).resolve())
    for split in ("train_dataset", "val_dataset", "pred_dataset"):
        try:
            init_args = data[split]["init_args"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"configuration lacks {split}") from exc
        if not isinstance(init_args, dict):
            raise ValueError(f"configuration {split} init_args must be a mapping")
        init_args["file_path"] = safe_lmdb
        init_args["lazy_storage"] = True
    data["batch_size"] = batch_size
    data["num_workers"] = 0
    data["pin_memory"] = False
    data["persistent_workers"] = False
    config["model"]["generation_xyz_filename"] = Path(output_path).name
    trainer.pop("callbacks", None)
    trainer.update(
        {
            "accelerator": accelerator,
            "devices": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
        }
    )
    runtime_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    rendered = runtime_config_path.read_text(encoding="utf-8")
    forbidden = ("data/mp_20", "test.lmdb", "train.lmdb", "val.lmdb")
    if any(token in rendered for token in forbidden):
        raise ValueError("safe runtime configuration still contains a reference path")
    return integration_steps


def build_predict_command(
    *,
    runtime_python: Path,
    runtime_config_path: Path,
    checkpoint_path: Path,
    seed: int,
) -> list[str]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative exact integer")
    return [
        str(Path(runtime_python).absolute()),
        "-m",
        "omg.main",
        "predict",
        f"--config={Path(runtime_config_path).resolve()}",
        f"--ckpt_path={Path(checkpoint_path).resolve()}",
        f"--seed_everything={seed}",
    ]


def _git_head(source_dir: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _verify_unchanged(paths: Mapping[str, Path], hashes: Mapping[str, str]) -> None:
    for role, path in paths.items():
        if _sha256(path) != hashes[role]:
            raise ValueError(f"{role} changed before generation publication")


def run_omatg_generation(
    *,
    composition_dir: Path,
    official_config_path: Path,
    checkpoint_path: Path,
    runtime_python: Path,
    omatg_source_dir: Path,
    output_dir: Path,
    batch_size: int = FORMAL_BATCH_SIZE,
    seed: int = FORMAL_SEED,
    accelerator: str = FORMAL_ACCELERATOR,
    cpu_threads: int = FORMAL_CPU_THREADS,
    require_formal_inputs: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Run OMatG once and publish its complete, unfiltered raw output."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    composition_dir = Path(composition_dir).resolve()
    official_config_path = Path(official_config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    # Keep the venv launcher path: resolving its symlink would silently invoke
    # the base interpreter and discard the venv's site-packages.
    runtime_python = Path(runtime_python).absolute()
    omatg_source_dir = Path(omatg_source_dir).resolve()
    for role, path in {
        "official_config": official_config_path,
        "checkpoint": checkpoint_path,
        "runtime_python": runtime_python,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    if not (omatg_source_dir / "omg" / "__init__.py").is_file():
        raise FileNotFoundError("OMatG source package is absent")
    cohort, source_manifest, source_hashes = _load_source(composition_dir)
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("accelerator must be cpu or gpu")
    if type(cpu_threads) is not int or cpu_threads <= 0:
        raise ValueError("cpu_threads must be a positive exact integer")
    input_paths = {
        "composition_cohort": composition_dir / COHORT_NAME,
        "composition_lmdb": composition_dir / COMPOSITIONS_LMDB_NAME,
        "composition_manifest": composition_dir / SOURCE_MANIFEST_NAME,
        "official_config": official_config_path,
        "checkpoint": checkpoint_path,
    }
    input_hashes = {role: _sha256(path) for role, path in input_paths.items()}
    git_head = _git_head(omatg_source_dir)
    formal_identity = (
        input_hashes["composition_manifest"]
        == FORMAL_COMPOSITION_MANIFEST_SHA256
        and input_hashes["official_config"] == FORMAL_CONFIG_SHA256
        and input_hashes["checkpoint"] == FORMAL_CHECKPOINT_SHA256
        and git_head == FORMAL_OMATG_GIT_COMMIT
        and batch_size == FORMAL_BATCH_SIZE
        and seed == FORMAL_SEED
        and accelerator == FORMAL_ACCELERATOR
        and cpu_threads == FORMAL_CPU_THREADS
    )
    if require_formal_inputs and not formal_identity:
        raise ValueError("formal NEXT25 generator inputs or constants differ")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        runtime_config_path = staging / RUNTIME_CONFIG_NAME
        generated_path = staging / GENERATED_NAME
        initial_path = staging / INITIAL_NAME
        integration_steps = write_safe_runtime_config(
            official_config_path=official_config_path,
            composition_lmdb_path=input_paths["composition_lmdb"],
            output_path=generated_path,
            runtime_config_path=runtime_config_path,
            batch_size=batch_size,
            accelerator=accelerator,
        )
        command = build_predict_command(
            runtime_python=runtime_python,
            runtime_config_path=runtime_config_path,
            checkpoint_path=checkpoint_path,
            seed=seed,
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(omatg_source_dir)
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[variable] = str(cpu_threads)
        completed = runner(
            command,
            cwd=staging,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        (staging / STDOUT_NAME).write_text(completed.stdout or "", encoding="utf-8")
        (staging / STDERR_NAME).write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no subprocess output").strip()
            detail = " ".join(detail.split())[-4000:]
            raise RuntimeError(
                f"OMatG exited with exit code {completed.returncode}: {detail}"
            )
        lightning_config_path = staging / LIGHTNING_CONFIG_NAME
        resolved_config_path = staging / RESOLVED_CONFIG_NAME
        if not lightning_config_path.is_file():
            raise RuntimeError("OMatG did not emit its resolved Lightning configuration")
        resolved_text = lightning_config_path.read_text(encoding="utf-8")
        forbidden_paths = ("data/mp_20", "test.lmdb", "train.lmdb", "val.lmdb")
        if any(token in resolved_text for token in forbidden_paths):
            raise ValueError("resolved OMatG configuration contains a reference path")
        os.replace(lightning_config_path, resolved_config_path)
        if not generated_path.is_file() or not initial_path.is_file():
            raise RuntimeError("OMatG did not produce both generated and initial XYZ files")
        try:
            generated_frames = read(generated_path, index=":", format="extxyz")
            initial_frames = read(initial_path, index=":", format="extxyz")
        except Exception as exc:
            raise ValueError("OMatG output is not readable extended XYZ") from exc
        if len(generated_frames) != len(cohort) or len(initial_frames) != len(cohort):
            raise ValueError("OMatG frame count differs from the complete composition cohort")

        safe_text = runtime_config_path.read_text(encoding="utf-8")
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "composition_only_raw_generator_output_freeze",
            "source_protocol": SOURCE_PROTOCOL,
            "source_generator": "omatg_mp20_csp_linear_ode",
            "source_revision": {
                "git_commit": git_head,
                "model_repository_revision": "87dcc2a222f849f4f3c381a8cfa47ede0971d364",
            },
            "input_role": "composition_only",
            "output_role": "raw_unrelaxed_generator_x0",
            "seed": seed,
            "batch_size": batch_size,
            "accelerator": accelerator,
            "cpu_threads": cpu_threads,
            "integration_time_steps": integration_steps,
            "command": command,
            "all_generator_outputs_retained": True,
            "post_generation_validity_filter_used": False,
            "reference_geometry_fields_accessed": False,
            "property_label_fields_accessed": False,
            "dft_or_relaxed_structures_accessed": False,
            "energy_or_force_model_used": False,
            "physical_relaxation_used": False,
            "same_composition_candidates_used": False,
            "runtime_config_contains_reference_paths": any(
                token in safe_text or token in resolved_text
                for token in forbidden_paths
            ),
            "dataset_paths_all_equal_composition_only_lmdb": True,
            "counts": {
                "composition_rows": len(cohort),
                "generated_frames": len(generated_frames),
            },
            "inputs_sha256": {
                role: {"path": str(path), "sha256": input_hashes[role]}
                for role, path in input_paths.items()
            },
            "composition_source_manifest_sha256": source_hashes["manifest"],
            "production_protocol_eligible": bool(formal_identity),
            "scientific_improvement_claim": False,
        }
        output_members = (
            GENERATED_NAME,
            INITIAL_NAME,
            RUNTIME_CONFIG_NAME,
            RESOLVED_CONFIG_NAME,
            STDOUT_NAME,
            STDERR_NAME,
        )
        manifest["outputs_sha256"] = {
            name: _sha256(staging / name) for name in output_members
        }
        repository_root = Path(__file__).resolve().parents[1]
        source_paths = {
            "src/next25_omatg_run.py": Path(__file__).resolve(),
            "src/next25_omatg_compositions.py": repository_root
            / "src/next25_omatg_compositions.py",
        }
        manifest["executed_source_sha256"] = {
            relative: _sha256(path) for relative, path in source_paths.items()
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _verify_unchanged(input_paths, input_hashes)
        for relative, path in source_paths.items():
            if _sha256(path) != manifest["executed_source_sha256"][relative]:
                raise ValueError(f"executed source changed before publication: {relative}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-dir", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--omatg-source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default=FORMAL_ACCELERATOR)
    parser.add_argument("--cpu-threads", type=int, default=FORMAL_CPU_THREADS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_omatg_generation(
        composition_dir=args.composition_dir,
        official_config_path=args.official_config,
        checkpoint_path=args.checkpoint,
        runtime_python=args.runtime_python,
        omatg_source_dir=args.omatg_source_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        accelerator=args.accelerator,
        cpu_threads=args.cpu_threads,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
