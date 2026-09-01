#!/usr/bin/env python3
"""Extract label-free fixed-cell MatterSim few-step trajectory features."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import importlib.metadata
import json
from numbers import Real
from pathlib import Path
import platform
import shlex
import zipfile

import numpy as np
import pandas as pd

from src.next6_mattersim_baseline import frame_to_atoms
from src.next7_mattersim_prerelax import (
    BatchPredictor,
    FIRE_PARAMETERS,
    SNAPSHOT_STEPS,
    SnapshotValidationError,
    assess_prerelax_support,
    make_mattersim_predictor,
    run_fixed_cell_fire,
    summarize_snapshots,
)


VALID_STAGES = (
    "search_calibration",
    "formula_selection",
    "threshold_calibration",
    "test",
)
DEVELOPMENT_FREEZE_PROTOCOL = (
    "2026-08-01-mattersim-fewstep-development-freeze-v1"
)
EVIDENCE_ROLE = "historically seen discovery; not confirmatory"

_NUMERIC_METRICS = (
    "energy_total_ev",
    "energy_ev_per_atom",
    "energy_change_from_previous_snapshot_ev_per_atom",
    "fmax_ev_per_a",
    "frms_ev_per_a",
    "stress_frobenius_ev_per_a3",
    "stress_max_abs_principal_ev_per_a3",
    "rms_mic_displacement_a",
    "max_mic_displacement_a",
    "min_pair_distance_a",
)


def _output_columns() -> list[str]:
    columns = [
        "sid",
        "rk",
        "material",
        "stage",
        "strict_x0_ok",
        "initial_ionic_step",
        "geom_min_pair_ratio",
        "evidence_role",
        "input_role",
        "fewstep_feature_ok",
        "fewstep_feature_error",
        "force_evaluations",
        "optimizer_updates",
        "retry_overhead_force_evaluations",
        "retry_overhead_optimizer_updates",
        "allocated_seconds",
    ]
    for step in SNAPSHOT_STEPS:
        columns.extend(f"k{step}_{metric}" for metric in _NUMERIC_METRICS)
        columns.extend((f"k{step}_supported", f"k{step}_support_reason"))
    return columns


def _output_dtypes() -> dict[str, str]:
    string_columns = {
        "sid",
        "rk",
        "material",
        "stage",
        "evidence_role",
        "input_role",
        "fewstep_feature_error",
        *(f"k{step}_support_reason" for step in SNAPSHOT_STEPS),
    }
    bool_columns = {
        "strict_x0_ok",
        "fewstep_feature_ok",
        *(f"k{step}_supported" for step in SNAPSHOT_STEPS),
    }
    int_columns = {
        "initial_ionic_step",
        "force_evaluations",
        "optimizer_updates",
        "retry_overhead_force_evaluations",
        "retry_overhead_optimizer_updates",
    }
    return {
        column: (
            "string"
            if column in string_columns
            else "bool"
            if column in bool_columns
            else "int64"
            if column in int_columns
            else "float64"
        )
        for column in _output_columns()
    }


def _validated_stages(stages: Sequence[str]) -> tuple[str, ...]:
    if isinstance(stages, (str, bytes)):
        raise ValueError("stages must be an explicit sequence")
    selected = tuple(stages)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("stages must be nonempty and unique")
    unknown = sorted(set(selected) - set(VALID_STAGES))
    if unknown:
        raise ValueError(f"stages contain unsupported values: {unknown}")
    if "test" in selected and len(selected) != 1:
        raise ValueError("test stage cannot run with development stages")
    return selected


def _failure_columns(reason: str) -> dict[str, object]:
    columns: dict[str, object] = {}
    for step in SNAPSHOT_STEPS:
        columns.update(
            {f"k{step}_{metric}": np.nan for metric in _NUMERIC_METRICS}
        )
        columns[f"k{step}_supported"] = False
        columns[f"k{step}_support_reason"] = reason
    return columns


def _normalized_initial_ionic_step(value: object) -> tuple[bool, int]:
    """Return whether metadata is numeric zero and a stable integer value."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False, -1
    numeric = float(value)
    if not np.isfinite(numeric):
        return False, -1
    if numeric == 0.0:
        return True, 0
    if numeric.is_integer() and np.iinfo(np.int64).min <= numeric <= np.iinfo(np.int64).max:
        return False, int(numeric)
    return False, -1


def _raw_initial_ionic_step(frame_text: str) -> int | None:
    """Parse the single ionic_step token from an extended-XYZ comment safely."""

    lines = frame_text.splitlines()
    if len(lines) < 2:
        return None
    try:
        tokens = shlex.split(lines[1])
    except ValueError:
        return None
    values = [
        token.split("=", 1)[1]
        for token in tokens
        if token.startswith("ionic_step=")
    ]
    if len(values) != 1:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cuda_device_index(device: str) -> int | None:
    text = str(device)
    if text == "cuda":
        return 0
    if not text.startswith("cuda:"):
        return None
    suffix = text.removeprefix("cuda:")
    return int(suffix) if suffix.isdigit() else None


def _reset_peak_cuda_memory(device: str) -> bool:
    index = _cuda_device_index(device)
    if index is None:
        return False
    try:
        import torch

        if not bool(torch.cuda.is_available()):
            return False
        torch.cuda.reset_peak_memory_stats(index)
    except Exception:
        return False
    return True


def _runtime_metadata(
    device: str, *, collect_cuda_metrics: bool
) -> tuple[dict[str, object], int | None]:
    runtime: dict[str, object] = {
        "python_version": platform.python_version(),
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "gpu_name": None,
    }
    peak: int | None = None
    try:
        import torch
    except Exception:
        return runtime, peak
    runtime["torch_version"] = str(torch.__version__)
    try:
        available = bool(torch.cuda.is_available())
        runtime["cuda_available"] = available
        runtime["cuda_version"] = (
            None if torch.version.cuda is None else str(torch.version.cuda)
        )
        device_index = _cuda_device_index(device)
        if available and collect_cuda_metrics and device_index is not None:
            runtime["gpu_name"] = str(
                torch.cuda.get_device_name(device_index)
            )
            peak = int(torch.cuda.max_memory_allocated(device_index))
    except Exception:
        pass
    return runtime, peak


def _validate_development_freeze(
    path: Path,
    *,
    checkpoint_sha256: str,
    feature_inputs_sha256: dict[str, str],
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid frozen protocol JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid frozen protocol document")
    if payload.get("protocol") != DEVELOPMENT_FREEZE_PROTOCOL:
        raise ValueError("frozen protocol identifier mismatch")
    if payload.get("state") != "frozen":
        raise ValueError("frozen protocol state mismatch")
    frozen_at = payload.get("frozen_at_utc")
    if not isinstance(frozen_at, str) or not frozen_at.strip():
        raise ValueError("frozen protocol timestamp is missing")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("frozen protocol checkpoint mismatch")
    if payload.get("feature_inputs_sha256") != feature_inputs_sha256:
        raise ValueError("frozen protocol feature-input mismatch")
    code_sha256 = {
        "next7_mattersim_prerelax.py": _sha256_file(
            Path(__file__).resolve().with_name("next7_mattersim_prerelax.py")
        ),
        "next7_mattersim_features.py": _sha256_file(Path(__file__).resolve()),
    }
    if payload.get("code_sha256") != code_sha256:
        raise ValueError("frozen protocol code mismatch")


def _summary_columns(snapshots: dict[int, object]) -> dict[str, object]:
    summaries = summarize_snapshots(snapshots)
    columns: dict[str, object] = {}
    for step in SNAPSHOT_STEPS:
        summary = summaries[step]
        decision = assess_prerelax_support(snapshots, cutoff_step=step)
        columns.update(
            {
                f"k{step}_energy_total_ev": summary.total_energy_ev,
                f"k{step}_energy_ev_per_atom": summary.energy_per_atom_ev,
                f"k{step}_energy_change_from_previous_snapshot_ev_per_atom": (
                    summary.energy_change_from_previous_snapshot_ev_per_atom
                ),
                f"k{step}_fmax_ev_per_a": summary.fmax_ev_per_a,
                f"k{step}_frms_ev_per_a": summary.frms_ev_per_a,
                f"k{step}_stress_frobenius_ev_per_a3": (
                    summary.stress_frobenius_ev_per_a3
                ),
                f"k{step}_stress_max_abs_principal_ev_per_a3": (
                    summary.stress_max_abs_eigenvalue_ev_per_a3
                ),
                f"k{step}_rms_mic_displacement_a": (
                    summary.rms_displacement_from_x0_a
                ),
                f"k{step}_max_mic_displacement_a": (
                    summary.max_displacement_from_x0_a
                ),
                f"k{step}_min_pair_distance_a": summary.min_pair_distance_a,
                f"k{step}_supported": decision.supported,
                f"k{step}_support_reason": decision.reason,
            }
        )
    return columns


def run_fewstep_features(
    elementa_dir: Path,
    p9_dir: Path,
    stage_assignments_path: Path,
    output_dir: Path,
    *,
    checkpoint: Path,
    stages: Sequence[str],
    device: str = "cuda",
    atom_budget: int = 4096,
    structure_cap: int | None = None,
    inference_batch_size: int = 64,
    structure_chunk_size: int = 512,
    predictor: BatchPredictor | None = None,
    frozen_protocol_path: Path | None = None,
) -> dict[str, object]:
    """Build one label-free few-step feature row per explicitly selected sid."""

    selected_stages = _validated_stages(stages)
    if selected_stages == ("test",):
        if frozen_protocol_path is None or not Path(
            frozen_protocol_path
            ).is_file():
            raise FileNotFoundError("test stage requires an existing frozen protocol")

    elementa_dir = Path(elementa_dir)
    p9_dir = Path(p9_dir)
    stage_assignments_path = Path(stage_assignments_path)
    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint)
    frames_path = elementa_dir / "elementa_initial_frames.zip"
    base_features_path = elementa_dir / "elementa_x0_features.parquet"
    metadata_path = p9_dir / "elementa_x0_p9_features.parquet"
    output_path = output_dir / "mattersim_fewstep_features.parquet"
    manifest_path = output_dir / "MANIFEST.json"

    existing_outputs = [
        path for path in (manifest_path, output_path) if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            f"refusing to overwrite existing output: {existing_outputs[0]}"
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = _sha256_file(checkpoint)
    for value, name in (
        (atom_budget, "atom_budget"),
        (inference_batch_size, "inference_batch_size"),
        (structure_chunk_size, "structure_chunk_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if structure_cap is not None and (
        isinstance(structure_cap, bool)
        or not isinstance(structure_cap, int)
        or structure_cap <= 0
    ):
        raise ValueError("structure_cap must be a positive integer or None")

    feature_inputs_sha256 = {
        frames_path.name: _sha256_file(frames_path),
        metadata_path.name: _sha256_file(metadata_path),
        base_features_path.name: _sha256_file(base_features_path),
        stage_assignments_path.name: _sha256_file(stage_assignments_path),
    }
    if selected_stages == ("test",):
        assert frozen_protocol_path is not None
        _validate_development_freeze(
            Path(frozen_protocol_path),
            checkpoint_sha256=checkpoint_sha256,
            feature_inputs_sha256=feature_inputs_sha256,
        )

    metadata = pd.read_parquet(
        metadata_path,
        columns=(
            "sid",
            "rk",
            "material",
            "strict_x0_ok",
            "initial_ionic_step",
        ),
    )
    stage_table = pd.read_parquet(
        stage_assignments_path, columns=("sid", "rk", "stage")
    )
    base_features = pd.read_parquet(
        base_features_path, columns=("sid", "rk", "geom_min_pair_ratio")
    )
    for table, name in (
        (metadata, "metadata"),
        (stage_table, "stage assignments"),
        (base_features, "base features"),
    ):
        if table["sid"].isna().any() or table["sid"].duplicated().any():
            raise ValueError(f"{name} sid values must be unique and nonmissing")
        if table["rk"].isna().any():
            raise ValueError(f"{name} rk values must be nonmissing")
        table["sid"] = table["sid"].astype(str)
        table["rk"] = table["rk"].astype(str)
    if set(metadata["sid"]) != set(stage_table["sid"]):
        raise ValueError("metadata and stage-assignment sid sets differ")
    if set(metadata["sid"]) != set(base_features["sid"]):
        raise ValueError("metadata and base-feature sid sets differ")
    joined = metadata.merge(
        stage_table,
        on="sid",
        how="inner",
        suffixes=("_metadata", "_stage"),
        validate="one_to_one",
    )
    if not joined["rk_metadata"].eq(joined["rk_stage"]).all():
        raise ValueError("metadata and stage-assignment rk values differ")
    joined = joined.rename(columns={"rk_metadata": "rk"}).drop(
        columns="rk_stage"
    )
    joined = joined.merge(
        base_features,
        on="sid",
        how="inner",
        suffixes=("", "_base"),
        validate="one_to_one",
    )
    if not joined["rk"].eq(joined["rk_base"]).all():
        raise ValueError("metadata and base-feature rk values differ")
    joined = joined.drop(columns="rk_base")
    joined["geom_min_pair_ratio"] = pd.to_numeric(
        joined["geom_min_pair_ratio"], errors="coerce"
    )
    if not set(joined["stage"]).issubset(set(VALID_STAGES)):
        raise ValueError("stage assignments contain unsupported stages")
    selected = joined.loc[joined["stage"].isin(selected_stages)].copy()

    base_rows: list[dict[str, object]] = []
    total_calls = 0
    total_elapsed = 0.0
    cuda_tracking_attempted = False
    cuda_tracking_started = False
    with zipfile.ZipFile(frames_path) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            sid = Path(name).stem
            if sid in members:
                raise ValueError(f"initial-frame archive has duplicate sid stem: {sid}")
            members[sid] = name
        if set(members) != set(metadata["sid"]):
            raise ValueError("initial-frame archive and metadata sid sets differ")

        selected_records = selected.to_dict("records")
        for start in range(0, len(selected_records), structure_chunk_size):
            selected_chunk = selected_records[start : start + structure_chunk_size]
            records_chunk: list[dict[str, object]] = []
            structures_chunk: list[object] = []
            for record in selected_chunk:
                metadata_step_ok, initial_ionic_step = (
                    _normalized_initial_ionic_step(record["initial_ionic_step"])
                )
                strict_value = record["strict_x0_ok"]
                strict_flag_ok = isinstance(strict_value, (bool, np.bool_)) and bool(
                    strict_value
                )
                base = {
                    "sid": str(record["sid"]),
                    "rk": str(record["rk"]),
                    "material": str(record["material"]),
                    "stage": str(record["stage"]),
                    "strict_x0_ok": strict_flag_ok and metadata_step_ok,
                    "initial_ionic_step": initial_ionic_step,
                    "geom_min_pair_ratio": float(record["geom_min_pair_ratio"]),
                    "evidence_role": EVIDENCE_ROLE,
                }
                if not base["strict_x0_ok"]:
                    base_rows.append(
                        {
                            **base,
                            "input_role": "trajectory_earliest_available",
                            "fewstep_feature_ok": False,
                            "fewstep_feature_error": "nonzero_initial_ionic_step",
                            "force_evaluations": 0,
                            "optimizer_updates": 0,
                            "retry_overhead_force_evaluations": 0,
                            "retry_overhead_optimizer_updates": 0,
                            "allocated_seconds": 0.0,
                            **_failure_columns("nonzero_initial_ionic_step"),
                        }
                    )
                    continue
                try:
                    frame_text = archive.read(members[base["sid"]]).decode("utf-8")
                except Exception:
                    base_rows.append(
                        {
                            **base,
                            "input_role": "unrelaxed_x0_only",
                            "fewstep_feature_ok": False,
                            "fewstep_feature_error": "invalid_initial_frame",
                            "force_evaluations": 0,
                            "optimizer_updates": 0,
                            "retry_overhead_force_evaluations": 0,
                            "retry_overhead_optimizer_updates": 0,
                            "allocated_seconds": 0.0,
                            **_failure_columns("invalid_initial_frame"),
                        }
                    )
                    continue
                if _raw_initial_ionic_step(frame_text) != 0:
                    base["strict_x0_ok"] = False
                    base_rows.append(
                        {
                            **base,
                            "input_role": "trajectory_earliest_available",
                            "fewstep_feature_ok": False,
                            "fewstep_feature_error": "nonzero_initial_ionic_step",
                            "force_evaluations": 0,
                            "optimizer_updates": 0,
                            "retry_overhead_force_evaluations": 0,
                            "retry_overhead_optimizer_updates": 0,
                            "allocated_seconds": 0.0,
                            **_failure_columns("nonzero_initial_ionic_step"),
                        }
                    )
                    continue
                try:
                    atoms = frame_to_atoms(frame_text)
                except Exception:
                    base_rows.append(
                        {
                            **base,
                            "input_role": "unrelaxed_x0_only",
                            "fewstep_feature_ok": False,
                            "fewstep_feature_error": "invalid_initial_frame",
                            "force_evaluations": 0,
                            "optimizer_updates": 0,
                            "retry_overhead_force_evaluations": 0,
                            "retry_overhead_optimizer_updates": 0,
                            "allocated_seconds": 0.0,
                            **_failure_columns("invalid_initial_frame"),
                        }
                    )
                    continue
                records_chunk.append(base)
                structures_chunk.append(atoms)

            if not structures_chunk:
                continue
            if predictor is None:
                predictor = make_mattersim_predictor(
                    checkpoint,
                    device=device,
                    batch_size=inference_batch_size,
                )
            if not cuda_tracking_attempted:
                cuda_tracking_started = _reset_peak_cuda_memory(device)
                cuda_tracking_attempted = True
            run = run_fixed_cell_fire(
                structures_chunk,
                predictor,
                atom_budget=atom_budget,
                structure_cap=structure_cap,
            )
            total_calls += run.predictor_forward_calls
            total_elapsed += run.elapsed_seconds
            allocated = run.elapsed_seconds / len(records_chunk)
            for record, result in zip(records_chunk, run.results, strict=True):
                if result.error is not None:
                    reason = str(result.error)
                    base_rows.append(
                        {
                            **record,
                            "input_role": "unrelaxed_x0_only",
                            "fewstep_feature_ok": False,
                            "fewstep_feature_error": reason,
                            "force_evaluations": result.force_evaluations,
                            "optimizer_updates": result.optimizer_updates,
                            "retry_overhead_force_evaluations": (
                                result.retry_overhead_force_evaluations
                            ),
                            "retry_overhead_optimizer_updates": (
                                result.retry_overhead_optimizer_updates
                            ),
                            "allocated_seconds": allocated,
                            **_failure_columns(reason),
                        }
                    )
                    continue
                try:
                    summary_columns = _summary_columns(result.snapshots)
                except SnapshotValidationError as exc:
                    reason = exc.reason
                except Exception:
                    reason = "invalid_snapshots"
                else:
                    reason = ""
                    pair_ratio = float(record["geom_min_pair_ratio"])
                    if not np.isfinite(pair_ratio) or pair_ratio < 0.45:
                        for step in SNAPSHOT_STEPS:
                            summary_columns[f"k{step}_supported"] = False
                            summary_columns[f"k{step}_support_reason"] = (
                                "unsafe_x0_pair_ratio"
                            )
                if reason:
                    base_rows.append(
                        {
                            **record,
                            "input_role": "unrelaxed_x0_only",
                            "fewstep_feature_ok": False,
                            "fewstep_feature_error": reason,
                            "force_evaluations": result.force_evaluations,
                            "optimizer_updates": result.optimizer_updates,
                            "retry_overhead_force_evaluations": (
                                result.retry_overhead_force_evaluations
                            ),
                            "retry_overhead_optimizer_updates": (
                                result.retry_overhead_optimizer_updates
                            ),
                            "allocated_seconds": allocated,
                            **_failure_columns(reason),
                        }
                    )
                    continue
                base_rows.append(
                    {
                        **record,
                        "input_role": "unrelaxed_x0_only",
                        "fewstep_feature_ok": True,
                        "fewstep_feature_error": "",
                        "force_evaluations": result.force_evaluations,
                        "optimizer_updates": result.optimizer_updates,
                        "retry_overhead_force_evaluations": (
                            result.retry_overhead_force_evaluations
                        ),
                        "retry_overhead_optimizer_updates": (
                            result.retry_overhead_optimizer_updates
                        ),
                        "allocated_seconds": allocated,
                        **summary_columns,
                    }
                )

    table = pd.DataFrame(base_rows, columns=_output_columns()).astype(
        _output_dtypes()
    )
    table = table.sort_values("sid", kind="stable").reset_index(drop=True)
    if len(table) != len(selected) or table["sid"].duplicated().any():
        raise ValueError("few-step output must contain one row per selected sid")
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    try:
        mattersim_version = importlib.metadata.version("mattersim")
    except importlib.metadata.PackageNotFoundError:
        mattersim_version = "unknown"
    runtime, peak_cuda_memory = _runtime_metadata(
        device, collect_cuda_metrics=cuda_tracking_started
    )
    manifest: dict[str, object] = {
        "protocol": "2026-08-01-mattersim-fewstep-prerelax-v1",
        "evidence_role": EVIDENCE_ROLE,
        "input_policy": (
            "label-free x0 frames only; non-strict initial frames are retained "
            "fail-open and never enter the predictor"
        ),
        "stages": list(selected_stages),
        "model": {
            "package": "mattersim",
            "version": mattersim_version,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "device": str(device),
            "inference_batch_size": inference_batch_size,
            "atom_budget": atom_budget,
            "structure_cap": structure_cap,
            "structure_chunk_size": structure_chunk_size,
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "fire_parameters": dict(FIRE_PARAMETERS),
        },
        "runtime": runtime,
        "counts": {
            "input_rows": len(metadata),
            "stage_assignment_rows": len(stage_table),
            "selected_rows": len(selected),
            "strict_x0_rows": int(table["strict_x0_ok"].sum()),
            "nonstrict_x0_rows": int((~table["strict_x0_ok"]).sum()),
            "successful_rows": int(table["fewstep_feature_ok"].sum()),
            "failed_rows": int((~table["fewstep_feature_ok"]).sum()),
            "force_evaluations": int(table["force_evaluations"].sum()),
            "optimizer_updates": int(table["optimizer_updates"].sum()),
            "retry_overhead_force_evaluations": int(
                table["retry_overhead_force_evaluations"].sum()
            ),
            "retry_overhead_optimizer_updates": int(
                table["retry_overhead_optimizer_updates"].sum()
            ),
            **{
                f"supported_at_k{step}": int(table[f"k{step}_supported"].sum())
                for step in SNAPSHOT_STEPS
            },
        },
        "execution": {
            "predictor_forward_calls": total_calls,
            "total_elapsed_seconds": total_elapsed,
            "peak_cuda_memory_bytes": peak_cuda_memory,
        },
        "inputs_sha256": {
            **feature_inputs_sha256,
            checkpoint.name: checkpoint_sha256,
        },
        "outputs_sha256": {output_path.name: _sha256_file(output_path)},
    }
    if selected_stages == ("test",):
        assert frozen_protocol_path is not None
        frozen_path = Path(frozen_protocol_path)
        frozen_sha256 = _sha256_file(frozen_path)
        manifest["frozen_protocol"] = {
            "path": str(frozen_path.resolve()),
            "sha256": frozen_sha256,
        }
        inputs_sha256 = manifest["inputs_sha256"]
        assert isinstance(inputs_sha256, dict)
        inputs_sha256[frozen_path.name] = frozen_sha256
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elementa", type=Path, required=True)
    parser.add_argument("--p9", type=Path, required=True)
    parser.add_argument("--stage-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=VALID_STAGES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atom-budget", type=int, default=4096)
    parser.add_argument("--structure-cap", type=int)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--structure-chunk-size", type=int, default=512)
    parser.add_argument("--frozen-protocol", type=Path)
    args = parser.parse_args(argv)
    manifest = run_fewstep_features(
        args.elementa,
        args.p9,
        args.stage_assignments,
        args.output,
        checkpoint=args.checkpoint,
        stages=tuple(args.stages),
        device=args.device,
        atom_budget=args.atom_budget,
        structure_cap=args.structure_cap,
        inference_batch_size=args.inference_batch_size,
        structure_chunk_size=args.structure_chunk_size,
        frozen_protocol_path=args.frozen_protocol,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VALID_STAGES", "main", "run_fewstep_features"]
