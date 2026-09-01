"""Label-free, fixed-protocol batched CHSC-v0 feature construction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from ase import Atoms

from src.next10_lrrc_mattersim_features import (
    BatchForcePredictor,
    FROZEN_M5_SHA256,
    _production_predictor,
    _runtime_identity,
    _sha256_file,
    _snapshot,
    _strict_json_document,
    _validated_builtin_telemetry,
    _validated_prediction,
)
from src.next11_geometry_only_frames import load_geometry_only_archive
from src.next12_chsc import (
    CHSC_VERSION,
    STEP_STRAIN,
    STRAIN_DIMENSION,
    CHSCResult,
    CHSCStatus,
    CHSCValidationError,
    _validated_geometry,
    analyze_strain_hessian_pair,
    deform_cell,
    direction_set,
    directional_curvatures_to_hessian,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next12-chsc-mattersim-features-v1"
UPSTREAM_PHSC_PROTOCOL = "2026-08-02-next11-phsc-mattersim-features-v1"
FROZEN_MODEL_BATCH_SIZE = 32
FROZEN_STRUCTURES_PER_CALL = 12
FROZEN_INPUT_SHA256 = {
    "phsc_features": "f3492a81cef37b9887ee86b784c172c5cf1667ab0a7206379f787cb54aec6875",
    "phsc_manifest": "2b1ffd28995747352fe5a2ec4263e5822bf5f104846a0624c78de466d15ef9f5",
    "geometry_only_frames": "9b99226a7dc5497fca2aaadbf6ac554c657cb5475705072bcd56b92db9515de9",
    "geometry_manifest": "2e5559595fa1dbc3f16470b005e1dc4f9dbe4a65de81a39a52f53c0af9b14901",
    "checkpoint": FROZEN_M5_SHA256,
}
FORMAL_EXPECTED_COUNTS = {"selected_rows": 2171, "strict_rows": 2164, "nonstrict_rows": 7}
CRITERION = {
    "name": CHSC_VERSION,
    "scope": "fixed_fractional_coordinate_homogeneous_strain_hessian",
    "strain_dimension": STRAIN_DIMENSION,
    "direction_count": 21,
    "step_strain": STEP_STRAIN,
    "probe_order": ["center", "(+h,-h,+h/2,-h/2) for each frozen direction"],
    "energy_evaluations_per_structure": 85,
    "primary_decision_proxy": "two_scale_strain_operator_difference",
    "numerical_consistency_proxies_are_confidence_bounds": False,
    "numerical_consistency_proxies_are_rigorous_error_bounds": False,
}
OUTPUT_NAME = "chsc_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
OUTPUT_COLUMNS = (
    "sid",
    "rk",
    "stage",
    "threshold_role",
    "strict_x0_ok",
    "natoms",
    "chsc_status",
    "chsc_negative",
    "h_strain",
    "lambda_h_ev_per_atom",
    "lambda_h2_ev_per_atom",
    "lambda_r_ev_per_atom",
    "e_num_ev_per_atom",
    "u_num_ev_per_atom",
    "l_num_ev_per_atom",
    "tau_alg_ev_per_atom",
    "antisymmetric_norm_h_ev_per_atom",
    "antisymmetric_norm_h2_ev_per_atom",
    "hessian_h_json",
    "hessian_h2_json",
    "energy_call_count",
    "error",
)
EXECUTED_SOURCE_RELATIVE = (
    "src/next12_chsc_mattersim_features.py",
    "src/next12_chsc.py",
    "src/next11_phsc.py",
    "src/next11_geometry_only_frames.py",
    "src/next10_lrrc_mattersim_features.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
    "src/next6_wbm_build.py",
    "src/next6_wbm_features.py",
    "src/next6_wbm_protocol.py",
)


class BatchCHSCError(RuntimeError):
    """Raised when batched CHSC cannot preserve exact cohort alignment."""


@dataclass(frozen=True, slots=True)
class BatchCHSCResult:
    """One sid-aligned CHSC result plus both reconstructed strain Hessians."""

    sid: str
    result: CHSCResult
    hessian_h: np.ndarray
    hessian_h2: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedCHSC:
    sid: str
    base: Atoms
    probes: tuple[Atoms, ...]


def _probe_group(atoms: Atoms) -> tuple[Atoms, ...]:
    base = _validated_geometry(atoms)
    probes: list[Atoms] = [base]
    for direction in direction_set():
        probes.extend(
            (
                deform_cell(base, direction, STEP_STRAIN),
                deform_cell(base, -direction, STEP_STRAIN),
                deform_cell(base, direction, STEP_STRAIN / 2.0),
                deform_cell(base, -direction, STEP_STRAIN / 2.0),
            )
        )
    if len(probes) != 85:
        raise AssertionError("CHSC-v0 probe group must contain exactly 85 structures")
    return tuple(probes)


def _result_from_energies(
    energies: Sequence[float], n_atoms: int
) -> tuple[CHSCResult, np.ndarray, np.ndarray]:
    values = np.asarray(energies, dtype=np.float64)
    if values.shape != (85,) or not np.all(np.isfinite(values)):
        raise BatchCHSCError("one CHSC probe group must contain 85 finite energies")
    center = float(values[0])
    samples = values[1:].reshape(21, 4)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        curvatures_h = (samples[:, 0] - 2.0 * center + samples[:, 1]) / (
            float(n_atoms) * STEP_STRAIN**2
        )
        curvatures_h2 = (samples[:, 2] - 2.0 * center + samples[:, 3]) / (
            float(n_atoms) * (STEP_STRAIN / 2.0) ** 2
        )
    hessian_h = directional_curvatures_to_hessian(curvatures_h)
    hessian_h2 = directional_curvatures_to_hessian(curvatures_h2)
    spectral = analyze_strain_hessian_pair(hessian_h, hessian_h2)
    result = CHSCResult(
        status=spectral.status,
        negative=spectral.negative,
        h=STEP_STRAIN,
        lambda_h=spectral.lambda_h,
        lambda_h2=spectral.lambda_h2,
        lambda_r=spectral.lambda_r,
        e_num=spectral.e_num,
        u_num=spectral.u_num,
        l_num=spectral.l_num,
        tau_alg=spectral.tau_alg,
        antisymmetric_norm_h=spectral.antisymmetric_norm_h,
        antisymmetric_norm_h2=spectral.antisymmetric_norm_h2,
        energy_call_count=85,
    )
    return result, hessian_h, hessian_h2


def evaluate_chsc_batch(
    sids: Sequence[str],
    structures: Sequence[Atoms],
    predictor: BatchForcePredictor,
    *,
    structures_per_call: int = FROZEN_STRUCTURES_PER_CALL,
) -> list[BatchCHSCResult]:
    """Evaluate complete 85-probe groups without splitting a structure group."""

    if len(sids) != len(structures):
        raise BatchCHSCError("sids and structures must align one-to-one")
    if type(structures_per_call) is not int or structures_per_call <= 0:
        raise ValueError("structures_per_call must be a positive exact integer")
    if not callable(predictor):
        raise ValueError("predictor must be callable")
    normalized_sids = [str(sid) for sid in sids]
    if len(set(normalized_sids)) != len(normalized_sids):
        raise BatchCHSCError("sid values must be unique")
    ordered = sorted(zip(normalized_sids, structures, strict=True), key=lambda pair: pair[0])
    prepared: list[_PreparedCHSC] = []
    completed: dict[str, BatchCHSCResult] = {}
    nan_hessian = np.full((STRAIN_DIMENSION, STRAIN_DIMENSION), np.nan)
    for sid, atoms in ordered:
        try:
            probes = _probe_group(atoms)
        except CHSCValidationError as exc:
            completed[sid] = BatchCHSCResult(
                sid=sid,
                result=CHSCResult(
                    status=CHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                hessian_h=nan_hessian.copy(),
                hessian_h2=nan_hessian.copy(),
            )
            continue
        except Exception as exc:
            completed[sid] = BatchCHSCResult(
                sid=sid,
                result=CHSCResult(
                    status=CHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                    h=STEP_STRAIN,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                hessian_h=nan_hessian.copy(),
                hessian_h2=nan_hessian.copy(),
            )
            continue
        prepared.append(_PreparedCHSC(sid=sid, base=probes[0], probes=probes))

    for start in range(0, len(prepared), structures_per_call):
        chunk = prepared[start : start + structures_per_call]
        flat = [probe for item in chunk for probe in item.probes]
        try:
            prediction = predictor(flat)
            energies, _forces, _stresses = _validated_prediction(prediction, flat)
        except Exception as exc:
            raise BatchCHSCError(f"batch predictor failed: {type(exc).__name__}: {exc}") from exc
        if len(energies) != 85 * len(chunk):
            raise BatchCHSCError("batch energy count differs from complete CHSC groups")
        for index, item in enumerate(chunk):
            lo = 85 * index
            hi = lo + 85
            result, hessian_h, hessian_h2 = _result_from_energies(
                energies[lo:hi], len(item.base)
            )
            completed[item.sid] = BatchCHSCResult(
                sid=item.sid,
                result=result,
                hessian_h=hessian_h,
                hessian_h2=hessian_h2,
            )
    if set(completed) != set(normalized_sids):
        raise BatchCHSCError("batch CHSC did not produce one result per sid")
    return [completed[sid] for sid in sorted(completed)]


def _validate_phsc_manifest(
    manifest: Mapping[str, object], *, phsc_name: str, phsc_sha256: str
) -> None:
    if manifest.get("protocol") != UPSTREAM_PHSC_PROTOCOL:
        raise ValueError("upstream PHSC manifest protocol is not frozen")
    if manifest.get("labels_opened") is not False:
        raise ValueError("upstream PHSC manifest does not prove label-free execution")
    isolation = manifest.get("input_isolation")
    if not isinstance(isolation, Mapping):
        raise ValueError("upstream PHSC input isolation is missing")
    for key in ("raw_x0_archive_opened", "endpoint_label_artifacts_opened"):
        if isolation.get(key) is not False:
            raise ValueError(f"upstream PHSC manifest violates {key}")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(phsc_name) != phsc_sha256:
        raise ValueError("upstream PHSC table hash does not match its manifest")


def _load_geometry_snapshot(
    archive_data: bytes, manifest_data: bytes, expected_sids: Sequence[str]
) -> tuple[list[str], list[Atoms]]:
    staging = Path(tempfile.mkdtemp(prefix=".next12-geometry-snapshot-"))
    try:
        archive = staging / "geometry_only_frames.zip"
        manifest = staging / "MANIFEST.json"
        archive.write_bytes(archive_data)
        manifest.write_bytes(manifest_data)
        return load_geometry_only_archive(
            archive_path=archive,
            manifest_path=manifest,
            expected_sids=expected_sids,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load_phsc_table(data: bytes) -> pd.DataFrame:
    table = pd.read_parquet(io.BytesIO(data))
    required = {
        "sid",
        "rk",
        "stage",
        "threshold_role",
        "strict_x0_ok",
        "phsc_status",
        "phsc_negative",
    }
    if not required.issubset(table.columns):
        raise ValueError("upstream PHSC table lacks required columns")
    table = table.loc[:, list(required)].copy()
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError("upstream PHSC sid values must be nonmissing and unique")
    table["sid"] = table["sid"].astype(str)
    table = table.sort_values("sid", kind="stable", ignore_index=True)
    return table


def _matrix_json(matrix: np.ndarray) -> str:
    return json.dumps(matrix.tolist(), allow_nan=False, separators=(",", ":"))


def _result_row(
    upstream: Mapping[str, object], item: BatchCHSCResult | None, natoms: int
) -> dict[str, object]:
    if item is None:
        result = CHSCResult(
            status=CHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            error="strict_x0_ok is false in the frozen upstream PHSC cohort",
        )
        hessian_h_json = None
        hessian_h2_json = None
    else:
        result = item.result
        if result.energy_call_count:
            hessian_h_json = _matrix_json(item.hessian_h)
            hessian_h2_json = _matrix_json(item.hessian_h2)
        else:
            hessian_h_json = None
            hessian_h2_json = None
    return {
        "sid": str(upstream["sid"]),
        "rk": str(upstream["rk"]),
        "stage": str(upstream["stage"]),
        "threshold_role": str(upstream["threshold_role"]),
        "strict_x0_ok": bool(upstream["strict_x0_ok"]),
        "natoms": int(natoms),
        "chsc_status": result.status.value,
        "chsc_negative": result.negative,
        "h_strain": result.h,
        "lambda_h_ev_per_atom": result.lambda_h,
        "lambda_h2_ev_per_atom": result.lambda_h2,
        "lambda_r_ev_per_atom": result.lambda_r,
        "e_num_ev_per_atom": result.e_num,
        "u_num_ev_per_atom": result.u_num,
        "l_num_ev_per_atom": result.l_num,
        "tau_alg_ev_per_atom": result.tau_alg,
        "antisymmetric_norm_h_ev_per_atom": result.antisymmetric_norm_h,
        "antisymmetric_norm_h2_ev_per_atom": result.antisymmetric_norm_h2,
        "hessian_h_json": hessian_h_json,
        "hessian_h2_json": hessian_h2_json,
        "energy_call_count": int(result.energy_call_count),
        "error": result.error,
    }


def _is_canonical_cuda_device(device: str) -> bool:
    if not device.startswith("cuda:"):
        return False
    index = device.removeprefix("cuda:")
    return bool(index.isdigit() and str(int(index)) == index)


def run_label_free_features(
    *,
    phsc_features_path: Path,
    phsc_manifest_path: Path,
    frames_zip_path: Path,
    geometry_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    predictor: BatchForcePredictor | None = None,
    device: str = "cuda:0",
    model_batch_size: int = FROZEN_MODEL_BATCH_SIZE,
    structures_per_call: int = FROZEN_STRUCTURES_PER_CALL,
) -> dict[str, object]:
    """Seal additive CHSC features without accepting any endpoint input."""

    target = Path(output_dir)
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    if type(model_batch_size) is not int or model_batch_size <= 0:
        raise ValueError("model_batch_size must be a positive exact integer")
    if type(structures_per_call) is not int or structures_per_call <= 0:
        raise ValueError("structures_per_call must be a positive exact integer")
    device = str(device).strip().lower()
    if not device:
        raise ValueError("device must be a nonempty string")
    if predictor is None and (
        model_batch_size != FROZEN_MODEL_BATCH_SIZE
        or structures_per_call != FROZEN_STRUCTURES_PER_CALL
        or not _is_canonical_cuda_device(device)
    ):
        raise ValueError(
            "production CHSC requires cuda:N, model_batch_size=32, and "
            "structures_per_call=12"
        )

    paths = {
        "phsc_features": Path(phsc_features_path),
        "phsc_manifest": Path(phsc_manifest_path),
        "geometry_only_frames": Path(frames_zip_path),
        "geometry_manifest": Path(geometry_manifest_path),
        "checkpoint": Path(checkpoint_path),
    }
    snapshots = {
        role: _snapshot(path, include_data=role != "checkpoint")
        for role, path in paths.items()
    }
    phsc_manifest = _strict_json_document(
        snapshots["phsc_manifest"].data or b"", role="PHSC manifest"
    )
    _validate_phsc_manifest(
        phsc_manifest,
        phsc_name=snapshots["phsc_features"].path.name,
        phsc_sha256=snapshots["phsc_features"].sha256,
    )
    table = _load_phsc_table(snapshots["phsc_features"].data or b"")
    strict_sids = table.loc[table["strict_x0_ok"].astype(bool), "sid"].tolist()
    geometry_sids, structures = _load_geometry_snapshot(
        snapshots["geometry_only_frames"].data or b"",
        snapshots["geometry_manifest"].data or b"",
        strict_sids,
    )
    observed_counts = {
        "selected_rows": len(table),
        "strict_rows": len(strict_sids),
        "nonstrict_rows": len(table) - len(strict_sids),
    }
    if predictor is None:
        observed_hashes = {role: snapshot.sha256 for role, snapshot in snapshots.items()}
        if observed_hashes != FROZEN_INPUT_SHA256:
            mismatched = sorted(
                role
                for role, expected in FROZEN_INPUT_SHA256.items()
                if observed_hashes.get(role) != expected
            )
            raise ValueError(f"production CHSC inputs differ from frozen identities: {mismatched}")
        if observed_counts != FORMAL_EXPECTED_COUNTS:
            raise ValueError("production CHSC cohort counts differ from frozen PHSC cohort")

    runtime = _runtime_identity(device)
    if predictor is None:
        if runtime.get("mattersim_version") != "1.2.3":
            raise RuntimeError("production CHSC requires MatterSim 1.2.3")
        if runtime.get("cuda_available") is not True:
            raise RuntimeError("production CHSC requires available CUDA")
        active_predictor, loaded_checkpoint_sha256 = _production_predictor(
            snapshots["checkpoint"].path,
            device=device,
            batch_size=model_batch_size,
        )
        adapter_mode = "builtin_indexed_mattersim"
    else:
        active_predictor = predictor
        loaded_checkpoint_sha256 = None
        adapter_mode = "injected_test_double"

    predictor_batch_sizes: list[int] = []

    def counting_predictor(batch: list[Atoms]):
        predictor_batch_sizes.append(len(batch))
        return active_predictor(batch)

    started = time.perf_counter()
    batch_results = evaluate_chsc_batch(
        geometry_sids,
        structures,
        counting_predictor,
        structures_per_call=structures_per_call,
    )
    elapsed = time.perf_counter() - started
    by_sid = {item.sid: item for item in batch_results}
    natoms = {sid: len(atoms) for sid, atoms in zip(geometry_sids, structures, strict=True)}
    rows = [
        _result_row(
            record,
            by_sid.get(str(record["sid"])) if bool(record["strict_x0_ok"]) else None,
            natoms.get(str(record["sid"]), 0),
        )
        for record in table.to_dict("records")
    ]
    output_table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    energy_evaluations = int(output_table["energy_call_count"].sum())
    if any(size <= 0 or size % 85 != 0 for size in predictor_batch_sizes):
        raise BatchCHSCError("predictor call split a complete 85-probe CHSC group")
    if sum(predictor_batch_sizes) != energy_evaluations:
        raise BatchCHSCError("predictor telemetry differs from row energy-call counts")

    if predictor is None:
        telemetry = _validated_builtin_telemetry(
            active_predictor,
            device=device,
            expected_evaluations=energy_evaluations,
        )
        expected_forward_calls = sum(
            math.ceil(size / model_batch_size) for size in predictor_batch_sizes
        )
        if int(telemetry["forward_calls"]) != expected_forward_calls:
            raise RuntimeError("MatterSim forward calls differ from frozen chunking")
        production_eligible = True
        forward_calls: int | None = int(telemetry["forward_calls"])
        peak_cuda_memory_bytes: int | None = int(telemetry["peak_cuda_memory_bytes"])
        adapter = {
            "mode": adapter_mode,
            "index_alignment_verified": True,
            "device": device,
            "model_batch_size": model_batch_size,
            "structures_per_call": structures_per_call,
            "evaluations": int(telemetry["evaluations"]),
            "model_parameter_device": telemetry["model_parameter_device"],
            "result_tensor_devices": telemetry["result_tensor_devices"],
        }
    else:
        production_eligible = False
        forward_calls = None
        peak_cuda_memory_bytes = None
        adapter = {
            "mode": adapter_mode,
            "index_alignment_verified": False,
            "device": device,
            "model_batch_size": model_batch_size,
            "structures_per_call": structures_per_call,
            "evaluations": energy_evaluations,
            "model_parameter_device": None,
            "result_tensor_devices": [],
        }

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        relative: repository_root / relative for relative in EXECUTED_SOURCE_RELATIVE
    }
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}

    def verify_unchanged() -> None:
        for role, snapshot in snapshots.items():
            if _sha256_file(snapshot.path) != snapshot.sha256:
                raise RuntimeError(f"input {role} changed after initial hash")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_hashes[relative]:
                raise RuntimeError(f"executed source {relative} changed after initial hash")

    statuses = output_table["chsc_status"].astype(str)
    strict = output_table["strict_x0_ok"].astype(bool)
    counts = {
        **observed_counts,
        "resolved_negative_rows": int((statuses == CHSCStatus.RESOLVED_NEGATIVE.value).sum()),
        "resolved_nonnegative_rows": int(
            (statuses == CHSCStatus.RESOLVED_NONNEGATIVE.value).sum()
        ),
        "near_zero_or_inconsistent_rows": int(
            (statuses == CHSCStatus.NEAR_ZERO_OR_INCONSISTENT.value).sum()
        ),
        "abstained_rows": int((~strict).sum())
        + int(statuses.str.startswith("abstain_")[strict].sum()),
        "energy_evaluations": energy_evaluations,
        "batch_predictor_calls": len(predictor_batch_sizes),
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "development_gate",
        "labels_opened": False,
        "input_isolation": {
            "geometry_only": True,
            "raw_x0_archive_opened": False,
            "endpoint_label_artifacts_opened": False,
        },
        "criterion": CRITERION,
        "adapter": adapter,
        "predictor_loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "runtime": runtime,
        "counts": counts,
        "execution": {
            "batch_predictor_calls": len(predictor_batch_sizes),
            "predictor_batch_sizes": predictor_batch_sizes,
            "max_predictor_batch_size": max(predictor_batch_sizes, default=0),
            "forward_calls": forward_calls,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "wall_time_seconds": elapsed,
        },
        "inputs_sha256": {
            role: {"path": str(snapshot.path.resolve()), "sha256": snapshot.sha256}
            for role, snapshot in snapshots.items()
        },
        "executed_source_sha256": source_hashes,
        "feature_columns": list(OUTPUT_COLUMNS),
        "production_protocol_eligible": production_eligible,
        "scientific_improvement_claim": False,
        "known_limitations": [
            "CHSC-v0 is a MatterSim fixed-fractional cell-curvature diagnostic, not DFT.",
            "Negative curvature is rejection evidence; nonnegative curvature is not a stability certificate.",
            "No DFT endpoint or label was opened in this run.",
        ],
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        table_path = staging / OUTPUT_NAME
        output_table.to_parquet(table_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(table_path)}
        manifest_path = staging / MANIFEST_NAME
        payload = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        verify_unchanged()
        _atomic_publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phsc-features", required=True, type=Path)
    parser.add_argument("--phsc-manifest", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-batch-size", type=int, default=FROZEN_MODEL_BATCH_SIZE)
    parser.add_argument(
        "--structures-per-call", type=int, default=FROZEN_STRUCTURES_PER_CALL
    )
    arguments = parser.parse_args(argv)
    run_label_free_features(
        phsc_features_path=arguments.phsc_features,
        phsc_manifest_path=arguments.phsc_manifest,
        frames_zip_path=arguments.frames_zip,
        geometry_manifest_path=arguments.geometry_manifest,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
        model_batch_size=arguments.model_batch_size,
        structures_per_call=arguments.structures_per_call,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
