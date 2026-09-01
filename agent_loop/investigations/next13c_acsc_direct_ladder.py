"""Nested three-amplitude energy confirmation for sealed ACSC mixed modes."""

from __future__ import annotations

import argparse
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
    _production_predictor,
    _runtime_identity,
    _sha256_file,
    _snapshot,
    _strict_json_document,
    _validated_builtin_telemetry,
    _validated_prediction,
)
from src.next11_geometry_only_frames import load_geometry_only_archive
from src.next13b_acsc_direct import (
    DIRECT_VERSION,
    DirectStatus,
    direct_curvature_from_energies,
    mixed_mode_probe,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


LADDER_VERSION = "ACSC-DIRECT-LADDER-v0"
PROTOCOL = "2026-08-02-next13c-acsc-direct-ladder-v1"
UPSTREAM_DIRECT_PROTOCOL = "2026-08-02-next13b-acsc-direct-mattersim-v1"
UPSTREAM_PHSC_PROTOCOL = "2026-08-02-next11-phsc-mattersim-features-v1"
LARGE_STEP = 2**-7
FROZEN_MODEL_BATCH_SIZE = 32
FROZEN_CANDIDATES_PER_CALL = 32
OUTPUT_NAME = "acsc_direct_ladder.parquet"
MANIFEST_NAME = "MANIFEST.json"
OUTPUT_COLUMNS = (
    "sid",
    "rk",
    "natoms",
    "recomputed_coupling_only_negative",
    "small_direct_status",
    "small_direct_confirmed",
    "small_q_h_ev_per_atom",
    "small_q_h2_ev_per_atom",
    "small_q_r_ev_per_atom",
    "small_e_num_ev_per_atom",
    "small_u_num_ev_per_atom",
    "large_direct_status",
    "large_direct_negative",
    "large_q_h_ev_per_atom",
    "large_q_h2_ev_per_atom",
    "large_q_r_ev_per_atom",
    "large_e_num_ev_per_atom",
    "large_u_num_ev_per_atom",
    "large_l_num_ev_per_atom",
    "large_tau_alg_ev_per_atom",
    "independent_middle_curvature_delta_ev_per_atom",
    "nested_three_scale_confirmed",
    "prediction_evaluations",
    "error",
)
EXECUTED_SOURCE_RELATIVE = (
    "src/next13c_acsc_direct_ladder.py",
    "src/next13b_acsc_direct.py",
    "src/next11_phsc.py",
    "src/next11_geometry_only_frames.py",
    "src/next10_lrrc_mattersim_features.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
    "src/next6_wbm_build.py",
    "src/next6_wbm_features.py",
    "src/next6_wbm_protocol.py",
)


class LadderBatchError(RuntimeError):
    """Raised when a five-probe ladder group loses alignment."""


def nested_confirmation(small_confirmed: bool, large_status: str) -> bool:
    """Require both independently executed direct pairs to resolve negative."""

    if type(small_confirmed) is not bool:
        raise ValueError("small_confirmed must be an exact bool")
    try:
        status = DirectStatus(large_status)
    except (TypeError, ValueError) as exc:
        raise ValueError("large_status must be a known direct status") from exc
    return small_confirmed and status is DirectStatus.RESOLVED_NEGATIVE


def ladder_probe_group(base: Atoms, mode: np.ndarray) -> tuple[Atoms, ...]:
    """Return center, +/-2^-7, and +/-2^-8 mixed-mode probes."""

    return (
        base.copy(),
        mixed_mode_probe(base, mode, amplitude=LARGE_STEP),
        mixed_mode_probe(base, mode, amplitude=-LARGE_STEP),
        mixed_mode_probe(base, mode, amplitude=LARGE_STEP / 2.0),
        mixed_mode_probe(base, mode, amplitude=-LARGE_STEP / 2.0),
    )


def sealed_mode_table(direct_table: pd.DataFrame) -> pd.DataFrame:
    """Validate all sealed direct rows and parse their canonical mode vectors."""

    required = {
        "sid",
        "rk",
        "natoms",
        "mode_json",
        "recomputed_coupling_only_negative",
        "direct_status",
        "direct_confirmed",
        "direct_q_h_ev_per_atom",
        "direct_q_h2_ev_per_atom",
        "direct_q_r_ev_per_atom",
        "direct_e_num_ev_per_atom",
        "direct_u_num_ev_per_atom",
    }
    if not required.issubset(direct_table.columns):
        raise ValueError("sealed direct table lacks required columns")
    table = direct_table.loc[:, sorted(required)].copy()
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError("sealed direct sid values must be unique")
    parsed: list[np.ndarray] = []
    for row in table.itertuples(index=False):
        try:
            vector = np.asarray(json.loads(str(row.mode_json)), dtype=np.float64)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("sealed mode JSON is invalid") from exc
        expected = 3 * int(row.natoms) + 3
        if (
            vector.shape != (expected,)
            or not np.all(np.isfinite(vector))
            or not np.isclose(np.linalg.norm(vector), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("sealed mode vector shape/norm is invalid")
        expected_confirmed = bool(row.recomputed_coupling_only_negative) and str(
            row.direct_status
        ) == DirectStatus.RESOLVED_NEGATIVE.value
        if bool(row.direct_confirmed) != expected_confirmed:
            raise ValueError("sealed direct confirmation semantics mismatch")
        parsed.append(vector)
    table["mode_vector"] = parsed
    return table.sort_values("sid", kind="stable", ignore_index=True)


def _feature_manifest(
    data: bytes, *, protocol: str, output_name: str, output_sha256: str
) -> dict[str, object]:
    manifest = dict(_strict_json_document(data, role=f"{protocol} manifest"))
    if manifest.get("protocol") != protocol or manifest.get("labels_opened") is not False:
        raise ValueError("upstream manifest protocol/label isolation mismatch")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(output_name) != output_sha256:
        raise ValueError("upstream output hash differs from manifest")
    return manifest


def _manifest_input_hash(manifest: Mapping[str, object], role: str) -> str:
    inputs = manifest.get("inputs_sha256")
    record = inputs.get(role) if isinstance(inputs, Mapping) else None
    if not isinstance(record, Mapping) or type(record.get("sha256")) is not str:
        raise ValueError(f"upstream direct manifest lacks {role} hash")
    return str(record["sha256"])


def run_ladder(
    *,
    direct_features_path: Path,
    direct_manifest_path: Path,
    phsc_features_path: Path,
    phsc_manifest_path: Path,
    frames_zip_path: Path,
    geometry_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    predictor: BatchForcePredictor | None = None,
    device: str = "cuda:0",
    model_batch_size: int = FROZEN_MODEL_BATCH_SIZE,
    candidates_per_call: int = FROZEN_CANDIDATES_PER_CALL,
) -> dict[str, object]:
    """Evaluate the larger direct pair and publish nested three-scale states."""

    target = Path(output_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(target)
    paths = {
        "direct_features": Path(direct_features_path),
        "direct_manifest": Path(direct_manifest_path),
        "phsc_features": Path(phsc_features_path),
        "phsc_manifest": Path(phsc_manifest_path),
        "geometry_only_frames": Path(frames_zip_path),
        "geometry_manifest": Path(geometry_manifest_path),
        "checkpoint": Path(checkpoint_path),
    }
    retained = {"direct_features", "direct_manifest", "phsc_features", "phsc_manifest"}
    snapshots = {
        role: _snapshot(path, include_data=role in retained)
        for role, path in paths.items()
    }
    direct_manifest = _feature_manifest(
        snapshots["direct_manifest"].data or b"",
        protocol=UPSTREAM_DIRECT_PROTOCOL,
        output_name=paths["direct_features"].name,
        output_sha256=snapshots["direct_features"].sha256,
    )
    if direct_manifest.get("endpoint_artifacts_opened") is not False:
        raise ValueError("upstream direct manifest does not prove endpoint isolation")
    _feature_manifest(
        snapshots["phsc_manifest"].data or b"",
        protocol=UPSTREAM_PHSC_PROTOCOL,
        output_name=paths["phsc_features"].name,
        output_sha256=snapshots["phsc_features"].sha256,
    )
    for role in (
        "phsc_features",
        "phsc_manifest",
        "geometry_only_frames",
        "geometry_manifest",
        "checkpoint",
    ):
        if _manifest_input_hash(direct_manifest, role) != snapshots[role].sha256:
            raise ValueError(f"upstream direct input identity differs for {role}")

    direct_table = pd.read_parquet(io.BytesIO(snapshots["direct_features"].data or b""))
    modes = sealed_mode_table(direct_table)
    phsc = pd.read_parquet(io.BytesIO(snapshots["phsc_features"].data or b""))
    strict_sids = sorted(phsc.loc[phsc["strict_x0_ok"].astype(bool), "sid"].astype(str))
    geometry_sids, structures = load_geometry_only_archive(
        archive_path=paths["geometry_only_frames"],
        manifest_path=paths["geometry_manifest"],
        expected_sids=strict_sids,
    )
    geometry = dict(zip(geometry_sids, structures, strict=True))
    if any(sid not in geometry for sid in modes["sid"].astype(str)):
        raise ValueError("sealed direct mode is missing from geometry archive")

    runtime = _runtime_identity(device)
    if predictor is None:
        if runtime.get("mattersim_version") != "1.2.3" or runtime.get("cuda_available") is not True:
            raise RuntimeError("production direct ladder requires MatterSim 1.2.3 with CUDA")
        active_predictor, loaded_checkpoint_sha256 = _production_predictor(
            paths["checkpoint"], device=device, batch_size=model_batch_size
        )
        adapter_mode = "builtin_indexed_mattersim"
    else:
        active_predictor = predictor
        loaded_checkpoint_sha256 = None
        adapter_mode = "injected_test_double"

    if type(candidates_per_call) is not int or candidates_per_call <= 0:
        raise ValueError("candidates_per_call must be a positive exact integer")
    batch_sizes: list[int] = []
    results: dict[str, tuple[object, ...]] = {}
    records = modes.to_dict("records")
    started = time.perf_counter()
    for start in range(0, len(records), candidates_per_call):
        chunk = records[start : start + candidates_per_call]
        groups = [
            ladder_probe_group(geometry[str(row["sid"])], np.asarray(row["mode_vector"]))
            for row in chunk
        ]
        flat = [probe for group in groups for probe in group]
        batch_sizes.append(len(flat))
        prediction = active_predictor(flat)
        energies, _forces, _stresses = _validated_prediction(prediction, flat)
        if len(energies) != 5 * len(chunk):
            raise LadderBatchError("predictor split a five-probe ladder group")
        for index, row in enumerate(chunk):
            values = energies[5 * index : 5 * index + 5]
            result = direct_curvature_from_energies(
                *values, n_atoms=int(row["natoms"]), h=LARGE_STEP
            )
            results[str(row["sid"])] = (result,)
    elapsed = time.perf_counter() - started

    rows: list[dict[str, object]] = []
    for row in records:
        sid = str(row["sid"])
        result = results[sid][0]
        assert hasattr(result, "status")
        nested = nested_confirmation(bool(row["direct_confirmed"]), result.status.value)
        rows.append(
            {
                "sid": sid,
                "rk": str(row["rk"]),
                "natoms": int(row["natoms"]),
                "recomputed_coupling_only_negative": bool(row["recomputed_coupling_only_negative"]),
                "small_direct_status": str(row["direct_status"]),
                "small_direct_confirmed": bool(row["direct_confirmed"]),
                "small_q_h_ev_per_atom": float(row["direct_q_h_ev_per_atom"]),
                "small_q_h2_ev_per_atom": float(row["direct_q_h2_ev_per_atom"]),
                "small_q_r_ev_per_atom": float(row["direct_q_r_ev_per_atom"]),
                "small_e_num_ev_per_atom": float(row["direct_e_num_ev_per_atom"]),
                "small_u_num_ev_per_atom": float(row["direct_u_num_ev_per_atom"]),
                "large_direct_status": result.status.value,
                "large_direct_negative": result.negative,
                "large_q_h_ev_per_atom": result.q_h,
                "large_q_h2_ev_per_atom": result.q_h2,
                "large_q_r_ev_per_atom": result.q_r,
                "large_e_num_ev_per_atom": result.e_num,
                "large_u_num_ev_per_atom": result.u_num,
                "large_l_num_ev_per_atom": result.l_num,
                "large_tau_alg_ev_per_atom": result.tau_alg,
                "independent_middle_curvature_delta_ev_per_atom": (
                    result.q_h2 - float(row["direct_q_h_ev_per_atom"])
                ),
                "nested_three_scale_confirmed": nested,
                "prediction_evaluations": result.energy_call_count,
                "error": "",
            }
        )
    output_table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        "sid", kind="stable", ignore_index=True
    )
    evaluations = int(output_table["prediction_evaluations"].sum())
    if sum(batch_sizes) != evaluations:
        raise RuntimeError("ladder batch sizes differ from row evaluations")

    if predictor is None:
        telemetry = _validated_builtin_telemetry(
            active_predictor, device=device, expected_evaluations=evaluations
        )
        expected_forwards = sum(math.ceil(size / model_batch_size) for size in batch_sizes)
        if int(telemetry["forward_calls"]) != expected_forwards:
            raise RuntimeError("MatterSim forwards differ from ladder chunking")
        production_eligible = True
        forward_calls: int | None = int(telemetry["forward_calls"])
        peak_cuda_memory_bytes: int | None = int(telemetry["peak_cuda_memory_bytes"])
    else:
        production_eligible = False
        forward_calls = None
        peak_cuda_memory_bytes = None

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {relative: repository_root / relative for relative in EXECUTED_SOURCE_RELATIVE}
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    large_status = output_table["large_direct_status"].astype(str)
    counts = {
        "sealed_modes": len(output_table),
        "small_direct_confirmed": int(output_table["small_direct_confirmed"].sum()),
        "large_resolved_negative": int(large_status.eq(DirectStatus.RESOLVED_NEGATIVE.value).sum()),
        "large_resolved_nonnegative": int(large_status.eq(DirectStatus.RESOLVED_NONNEGATIVE.value).sum()),
        "large_near_zero_or_inconsistent": int(large_status.eq(DirectStatus.NEAR_ZERO_OR_INCONSISTENT.value).sum()),
        "nested_three_scale_confirmed": int(output_table["nested_three_scale_confirmed"].sum()),
        "prediction_evaluations": evaluations,
        "batch_predictor_calls": len(batch_sizes),
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "version": LADDER_VERSION,
        "upstream_direct_version": DIRECT_VERSION,
        "mode": "sealed_label_free_nested_three_scale_confirmation",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "criterion": {
            "amplitudes": [2**-7, 2**-8, 2**-9],
            "nested_confirmation": (
                "small pair (2^-8,2^-9) resolved_negative and "
                "large pair (2^-7,2^-8) resolved_negative"
            ),
            "thresholds_refit": False,
        },
        "counts": counts,
        "adapter": {
            "mode": adapter_mode,
            "device": device,
            "model_batch_size": model_batch_size,
            "candidates_per_call": candidates_per_call,
        },
        "predictor_loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "runtime": runtime,
        "execution": {
            "predictor_batch_sizes": batch_sizes,
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
            "All three amplitudes use the same MatterSim checkpoint.",
            "Energy quantization remains visible and is handled by abstaining, not threshold fitting.",
            "No DFT endpoint or label was opened.",
        ],
    }

    def verify_unchanged() -> None:
        for role, snapshot in snapshots.items():
            if _sha256_file(snapshot.path) != snapshot.sha256:
                raise RuntimeError(f"input {role} changed after initial hash")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_hashes[relative]:
                raise RuntimeError(f"executed source {relative} changed after initial hash")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        table_path = staging / OUTPUT_NAME
        output_table.to_parquet(table_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(table_path)}
        payload = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        manifest_path = staging / MANIFEST_NAME
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
    parser.add_argument("--direct-features", required=True, type=Path)
    parser.add_argument("--direct-manifest", required=True, type=Path)
    parser.add_argument("--phsc-features", required=True, type=Path)
    parser.add_argument("--phsc-manifest", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-batch-size", type=int, default=FROZEN_MODEL_BATCH_SIZE)
    parser.add_argument("--candidates-per-call", type=int, default=FROZEN_CANDIDATES_PER_CALL)
    arguments = parser.parse_args(argv)
    run_ladder(
        direct_features_path=arguments.direct_features,
        direct_manifest_path=arguments.direct_manifest,
        phsc_features_path=arguments.phsc_features,
        phsc_manifest_path=arguments.phsc_manifest,
        frames_zip_path=arguments.frames_zip,
        geometry_manifest_path=arguments.geometry_manifest,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
        model_batch_size=arguments.model_batch_size,
        candidates_per_call=arguments.candidates_per_call,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
