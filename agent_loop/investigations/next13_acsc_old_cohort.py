"""Frozen label-free ACSC-v0 audit on the old PHSC/CHSC cohort.

Only rows whose sealed upstream PHSC-v0 and CHSC-v0 states are both
``resolved_nonnegative`` are evaluated.  This is a prespecified incremental
coupling test and never opens DFT endpoints or label artifacts.
"""

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

from src.next10_lrrc_mattersim_features import (
    BatchForcePredictor,
    _production_predictor,
    _runtime_identity,
    _sha256_file,
    _snapshot,
    _strict_json_document,
    _validated_builtin_telemetry,
)
from src.next11_geometry_only_frames import load_geometry_only_archive
from src.next11_phsc import PHSCStatus
from src.next12_chsc import CHSCStatus
from src.next13_acsc import ACSC_VERSION, ACSCStatus
from src.next13_acsc_mattersim_features import (
    FROZEN_STRUCTURES_PER_CALL,
    BatchACSCResult,
    evaluate_acsc_batch,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next13-acsc-old-cohort-v1"
UPSTREAM_PHSC_PROTOCOL = "2026-08-02-next11-phsc-mattersim-features-v1"
UPSTREAM_CHSC_PROTOCOL = "2026-08-02-next12-chsc-mattersim-features-v1"
FROZEN_MODEL_BATCH_SIZE = 32
OUTPUT_NAME = "acsc_incremental_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
OUTPUT_COLUMNS = (
    "sid",
    "rk",
    "stage",
    "threshold_role",
    "natoms",
    "upstream_phsc_status",
    "upstream_chsc_status",
    "recomputed_phsc_status",
    "recomputed_chsc_status",
    "pure_status_drift",
    "acsc_status",
    "acsc_negative",
    "coupling_only_negative",
    "d_star_angstrom",
    "lambda_h_ev_per_atom",
    "lambda_h2_ev_per_atom",
    "lambda_r_ev_per_atom",
    "e_num_ev_per_atom",
    "u_num_ev_per_atom",
    "l_num_ev_per_atom",
    "tau_alg_ev_per_atom",
    "antisymmetric_norm_h_ev_per_atom",
    "antisymmetric_norm_h2_ev_per_atom",
    "cross_norm_h_ev_per_a",
    "cross_norm_h2_ev_per_a",
    "prediction_evaluation_count",
    "error",
)
EXECUTED_SOURCE_RELATIVE = (
    "src/next13_acsc_old_cohort.py",
    "src/next13_acsc_mattersim_features.py",
    "src/next13_acsc.py",
    "src/next12_chsc_mattersim_features.py",
    "src/next12_chsc.py",
    "src/next11_phsc_mattersim_features.py",
    "src/next11_phsc.py",
    "src/next11_geometry_only_frames.py",
    "src/next10_lrrc_mattersim_features.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
    "src/next6_wbm_build.py",
    "src/next6_wbm_features.py",
    "src/next6_wbm_protocol.py",
)
_IN_MEMORY_INPUT_ROLES = frozenset(
    {"phsc_features", "phsc_manifest", "chsc_features", "chsc_manifest"}
)


def _snapshot_inputs(paths: Mapping[str, Path]) -> dict[str, object]:
    """Hash every input while retaining bytes only for in-memory parsers."""

    return {
        role: _snapshot(path, include_data=role in _IN_MEMORY_INPUT_ROLES)
        for role, path in paths.items()
    }


def _required_table(table: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    status_column = f"{kind}_status"
    required = {
        "sid",
        "rk",
        "stage",
        "threshold_role",
        "strict_x0_ok",
        "natoms",
        status_column,
    }
    if not required.issubset(table.columns):
        raise ValueError(f"upstream {kind.upper()} table lacks required columns")
    selected = table.loc[:, sorted(required)].copy()
    if selected["sid"].isna().any() or selected["sid"].astype(str).duplicated().any():
        raise ValueError(f"upstream {kind.upper()} sid values must be unique")
    selected["sid"] = selected["sid"].astype(str)
    return selected.sort_values("sid", kind="stable", ignore_index=True)


def eligible_upstream_table(
    phsc_table: pd.DataFrame, chsc_table: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the frozen both-nonnegative intersection and audited counts."""

    phsc = _required_table(phsc_table, kind="phsc")
    chsc = _required_table(chsc_table, kind="chsc")
    if phsc["sid"].tolist() != chsc["sid"].tolist():
        raise ValueError("upstream PHSC/CHSC sid alignment differs")
    metadata = ("rk", "stage", "threshold_role", "strict_x0_ok", "natoms")
    for column in metadata:
        left = phsc[column].astype(str).tolist()
        right = chsc[column].astype(str).tolist()
        if left != right:
            raise ValueError(f"upstream metadata alignment differs for {column}")
    strict = phsc["strict_x0_ok"].astype(bool)
    eligible_mask = (
        strict
        & phsc["phsc_status"].astype(str).eq(PHSCStatus.RESOLVED_NONNEGATIVE.value)
        & chsc["chsc_status"].astype(str).eq(CHSCStatus.RESOLVED_NONNEGATIVE.value)
    )
    eligible = pd.DataFrame(
        {
            "sid": phsc.loc[eligible_mask, "sid"].astype(str),
            "rk": phsc.loc[eligible_mask, "rk"].astype(str),
            "stage": phsc.loc[eligible_mask, "stage"].astype(str),
            "threshold_role": phsc.loc[eligible_mask, "threshold_role"].astype(str),
            "natoms": phsc.loc[eligible_mask, "natoms"].astype("int64"),
            "upstream_phsc_status": phsc.loc[eligible_mask, "phsc_status"].astype(str),
            "upstream_chsc_status": chsc.loc[eligible_mask, "chsc_status"].astype(str),
        }
    ).reset_index(drop=True)
    counts = {
        "upstream_rows": len(phsc),
        "upstream_strict_rows": int(strict.sum()),
        "eligible_both_resolved_nonnegative_rows": len(eligible),
    }
    return eligible, counts


def _validate_feature_manifest(
    data: bytes,
    *,
    protocol: str,
    output_name: str,
    output_sha256: str,
) -> Mapping[str, object]:
    manifest = _strict_json_document(data, role=f"{protocol} manifest")
    if manifest.get("protocol") != protocol:
        raise ValueError("upstream feature manifest protocol mismatch")
    if manifest.get("labels_opened") is not False:
        raise ValueError("upstream feature manifest is not label-free")
    isolation = manifest.get("input_isolation")
    if not isinstance(isolation, Mapping) or isolation.get("endpoint_label_artifacts_opened") is not False:
        raise ValueError("upstream feature manifest does not seal endpoint isolation")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(output_name) != output_sha256:
        raise ValueError("upstream feature table hash differs from its manifest")
    return manifest


def _result_row(upstream: Mapping[str, object], item: BatchACSCResult) -> dict[str, object]:
    acsc = item.acsc
    pure_drift = bool(
        item.phsc.status.value != str(upstream["upstream_phsc_status"])
        or item.chsc.status.value != str(upstream["upstream_chsc_status"])
    )
    cross_norm_h = (
        float(np.linalg.norm(item.cross_h, ord=2))
        if item.cross_h.size and np.all(np.isfinite(item.cross_h))
        else np.nan
    )
    cross_norm_h2 = (
        float(np.linalg.norm(item.cross_h2, ord=2))
        if item.cross_h2.size and np.all(np.isfinite(item.cross_h2))
        else np.nan
    )
    coupling_only = bool(acsc.coupling_only_negative) and not pure_drift
    return {
        **{key: upstream[key] for key in ("sid", "rk", "stage", "threshold_role", "natoms")},
        "upstream_phsc_status": upstream["upstream_phsc_status"],
        "upstream_chsc_status": upstream["upstream_chsc_status"],
        "recomputed_phsc_status": item.phsc.status.value,
        "recomputed_chsc_status": item.chsc.status.value,
        "pure_status_drift": pure_drift,
        "acsc_status": acsc.status.value,
        "acsc_negative": acsc.negative,
        "coupling_only_negative": coupling_only,
        "d_star_angstrom": acsc.d_star,
        "lambda_h_ev_per_atom": acsc.lambda_h,
        "lambda_h2_ev_per_atom": acsc.lambda_h2,
        "lambda_r_ev_per_atom": acsc.lambda_r,
        "e_num_ev_per_atom": acsc.e_num,
        "u_num_ev_per_atom": acsc.u_num,
        "l_num_ev_per_atom": acsc.l_num,
        "tau_alg_ev_per_atom": acsc.tau_alg,
        "antisymmetric_norm_h_ev_per_atom": acsc.antisymmetric_norm_h,
        "antisymmetric_norm_h2_ev_per_atom": acsc.antisymmetric_norm_h2,
        "cross_norm_h_ev_per_a": cross_norm_h,
        "cross_norm_h2_ev_per_a": cross_norm_h2,
        "prediction_evaluation_count": acsc.prediction_evaluation_count,
        "error": acsc.error,
    }


def run_label_free_old_cohort(
    *,
    phsc_features_path: Path,
    phsc_manifest_path: Path,
    chsc_features_path: Path,
    chsc_manifest_path: Path,
    frames_zip_path: Path,
    geometry_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    predictor: BatchForcePredictor | None = None,
    device: str = "cuda:0",
    model_batch_size: int = FROZEN_MODEL_BATCH_SIZE,
    structures_per_call: int = FROZEN_STRUCTURES_PER_CALL,
) -> dict[str, object]:
    """Run the prespecified ACSC incremental audit and atomically publish it."""

    target = Path(output_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(target)
    paths = {
        "phsc_features": Path(phsc_features_path),
        "phsc_manifest": Path(phsc_manifest_path),
        "chsc_features": Path(chsc_features_path),
        "chsc_manifest": Path(chsc_manifest_path),
        "geometry_only_frames": Path(frames_zip_path),
        "geometry_manifest": Path(geometry_manifest_path),
        "checkpoint": Path(checkpoint_path),
    }
    snapshots = _snapshot_inputs(paths)
    phsc_manifest = _validate_feature_manifest(
        snapshots["phsc_manifest"].data or b"",
        protocol=UPSTREAM_PHSC_PROTOCOL,
        output_name=paths["phsc_features"].name,
        output_sha256=snapshots["phsc_features"].sha256,
    )
    chsc_manifest = _validate_feature_manifest(
        snapshots["chsc_manifest"].data or b"",
        protocol=UPSTREAM_CHSC_PROTOCOL,
        output_name=paths["chsc_features"].name,
        output_sha256=snapshots["chsc_features"].sha256,
    )
    loaded_hashes = {
        phsc_manifest.get("predictor_loaded_checkpoint_sha256"),
        chsc_manifest.get("predictor_loaded_checkpoint_sha256"),
    }
    if loaded_hashes != {snapshots["checkpoint"].sha256}:
        raise ValueError("upstream model identity or requested checkpoint differs")

    phsc_table = pd.read_parquet(io.BytesIO(snapshots["phsc_features"].data or b""))
    chsc_table = pd.read_parquet(io.BytesIO(snapshots["chsc_features"].data or b""))
    eligible, selection_counts = eligible_upstream_table(phsc_table, chsc_table)
    strict_sids = sorted(
        phsc_table.loc[phsc_table["strict_x0_ok"].astype(bool), "sid"].astype(str)
    )
    geometry_sids, structures = load_geometry_only_archive(
        archive_path=paths["geometry_only_frames"],
        manifest_path=paths["geometry_manifest"],
        expected_sids=strict_sids,
    )
    geometry = dict(zip(geometry_sids, structures, strict=True))
    eligible_sids = eligible["sid"].astype(str).tolist()
    if any(sid not in geometry for sid in eligible_sids):
        raise ValueError("eligible sid is missing from sealed geometry archive")
    eligible_structures = [geometry[sid] for sid in eligible_sids]
    for sid, expected_natoms, atoms in zip(
        eligible_sids, eligible["natoms"].astype(int), eligible_structures, strict=True
    ):
        if len(atoms) != expected_natoms:
            raise ValueError(f"eligible natoms differs from sealed geometry for {sid}")

    runtime = _runtime_identity(device)
    if predictor is None:
        if runtime.get("mattersim_version") != "1.2.3" or runtime.get("cuda_available") is not True:
            raise RuntimeError("production ACSC requires MatterSim 1.2.3 with CUDA")
        active_predictor, loaded_checkpoint_sha256 = _production_predictor(
            paths["checkpoint"], device=device, batch_size=model_batch_size
        )
        adapter_mode = "builtin_indexed_mattersim"
    else:
        active_predictor = predictor
        loaded_checkpoint_sha256 = None
        adapter_mode = "injected_test_double"

    predictor_batch_sizes: list[int] = []

    def counting_predictor(batch: list[object]):
        predictor_batch_sizes.append(len(batch))
        return active_predictor(batch)  # type: ignore[arg-type]

    started = time.perf_counter()
    batch_results = evaluate_acsc_batch(
        eligible_sids,
        eligible_structures,
        counting_predictor,
        structures_per_call=structures_per_call,
    )
    elapsed = time.perf_counter() - started
    by_sid = {item.sid: item for item in batch_results}
    rows = [
        _result_row(record, by_sid[str(record["sid"])])
        for record in eligible.to_dict("records")
    ]
    output_table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    evaluations = int(output_table["prediction_evaluation_count"].sum())
    if sum(predictor_batch_sizes) != evaluations:
        raise RuntimeError("predictor batch sizes differ from ACSC row evaluations")

    if predictor is None:
        telemetry = _validated_builtin_telemetry(
            active_predictor, device=device, expected_evaluations=evaluations
        )
        expected_forwards = sum(
            math.ceil(size / model_batch_size) for size in predictor_batch_sizes
        )
        if int(telemetry["forward_calls"]) != expected_forwards:
            raise RuntimeError("MatterSim forward calls differ from frozen ACSC chunking")
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
    statuses = output_table["acsc_status"].astype(str)
    counts = {
        **selection_counts,
        "recomputed_pure_status_drift_rows": int(output_table["pure_status_drift"].sum()),
        "acsc_resolved_negative_rows": int(statuses.eq(ACSCStatus.RESOLVED_NEGATIVE.value).sum()),
        "acsc_resolved_nonnegative_rows": int(statuses.eq(ACSCStatus.RESOLVED_NONNEGATIVE.value).sum()),
        "acsc_near_zero_or_inconsistent_rows": int(statuses.eq(ACSCStatus.NEAR_ZERO_OR_INCONSISTENT.value).sum()),
        "acsc_abstained_rows": int(statuses.str.startswith("abstain_").sum()),
        "incremental_coupling_only_negative_rows": int(output_table["coupling_only_negative"].sum()),
        "prediction_evaluations": evaluations,
        "batch_predictor_calls": len(predictor_batch_sizes),
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "development_gate_incremental_label_free",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "input_isolation": {
            "geometry_only": True,
            "raw_x0_archive_opened": False,
            "endpoint_label_artifacts_opened": False,
        },
        "criterion": {
            "name": ACSC_VERSION,
            "selection": "sealed upstream PHSC and CHSC both resolved_nonnegative",
            "incremental_reject": (
                "upstream and recomputed pure states resolved_nonnegative; "
                "ACSC resolved_negative"
            ),
            "additional_probe_structures_beyond_phsc_union_chsc": 0,
            "numerical_proxy_is_confidence_bound": False,
        },
        "adapter": {
            "mode": adapter_mode,
            "device": device,
            "model_batch_size": model_batch_size,
            "structures_per_call": structures_per_call,
        },
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
            "ACSC-v0 is a MatterSim Gamma-point curvature diagnostic, not DFT.",
            "The old cohort is development-only and cannot establish prospective quality.",
            "No DFT endpoint or label was opened in this run.",
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
    parser.add_argument("--phsc-features", required=True, type=Path)
    parser.add_argument("--phsc-manifest", required=True, type=Path)
    parser.add_argument("--chsc-features", required=True, type=Path)
    parser.add_argument("--chsc-manifest", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-batch-size", type=int, default=FROZEN_MODEL_BATCH_SIZE)
    parser.add_argument("--structures-per-call", type=int, default=FROZEN_STRUCTURES_PER_CALL)
    arguments = parser.parse_args(argv)
    run_label_free_old_cohort(
        phsc_features_path=arguments.phsc_features,
        phsc_manifest_path=arguments.phsc_manifest,
        chsc_features_path=arguments.chsc_features,
        chsc_manifest_path=arguments.chsc_manifest,
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
