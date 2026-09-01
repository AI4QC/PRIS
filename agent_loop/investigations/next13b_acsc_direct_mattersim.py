"""Sealed MatterSim execution of ACSC-DIRECT-v0 on 123 ACSC candidates."""

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
    _production_predictor,
    _runtime_identity,
    _sha256_file,
    _snapshot,
    _strict_json_document,
    _validated_builtin_telemetry,
    _validated_prediction,
)
from src.next11_geometry_only_frames import load_geometry_only_archive
from src.next11_phsc import (
    PHSCStatus,
    analyze_hessian_pair,
    hessian_columns_from_force_samples,
)
from src.next12_chsc import CHSCStatus
from src.next12_chsc_mattersim_features import _result_from_energies
from src.next13_acsc import (
    ACSCSpectralResult,
    ACSCStatus,
    analyze_coupled_hessian_pair,
    cross_hessians_from_strain_forces,
    scaled_internal_coupled_hessian,
)
from src.next13_acsc_mattersim_features import (
    FROZEN_STRUCTURES_PER_CALL,
    _prepare,
)
from src.next13b_acsc_direct import (
    DIRECT_STEP,
    DIRECT_VERSION,
    DirectCurvatureResult,
    DirectStatus,
    MinimumMode,
    direct_curvature_from_energies,
    minimum_richardson_mode,
    mixed_mode_probe,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next13b-acsc-direct-mattersim-v1"
UPSTREAM_ACSC_PROTOCOL = "2026-08-02-next13-acsc-old-cohort-v1"
UPSTREAM_PHSC_PROTOCOL = "2026-08-02-next11-phsc-mattersim-features-v1"
FROZEN_MODEL_BATCH_SIZE = 32
FROZEN_DIRECT_CANDIDATES_PER_CALL = 32
OUTPUT_NAME = "acsc_direct_confirmation.parquet"
MANIFEST_NAME = "MANIFEST.json"
OUTPUT_COLUMNS = (
    "sid",
    "rk",
    "natoms",
    "formal_lambda_r_ev_per_atom",
    "formal_e_num_ev_per_atom",
    "formal_u_num_ev_per_atom",
    "recomputed_phsc_status",
    "recomputed_chsc_status",
    "recomputed_acsc_status",
    "recomputed_coupling_only_negative",
    "recomputed_lambda_r_ev_per_atom",
    "recomputed_e_num_ev_per_atom",
    "recomputed_u_num_ev_per_atom",
    "mode_spectral_gap_ev_per_atom",
    "mode_atomic_fraction",
    "mode_strain_fraction",
    "mode_json",
    "direct_status",
    "direct_negative",
    "direct_confirmed",
    "direct_h",
    "direct_q_h_ev_per_atom",
    "direct_q_h2_ev_per_atom",
    "direct_q_r_ev_per_atom",
    "direct_e_num_ev_per_atom",
    "direct_u_num_ev_per_atom",
    "direct_l_num_ev_per_atom",
    "direct_tau_alg_ev_per_atom",
    "direct_minus_matrix_lambda_r_ev_per_atom",
    "center_energy_repeat_delta_ev",
    "reconstruction_prediction_evaluations",
    "direct_prediction_evaluations",
    "error",
)
EXECUTED_SOURCE_RELATIVE = (
    "src/next13b_acsc_direct_mattersim.py",
    "src/next13b_acsc_direct.py",
    "src/next13_acsc_mattersim_features.py",
    "src/next13_acsc.py",
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


class DirectBatchError(RuntimeError):
    """Raised when sealed reconstruction or direct groups lose alignment."""


@dataclass(frozen=True, slots=True)
class ReconstructedCandidate:
    sid: str
    base: Atoms
    d_star: float
    phsc_status: PHSCStatus
    chsc_status: CHSCStatus
    acsc: ACSCSpectralResult
    mode: MinimumMode
    center_energy: float
    prediction_evaluations: int


@dataclass(frozen=True, slots=True)
class DirectCandidateResult:
    sid: str
    result: DirectCurvatureResult
    center_repeat_delta: float


def sealed_candidate_table(formal_table: pd.DataFrame) -> pd.DataFrame:
    """Select exact sealed coupling-only negatives and reject semantic drift."""

    required = {
        "sid",
        "rk",
        "natoms",
        "upstream_phsc_status",
        "upstream_chsc_status",
        "recomputed_phsc_status",
        "recomputed_chsc_status",
        "pure_status_drift",
        "acsc_status",
        "coupling_only_negative",
        "lambda_r_ev_per_atom",
        "e_num_ev_per_atom",
        "u_num_ev_per_atom",
    }
    if not required.issubset(formal_table.columns):
        raise ValueError("formal ACSC table lacks required columns")
    table = formal_table.loc[:, sorted(required)].copy()
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError("formal ACSC sid values must be unique")
    table["sid"] = table["sid"].astype(str)
    selected = table.loc[table["coupling_only_negative"].astype(bool)].copy()
    valid = (
        ~selected["pure_status_drift"].astype(bool)
        & selected["upstream_phsc_status"].astype(str).eq(PHSCStatus.RESOLVED_NONNEGATIVE.value)
        & selected["upstream_chsc_status"].astype(str).eq(CHSCStatus.RESOLVED_NONNEGATIVE.value)
        & selected["recomputed_phsc_status"].astype(str).eq(PHSCStatus.RESOLVED_NONNEGATIVE.value)
        & selected["recomputed_chsc_status"].astype(str).eq(CHSCStatus.RESOLVED_NONNEGATIVE.value)
        & selected["acsc_status"].astype(str).eq(ACSCStatus.RESOLVED_NEGATIVE.value)
    )
    if not valid.all():
        raise ValueError("true coupling-only flag conflicts with sealed ACSC semantics")
    return selected.sort_values("sid", kind="stable", ignore_index=True)


def direct_probe_group(base: Atoms, mode: np.ndarray) -> tuple[Atoms, ...]:
    """Return center, +/-h, +/-h/2 in the frozen direct order."""

    return (
        base.copy(),
        mixed_mode_probe(base, mode, amplitude=DIRECT_STEP),
        mixed_mode_probe(base, mode, amplitude=-DIRECT_STEP),
        mixed_mode_probe(base, mode, amplitude=DIRECT_STEP / 2.0),
        mixed_mode_probe(base, mode, amplitude=-DIRECT_STEP / 2.0),
    )


def _reconstruct_one(
    sid: str,
    prepared: object,
    energies: Sequence[float],
    forces: Sequence[np.ndarray],
) -> ReconstructedCandidate:
    base = prepared.base
    d_star = float(prepared.d_star)
    h_atomic = float(prepared.h_atomic)
    n_atoms = len(base)
    atomic_count = 12 * n_atoms
    expected = atomic_count + 85
    if len(energies) != expected or len(forces) != expected:
        raise DirectBatchError(f"incomplete reconstruction prediction for {sid}")
    dimension = 3 * n_atoms
    atomic_h = np.empty((dimension, dimension), dtype=np.float64)
    atomic_h2 = np.empty((dimension, dimension), dtype=np.float64)
    for coordinate in range(dimension):
        offset = 4 * coordinate
        column_h, column_h2 = hessian_columns_from_force_samples(
            *forces[offset : offset + 4], h=h_atomic
        )
        atomic_h[:, coordinate] = column_h
        atomic_h2[:, coordinate] = column_h2
    phsc = analyze_hessian_pair(atomic_h, atomic_h2)

    strain_energies = energies[atomic_count:]
    chsc, strain_h, strain_h2 = _result_from_energies(strain_energies, n_atoms)
    strain_forces = forces[atomic_count:]
    axial = tuple(
        np.stack([strain_forces[1 + 4 * axis + offset] for axis in range(6)])
        for offset in range(4)
    )
    cross_h, cross_h2 = cross_hessians_from_strain_forces(*axial, h=2**-7)
    k_h = scaled_internal_coupled_hessian(
        atomic_h, strain_h, cross_h, d_star=d_star
    )
    k_h2 = scaled_internal_coupled_hessian(
        atomic_h2, strain_h2, cross_h2, d_star=d_star
    )
    acsc = analyze_coupled_hessian_pair(k_h, k_h2)
    mode = minimum_richardson_mode(k_h, k_h2)
    if not np.isclose(mode.lambda_r, acsc.lambda_r, rtol=0.0, atol=2e-12):
        raise DirectBatchError(f"minimum mode and ACSC lambda_R differ for {sid}")
    return ReconstructedCandidate(
        sid=sid,
        base=base.copy(),
        d_star=d_star,
        phsc_status=phsc.status,
        chsc_status=chsc.status,
        acsc=acsc,
        mode=mode,
        center_energy=float(strain_energies[0]),
        prediction_evaluations=expected,
    )


def reconstruct_candidates(
    sids: Sequence[str],
    structures: Sequence[Atoms],
    predictor: BatchForcePredictor,
    *,
    structures_per_call: int = FROZEN_STRUCTURES_PER_CALL,
) -> tuple[ReconstructedCandidate, ...]:
    """Rebuild full coupled matrices for complete sorted candidate groups."""

    if len(sids) != len(structures) or len(set(sids)) != len(sids):
        raise DirectBatchError("reconstruction sids/structures must align uniquely")
    if type(structures_per_call) is not int or structures_per_call <= 0:
        raise ValueError("structures_per_call must be a positive exact integer")
    ordered = sorted(zip(sids, structures, strict=True), key=lambda pair: pair[0])
    prepared = [(sid, _prepare(sid, atoms)) for sid, atoms in ordered]
    completed: dict[str, ReconstructedCandidate] = {}
    for start in range(0, len(prepared), structures_per_call):
        chunk = prepared[start : start + structures_per_call]
        flat = [probe for _sid, item in chunk for probe in item.probes]
        prediction = predictor(flat)
        energies, forces, _stresses = _validated_prediction(prediction, flat)
        offset = 0
        for sid, item in chunk:
            count = len(item.probes)
            completed[sid] = _reconstruct_one(
                sid,
                item,
                energies[offset : offset + count],
                forces[offset : offset + count],
            )
            offset += count
        if offset != len(flat):
            raise DirectBatchError("unused reconstruction output remains")
    return tuple(completed[sid] for sid, _atoms in ordered)


def evaluate_direct_candidates(
    candidates: Sequence[ReconstructedCandidate],
    predictor: BatchForcePredictor,
    *,
    candidates_per_call: int = FROZEN_DIRECT_CANDIDATES_PER_CALL,
) -> tuple[DirectCandidateResult, ...]:
    """Evaluate complete five-structure mixed-mode groups."""

    if type(candidates_per_call) is not int or candidates_per_call <= 0:
        raise ValueError("candidates_per_call must be a positive exact integer")
    ordered = sorted(candidates, key=lambda item: item.sid)
    if len({item.sid for item in ordered}) != len(ordered):
        raise DirectBatchError("direct candidate sids must be unique")
    completed: dict[str, DirectCandidateResult] = {}
    for start in range(0, len(ordered), candidates_per_call):
        chunk = ordered[start : start + candidates_per_call]
        groups = [direct_probe_group(item.base, item.mode.vector) for item in chunk]
        flat = [probe for group in groups for probe in group]
        prediction = predictor(flat)
        energies, _forces, _stresses = _validated_prediction(prediction, flat)
        if len(energies) != 5 * len(chunk):
            raise DirectBatchError("direct predictor split a five-structure group")
        for index, item in enumerate(chunk):
            values = energies[5 * index : 5 * index + 5]
            result = direct_curvature_from_energies(
                *values, n_atoms=len(item.base), h=DIRECT_STEP
            )
            completed[item.sid] = DirectCandidateResult(
                sid=item.sid,
                result=result,
                center_repeat_delta=float(values[0] - item.center_energy),
            )
    return tuple(completed[item.sid] for item in ordered)


def _validated_feature_manifest(
    data: bytes,
    *,
    protocol: str,
    output_name: str,
    output_sha256: str,
) -> dict[str, object]:
    manifest = dict(_strict_json_document(data, role=f"{protocol} manifest"))
    if manifest.get("protocol") != protocol:
        raise ValueError("upstream protocol mismatch")
    if manifest.get("labels_opened") is not False:
        raise ValueError("upstream manifest is not label-free")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(output_name) != output_sha256:
        raise ValueError("upstream table hash differs from its manifest")
    return manifest


def _snapshot_inputs(paths: Mapping[str, Path]) -> dict[str, object]:
    retained = {"acsc_features", "acsc_manifest", "phsc_features", "phsc_manifest"}
    return {
        role: _snapshot(path, include_data=role in retained)
        for role, path in paths.items()
    }


def _input_hash_from_manifest(manifest: Mapping[str, object], role: str) -> str:
    inputs = manifest.get("inputs_sha256")
    if not isinstance(inputs, Mapping):
        raise ValueError("formal ACSC manifest lacks input hashes")
    record = inputs.get(role)
    if not isinstance(record, Mapping) or type(record.get("sha256")) is not str:
        raise ValueError(f"formal ACSC manifest lacks {role} hash")
    return str(record["sha256"])


def _row(
    formal: Mapping[str, object],
    reconstructed: ReconstructedCandidate,
    direct: DirectCandidateResult,
) -> dict[str, object]:
    internal_dim = 3 * len(reconstructed.base) - 3
    vector = reconstructed.mode.vector
    atomic_fraction = float(vector[:internal_dim] @ vector[:internal_dim])
    strain_fraction = float(vector[internal_dim:] @ vector[internal_dim:])
    recomputed_coupling = bool(
        reconstructed.phsc_status is PHSCStatus.RESOLVED_NONNEGATIVE
        and reconstructed.chsc_status is CHSCStatus.RESOLVED_NONNEGATIVE
        and reconstructed.acsc.status is ACSCStatus.RESOLVED_NEGATIVE
    )
    result = direct.result
    return {
        "sid": str(formal["sid"]),
        "rk": str(formal["rk"]),
        "natoms": int(formal["natoms"]),
        "formal_lambda_r_ev_per_atom": float(formal["lambda_r_ev_per_atom"]),
        "formal_e_num_ev_per_atom": float(formal["e_num_ev_per_atom"]),
        "formal_u_num_ev_per_atom": float(formal["u_num_ev_per_atom"]),
        "recomputed_phsc_status": reconstructed.phsc_status.value,
        "recomputed_chsc_status": reconstructed.chsc_status.value,
        "recomputed_acsc_status": reconstructed.acsc.status.value,
        "recomputed_coupling_only_negative": recomputed_coupling,
        "recomputed_lambda_r_ev_per_atom": reconstructed.acsc.lambda_r,
        "recomputed_e_num_ev_per_atom": reconstructed.acsc.e_num,
        "recomputed_u_num_ev_per_atom": reconstructed.acsc.u_num,
        "mode_spectral_gap_ev_per_atom": reconstructed.mode.spectral_gap,
        "mode_atomic_fraction": atomic_fraction,
        "mode_strain_fraction": strain_fraction,
        "mode_json": json.dumps(vector.tolist(), allow_nan=False, separators=(",", ":")),
        "direct_status": result.status.value,
        "direct_negative": result.negative,
        "direct_confirmed": recomputed_coupling and result.status is DirectStatus.RESOLVED_NEGATIVE,
        "direct_h": result.h,
        "direct_q_h_ev_per_atom": result.q_h,
        "direct_q_h2_ev_per_atom": result.q_h2,
        "direct_q_r_ev_per_atom": result.q_r,
        "direct_e_num_ev_per_atom": result.e_num,
        "direct_u_num_ev_per_atom": result.u_num,
        "direct_l_num_ev_per_atom": result.l_num,
        "direct_tau_alg_ev_per_atom": result.tau_alg,
        "direct_minus_matrix_lambda_r_ev_per_atom": result.q_r - reconstructed.mode.lambda_r,
        "center_energy_repeat_delta_ev": direct.center_repeat_delta,
        "reconstruction_prediction_evaluations": reconstructed.prediction_evaluations,
        "direct_prediction_evaluations": result.energy_call_count,
        "error": "",
    }


def run_direct_confirmation(
    *,
    acsc_features_path: Path,
    acsc_manifest_path: Path,
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
    direct_candidates_per_call: int = FROZEN_DIRECT_CANDIDATES_PER_CALL,
) -> dict[str, object]:
    """Reconstruct and directly confirm the sealed ACSC candidate set."""

    target = Path(output_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(target)
    paths = {
        "acsc_features": Path(acsc_features_path),
        "acsc_manifest": Path(acsc_manifest_path),
        "phsc_features": Path(phsc_features_path),
        "phsc_manifest": Path(phsc_manifest_path),
        "geometry_only_frames": Path(frames_zip_path),
        "geometry_manifest": Path(geometry_manifest_path),
        "checkpoint": Path(checkpoint_path),
    }
    snapshots = _snapshot_inputs(paths)
    acsc_manifest = _validated_feature_manifest(
        snapshots["acsc_manifest"].data or b"",
        protocol=UPSTREAM_ACSC_PROTOCOL,
        output_name=paths["acsc_features"].name,
        output_sha256=snapshots["acsc_features"].sha256,
    )
    if acsc_manifest.get("endpoint_artifacts_opened") is not False:
        raise ValueError("formal ACSC manifest does not prove endpoint isolation")
    phsc_manifest = _validated_feature_manifest(
        snapshots["phsc_manifest"].data or b"",
        protocol=UPSTREAM_PHSC_PROTOCOL,
        output_name=paths["phsc_features"].name,
        output_sha256=snapshots["phsc_features"].sha256,
    )
    del phsc_manifest
    for role in (
        "phsc_features",
        "phsc_manifest",
        "geometry_only_frames",
        "geometry_manifest",
        "checkpoint",
    ):
        if _input_hash_from_manifest(acsc_manifest, role) != snapshots[role].sha256:
            raise ValueError(f"formal ACSC input identity differs for {role}")

    formal_all = pd.read_parquet(io.BytesIO(snapshots["acsc_features"].data or b""))
    formal = sealed_candidate_table(formal_all)
    phsc_table = pd.read_parquet(io.BytesIO(snapshots["phsc_features"].data or b""))
    if not {"sid", "strict_x0_ok"}.issubset(phsc_table.columns):
        raise ValueError("upstream PHSC table lacks geometry selection columns")
    strict_sids = sorted(
        phsc_table.loc[phsc_table["strict_x0_ok"].astype(bool), "sid"].astype(str)
    )
    geometry_sids, structures = load_geometry_only_archive(
        archive_path=paths["geometry_only_frames"],
        manifest_path=paths["geometry_manifest"],
        expected_sids=strict_sids,
    )
    geometry = dict(zip(geometry_sids, structures, strict=True))
    candidate_sids = formal["sid"].astype(str).tolist()
    if any(sid not in geometry for sid in candidate_sids):
        raise ValueError("sealed ACSC candidate is missing from geometry archive")
    candidate_structures = [geometry[sid] for sid in candidate_sids]
    for sid, natoms, atoms in zip(
        candidate_sids, formal["natoms"].astype(int), candidate_structures, strict=True
    ):
        if len(atoms) != natoms:
            raise ValueError(f"candidate natoms differs from sealed geometry for {sid}")

    runtime = _runtime_identity(device)
    if predictor is None:
        if runtime.get("mattersim_version") != "1.2.3" or runtime.get("cuda_available") is not True:
            raise RuntimeError("production ACSC-DIRECT requires MatterSim 1.2.3 with CUDA")
        active_predictor, loaded_checkpoint_sha256 = _production_predictor(
            paths["checkpoint"], device=device, batch_size=model_batch_size
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
    reconstructed = reconstruct_candidates(
        candidate_sids,
        candidate_structures,
        counting_predictor,
        structures_per_call=structures_per_call,
    )
    reconstruction_call_count = len(predictor_batch_sizes)
    direct_results = evaluate_direct_candidates(
        reconstructed,
        counting_predictor,
        candidates_per_call=direct_candidates_per_call,
    )
    elapsed = time.perf_counter() - started
    by_reconstructed = {item.sid: item for item in reconstructed}
    by_direct = {item.sid: item for item in direct_results}
    output_table = pd.DataFrame(
        [
            _row(
                record,
                by_reconstructed[str(record["sid"])],
                by_direct[str(record["sid"])],
            )
            for record in formal.to_dict("records")
        ],
        columns=OUTPUT_COLUMNS,
    ).sort_values("sid", kind="stable", ignore_index=True)
    reconstruction_evaluations = int(
        output_table["reconstruction_prediction_evaluations"].sum()
    )
    direct_evaluations = int(output_table["direct_prediction_evaluations"].sum())
    total_evaluations = reconstruction_evaluations + direct_evaluations
    if sum(predictor_batch_sizes) != total_evaluations:
        raise RuntimeError("predictor batch sizes differ from direct-confirmation evaluations")

    if predictor is None:
        telemetry = _validated_builtin_telemetry(
            active_predictor, device=device, expected_evaluations=total_evaluations
        )
        expected_forwards = sum(
            math.ceil(size / model_batch_size) for size in predictor_batch_sizes
        )
        if int(telemetry["forward_calls"]) != expected_forwards:
            raise RuntimeError("MatterSim forwards differ from direct-confirmation chunking")
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
    direct_statuses = output_table["direct_status"].astype(str)
    counts = {
        "sealed_candidates": len(output_table),
        "recomputed_coupling_only_negative": int(output_table["recomputed_coupling_only_negative"].sum()),
        "direct_resolved_negative": int(direct_statuses.eq(DirectStatus.RESOLVED_NEGATIVE.value).sum()),
        "direct_resolved_nonnegative": int(direct_statuses.eq(DirectStatus.RESOLVED_NONNEGATIVE.value).sum()),
        "direct_near_zero_or_inconsistent": int(direct_statuses.eq(DirectStatus.NEAR_ZERO_OR_INCONSISTENT.value).sum()),
        "direct_confirmed": int(output_table["direct_confirmed"].sum()),
        "reconstruction_prediction_evaluations": reconstruction_evaluations,
        "direct_prediction_evaluations": direct_evaluations,
        "total_prediction_evaluations": total_evaluations,
        "reconstruction_batch_predictor_calls": reconstruction_call_count,
        "direct_batch_predictor_calls": len(predictor_batch_sizes) - reconstruction_call_count,
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "version": DIRECT_VERSION,
        "mode": "sealed_candidate_label_free_direct_energy_confirmation",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "selection": (
            "exact 123 coupling_only_negative rows from sealed NEXT13 ACSC artifact; "
            "no label-dependent selection"
        ),
        "criterion": {
            "path": "r(t)=f*A(t)+d_star*Q*(t*z); A(t)=A0*exp(t*sum eta_a*B_a).T",
            "step": DIRECT_STEP,
            "probe_order": ["center", "+h", "-h", "+h/2", "-h/2"],
            "decision": "q_h<0 and q_h2<0 and q_R+e_num<0 under strict tau_alg",
            "thresholds_refit": False,
            "independent_model_confirmation": False,
        },
        "counts": counts,
        "adapter": {
            "mode": adapter_mode,
            "device": device,
            "model_batch_size": model_batch_size,
            "structures_per_call": structures_per_call,
            "direct_candidates_per_call": direct_candidates_per_call,
        },
        "predictor_loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "runtime": runtime,
        "execution": {
            "predictor_batch_sizes": predictor_batch_sizes,
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
            "Direct confirmation uses the same MatterSim checkpoint as ACSC reconstruction.",
            "The selected eigenmode is data-dependent; this is a numerical consistency test, not a prospective metric.",
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
    parser.add_argument("--acsc-features", required=True, type=Path)
    parser.add_argument("--acsc-manifest", required=True, type=Path)
    parser.add_argument("--phsc-features", required=True, type=Path)
    parser.add_argument("--phsc-manifest", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-batch-size", type=int, default=FROZEN_MODEL_BATCH_SIZE)
    parser.add_argument("--structures-per-call", type=int, default=FROZEN_STRUCTURES_PER_CALL)
    parser.add_argument(
        "--direct-candidates-per-call",
        type=int,
        default=FROZEN_DIRECT_CANDIDATES_PER_CALL,
    )
    arguments = parser.parse_args(argv)
    run_direct_confirmation(
        acsc_features_path=arguments.acsc_features,
        acsc_manifest_path=arguments.acsc_manifest,
        phsc_features_path=arguments.phsc_features,
        phsc_manifest_path=arguments.phsc_manifest,
        frames_zip_path=arguments.frames_zip,
        geometry_manifest_path=arguments.geometry_manifest,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
        model_batch_size=arguments.model_batch_size,
        structures_per_call=arguments.structures_per_call,
        direct_candidates_per_call=arguments.direct_candidates_per_call,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
