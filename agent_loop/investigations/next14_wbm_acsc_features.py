"""Run frozen PHSC/CHSC/ACSC and three-scale confirmation on WBM x0."""

from __future__ import annotations

import argparse
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
    _validated_builtin_telemetry,
    _validated_prediction,
)
from src.next12_prospective_gates import _compose_decision
from src.next13_acsc import ACSCStatus
from src.next13_acsc_mattersim_features import (
    FROZEN_STRUCTURES_PER_CALL,
    evaluate_acsc_batch,
)
from src.next13b_acsc_direct import DirectStatus, direct_curvature_from_energies
from src.next13b_acsc_direct_mattersim import (
    FROZEN_DIRECT_CANDIDATES_PER_CALL,
    evaluate_direct_candidates,
    reconstruct_candidates,
)
from src.next13c_acsc_direct_ladder import LARGE_STEP, ladder_probe_group
from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next14_wbm_pauling import _load_holdout


PROTOCOL = "2026-08-02-next14-wbm-acsc-three-scale-features-v1"
OUTPUT_NAME = "acsc_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_MODEL_BATCH_SIZE = 32
FROZEN_LADDER_CANDIDATES_PER_CALL = 32
DECISIONS = ("KEEP", "REJECT", "ABSTAIN")
FROZEN_FORMAL_INPUT_SHA256: Mapping[str, str] = {
    "metadata": "ace914af28d6d1e82bbdd2a4ca0d7be39dc024fa9a98192c8ce770dfc5c75861",
    "geometry_only_frames": "e79dd74d93261029398565ffed68c57b4ff1b99821129236f7e771d6bff838e3",
    "holdout_manifest": "78bfc3aa4876887e9c683ec37571b55d42189e44ffc104e8921f27c5fa3b74db",
    "checkpoint": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5",
}


def mechanical_decisions(
    phsc_status: str,
    chsc_status: str,
    coupling_only_negative: bool,
    nested_three_scale_confirmed: bool,
) -> tuple[str, str, str]:
    """Return pure, formal-coupled, and conservative nested decisions."""

    if type(coupling_only_negative) is not bool or type(nested_three_scale_confirmed) is not bool:
        raise ValueError("coupling and nested flags must be exact booleans")
    if nested_three_scale_confirmed and not coupling_only_negative:
        raise ValueError("nested confirmation requires a coupling-only candidate")
    pure = _compose_decision("KEEP", phsc_status, chsc_status)
    formal = "REJECT" if coupling_only_negative else pure
    nested = "REJECT" if nested_three_scale_confirmed else pure
    return pure, formal, nested


def _empty_direct_fields() -> dict[str, object]:
    return {
        "recomputed_phsc_status": "not_evaluated_not_coupling_only",
        "recomputed_chsc_status": "not_evaluated_not_coupling_only",
        "recomputed_acsc_status": "not_evaluated_not_coupling_only",
        "recomputed_coupling_only_negative": False,
        "mode_spectral_gap_ev_per_atom": np.nan,
        "mode_atomic_fraction": np.nan,
        "mode_strain_fraction": np.nan,
        "mode_json": "",
        "small_direct_status": "not_evaluated_not_coupling_only",
        "small_direct_confirmed": False,
        "small_q_h_ev_per_atom": np.nan,
        "small_q_h2_ev_per_atom": np.nan,
        "small_q_r_ev_per_atom": np.nan,
        "large_direct_status": "not_evaluated_not_coupling_only",
        "large_direct_negative": False,
        "large_q_h_ev_per_atom": np.nan,
        "large_q_h2_ev_per_atom": np.nan,
        "large_q_r_ev_per_atom": np.nan,
        "nested_three_scale_confirmed": False,
        "reconstruction_prediction_evaluations": 0,
        "small_direct_prediction_evaluations": 0,
        "large_direct_prediction_evaluations": 0,
    }


def run_wbm_acsc_features(
    *,
    metadata_path: Path,
    frames_zip_path: Path,
    holdout_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    predictor: BatchForcePredictor | None = None,
    device: str = "cuda:0",
    model_batch_size: int = FROZEN_MODEL_BATCH_SIZE,
    structures_per_call: int = FROZEN_STRUCTURES_PER_CALL,
    direct_candidates_per_call: int = FROZEN_DIRECT_CANDIDATES_PER_CALL,
    ladder_candidates_per_call: int = FROZEN_LADDER_CANDIDATES_PER_CALL,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Execute every frozen label-free probe before WBM DFT labels are opened."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    for value, name in (
        (model_batch_size, "model_batch_size"),
        (structures_per_call, "structures_per_call"),
        (direct_candidates_per_call, "direct_candidates_per_call"),
        (ladder_candidates_per_call, "ladder_candidates_per_call"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive exact integer")
    paths = {
        "metadata": Path(metadata_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "holdout_manifest": Path(holdout_manifest_path).resolve(),
        "checkpoint": Path(checkpoint_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs and input_hashes != dict(FROZEN_FORMAL_INPUT_SHA256):
        raise ValueError("formal WBM ACSC inputs differ from frozen identities")
    metadata, structures = _load_holdout(
        metadata_path=paths["metadata"],
        frames_zip_path=paths["geometry_only_frames"],
        manifest_path=paths["holdout_manifest"],
    )
    sids = metadata["material_id"].astype(str).tolist()
    runtime = _runtime_identity(device)
    if predictor is None:
        if runtime.get("mattersim_version") != "1.2.3" or runtime.get("cuda_available") is not True:
            raise RuntimeError("formal WBM ACSC execution requires MatterSim 1.2.3 with CUDA")
        active_predictor, loaded_checkpoint_sha256 = _production_predictor(
            paths["checkpoint"], device=device, batch_size=model_batch_size
        )
        adapter_mode = "builtin_indexed_mattersim"
    else:
        active_predictor = predictor
        loaded_checkpoint_sha256 = None
        adapter_mode = "injected_test_double"

    started = time.perf_counter()
    combined = evaluate_acsc_batch(
        sids, structures, active_predictor, structures_per_call=structures_per_call
    )
    combined_by_sid = {item.sid: item for item in combined}
    geometry = dict(zip(sids, structures, strict=True))
    candidate_sids = sorted(
        item.sid for item in combined if item.acsc.coupling_only_negative is True
    )
    candidates = reconstruct_candidates(
        candidate_sids,
        [geometry[sid] for sid in candidate_sids],
        active_predictor,
        structures_per_call=structures_per_call,
    )
    small_results = evaluate_direct_candidates(
        candidates,
        active_predictor,
        candidates_per_call=direct_candidates_per_call,
    )
    small_by_sid = {item.sid: item for item in small_results}

    large_by_sid: dict[str, object] = {}
    for start in range(0, len(candidates), ladder_candidates_per_call):
        chunk = candidates[start : start + ladder_candidates_per_call]
        groups = [ladder_probe_group(item.base, item.mode.vector) for item in chunk]
        flat = [probe for group in groups for probe in group]
        prediction = active_predictor(flat)
        energies, _forces, _stresses = _validated_prediction(prediction, flat)
        if len(energies) != 5 * len(chunk):
            raise RuntimeError("WBM ACSC ladder split a five-probe group")
        for index, item in enumerate(chunk):
            values = energies[5 * index : 5 * index + 5]
            large_by_sid[item.sid] = direct_curvature_from_energies(
                *values, n_atoms=len(item.base), h=LARGE_STEP
            )
    candidate_by_sid = {item.sid: item for item in candidates}

    rows: list[dict[str, object]] = []
    for upstream in metadata.to_dict("records"):
        sid = str(upstream["material_id"])
        item = combined_by_sid[sid]
        row: dict[str, object] = {
            "material_id": sid,
            "rk": str(upstream["rk"]),
            "formula": str(upstream["formula"]),
            "natoms": int(upstream["natoms"]),
            "phsc_status": item.phsc.status.value,
            "phsc_negative": item.phsc.negative,
            "phsc_lambda_r_ev_per_atom": item.phsc.lambda_r,
            "phsc_u_num_ev_per_atom": item.phsc.u_num,
            "chsc_status": item.chsc.status.value,
            "chsc_negative": item.chsc.negative,
            "chsc_lambda_r_ev_per_atom": item.chsc.lambda_r,
            "chsc_u_num_ev_per_atom": item.chsc.u_num,
            "acsc_status": item.acsc.status.value,
            "acsc_negative": item.acsc.negative,
            "coupling_only_negative": bool(item.acsc.coupling_only_negative),
            "acsc_lambda_r_ev_per_atom": item.acsc.lambda_r,
            "acsc_u_num_ev_per_atom": item.acsc.u_num,
            "combined_prediction_evaluations": item.acsc.prediction_evaluation_count,
            "error": item.acsc.error or item.phsc.error or item.chsc.error or "",
            **_empty_direct_fields(),
        }
        if sid in candidate_by_sid:
            reconstructed = candidate_by_sid[sid]
            small = small_by_sid[sid].result
            large = large_by_sid[sid]
            internal_dim = 3 * len(reconstructed.base) - 3
            vector = reconstructed.mode.vector
            recomputed_coupling = bool(
                reconstructed.phsc_status.value == "resolved_nonnegative"
                and reconstructed.chsc_status.value == "resolved_nonnegative"
                and reconstructed.acsc.status is ACSCStatus.RESOLVED_NEGATIVE
            )
            small_confirmed = bool(
                recomputed_coupling and small.status is DirectStatus.RESOLVED_NEGATIVE
            )
            nested_confirmed = bool(
                small_confirmed and large.status is DirectStatus.RESOLVED_NEGATIVE
            )
            row.update(
                {
                    "recomputed_phsc_status": reconstructed.phsc_status.value,
                    "recomputed_chsc_status": reconstructed.chsc_status.value,
                    "recomputed_acsc_status": reconstructed.acsc.status.value,
                    "recomputed_coupling_only_negative": recomputed_coupling,
                    "mode_spectral_gap_ev_per_atom": reconstructed.mode.spectral_gap,
                    "mode_atomic_fraction": float(vector[:internal_dim] @ vector[:internal_dim]),
                    "mode_strain_fraction": float(vector[internal_dim:] @ vector[internal_dim:]),
                    "mode_json": json.dumps(vector.tolist(), allow_nan=False, separators=(",", ":")),
                    "small_direct_status": small.status.value,
                    "small_direct_confirmed": small_confirmed,
                    "small_q_h_ev_per_atom": small.q_h,
                    "small_q_h2_ev_per_atom": small.q_h2,
                    "small_q_r_ev_per_atom": small.q_r,
                    "large_direct_status": large.status.value,
                    "large_direct_negative": large.negative,
                    "large_q_h_ev_per_atom": large.q_h,
                    "large_q_h2_ev_per_atom": large.q_h2,
                    "large_q_r_ev_per_atom": large.q_r,
                    "nested_three_scale_confirmed": nested_confirmed,
                    "reconstruction_prediction_evaluations": reconstructed.prediction_evaluations,
                    "small_direct_prediction_evaluations": small.energy_call_count,
                    "large_direct_prediction_evaluations": large.energy_call_count,
                }
            )
        pure, formal, nested = mechanical_decisions(
            str(row["phsc_status"]),
            str(row["chsc_status"]),
            bool(row["coupling_only_negative"]),
            bool(row["nested_three_scale_confirmed"]),
        )
        row["phsc_chsc_decision"] = pure
        row["phsc_chsc_acsc_formal_decision"] = formal
        row["phsc_chsc_acsc_nested_decision"] = nested
        rows.append(row)
    elapsed = time.perf_counter() - started
    table = pd.DataFrame(rows).sort_values("material_id", kind="stable", ignore_index=True)
    if len(table) != len(metadata) or table["material_id"].duplicated().any():
        raise RuntimeError("WBM ACSC feature accounting differs")
    evaluations = int(
        table[
            [
                "combined_prediction_evaluations",
                "reconstruction_prediction_evaluations",
                "small_direct_prediction_evaluations",
                "large_direct_prediction_evaluations",
            ]
        ].sum().sum()
    )
    if predictor is None:
        telemetry = _validated_builtin_telemetry(
            active_predictor, device=device, expected_evaluations=evaluations
        )
        production_eligible = bool(require_formal_inputs)
    else:
        telemetry = None
        production_eligible = False

    repository_root = Path(__file__).resolve().parents[1]
    source_relatives = (
        "src/next14_wbm_acsc_features.py",
        "src/next13_acsc_mattersim_features.py",
        "src/next13_acsc.py",
        "src/next13b_acsc_direct_mattersim.py",
        "src/next13b_acsc_direct.py",
        "src/next13c_acsc_direct_ladder.py",
        "src/next12_chsc_mattersim_features.py",
        "src/next12_chsc.py",
        "src/next11_phsc.py",
        "src/next10_lrrc_mattersim_features.py",
    )
    source_paths = {relative: repository_root / relative for relative in source_relatives}
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    counts: dict[str, object] = {
        "rows": len(table),
        "coupling_only_candidates": int(table["coupling_only_negative"].sum()),
        "recomputed_coupling_only_candidates": int(table["recomputed_coupling_only_negative"].sum()),
        "small_direct_confirmed": int(table["small_direct_confirmed"].sum()),
        "nested_three_scale_confirmed": int(table["nested_three_scale_confirmed"].sum()),
        "prediction_evaluations": evaluations,
    }
    for column in (
        "phsc_chsc_decision",
        "phsc_chsc_acsc_formal_decision",
        "phsc_chsc_acsc_nested_decision",
    ):
        counts[column] = {decision: int(table[column].eq(decision).sum()) for decision in DECISIONS}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "external_wbm_label_free_frozen_three_scale_acsc",
        "evidence_role": "label-free method execution before NEXT14 WBM label opening",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "thresholds_refit": False,
        "criterion": {
            "amplitudes": [2**-7, 2**-8, 2**-9],
            "nested_confirmation": "coupling-only ACSC plus resolved-negative small and large direct pairs",
        },
        "counts": counts,
        "adapter": {
            "mode": adapter_mode,
            "device": device,
            "model_batch_size": model_batch_size,
            "structures_per_call": structures_per_call,
            "direct_candidates_per_call": direct_candidates_per_call,
            "ladder_candidates_per_call": ladder_candidates_per_call,
        },
        "predictor_loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "runtime": runtime,
        "telemetry": telemetry,
        "execution": {"wall_time_seconds": elapsed},
        "inputs_sha256": {role: {"path": str(path), "sha256": input_hashes[role]} for role, path in paths.items()},
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": production_eligible,
        "scientific_improvement_claim": False,
        "known_limitations": [
            "All mechanical probes use the same frozen MatterSim checkpoint.",
            "WBM is an external source but its test labels were opened by older unrelated workflows.",
            "No WBM label or relaxed structure was opened during this execution.",
        ],
    }

    def verify_unchanged() -> None:
        for role, path in paths.items():
            if _sha256_file(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before publication")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / OUTPUT_NAME
        table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        verify_unchanged()
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--holdout-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args(argv)
    run_wbm_acsc_features(
        metadata_path=arguments.metadata,
        frames_zip_path=arguments.frames_zip,
        holdout_manifest_path=arguments.holdout_manifest,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
