"""Apply frozen M5, PHSC-v0, and CHSC-v0 gates to a prospective x0 cohort."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence
import zipfile

from ase import Atoms
from ase.io import read
import numpy as np
import pandas as pd

from src.next10_lrrc_mattersim_features import (
    BatchForcePredictor,
    FROZEN_M5_SHA256,
    _production_predictor,
    _runtime_identity,
    _validated_builtin_telemetry,
    _validated_prediction,
)
from src.next11_phsc import PHSCStatus
from src.next11_phsc_mattersim_features import (
    FROZEN_GROUPS_PER_CALL as FROZEN_PHSC_GROUPS_PER_CALL,
    BatchPHSCResult,
    evaluate_phsc_batch,
)
from src.next12_chsc import CHSCStatus
from src.next12_chsc_mattersim_features import (
    FROZEN_STRUCTURES_PER_CALL as FROZEN_CHSC_STRUCTURES_PER_CALL,
    BatchCHSCResult,
    evaluate_chsc_batch,
)
from src.next12_prospective_cohort import (
    ARCHIVE_NAME as UPSTREAM_ARCHIVE_NAME,
    COHORT_COLUMNS,
    COHORT_NAME as UPSTREAM_COHORT_NAME,
    MANIFEST_NAME as UPSTREAM_MANIFEST_NAME,
    PROTOCOL as UPSTREAM_COHORT_PROTOCOL,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next12-prospective-m5-phsc-chsc-v1"
FROZEN_DEVELOPMENT_PROTOCOL = (
    "2026-08-01-mattersim-committee-development-freeze-v1"
)
FROZEN_PRIMARY_M5_THRESHOLD = 0.12119269371032715
FROZEN_MODEL_BATCH_SIZE = 32
FROZEN_DEVICE = "cuda:0"
OUTPUT_NAME = "prospective_gate_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_FORMAL_INPUT_SHA256 = {
    "cohort": "fc08be4f1b28dc82f4a26aeb49819b914ad8df7c7c7ee3887dea7a6c61095215",
    "geometry_only_frames": "3b392bdd38120dae579dc22b1b51e7c30bcbbed0e72c9b462c1bce16eda96959",
    "cohort_manifest": "8649853dcbb40a081183b671101fdf2933f30358ad1e5cb5b8694e8e451a846a",
    "frozen_protocol": "b8049ad2f627ad91973ae86178c704871086097462f287b21c5330e3d4916fd4",
    "checkpoint": FROZEN_M5_SHA256,
}
FROZEN_FORMAL_COUNTS = {"attempts": 256, "generated": 256, "total_atoms": 2987}
DECISIONS = ("KEEP", "REJECT", "ABSTAIN")
_NONABSTAIN_PHSC = frozenset(
    {
        PHSCStatus.RESOLVED_NEGATIVE.value,
        PHSCStatus.RESOLVED_NONNEGATIVE.value,
        PHSCStatus.NEAR_ZERO_OR_INCONSISTENT.value,
    }
)
_NONABSTAIN_CHSC = frozenset(
    {
        CHSCStatus.RESOLVED_NEGATIVE.value,
        CHSCStatus.RESOLVED_NONNEGATIVE.value,
        CHSCStatus.NEAR_ZERO_OR_INCONSISTENT.value,
    }
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(payload: bytes, *, role: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"{role} contains nonstandard JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _extract_frozen_primary_m5_rule(
    protocol: Mapping[str, object],
) -> dict[str, object]:
    if protocol.get("protocol") != FROZEN_DEVELOPMENT_PROTOCOL:
        raise ValueError("frozen development protocol identity differs")
    raw_rules = protocol.get("final_rules")
    if not isinstance(raw_rules, list):
        raise ValueError("frozen protocol final_rules is missing")
    matches = [
        value
        for value in raw_rules
        if isinstance(value, Mapping)
        and value.get("track") == "primary"
        and value.get("formula") == "M5"
    ]
    if len(matches) != 1:
        raise ValueError("frozen primary M5 rule is not unique")
    rule = matches[0]
    expected = {
        "operator": "score > threshold",
        "threshold": FROZEN_PRIMARY_M5_THRESHOLD,
        "unsupported_decision": "ABSTAIN",
        "within_group": "max",
    }
    observed = {key: rule.get(key) for key in expected}
    if observed != expected:
        raise ValueError("frozen primary M5 rule differs from the sealed rule")
    if rule.get("threshold_state") != "finite" or rule.get(
        "threshold_source_role"
    ) != "threshold_fit":
        raise ValueError("frozen primary M5 threshold provenance differs")
    return expected


def _m5_table(
    *,
    sids: Sequence[str],
    formulas: Sequence[str],
    natoms: Sequence[int],
    total_energies: Sequence[float],
    threshold: float,
) -> pd.DataFrame:
    lengths = {len(sids), len(formulas), len(natoms), len(total_energies)}
    if len(lengths) != 1:
        raise ValueError("M5 inputs must align one-to-one")
    if len(set(sids)) != len(sids):
        raise ValueError("M5 sid values must be unique")
    atoms = np.asarray(natoms, dtype=int)
    energies = np.asarray(total_energies, dtype=float)
    if (atoms <= 0).any() or not np.isfinite(energies).all():
        raise ValueError("M5 atom counts and total energies must be finite and valid")
    table = pd.DataFrame(
        {
            "sid": list(sids),
            "formula": list(formulas),
            "m5_energy_total_ev": energies,
            "m5_energy_ev_per_atom": energies / atoms,
        }
    )
    table["m5_group_size"] = table.groupby("formula", sort=False)["sid"].transform(
        "size"
    )
    table["m5_has_competitor"] = table["m5_group_size"] > 1
    minimum = table.groupby("formula", sort=False)[
        "m5_energy_ev_per_atom"
    ].transform("min")
    table["m5_gap_ev_per_atom"] = table["m5_energy_ev_per_atom"] - minimum
    gaps = table["m5_gap_ev_per_atom"].to_numpy(dtype=float)
    if (~np.isfinite(gaps) | (gaps < 0.0)).any():
        raise ValueError("M5 within-composition gaps must be finite and nonnegative")
    table["m5_decision"] = np.where(gaps > threshold, "REJECT", "KEEP")
    return table


def _diagnostic_is_abstain(status: str, allowed: frozenset[str]) -> bool:
    if status in allowed:
        return False
    if status.startswith("abstain_"):
        return True
    raise ValueError(f"unknown diagnostic status: {status}")


def _compose_phsc_decision(baseline: str, phsc_status: str) -> str:
    if baseline not in DECISIONS:
        raise ValueError(f"unknown M5 decision: {baseline}")
    if baseline == "ABSTAIN" or _diagnostic_is_abstain(
        phsc_status, _NONABSTAIN_PHSC
    ):
        return "ABSTAIN"
    if baseline == "REJECT" or phsc_status == PHSCStatus.RESOLVED_NEGATIVE.value:
        return "REJECT"
    return "KEEP"


def _compose_decision(baseline: str, phsc_status: str, chsc_status: str) -> str:
    after_phsc = _compose_phsc_decision(baseline, phsc_status)
    if after_phsc == "ABSTAIN" or _diagnostic_is_abstain(
        chsc_status, _NONABSTAIN_CHSC
    ):
        return "ABSTAIN"
    if after_phsc == "REJECT" or chsc_status == CHSCStatus.RESOLVED_NEGATIVE.value:
        return "REJECT"
    return "KEEP"


def _load_cohort(
    *, cohort_data: bytes, archive_data: bytes, manifest_data: bytes
) -> tuple[pd.DataFrame, list[str], list[Atoms], dict[str, object]]:
    manifest = _strict_json(manifest_data, role="prospective cohort manifest")
    if manifest.get("protocol") != UPSTREAM_COHORT_PROTOCOL:
        raise ValueError("prospective cohort protocol identity differs")
    for key in ("labels_opened", "energy_or_force_models_called"):
        if manifest.get(key) is not False:
            raise ValueError(f"prospective cohort does not prove {key}=false")
    if manifest.get("all_attempts_retained") is not True:
        raise ValueError("prospective cohort did not retain every attempt")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping):
        raise ValueError("prospective cohort output hashes are missing")
    expected_outputs = {
        UPSTREAM_COHORT_NAME: _sha256_bytes(cohort_data),
        UPSTREAM_ARCHIVE_NAME: _sha256_bytes(archive_data),
    }
    if {key: outputs.get(key) for key in expected_outputs} != expected_outputs:
        raise ValueError("prospective cohort output hashes differ from its manifest")
    table = pd.read_parquet(io.BytesIO(cohort_data))
    if list(table.columns) != list(COHORT_COLUMNS):
        raise ValueError("prospective cohort columns differ from geometry-only schema")
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError("prospective cohort sid values must be nonmissing and unique")
    table["sid"] = table["sid"].astype(str)
    if table["attempt_index"].tolist() != list(range(len(table))):
        raise ValueError("prospective cohort attempt order is incomplete")
    valid_statuses = {"generated", "failed"}
    if not set(table["generation_status"].astype(str)).issubset(valid_statuses):
        raise ValueError("prospective cohort generation status is invalid")
    generated = table.loc[table["generation_status"].eq("generated")].copy()
    expected_members = generated["archive_member"].astype(str).tolist()
    sids: list[str] = []
    structures: list[Atoms] = []
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if members != expected_members or len(members) != len(set(members)):
            raise ValueError("prospective geometry members do not exactly align")
        for record in generated.to_dict("records"):
            member = str(record["archive_member"])
            payload = archive.read(member)
            if _sha256_bytes(payload) != record["geometry_sha256"]:
                raise ValueError(f"prospective geometry hash differs for {record['sid']}")
            lowered = payload.lower()
            for forbidden in (b"energy=", b"forces", b"stress", b"endpoint"):
                if forbidden in lowered:
                    raise ValueError("prospective archive is not geometry-only")
            atoms = read(
                io.StringIO(payload.decode("utf-8")),
                format="extxyz",
                index=0,
                parallel=False,
                do_not_split_by_at_sign=True,
            )
            if set(atoms.arrays) != {"numbers", "positions"} or atoms.info:
                raise ValueError("prospective geometry contains non-geometric fields")
            if len(atoms) != int(record["natoms"]):
                raise ValueError("prospective geometry atom count differs")
            if atoms.get_chemical_formula(mode="hill") != str(record["formula"]):
                raise ValueError("prospective geometry formula differs")
            sids.append(str(record["sid"]))
            structures.append(atoms)
    return table, sids, structures, manifest


def _phsc_fields(item: BatchPHSCResult | None) -> dict[str, object]:
    if item is None:
        return {
            "phsc_status": "abstain_generation_failed",
            "phsc_negative": None,
            "phsc_lambda_r_ev_per_a2": np.nan,
            "phsc_e_num_ev_per_a2": np.nan,
            "phsc_u_num_ev_per_a2": np.nan,
            "phsc_force_call_count": 0,
            "phsc_error": "generation failed before geometry freeze",
        }
    result = item.result
    return {
        "phsc_status": result.status.value,
        "phsc_negative": result.negative,
        "phsc_lambda_r_ev_per_a2": result.lambda_r,
        "phsc_e_num_ev_per_a2": result.e_num,
        "phsc_u_num_ev_per_a2": result.u_num,
        "phsc_force_call_count": int(result.force_call_count),
        "phsc_error": result.error,
    }


def _matrix_json(matrix: np.ndarray, *, enabled: bool) -> str | None:
    if not enabled:
        return None
    return json.dumps(np.asarray(matrix, dtype=float).tolist(), allow_nan=False, separators=(",", ":"))


def _chsc_fields(item: BatchCHSCResult | None) -> dict[str, object]:
    if item is None:
        return {
            "chsc_status": "abstain_generation_failed",
            "chsc_negative": None,
            "chsc_lambda_r_ev_per_atom": np.nan,
            "chsc_e_num_ev_per_atom": np.nan,
            "chsc_u_num_ev_per_atom": np.nan,
            "chsc_energy_call_count": 0,
            "chsc_hessian_h_json": None,
            "chsc_hessian_h2_json": None,
            "chsc_error": "generation failed before geometry freeze",
        }
    result = item.result
    enabled = bool(result.energy_call_count)
    return {
        "chsc_status": result.status.value,
        "chsc_negative": result.negative,
        "chsc_lambda_r_ev_per_atom": result.lambda_r,
        "chsc_e_num_ev_per_atom": result.e_num,
        "chsc_u_num_ev_per_atom": result.u_num,
        "chsc_energy_call_count": int(result.energy_call_count),
        "chsc_hessian_h_json": _matrix_json(item.hessian_h, enabled=enabled),
        "chsc_hessian_h2_json": _matrix_json(item.hessian_h2, enabled=enabled),
        "chsc_error": result.error,
    }


def _decision_counts(table: pd.DataFrame, column: str) -> dict[str, int]:
    return {decision: int(table[column].eq(decision).sum()) for decision in DECISIONS}


def run_prospective_gates(
    *,
    cohort_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    frozen_protocol_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    predictor: BatchForcePredictor | None = None,
    device: str = FROZEN_DEVICE,
    model_batch_size: int = FROZEN_MODEL_BATCH_SIZE,
    phsc_groups_per_call: int = FROZEN_PHSC_GROUPS_PER_CALL,
    chsc_structures_per_call: int = FROZEN_CHSC_STRUCTURES_PER_CALL,
) -> dict[str, object]:
    """Evaluate the already frozen prospective cohort without endpoint access."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    if any(
        type(value) is not int or value <= 0
        for value in (model_batch_size, phsc_groups_per_call, chsc_structures_per_call)
    ):
        raise ValueError("all batch and grouping parameters must be positive exact integers")
    device = str(device).strip().lower()
    if not device:
        raise ValueError("device must be a nonempty string")
    paths = {
        "cohort": Path(cohort_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "frozen_protocol": Path(frozen_protocol_path).resolve(),
        "checkpoint": Path(checkpoint_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    cohort_data = paths["cohort"].read_bytes()
    archive_data = paths["geometry_only_frames"].read_bytes()
    cohort_manifest_data = paths["cohort_manifest"].read_bytes()
    frozen_protocol_data = paths["frozen_protocol"].read_bytes()
    table, generated_sids, structures, upstream_manifest = _load_cohort(
        cohort_data=cohort_data,
        archive_data=archive_data,
        manifest_data=cohort_manifest_data,
    )
    frozen_protocol = _strict_json(frozen_protocol_data, role="frozen protocol")
    rule = _extract_frozen_primary_m5_rule(frozen_protocol)
    generated_rows = table.loc[table["generation_status"].eq("generated")].copy()
    observed_counts = {
        "attempts": len(table),
        "generated": len(generated_rows),
        "total_atoms": int(generated_rows["natoms"].sum()),
    }
    production = predictor is None
    if production:
        if input_hashes != FROZEN_FORMAL_INPUT_SHA256:
            mismatched = sorted(
                role
                for role, expected in FROZEN_FORMAL_INPUT_SHA256.items()
                if input_hashes.get(role) != expected
            )
            raise ValueError(f"formal prospective gate inputs differ: {mismatched}")
        if upstream_manifest.get("production_protocol_eligible") is not True:
            raise ValueError("formal prospective cohort is not production eligible")
        if observed_counts != FROZEN_FORMAL_COUNTS:
            raise ValueError("formal prospective cohort counts differ")
        if (
            device != FROZEN_DEVICE
            or model_batch_size != FROZEN_MODEL_BATCH_SIZE
            or phsc_groups_per_call != FROZEN_PHSC_GROUPS_PER_CALL
            or chsc_structures_per_call != FROZEN_CHSC_STRUCTURES_PER_CALL
        ):
            raise ValueError("formal prospective gate execution parameters are frozen")
        runtime = _runtime_identity(device)
        if runtime.get("mattersim_version") != "1.2.3" or runtime.get(
            "cuda_available"
        ) is not True:
            raise RuntimeError("formal prospective gates require MatterSim 1.2.3 CUDA")
        active_predictor, loaded_checkpoint_sha256 = _production_predictor(
            paths["checkpoint"], device=device, batch_size=model_batch_size
        )
        adapter_mode = "builtin_indexed_mattersim"
    else:
        active_predictor = predictor
        loaded_checkpoint_sha256 = None
        runtime = _runtime_identity(device)
        adapter_mode = "injected_test_double"

    predictor_batch_sizes: list[int] = []

    def counting_predictor(batch: list[Atoms]):
        predictor_batch_sizes.append(len(batch))
        return active_predictor(batch)

    started = time.perf_counter()
    base_prediction = counting_predictor(structures)
    base_energies, _base_forces, _base_stresses = _validated_prediction(
        base_prediction, structures
    )
    m5 = _m5_table(
        sids=generated_sids,
        formulas=generated_rows["formula"].astype(str).tolist(),
        natoms=generated_rows["natoms"].astype(int).tolist(),
        total_energies=base_energies,
        threshold=float(rule["threshold"]),
    )
    phsc_results = evaluate_phsc_batch(
        generated_sids,
        structures,
        counting_predictor,
        groups_per_call=phsc_groups_per_call,
    )
    chsc_results = evaluate_chsc_batch(
        generated_sids,
        structures,
        counting_predictor,
        structures_per_call=chsc_structures_per_call,
    )
    elapsed = time.perf_counter() - started
    m5_by_sid = {str(row["sid"]): row for row in m5.to_dict("records")}
    phsc_by_sid = {item.sid: item for item in phsc_results}
    chsc_by_sid = {item.sid: item for item in chsc_results}
    if set(phsc_by_sid) != set(generated_sids) or set(chsc_by_sid) != set(
        generated_sids
    ):
        raise RuntimeError("prospective diagnostic sid alignment is incomplete")

    rows: list[dict[str, object]] = []
    for upstream in table.to_dict("records"):
        sid = str(upstream["sid"])
        base = m5_by_sid.get(sid)
        if base is None:
            base_fields = {
                "m5_energy_total_ev": np.nan,
                "m5_energy_ev_per_atom": np.nan,
                "m5_group_size": 0,
                "m5_has_competitor": False,
                "m5_gap_ev_per_atom": np.nan,
                "m5_decision": "ABSTAIN",
            }
        else:
            base_fields = {
                key: base[key]
                for key in (
                    "m5_energy_total_ev",
                    "m5_energy_ev_per_atom",
                    "m5_group_size",
                    "m5_has_competitor",
                    "m5_gap_ev_per_atom",
                    "m5_decision",
                )
            }
        phsc_fields = _phsc_fields(phsc_by_sid.get(sid))
        chsc_fields = _chsc_fields(chsc_by_sid.get(sid))
        m5_decision = str(base_fields["m5_decision"])
        phsc_status = str(phsc_fields["phsc_status"])
        chsc_status = str(chsc_fields["chsc_status"])
        m5_phsc_decision = _compose_phsc_decision(m5_decision, phsc_status)
        composed_decision = _compose_decision(
            m5_decision, phsc_status, chsc_status
        )
        rows.append(
            {
                "attempt_index": int(upstream["attempt_index"]),
                "sid": sid,
                "formula": str(upstream["formula"]),
                "natoms": int(upstream["natoms"]),
                "generation_status": str(upstream["generation_status"]),
                **base_fields,
                **phsc_fields,
                **chsc_fields,
                "m5_phsc_decision": m5_phsc_decision,
                "composed_decision": composed_decision,
            }
        )
    output_table = pd.DataFrame(rows).sort_values(
        "sid", kind="stable", ignore_index=True
    )
    if len(output_table) != len(table) or output_table["sid"].duplicated().any():
        raise RuntimeError("prospective output did not retain every unique attempt")

    expected_evaluations = int(
        len(structures)
        + output_table["phsc_force_call_count"].sum()
        + output_table["chsc_energy_call_count"].sum()
    )
    if sum(predictor_batch_sizes) != expected_evaluations:
        raise RuntimeError("predictor telemetry differs from exact diagnostic calls")
    if production:
        telemetry = _validated_builtin_telemetry(
            active_predictor,
            device=device,
            expected_evaluations=expected_evaluations,
        )
        expected_forwards = sum(
            math.ceil(size / model_batch_size) for size in predictor_batch_sizes
        )
        if int(telemetry["forward_calls"]) != expected_forwards:
            raise RuntimeError("MatterSim forward calls differ from frozen batching")
    else:
        telemetry = None

    repository_root = Path(__file__).resolve().parents[1]
    source_relatives = (
        "src/next12_prospective_gates.py",
        "src/next12_prospective_cohort.py",
        "src/next11_phsc_mattersim_features.py",
        "src/next11_phsc.py",
        "src/next12_chsc_mattersim_features.py",
        "src/next12_chsc.py",
        "src/next10_lrrc_mattersim_features.py",
    )
    source_paths = {relative: repository_root / relative for relative in source_relatives}
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    counts = {
        "rows": len(output_table),
        "generated_rows": len(generated_rows),
        "failed_generation_rows": len(table) - len(generated_rows),
        "formula_groups": int(generated_rows["formula"].nunique()),
        "multi_candidate_formula_groups": int(
            (generated_rows.groupby("formula").size() > 1).sum()
        ),
        "m5": _decision_counts(output_table, "m5_decision"),
        "m5_phsc": _decision_counts(output_table, "m5_phsc_decision"),
        "composed": _decision_counts(output_table, "composed_decision"),
        "phsc_resolved_negative": int(
            output_table["phsc_status"].eq(PHSCStatus.RESOLVED_NEGATIVE.value).sum()
        ),
        "chsc_resolved_negative": int(
            output_table["chsc_status"].eq(CHSCStatus.RESOLVED_NEGATIVE.value).sum()
        ),
        "composed_reject": int(output_table["composed_decision"].eq("REJECT").sum()),
        "predictor_evaluations": expected_evaluations,
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "prospective_x0_frozen_gates",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "thresholds_refit": False,
        "input_isolation": {
            "prospective_geometry_only": True,
            "cohort_frozen_before_gate_evaluation": True,
            "endpoint_labels_available_to_process": False,
        },
        "frozen_rule": rule,
        "composition_rule": {
            "order": ["M5", "PHSC-v0", "CHSC-v0"],
            "negative_curvature_action": "REJECT",
            "nonnegative_action": "preserve_prior_decision",
            "diagnostic_abstention_action": "ABSTAIN",
        },
        "adapter": {
            "mode": adapter_mode,
            "device": device,
            "model_batch_size": model_batch_size,
            "phsc_groups_per_call": phsc_groups_per_call,
            "chsc_structures_per_call": chsc_structures_per_call,
        },
        "predictor_loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "runtime": runtime,
        "counts": counts,
        "execution": {
            "predictor_calls": len(predictor_batch_sizes),
            "predictor_batch_sizes": predictor_batch_sizes,
            "max_predictor_batch_size": max(predictor_batch_sizes, default=0),
            "telemetry": telemetry,
            "wall_time_seconds": elapsed,
        },
        "inputs_sha256": {
            role: {"path": str(paths[role]), "sha256": digest}
            for role, digest in input_hashes.items()
        },
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(production),
        "scientific_improvement_claim": False,
        "known_limitations": [
            "M5, PHSC-v0, and CHSC-v0 use MatterSim rather than DFT endpoints.",
            "Negative curvature is rejection evidence; nonnegative curvature is not a stability certificate.",
            "The SSAGEN checkpoint was trained on only 500 structures.",
            "DFT safety and superiority over Pauling rules remain unmeasured.",
        ],
    }

    def verify_unchanged() -> None:
        for role, path in paths.items():
            if _sha256_file(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_hashes[relative]:
                raise RuntimeError(f"executed source {relative} changed before publication")

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
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--frozen-protocol", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=FROZEN_DEVICE)
    parser.add_argument("--model-batch-size", type=int, default=FROZEN_MODEL_BATCH_SIZE)
    parser.add_argument(
        "--phsc-groups-per-call", type=int, default=FROZEN_PHSC_GROUPS_PER_CALL
    )
    parser.add_argument(
        "--chsc-structures-per-call",
        type=int,
        default=FROZEN_CHSC_STRUCTURES_PER_CALL,
    )
    arguments = parser.parse_args(argv)
    run_prospective_gates(
        cohort_path=arguments.cohort,
        frames_zip_path=arguments.frames_zip,
        cohort_manifest_path=arguments.cohort_manifest,
        frozen_protocol_path=arguments.frozen_protocol,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
        model_batch_size=arguments.model_batch_size,
        phsc_groups_per_call=arguments.phsc_groups_per_call,
        chsc_structures_per_call=arguments.chsc_structures_per_call,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
