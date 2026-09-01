"""Strict label-free necessary-condition stop for PHSC-v0.

This module consumes only sealed x0 feature artifacts and the frozen next8
protocol.  It cannot accept an endpoint-label path and computes no quality or
energy metric.  A passing result means only that the PHSC OR policy has enough
net deterministic rejections to make the predeclared +3 percentage-point
savings target arithmetically possible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from src.next6_elementa_protocol import apply_group_threshold
from src.next11_geometry_only_frames import (
    EXECUTED_SOURCE_RELATIVE as GEOMETRY_EXECUTED_SOURCE_RELATIVE,
    MANIFEST_NAME as GEOMETRY_MANIFEST_NAME,
    OUTPUT_ARCHIVE_NAME as GEOMETRY_ARCHIVE_NAME,
    PROTOCOL as GEOMETRY_PROTOCOL,
    validate_geometry_only_archive,
)
from src.next11_phsc import PHSCStatus, classify_phsc_state
from src.next8_mattersim_committee_features import (
    PROTOCOL as COMMITTEE_FEATURE_PROTOCOL,
    _atomic_publish_directory_no_replace,
)
from src.next8_mattersim_committee_protocol import (
    DEVELOPMENT_FREEZE_PROTOCOL,
    THRESHOLD_SPLIT_SALT,
    TRACKS,
    _validated_final_thresholds,
    construct_committee_scores,
    derive_disagreement_cutoffs,
    serialize_formula_catalog,
)


PROTOCOL = "2026-08-02-next11-phsc-label-free-stop-v1"
RESULT_NAME = "LABEL_FREE_STOP.json"
MANIFEST_NAME = "MANIFEST.json"
PHSC_FEATURE_PROTOCOL = "2026-08-02-next11-phsc-mattersim-features-v1"
MIN_NET_REJECT_DELTA = 66
FROZEN_FEATURE_ROWS = 12_990
FROZEN_ROLE_ROWS = 4_341
FROZEN_GATE_ROWS = 2_171
FROZEN_STRICT_ROWS = 2_164
FROZEN_NONSTRICT_ROWS = 7
FROZEN_RAW_ARCHIVE_FILE_MEMBERS = 21_755
FROZEN_GEOMETRY_TOTAL_ATOMS = 15_782
DECISIONS = ("KEEP", "REJECT", "ABSTAIN")
FORMULAS = ("M5", "AGREE995")
TRACK_ORDER = ("primary", "comparator")

FROZEN_INPUT_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "committee_features": "65f0234010f17f43a96789bde7858bae038ffaa4aaa2130eaee163fd3245bc8c",
        "committee_manifest": "e59848270c0fd1693d6f7d579ee327aebf4f34399ee73d27eb2c97f947cab9dd",
        "threshold_roles": "e6de5f5b5fc9545944043bda46e313fa2060833f1baa31dd93dcca12e4769602",
        "frozen_protocol": "b8049ad2f627ad91973ae86178c704871086097462f287b21c5330e3d4916fd4",
    }
)

PHSC_FROZEN_SOURCE_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "src/next11_phsc_mattersim_features.py": "35f3e4435034d5915c048a53bf7ff47aa5c404fec23f69f2656eeab4a55c3d19",
        "src/next11_phsc.py": "0016d021e6109c224f33b938f20e28bc2fe5063170dc1af3362695dddc6c3fda",
        "src/next11_geometry_only_frames.py": "2d8a9129140ff30258fdf6d40ccc1f9ecd5a1d4db50eb23278080ba58083d2cf",
        "src/next10_lrrc_mattersim_features.py": "9de42b45e6b526dfe2807921dbd680229a887c1a0c1f0cee1b1ed9ff47da44f1",
        "src/next9_lrrc.py": "16f70dbdcfbe17e45157be79db33077d81ffb2ea841c7a3fe13a308c347a1c90",
        "src/next8_mattersim_committee_features.py": "32153365d4e22a253ddb1869d9cd9b0a2b658dff3475639a27e3fbe576317909",
        "src/next6_mattersim_baseline.py": "fd874b08f17e489d438e57db984c711265b8a14236eeed58098c4914e94bfecb",
        "src/next6_wbm_build.py": "3edb1e24bb515e9a4057658974836e71f19851840cfef8f6cd053d7016a16d9a",
        "src/next6_wbm_features.py": "c6a71370a5108a562452c7670d72d364e521fa4017842725b9a661dcde65f55f",
        "src/next6_wbm_protocol.py": "73a538df7bfa046d3aed791dd54b6a79923f9dc9c33f19196185e7bc4004e299",
    }
)

FROZEN_GEOMETRY_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "geometry_only_frames": "9b99226a7dc5497fca2aaadbf6ac554c657cb5475705072bcd56b92db9515de9",
        "geometry_manifest": "2e5559595fa1dbc3f16470b005e1dc4f9dbe4a65de81a39a52f53c0af9b14901",
    }
)

PHSC_CRITERION: Mapping[str, object] = MappingProxyType(
    {
        "name": "PHSC-v0",
        "scope": "fixed_cell_gamma_point_atomic_hessian",
        "step_fraction": 0.00390625,
        "probe_order": ["+h", "-h", "+h/2", "-h/2"],
        "force_evaluations_per_atom": 12,
        "primary_decision_proxy": "two_scale_projected_operator_difference",
        "numerical_consistency_proxies_are_confidence_bounds": False,
        "numerical_consistency_proxies_are_rigorous_error_bounds": False,
    }
)

PHSC_FEATURE_COLUMNS = (
    "sid",
    "rk",
    "stage",
    "threshold_role",
    "strict_x0_ok",
    "natoms",
    "internal_dim",
    "phsc_status",
    "phsc_negative",
    "d_star_angstrom",
    "h_angstrom",
    "lambda_h_ev_per_a2",
    "lambda_h2_ev_per_a2",
    "lambda_r_ev_per_a2",
    "e_num_ev_per_a2",
    "u_num_ev_per_a2",
    "l_num_ev_per_a2",
    "tau_alg_ev_per_a2",
    "antisymmetric_norm_h_ev_per_a2",
    "antisymmetric_norm_h2_ev_per_a2",
    "acoustic_residual_h_ev_per_a2",
    "acoustic_residual_h2_ev_per_a2",
    "force_call_count",
    "error",
)
NUMERIC_DIAGNOSTIC_COLUMNS = PHSC_FEATURE_COLUMNS[9:22]
PHSC_EXECUTED_SOURCE_RELATIVE = (
    "src/next11_phsc_mattersim_features.py",
    "src/next11_phsc.py",
    "src/next10_lrrc_mattersim_features.py",
    "src/next9_lrrc.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
    "src/next6_wbm_build.py",
    "src/next6_wbm_features.py",
    "src/next6_wbm_protocol.py",
    "src/next11_geometry_only_frames.py",
)
EXECUTED_SOURCE_RELATIVE = (
    "src/next11_phsc_label_free_stop.py",
    "src/next11_geometry_only_frames.py",
    "src/next11_phsc.py",
    "src/next8_mattersim_committee_protocol.py",
    "src/next6_elementa_protocol.py",
    "src/next6_elementa_diagnostics.py",
    "src/next6_wbm_build.py",
    "src/next6_wbm_protocol.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
    "src/next6_wbm_features.py",
)

_PHSC_MANIFEST_KEYS = frozenset(
    {
        "protocol",
        "mode",
        "labels_opened",
        "selection",
        "input_isolation",
        "adapter",
        "predictor_loaded_checkpoint_sha256",
        "production_protocol_eligible",
        "evidence_role",
        "runtime",
        "inputs_sha256",
        "executed_source_sha256",
        "integrity",
        "feature_columns",
        "criterion",
        "formal_expectations",
        "counts",
        "execution",
        "scientific_improvement_claim",
        "outputs_sha256",
    }
)
_GEOMETRY_MANIFEST_KEYS = frozenset(
    {
        "protocol",
        "mode",
        "endpoint_label_artifacts_opened",
        "raw_x0_archive_bytes_read",
        "raw_x0_nongeometry_values_converted_or_exported",
        "input_role",
        "selection",
        "geometry_schema",
        "dropped_field_names",
        "inputs_sha256",
        "executed_source_sha256",
        "integrity",
        "counts",
        "sid_order_sha256",
        "outputs_sha256",
        "scientific_improvement_claim",
    }
)
_GEOMETRY_COUNT_KEYS = frozenset(
    {
        "feature_rows",
        "role_assignment_rows",
        "development_gate_rows",
        "strict_rows",
        "output_frames",
        "total_atoms",
        "raw_archive_file_members",
    }
)
_PHSC_COUNT_KEYS = frozenset(
    {
        "feature_rows",
        "role_assignment_rows",
        "selected_rows",
        "strict_rows",
        "nonstrict_rows",
        "probe_eligible_rows",
        "resolved_negative_rows",
        "resolved_nonnegative_rows",
        "near_zero_or_inconsistent_rows",
        "abstained_rows",
        "coordinate_groups",
        "probe_evaluations",
        "batch_predictor_calls",
    }
)
_SUCCESS_STATUSES = frozenset(
    {"resolved_negative", "resolved_nonnegative", "near_zero_or_inconsistent"}
)
_ABSTAIN_STATUSES = frozenset(
    {
        "abstain_unsupported_geometry",
        "abstain_force_failure",
        "abstain_invalid_force",
        "abstain_numerical_failure",
    }
)
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_CUDA_RE = re.compile(r"^cuda:(?:0|[1-9][0-9]*)$")


@dataclass(frozen=True, slots=True)
class _Snapshot:
    role: str
    path: Path
    payload: bytes
    sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(role: str, path: Path, *, expected_sha256: str | None = None) -> _Snapshot:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} is not a file: {resolved}")
    payload = resolved.read_bytes()
    digest = _sha256_bytes(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"{role} SHA-256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    return _Snapshot(role=role, path=resolved, payload=payload, sha256=digest)


def _reject_nonstandard_json(token: str) -> None:
    raise ValueError(f"nonstandard JSON token: {token}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(snapshot: _Snapshot) -> dict[str, object]:
    try:
        text = snapshot.payload.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{snapshot.role} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{snapshot.role} root must be a JSON object")
    return parsed


def _strict_parquet(snapshot: _Snapshot) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(snapshot.payload))
    except Exception as exc:
        raise ValueError(f"could not parse {snapshot.role} parquet: {exc}") from exc


def _is_sha256(value: object) -> bool:
    return type(value) is str and _HEX_RE.fullmatch(value) is not None


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _exact_hash_record(
    value: object,
    *,
    name: str,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must be an exact path/SHA-256 record")
    raw_path = value["path"]
    digest = value["sha256"]
    if type(raw_path) is not str or not raw_path:
        raise ValueError(f"{name}.path must be a nonempty exact string")
    if not _is_sha256(digest):
        raise ValueError(f"{name}.sha256 must be 64 lowercase hex digits")
    resolved = Path(raw_path).resolve()
    if expected_path is not None and resolved != expected_path.resolve():
        raise ValueError(f"{name} path does not match the explicit input")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{name} SHA-256 does not match the explicit input")
    return resolved, str(digest)


def _validate_string_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, role: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{role} is missing columns: {missing}")
    for column in columns:
        if not frame[column].map(lambda value: type(value) is str and bool(value)).all():
            raise ValueError(f"{role} {column} values must be nonempty exact strings")


def _validate_roles(
    committee: pd.DataFrame, roles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_string_columns(committee, ("sid", "rk", "stage"), role="committee features")
    _validate_string_columns(
        roles,
        ("sid", "rk", "stage", "threshold_role", "split_salt"),
        role="threshold roles",
    )
    if committee["sid"].duplicated().any() or roles["sid"].duplicated().any():
        raise ValueError("committee features and threshold roles require unique sid values")
    if set(committee["stage"]) != {
        "search_calibration",
        "formula_selection",
        "threshold_calibration",
    }:
        raise ValueError("committee features must contain exactly the development stages")
    if set(roles["stage"]) != {"threshold_calibration"}:
        raise ValueError("threshold roles must contain only threshold_calibration")
    if set(roles["threshold_role"]) != {"threshold_fit", "development_gate"}:
        raise ValueError("threshold roles must contain exact fit and gate assignments")
    if set(roles["split_salt"]) != {THRESHOLD_SPLIT_SALT}:
        raise ValueError("threshold role split salt differs from the frozen protocol")
    if (roles.groupby("rk", sort=False)["threshold_role"].nunique() != 1).any():
        raise ValueError("a composition group spans threshold roles")
    threshold = committee.loc[committee["stage"].eq("threshold_calibration")].copy()
    feature_keys = set(zip(threshold.sid, threshold.rk, threshold.stage, strict=True))
    role_keys = set(zip(roles.sid, roles.rk, roles.stage, strict=True))
    if feature_keys != role_keys:
        raise ValueError("threshold roles do not exactly cover threshold feature rows")
    joined = threshold.merge(
        roles[["sid", "rk", "stage", "threshold_role"]],
        on=["sid", "rk", "stage"],
        how="inner",
        validate="one_to_one",
    )
    gate = joined.loc[joined["threshold_role"].eq("development_gate")].copy()
    if len(gate) != FROZEN_GATE_ROWS:
        raise ValueError(
            f"development gate must contain {FROZEN_GATE_ROWS} rows; got {len(gate)}"
        )
    return joined, gate


def _validate_committee_manifest(
    manifest: Mapping[str, object],
    *,
    committee_snapshot: _Snapshot,
    manifest_snapshot: _Snapshot,
) -> tuple[Path, str, Path, str]:
    if manifest.get("protocol") != COMMITTEE_FEATURE_PROTOCOL:
        raise ValueError("committee manifest protocol mismatch")
    if manifest.get("mode") != "development":
        raise ValueError("committee manifest mode must be development")
    if manifest.get("production_protocol_eligible") is not True:
        raise ValueError("committee manifest must be production protocol eligible")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(committee_snapshot.path.name) != (
        committee_snapshot.sha256
    ):
        raise ValueError("committee manifest output hash mismatch")
    inputs = manifest.get("inputs_sha256")
    if not isinstance(inputs, Mapping) or "frames" not in inputs:
        raise ValueError("committee manifest lacks frames provenance")
    frames_path, frames_sha = _exact_hash_record(
        inputs["frames"], name="committee manifest frames"
    )
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or "m5" not in checkpoints:
        raise ValueError("committee manifest lacks M5 checkpoint provenance")
    checkpoint_path, checkpoint_sha = _exact_hash_record(
        checkpoints["m5"], name="committee manifest M5 checkpoint"
    )
    loaded = manifest.get("predictor_loaded_checkpoint_sha256")
    if not isinstance(loaded, Mapping) or loaded.get("m5") != checkpoint_sha:
        raise ValueError("committee manifest loaded M5 hash mismatch")
    if manifest_snapshot.sha256 != FROZEN_INPUT_SHA256["committee_manifest"]:
        raise ValueError("committee manifest is not the frozen byte identity")
    return frames_path, frames_sha, checkpoint_path, checkpoint_sha


def _validate_frozen_protocol(
    protocol: Mapping[str, object],
    *,
    committee: pd.DataFrame,
    committee_snapshot: _Snapshot,
    committee_manifest_snapshot: _Snapshot,
    roles: pd.DataFrame,
    roles_snapshot: _Snapshot,
) -> tuple[Any, pd.DataFrame]:
    if protocol.get("protocol") != DEVELOPMENT_FREEZE_PROTOCOL or protocol.get("state") != "frozen":
        raise ValueError("frozen protocol identity/state mismatch")
    provenance = protocol.get("cutoff_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("frozen protocol lacks cutoff provenance")
    if provenance.get("feature_sha256") != committee_snapshot.sha256:
        raise ValueError("frozen protocol committee feature hash mismatch")
    if provenance.get("feature_manifest_sha256") != committee_manifest_snapshot.sha256:
        raise ValueError("frozen protocol committee manifest hash mismatch")
    repo_root = Path(__file__).resolve().parents[1]
    protocol_source_sha = _sha256_file(
        repo_root / "src/next8_mattersim_committee_protocol.py"
    )
    if provenance.get("protocol_code_sha256") != protocol_source_sha:
        raise ValueError("frozen protocol score implementation hash mismatch")
    development_hashes = protocol.get("development_artifacts_sha256")
    if (
        not isinstance(development_hashes, Mapping)
        or development_hashes.get(roles_snapshot.path.name) != roles_snapshot.sha256
    ):
        raise ValueError("frozen protocol threshold-role hash mismatch")

    search = committee.loc[committee["stage"].eq("search_calibration")].copy()
    cutoffs = derive_disagreement_cutoffs(search)
    serialized = serialize_formula_catalog(cutoffs)
    catalog = protocol.get("catalog")
    serialized_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if (
        not isinstance(catalog, Mapping)
        or catalog.get("serialized") != serialized
        or catalog.get("sha256") != serialized_sha
        or provenance.get("catalog_serialization_sha256") != serialized_sha
    ):
        raise ValueError("frozen protocol catalog does not match label-free reconstruction")
    selection = protocol.get("selection")
    final_rules = protocol.get("final_rules")
    if not isinstance(selection, Mapping) or not isinstance(final_rules, list):
        raise ValueError("frozen protocol lacks selection/final rules")
    rules = _validated_final_thresholds(
        pd.DataFrame(final_rules), selection=selection
    )
    split = protocol.get("split")
    if not isinstance(split, Mapping) or split.get("salt") != THRESHOLD_SPLIT_SALT:
        raise ValueError("frozen protocol split identity mismatch")
    gate_groups = roles.loc[roles["threshold_role"].eq("development_gate"), "rk"].nunique()
    fit_groups = roles.loc[roles["threshold_role"].eq("threshold_fit"), "rk"].nunique()
    if split.get("development_gate_groups") != int(gate_groups) or split.get(
        "threshold_fit_groups"
    ) != int(fit_groups):
        raise ValueError("frozen protocol split group counts mismatch")
    return cutoffs, rules


def _validate_phsc_table(phsc: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    if tuple(phsc.columns) != PHSC_FEATURE_COLUMNS:
        raise ValueError("PHSC feature columns/order differ from the frozen schema")
    _validate_string_columns(
        phsc,
        ("sid", "rk", "stage", "threshold_role", "phsc_status"),
        role="PHSC features",
    )
    if not phsc["error"].map(lambda value: type(value) is str).all():
        raise ValueError("PHSC features error values must be exact strings")
    if phsc["sid"].duplicated().any():
        raise ValueError("PHSC features contain duplicate sid values")
    if set(phsc["stage"]) != {"threshold_calibration"} or set(
        phsc["threshold_role"]
    ) != {"development_gate"}:
        raise ValueError("PHSC features must contain only the development gate")
    gate_keys = set(zip(gate.sid, gate.rk, gate.stage, strict=True))
    phsc_keys = set(zip(phsc.sid, phsc.rk, phsc.stage, strict=True))
    if gate_keys != phsc_keys:
        raise ValueError("PHSC and development-gate keys differ")
    merged = gate[["sid", "rk", "stage", "strict_x0_ok"]].merge(
        phsc,
        on=["sid", "rk", "stage"],
        how="inner",
        validate="one_to_one",
        suffixes=("_committee", "_phsc"),
    )
    if len(merged) != len(gate):
        raise ValueError("PHSC merge lost development-gate rows")
    for column in ("strict_x0_ok_committee", "strict_x0_ok_phsc"):
        if not merged[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"{column} must be boolean")
    if not np.array_equal(
        merged["strict_x0_ok_committee"].to_numpy(dtype=bool),
        merged["strict_x0_ok_phsc"].to_numpy(dtype=bool),
    ):
        raise ValueError("PHSC strict_x0 flags differ from committee features")

    allowed = _SUCCESS_STATUSES | _ABSTAIN_STATUSES
    if not set(merged["phsc_status"]).issubset(allowed):
        raise ValueError("PHSC table contains an unknown status")
    strict_count = int(merged["strict_x0_ok_phsc"].astype(bool).sum())
    if strict_count != FROZEN_STRICT_ROWS or len(merged) - strict_count != FROZEN_NONSTRICT_ROWS:
        raise ValueError("PHSC strict/nonstrict counts differ from the frozen cohort")
    for row in merged.itertuples(index=False):
        strict = bool(row.strict_x0_ok_phsc)
        status = str(row.phsc_status)
        natoms = _exact_int(row.natoms, name="PHSC natoms")
        internal_dim = _exact_int(row.internal_dim, name="PHSC internal_dim")
        calls = _exact_int(row.force_call_count, name="PHSC force_call_count")
        diagnostics = np.asarray(
            [getattr(row, column) for column in NUMERIC_DIAGNOSTIC_COLUMNS],
            dtype=float,
        )
        negative = row.phsc_negative
        error = str(row.error)
        if not strict:
            if (
                natoms != 0
                or internal_dim != 0
                or calls != 0
                or status != "abstain_unsupported_geometry"
                or not pd.isna(negative)
                or error != "nonstrict_x0"
                or not np.isnan(diagnostics).all()
            ):
                raise ValueError("nonstrict PHSC sentinel semantics mismatch")
            continue
        if natoms < 2 or internal_dim != 3 * natoms - 3:
            raise ValueError("strict PHSC natoms/internal_dim mismatch")
        if status in _SUCCESS_STATUSES:
            if (
                calls != 12 * natoms
                or pd.isna(negative)
                or bool(negative) != (status == "resolved_negative")
                or not np.isfinite(diagnostics).all()
                or error != ""
            ):
                raise ValueError("successful PHSC row semantics mismatch")
            d_star = float(row.d_star_angstrom)
            h = float(row.h_angstrom)
            lambda_h = float(row.lambda_h_ev_per_a2)
            lambda_h2 = float(row.lambda_h2_ev_per_a2)
            lambda_r = float(row.lambda_r_ev_per_a2)
            e_num = float(row.e_num_ev_per_a2)
            u_num = float(row.u_num_ev_per_a2)
            l_num = float(row.l_num_ev_per_a2)
            tau_alg = float(row.tau_alg_ev_per_a2)
            norm_diagnostics = (
                float(row.antisymmetric_norm_h_ev_per_a2),
                float(row.antisymmetric_norm_h2_ev_per_a2),
                float(row.acoustic_residual_h_ev_per_a2),
                float(row.acoustic_residual_h2_ev_per_a2),
            )
            if (
                d_star <= 0.0
                or h <= 0.0
                or h != 0.00390625 * d_star
                or e_num < 0.0
                or tau_alg <= 0.0
                or any(value < 0.0 for value in norm_diagnostics)
                or u_num != lambda_r + e_num
                or l_num != lambda_r - e_num
            ):
                raise ValueError("PHSC successful-row numerical invariants mismatch")
            frozen_status = classify_phsc_state(
                lambda_h,
                lambda_h2,
                u_num,
                l_num,
                tau_alg,
            )
            if status != frozen_status.value:
                raise ValueError("PHSC status contradicts the frozen classifier")
            if bool(negative) != (frozen_status is PHSCStatus.RESOLVED_NEGATIVE):
                raise ValueError("PHSC negative flag contradicts the frozen classifier")
        else:
            if (
                not pd.isna(negative)
                or calls not in (0, 12 * natoms)
                or not np.isnan(diagnostics).all()
                or not error
            ):
                raise ValueError("abstained PHSC row semantics mismatch")
    return merged.sort_values("sid", kind="stable", ignore_index=True)


def _validate_geometry_artifact(
    *,
    archive_snapshot: _Snapshot,
    manifest_snapshot: _Snapshot,
    source_frames_path: Path,
    source_frames_sha: str,
    committee_snapshot: _Snapshot,
    roles_snapshot: _Snapshot,
    expected_sids: tuple[str, ...],
) -> None:
    manifest = _strict_json(manifest_snapshot)
    if set(manifest) != _GEOMETRY_MANIFEST_KEYS:
        raise ValueError("geometry manifest top-level schema mismatch")
    if (
        manifest.get("protocol") != GEOMETRY_PROTOCOL
        or manifest.get("mode") != "development_gate"
        or manifest.get("endpoint_label_artifacts_opened") is not False
        or manifest.get("raw_x0_archive_bytes_read") is not True
        or manifest.get("raw_x0_nongeometry_values_converted_or_exported") is not False
        or manifest.get("input_role") != "unrelaxed_x0_geometry_only"
        or manifest.get("scientific_improvement_claim") is not False
    ):
        raise ValueError("geometry manifest isolation identity mismatch")
    if manifest.get("selection") != {
        "stage": "threshold_calibration",
        "threshold_role": "development_gate",
        "strict_x0_ok": True,
    }:
        raise ValueError("geometry manifest selection mismatch")
    if manifest.get("integrity") != {"prepublish_rehash": "passed"}:
        raise ValueError("geometry manifest prepublication integrity mismatch")

    inputs = manifest.get("inputs_sha256")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "raw_frames",
        "committee_features",
        "threshold_roles",
    }:
        raise ValueError("geometry manifest input schema mismatch")
    _exact_hash_record(
        inputs["raw_frames"],
        name="geometry manifest raw frames provenance",
        expected_path=source_frames_path,
        expected_sha256=source_frames_sha,
    )
    _exact_hash_record(
        inputs["committee_features"],
        name="geometry manifest committee features",
        expected_path=committee_snapshot.path,
        expected_sha256=committee_snapshot.sha256,
    )
    _exact_hash_record(
        inputs["threshold_roles"],
        name="geometry manifest threshold roles",
        expected_path=roles_snapshot.path,
        expected_sha256=roles_snapshot.sha256,
    )

    outputs = manifest.get("outputs_sha256")
    if outputs != {GEOMETRY_ARCHIVE_NAME: archive_snapshot.sha256}:
        raise ValueError("geometry manifest output archive hash mismatch")
    sources = manifest.get("executed_source_sha256")
    frozen_geometry_source = PHSC_FROZEN_SOURCE_SHA256.get(
        GEOMETRY_EXECUTED_SOURCE_RELATIVE
    )
    if (
        not _is_sha256(frozen_geometry_source)
        or sources
        != {GEOMETRY_EXECUTED_SOURCE_RELATIVE: frozen_geometry_source}
    ):
        raise ValueError("geometry manifest source is not the frozen PHSC source")

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != _GEOMETRY_COUNT_KEYS:
        raise ValueError("geometry manifest count schema mismatch")
    parsed_counts = {
        key: _exact_int(value, name=f"geometry count {key}")
        for key, value in counts.items()
    }
    expected_counts = {
        "feature_rows": FROZEN_FEATURE_ROWS,
        "role_assignment_rows": FROZEN_ROLE_ROWS,
        "development_gate_rows": FROZEN_GATE_ROWS,
        "strict_rows": FROZEN_STRICT_ROWS,
        "output_frames": FROZEN_STRICT_ROWS,
        "total_atoms": FROZEN_GEOMETRY_TOTAL_ATOMS,
        "raw_archive_file_members": FROZEN_RAW_ARCHIVE_FILE_MEMBERS,
    }
    if parsed_counts != expected_counts or len(expected_sids) != FROZEN_STRICT_ROWS:
        raise ValueError("geometry manifest counts differ from the frozen cohort")

    with tempfile.TemporaryDirectory(prefix="next11-phsc-geometry-validate-") as raw:
        validation_dir = Path(raw)
        archive_copy = validation_dir / GEOMETRY_ARCHIVE_NAME
        manifest_copy = validation_dir / GEOMETRY_MANIFEST_NAME
        archive_copy.write_bytes(archive_snapshot.payload)
        manifest_copy.write_bytes(manifest_snapshot.payload)
        observed_sids = validate_geometry_only_archive(
            archive_path=archive_copy,
            manifest_path=manifest_copy,
            expected_sids=expected_sids,
        )
    if observed_sids != tuple(sorted(expected_sids)):
        raise ValueError("geometry archive sid binding mismatch")


def _validate_phsc_manifest(
    manifest: Mapping[str, object],
    *,
    phsc_snapshot: _Snapshot,
    manifest_snapshot: _Snapshot,
    committee_snapshot: _Snapshot,
    committee_manifest_snapshot: _Snapshot,
    roles_snapshot: _Snapshot,
    frames_path: Path,
    frames_sha: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    committee_rows: int,
    role_rows: int,
    phsc: pd.DataFrame,
    expected_geometry_sids: tuple[str, ...],
) -> tuple[_Snapshot, _Snapshot, _Snapshot]:
    if set(manifest) != _PHSC_MANIFEST_KEYS:
        raise ValueError("PHSC manifest top-level schema mismatch")
    if manifest.get("protocol") != PHSC_FEATURE_PROTOCOL or manifest.get("mode") != (
        "development_gate"
    ):
        raise ValueError("PHSC manifest protocol/mode mismatch")
    if manifest.get("labels_opened") is not False:
        raise ValueError("PHSC manifest must state that endpoint labels were not opened")
    if manifest.get("production_protocol_eligible") is not True:
        raise ValueError("PHSC manifest is not a production artifact")
    if manifest.get("scientific_improvement_claim") is not False:
        raise ValueError("PHSC manifest makes an invalid scientific claim")
    if manifest.get("evidence_role") != "label_free_phsc_feature_generation":
        raise ValueError("PHSC manifest evidence role mismatch")
    if manifest.get("selection") != {
        "stage": "threshold_calibration",
        "threshold_role": "development_gate",
    }:
        raise ValueError("PHSC manifest selection mismatch")
    if manifest.get("input_isolation") != {
        "geometry_only": True,
        "geometry_protocol": GEOMETRY_PROTOCOL,
        "raw_x0_archive_opened": False,
        "endpoint_label_artifacts_opened": False,
    }:
        raise ValueError("PHSC input isolation contract mismatch")
    if manifest.get("integrity") != {"prepublish_rehash": "passed"}:
        raise ValueError("PHSC manifest prepublication integrity mismatch")
    if manifest.get("feature_columns") != list(PHSC_FEATURE_COLUMNS):
        raise ValueError("PHSC manifest feature schema mismatch")
    if manifest.get("criterion") != dict(PHSC_CRITERION):
        raise ValueError("PHSC manifest criterion mismatch")
    expected_formal = {
        "feature_rows": FROZEN_FEATURE_ROWS,
        "role_assignment_rows": FROZEN_ROLE_ROWS,
        "selected_rows": FROZEN_GATE_ROWS,
        "strict_rows": FROZEN_STRICT_ROWS,
        "nonstrict_rows": FROZEN_NONSTRICT_ROWS,
        "model_batch_size": 32,
        "groups_per_call": 256,
        "device_contract": "canonical_cuda:N",
        "checkpoint_sha256": checkpoint_sha,
        "geometry_protocol": GEOMETRY_PROTOCOL,
        "geometry_only_frames_sha256": FROZEN_GEOMETRY_SHA256.get(
            "geometry_only_frames"
        ),
        "geometry_manifest_sha256": FROZEN_GEOMETRY_SHA256.get(
            "geometry_manifest"
        ),
    }
    if manifest.get("formal_expectations") != expected_formal:
        raise ValueError("PHSC manifest formal_expectations mismatch")
    outputs = manifest.get("outputs_sha256")
    if outputs != {phsc_snapshot.path.name: phsc_snapshot.sha256}:
        raise ValueError("PHSC manifest output hash mismatch")

    inputs = manifest.get("inputs_sha256")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "committee_features",
        "threshold_roles",
        "geometry_only_frames",
        "geometry_manifest",
        "source_frames_provenance",
        "feature_manifest",
        "checkpoint",
    }:
        raise ValueError("PHSC manifest input schema mismatch")
    _exact_hash_record(
        inputs["committee_features"],
        name="PHSC committee features",
        expected_path=committee_snapshot.path,
        expected_sha256=committee_snapshot.sha256,
    )
    _exact_hash_record(
        inputs["threshold_roles"],
        name="PHSC threshold roles",
        expected_path=roles_snapshot.path,
        expected_sha256=roles_snapshot.sha256,
    )
    _exact_hash_record(
        inputs["feature_manifest"],
        name="PHSC committee manifest",
        expected_path=committee_manifest_snapshot.path,
        expected_sha256=committee_manifest_snapshot.sha256,
    )
    if set(FROZEN_GEOMETRY_SHA256) != {
        "geometry_only_frames",
        "geometry_manifest",
    } or not all(_is_sha256(value) for value in FROZEN_GEOMETRY_SHA256.values()):
        raise ValueError("frozen geometry artifact hash schema mismatch")
    geometry_archive_path, geometry_archive_sha = _exact_hash_record(
        inputs["geometry_only_frames"],
        name="PHSC geometry-only frames",
        expected_sha256=FROZEN_GEOMETRY_SHA256["geometry_only_frames"],
    )
    geometry_manifest_path, geometry_manifest_sha = _exact_hash_record(
        inputs["geometry_manifest"],
        name="PHSC geometry manifest",
        expected_sha256=FROZEN_GEOMETRY_SHA256["geometry_manifest"],
    )
    _exact_hash_record(
        inputs["source_frames_provenance"],
        name="PHSC source frames provenance",
        expected_path=frames_path,
        expected_sha256=frames_sha,
    )
    _exact_hash_record(
        inputs["checkpoint"],
        name="PHSC checkpoint",
        expected_path=checkpoint_path,
        expected_sha256=checkpoint_sha,
    )
    if manifest.get("predictor_loaded_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("PHSC loaded checkpoint hash mismatch")
    geometry_archive_snapshot = _snapshot(
        "geometry_only_frames",
        geometry_archive_path,
        expected_sha256=geometry_archive_sha,
    )
    geometry_manifest_snapshot = _snapshot(
        "geometry_manifest",
        geometry_manifest_path,
        expected_sha256=geometry_manifest_sha,
    )
    _validate_geometry_artifact(
        archive_snapshot=geometry_archive_snapshot,
        manifest_snapshot=geometry_manifest_snapshot,
        source_frames_path=frames_path,
        source_frames_sha=frames_sha,
        committee_snapshot=committee_snapshot,
        roles_snapshot=roles_snapshot,
        expected_sids=expected_geometry_sids,
    )

    adapter = manifest.get("adapter")
    adapter_keys = {
        "mode",
        "index_alignment",
        "index_alignment_verified",
        "device",
        "model_batch_size",
        "groups_per_call",
        "model_parameter_device",
        "result_tensor_devices",
        "evaluations",
    }
    if not isinstance(adapter, Mapping) or set(adapter) != adapter_keys:
        raise ValueError("PHSC production adapter schema mismatch")
    device = adapter.get("device")
    if type(device) is not str or _CUDA_RE.fullmatch(device) is None:
        raise ValueError("PHSC production device must be canonical cuda:N")
    if (
        adapter.get("mode") != "builtin_indexed_mattersim"
        or adapter.get("index_alignment") != "sid_indexed_exact_one_to_one"
        or adapter.get("index_alignment_verified") is not True
        or adapter.get("model_batch_size") != 32
        or adapter.get("groups_per_call") != 256
        or adapter.get("model_parameter_device") != device
        or adapter.get("result_tensor_devices") != [device]
    ):
        raise ValueError("PHSC manifest is not the frozen indexed production adapter")

    runtime = manifest.get("runtime")
    runtime_keys = {
        "python_version",
        "python_implementation",
        "platform",
        "numpy_version",
        "pandas_version",
        "ase_version",
        "mattersim_version",
        "device",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "gpu_name",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys:
        raise ValueError("PHSC runtime schema mismatch")
    if runtime.get("device") != device or runtime.get("cuda_available") is not True:
        raise ValueError("PHSC runtime does not prove CUDA execution")
    if runtime.get("mattersim_version") != "1.2.3":
        raise ValueError("PHSC MatterSim version mismatch")
    for key in runtime_keys - {"cuda_available"}:
        value = runtime.get(key)
        if type(value) is not str or not value or value == "unavailable":
            raise ValueError(f"PHSC runtime {key} is unavailable")

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != _PHSC_COUNT_KEYS:
        raise ValueError("PHSC count schema mismatch")
    parsed_counts = {
        key: _exact_int(value, name=f"PHSC count {key}") for key, value in counts.items()
    }
    statuses = phsc["phsc_status"].astype(str)
    strict = phsc["strict_x0_ok"].astype(bool)
    calls = phsc["force_call_count"].to_numpy(dtype=int)
    expected_counts = {
        "feature_rows": committee_rows,
        "role_assignment_rows": role_rows,
        "selected_rows": len(phsc),
        "strict_rows": int(strict.sum()),
        "nonstrict_rows": int((~strict).sum()),
        "probe_eligible_rows": int(np.count_nonzero(calls > 0)),
        "resolved_negative_rows": int(statuses.eq("resolved_negative").sum()),
        "resolved_nonnegative_rows": int(statuses.eq("resolved_nonnegative").sum()),
        "near_zero_or_inconsistent_rows": int(
            statuses.eq("near_zero_or_inconsistent").sum()
        ),
        "abstained_rows": int(statuses.str.startswith("abstain_").sum()),
        "coordinate_groups": int(calls.sum() // 4),
        "probe_evaluations": int(calls.sum()),
        "batch_predictor_calls": parsed_counts["batch_predictor_calls"],
    }
    if parsed_counts != expected_counts:
        raise ValueError("PHSC manifest counts do not match the feature table")
    expected_predictor_calls = math.ceil(expected_counts["coordinate_groups"] / 256)
    if parsed_counts["batch_predictor_calls"] != expected_predictor_calls:
        raise ValueError("PHSC predictor-call count differs from frozen chunking")
    if adapter.get("evaluations") != expected_counts["probe_evaluations"]:
        raise ValueError("PHSC adapter evaluations differ from feature calls")

    execution = manifest.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {
        "batch_predictor_calls",
        "predictor_batch_sizes",
        "max_predictor_batch_size",
        "forward_calls",
        "peak_cuda_memory_bytes",
        "wall_time_seconds",
    }:
        raise ValueError("PHSC execution schema mismatch")
    execution_calls = _exact_int(
        execution.get("batch_predictor_calls"),
        name="PHSC execution predictor-call count",
    )
    if execution_calls != expected_predictor_calls:
        raise ValueError("PHSC execution predictor-call count mismatch")
    raw_batch_sizes = execution.get("predictor_batch_sizes")
    if type(raw_batch_sizes) is not list:
        raise ValueError("PHSC predictor batch sizes must be an exact JSON list")
    batch_sizes = [
        _exact_int(value, name=f"PHSC predictor batch size {index}", minimum=1)
        for index, value in enumerate(raw_batch_sizes)
    ]
    if len(batch_sizes) != execution_calls:
        raise ValueError("PHSC predictor batch sequence length mismatch")
    if any(size % 4 != 0 or size > 4 * 256 for size in batch_sizes):
        raise ValueError("PHSC predictor batch sizes must contain 1-256 complete probes")
    if any(size != 4 * 256 for size in batch_sizes[:-1]):
        raise ValueError("PHSC predictor batch sequence has a short non-final chunk")
    if sum(batch_sizes) != expected_counts["probe_evaluations"]:
        raise ValueError("PHSC predictor batch sizes do not sum to probe evaluations")
    observed_max_batch = _exact_int(
        execution.get("max_predictor_batch_size"),
        name="PHSC predictor batch maximum",
        minimum=1,
    )
    if observed_max_batch != max(batch_sizes):
        raise ValueError("PHSC predictor batch maximum mismatch")
    expected_forward_calls = sum(math.ceil(size / 32) for size in batch_sizes)
    observed_forward_calls = _exact_int(
        execution.get("forward_calls"),
        name="PHSC execution forward-call count",
        minimum=1,
    )
    if observed_forward_calls != expected_forward_calls:
        raise ValueError("PHSC execution forward-call count mismatch")
    _exact_int(
        execution.get("peak_cuda_memory_bytes"),
        name="PHSC execution peak_cuda_memory_bytes",
        minimum=1,
    )
    wall = execution.get("wall_time_seconds")
    if isinstance(wall, (bool, np.bool_)) or not isinstance(wall, (int, float)):
        raise ValueError("PHSC wall time must be a non-boolean real number")
    if not np.isfinite(float(wall)) or float(wall) <= 0.0:
        raise ValueError("PHSC wall time must be positive and finite")

    sources = manifest.get("executed_source_sha256")
    if not isinstance(sources, Mapping) or set(sources) != set(
        PHSC_EXECUTED_SOURCE_RELATIVE
    ):
        raise ValueError("PHSC executed-source closure mismatch")
    frozen_sources = PHSC_FROZEN_SOURCE_SHA256
    if not isinstance(frozen_sources, Mapping) or set(frozen_sources) != set(
        PHSC_EXECUTED_SOURCE_RELATIVE
    ):
        raise ValueError("frozen PHSC source allowlist is not sealed")
    repo_root = Path(__file__).resolve().parents[1]
    for relative in PHSC_EXECUTED_SOURCE_RELATIVE:
        digest = sources.get(relative)
        frozen_digest = frozen_sources.get(relative)
        if not _is_sha256(frozen_digest) or digest != frozen_digest:
            raise ValueError(f"frozen PHSC source hash mismatch: {relative}")
        if _sha256_file(repo_root / relative) != frozen_digest:
            raise ValueError(f"frozen PHSC source differs from checkout: {relative}")

    checkpoint_snapshot = _snapshot(
        "checkpoint", checkpoint_path, expected_sha256=checkpoint_sha
    )
    if manifest_snapshot.path.name == MANIFEST_NAME and manifest_snapshot.path.parent == (
        phsc_snapshot.path.parent
    ):
        # The filename may be MANIFEST.json in a formal directory; role binding,
        # not basename, is authoritative.  This branch intentionally does nothing.
        pass
    return geometry_archive_snapshot, geometry_manifest_snapshot, checkpoint_snapshot


def _base_decisions(
    gate: pd.DataFrame, *, cutoffs: Any, rules: pd.DataFrame
) -> dict[tuple[str, str], pd.DataFrame]:
    scores = construct_committee_scores(
        gate, cutoffs=cutoffs, expected_stage="threshold_calibration"
    )
    scores = scores.loc[scores["formula"].isin(FORMULAS)].copy()
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for track_name in TRACK_ORDER:
        if track_name not in TRACKS:
            raise ValueError(f"unknown frozen track: {track_name}")
        for formula in FORMULAS:
            matching = rules.loc[
                rules["track"].eq(track_name) & rules["formula"].eq(formula)
            ]
            if len(matching) != 1:
                raise ValueError("final rule grid does not identify one formula/track row")
            threshold = float(matching.iloc[0]["threshold"])
            rows = scores.loc[scores["formula"].eq(formula)].copy()
            finite = np.isfinite(rows["score_ev_per_atom"].to_numpy(dtype=float))
            supported = rows["state"].eq("KEEP").to_numpy(dtype=bool) & finite
            rows["baseline_decision"] = apply_group_threshold(
                rows["score_ev_per_atom"].to_numpy(dtype=float),
                supported,
                threshold,
            )
            rows["track"] = track_name
            rows["threshold"] = threshold
            result[(track_name, formula)] = rows.sort_values(
                "sid", kind="stable", ignore_index=True
            )
    return result


def _compose_decision(baseline: str, status: str) -> str:
    if baseline not in DECISIONS:
        raise ValueError(f"unknown baseline decision: {baseline}")
    if baseline == "ABSTAIN":
        return "ABSTAIN"
    if status == "resolved_negative":
        return "REJECT"
    if status in {"resolved_nonnegative", "near_zero_or_inconsistent"}:
        return baseline
    if status in _ABSTAIN_STATUSES:
        return "ABSTAIN"
    raise ValueError(f"unknown PHSC status: {status}")


def _transition_summary(baseline: pd.DataFrame, phsc: pd.DataFrame) -> dict[str, object]:
    joined = baseline[["sid", "rk", "baseline_decision"]].merge(
        phsc[["sid", "rk", "phsc_status"]],
        on=["sid", "rk"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(baseline):
        raise ValueError("baseline and PHSC rows are not exactly aligned")
    joined["composed_decision"] = [
        _compose_decision(str(base), str(status))
        for base, status in zip(
            joined["baseline_decision"], joined["phsc_status"], strict=True
        )
    ]
    matrix = {
        source: {
            target: int(
                (
                    joined["baseline_decision"].eq(source)
                    & joined["composed_decision"].eq(target)
                ).sum()
            )
            for target in DECISIONS
        }
        for source in DECISIONS
    }
    baseline_counts = {
        decision: int(joined["baseline_decision"].eq(decision).sum())
        for decision in DECISIONS
    }
    composed_counts = {
        decision: int(joined["composed_decision"].eq(decision).sum())
        for decision in DECISIONS
    }
    nonreject_to_reject = sum(
        matrix[source]["REJECT"] for source in DECISIONS if source != "REJECT"
    )
    reject_to_nonreject = sum(
        matrix["REJECT"][target] for target in DECISIONS if target != "REJECT"
    )
    net = composed_counts["REJECT"] - baseline_counts["REJECT"]
    if net != nonreject_to_reject - reject_to_nonreject:
        raise RuntimeError("transition decomposition does not close net reject delta")
    return {
        "baseline_counts": baseline_counts,
        "composed_counts": composed_counts,
        "transition_matrix": matrix,
        "nonreject_to_reject": int(nonreject_to_reject),
        "reject_to_nonreject": int(reject_to_nonreject),
        "net_reject_delta": int(net),
    }


def _build_result(
    *, gate: pd.DataFrame, phsc: pd.DataFrame, cutoffs: Any, rules: pd.DataFrame
) -> dict[str, object]:
    baselines = _base_decisions(gate, cutoffs=cutoffs, rules=rules)
    policies: dict[str, dict[str, object]] = {}
    for track in TRACK_ORDER:
        policies[track] = {
            formula: _transition_summary(baselines[(track, formula)], phsc)
            for formula in FORMULAS
        }
    primary_delta = int(policies["primary"]["M5"]["net_reject_delta"])
    return {
        "protocol": PROTOCOL,
        "evidence_scope": {
            "label_free": True,
            "endpoint_metrics_computed": False,
            "scientific_improvement_claim": False,
        },
        "policies": policies,
        "necessary_condition": {
            "cohort_rows": len(gate),
            "required_net_reject_delta": MIN_NET_REJECT_DELTA,
            "observed_primary_m5_net_reject_delta": primary_delta,
            "passes": bool(primary_delta >= MIN_NET_REJECT_DELTA),
            "labels_opened": False,
        },
    }


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _source_hashes() -> tuple[dict[str, Path], dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[1]
    paths = {relative: repo_root / relative for relative in EXECUTED_SOURCE_RELATIVE}
    for relative, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"executed source is missing: {relative}")
    return paths, {relative: _sha256_file(path) for relative, path in paths.items()}


def _hash_record(snapshot: _Snapshot) -> dict[str, str]:
    return {"path": str(snapshot.path), "sha256": snapshot.sha256}


def run_label_free_stop(
    committee_features_path: Path,
    committee_manifest_path: Path,
    role_assignments_path: Path,
    phsc_features_path: Path,
    phsc_manifest_path: Path,
    frozen_protocol_path: Path,
    output_dir: Path,
) -> Path:
    """Publish the sealed PHSC label-free 66-net-reject necessary condition."""

    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"output directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    explicit = {
        "committee_features": _snapshot(
            "committee_features",
            Path(committee_features_path),
            expected_sha256=FROZEN_INPUT_SHA256["committee_features"],
        ),
        "committee_manifest": _snapshot(
            "committee_manifest",
            Path(committee_manifest_path),
            expected_sha256=FROZEN_INPUT_SHA256["committee_manifest"],
        ),
        "threshold_roles": _snapshot(
            "threshold_roles",
            Path(role_assignments_path),
            expected_sha256=FROZEN_INPUT_SHA256["threshold_roles"],
        ),
        "phsc_features": _snapshot("phsc_features", Path(phsc_features_path)),
        "phsc_manifest": _snapshot("phsc_manifest", Path(phsc_manifest_path)),
        "frozen_protocol": _snapshot(
            "frozen_protocol",
            Path(frozen_protocol_path),
            expected_sha256=FROZEN_INPUT_SHA256["frozen_protocol"],
        ),
    }
    committee_manifest = _strict_json(explicit["committee_manifest"])
    phsc_manifest = _strict_json(explicit["phsc_manifest"])
    frozen_protocol = _strict_json(explicit["frozen_protocol"])
    committee = _strict_parquet(explicit["committee_features"])
    roles = _strict_parquet(explicit["threshold_roles"])
    phsc = _strict_parquet(explicit["phsc_features"])

    _joined, gate = _validate_roles(committee, roles)
    frames_path, frames_sha, checkpoint_path, checkpoint_sha = (
        _validate_committee_manifest(
            committee_manifest,
            committee_snapshot=explicit["committee_features"],
            manifest_snapshot=explicit["committee_manifest"],
        )
    )
    cutoffs, rules = _validate_frozen_protocol(
        frozen_protocol,
        committee=committee,
        committee_snapshot=explicit["committee_features"],
        committee_manifest_snapshot=explicit["committee_manifest"],
        roles=roles,
        roles_snapshot=explicit["threshold_roles"],
    )
    aligned_phsc = _validate_phsc_table(phsc, gate)
    expected_geometry_sids = tuple(
        sorted(
            phsc.loc[phsc["strict_x0_ok"].astype(bool), "sid"].astype(str).tolist()
        )
    )
    (
        geometry_archive_snapshot,
        geometry_manifest_snapshot,
        checkpoint_snapshot,
    ) = _validate_phsc_manifest(
        phsc_manifest,
        phsc_snapshot=explicit["phsc_features"],
        manifest_snapshot=explicit["phsc_manifest"],
        committee_snapshot=explicit["committee_features"],
        committee_manifest_snapshot=explicit["committee_manifest"],
        roles_snapshot=explicit["threshold_roles"],
        frames_path=frames_path,
        frames_sha=frames_sha,
        checkpoint_path=checkpoint_path,
        checkpoint_sha=checkpoint_sha,
        committee_rows=len(committee),
        role_rows=len(roles),
        phsc=phsc,
        expected_geometry_sids=expected_geometry_sids,
    )
    all_inputs = {
        **explicit,
        "geometry_only_frames": geometry_archive_snapshot,
        "geometry_manifest": geometry_manifest_snapshot,
        "checkpoint": checkpoint_snapshot,
    }
    input_records = {
        role: _hash_record(snapshot) for role, snapshot in all_inputs.items()
    }
    input_records["source_frames_provenance"] = {
        "path": str(frames_path.resolve()),
        "sha256": frames_sha,
    }
    source_paths, source_sha = _source_hashes()
    result = _build_result(
        gate=gate,
        phsc=aligned_phsc,
        cutoffs=cutoffs,
        rules=rules,
    )
    result_bytes = _json_bytes(result)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "development_gate_label_free_stop",
        "labels_opened": False,
        "production_protocol_eligible": True,
        "scientific_improvement_claim": False,
        "evidence_scope": {
            "label_free": True,
            "endpoint_metrics_computed": False,
            "necessary_condition_only": True,
        },
        "inputs_sha256": input_records,
        "executed_source_sha256": source_sha,
        "counts": {
            "committee_feature_rows": len(committee),
            "threshold_role_rows": len(roles),
            "development_gate_rows": len(gate),
            "phsc_rows": len(phsc),
        },
        "necessary_condition": result["necessary_condition"],
        "integrity": {"prepublish_rehash": "passed"},
        "outputs_sha256": {RESULT_NAME: _sha256_bytes(result_bytes)},
    }

    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        (staging / RESULT_NAME).write_bytes(result_bytes)
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256_file(staging / RESULT_NAME) != manifest["outputs_sha256"][RESULT_NAME]:
            raise RuntimeError("staged label-free result hash mismatch")
        for snapshot in all_inputs.values():
            if _sha256_file(snapshot.path) != snapshot.sha256:
                raise RuntimeError(f"input {snapshot.role} changed before publication")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_sha[relative]:
                raise RuntimeError(f"executed source changed before publication: {relative}")
        _atomic_publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


__all__ = [
    "DECISIONS",
    "EXECUTED_SOURCE_RELATIVE",
    "FROZEN_FEATURE_ROWS",
    "FROZEN_GATE_ROWS",
    "FROZEN_INPUT_SHA256",
    "FROZEN_NONSTRICT_ROWS",
    "FROZEN_STRICT_ROWS",
    "FROZEN_ROLE_ROWS",
    "MANIFEST_NAME",
    "MIN_NET_REJECT_DELTA",
    "PHSC_FEATURE_COLUMNS",
    "PHSC_FROZEN_SOURCE_SHA256",
    "PHSC_CRITERION",
    "PROTOCOL",
    "RESULT_NAME",
    "run_label_free_stop",
]
