"""Label-free, fixed-protocol batched PHSC-v0 feature construction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from itertools import islice
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from ase import Atoms

from src.next10_lrrc_mattersim_features import (
    BatchForcePredictor,
    FROZEN_M5_SHA256,
    _InputSnapshot,
    _hash_record,
    _production_predictor,
    _runtime_identity,
    _selected_gate_rows,
    _sha256_file,
    _snapshot,
    _strict_json_document,
    _validate_upstream_manifest,
    _validated_builtin_telemetry,
    _validated_prediction,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)
from src.next11_geometry_only_frames import (
    PROTOCOL as GEOMETRY_ONLY_PROTOCOL,
    load_geometry_only_archive,
)
from src.next11_phsc import (
    PHSCNumericalError,
    PHSCResult,
    PHSCStatus,
    PHSCValidationError,
    PHSC_VERSION,
    STEP_FRACTION,
    analyze_hessian_pair,
    canonicalize_phsc_geometry,
    hessian_columns_from_force_samples,
    phsc_probe_group,
)


FROZEN_GROUPS_PER_CALL = 256
FROZEN_MODEL_BATCH_SIZE = 32
FROZEN_ENGINEERING_SMOKE_COUNT = 8
FROZEN_NEXT11_INPUT_SHA256 = {
    "committee_features": "65f0234010f17f43a96789bde7858bae038ffaa4aaa2130eaee163fd3245bc8c",
    "threshold_roles": "e6de5f5b5fc9545944043bda46e313fa2060833f1baa31dd93dcca12e4769602",
    "geometry_only_frames": "9b99226a7dc5497fca2aaadbf6ac554c657cb5475705072bcd56b92db9515de9",
    "geometry_manifest": "2e5559595fa1dbc3f16470b005e1dc4f9dbe4a65de81a39a52f53c0af9b14901",
    "feature_manifest": "e59848270c0fd1693d6f7d579ee327aebf4f34399ee73d27eb2c97f947cab9dd",
}
FORMAL_EXPECTED_COUNTS = {
    "feature_rows": 12_990,
    "role_assignment_rows": 4_341,
    "selected_rows": 2_171,
    "strict_rows": 2_164,
    "nonstrict_rows": 7,
}
CRITERION = {
    "name": PHSC_VERSION,
    "scope": "fixed_cell_gamma_point_atomic_hessian",
    "step_fraction": STEP_FRACTION,
    "probe_order": ["+h", "-h", "+h/2", "-h/2"],
    "force_evaluations_per_atom": 12,
    "primary_decision_proxy": "two_scale_projected_operator_difference",
    "numerical_consistency_proxies_are_confidence_bounds": False,
    "numerical_consistency_proxies_are_rigorous_error_bounds": False,
}
PROTOCOL = "2026-08-02-next11-phsc-mattersim-features-v1"
OUTPUT_NAME = "phsc_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
OUTPUT_COLUMNS = (
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
NUMERIC_DIAGNOSTIC_COLUMNS = OUTPUT_COLUMNS[9:22]
EXECUTED_SOURCE_RELATIVE = (
    "src/next11_phsc_mattersim_features.py",
    "src/next11_phsc.py",
    "src/next11_geometry_only_frames.py",
    "src/next10_lrrc_mattersim_features.py",
    "src/next9_lrrc.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
    "src/next6_wbm_build.py",
    "src/next6_wbm_features.py",
    "src/next6_wbm_protocol.py",
)
_SUCCESS_STATUSES = frozenset(
    {
        PHSCStatus.RESOLVED_NEGATIVE,
        PHSCStatus.RESOLVED_NONNEGATIVE,
        PHSCStatus.NEAR_ZERO_OR_INCONSISTENT,
    }
)


class BatchPHSCError(RuntimeError):
    """Raised when a batch PHSC run cannot preserve exact cohort alignment."""


@dataclass(frozen=True, slots=True)
class BatchPHSCResult:
    """One sid-aligned PHSC-v0 result."""

    sid: str
    result: PHSCResult


@dataclass(slots=True)
class _PreparedPHSC:
    sid: str
    base: Atoms
    d_star: float
    h: float
    h_h: np.ndarray
    h_h2: np.ndarray
    seen_columns: np.ndarray
    numerical_error: str | None = None


@dataclass(frozen=True, slots=True)
class _ProbeGroup:
    prepared: _PreparedPHSC
    coordinate: int
    probes: tuple[Atoms, ...]


def _probe_groups(prepared: Sequence[_PreparedPHSC]) -> Iterator[_ProbeGroup]:
    for item in prepared:
        for coordinate in range(3 * len(item.base)):
            try:
                probes = phsc_probe_group(item.base, coordinate, item.h)
            except (PHSCValidationError, PHSCNumericalError) as exc:
                raise BatchPHSCError(
                    f"could not construct probe group for {item.sid} coordinate "
                    f"{coordinate}: {exc}"
                ) from None
            if len(probes) != 4:
                raise BatchPHSCError("PHSC core returned an incomplete probe group")
            yield _ProbeGroup(item, coordinate, probes)


def _preflight_probe_representability(
    prepared: Sequence[_PreparedPHSC],
    completed: dict[str, PHSCResult],
) -> list[_PreparedPHSC]:
    """Exclude rows with unrepresentable probes before any predictor call."""

    active: list[_PreparedPHSC] = []
    for item in prepared:
        try:
            for coordinate in range(3 * len(item.base)):
                probes = phsc_probe_group(item.base, coordinate, item.h)
                if len(probes) != 4:
                    raise BatchPHSCError("PHSC core returned an incomplete probe group")
        except PHSCNumericalError as exc:
            completed[item.sid] = PHSCResult(
                status=PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                force_call_count=0,
                error=f"PHSC probe preflight failed: {exc}",
            )
            continue
        except PHSCValidationError as exc:
            raise BatchPHSCError(
                f"validated PHSC geometry failed probe preflight for {item.sid}: {exc}"
            ) from None
        active.append(item)
    return active


def _chunks(iterator: Iterator[_ProbeGroup], size: int) -> Iterator[tuple[_ProbeGroup, ...]]:
    while chunk := tuple(islice(iterator, size)):
        yield chunk


def _predict_group_chunk(
    predictor: BatchForcePredictor,
    chunk: Sequence[_ProbeGroup],
) -> tuple[np.ndarray, ...]:
    structures = [probe.copy() for group in chunk for probe in group.probes]
    if len(structures) != 4 * len(chunk):
        raise BatchPHSCError("a predictor call would split a four-probe group")
    try:
        prediction = predictor(structures)
        _energies, forces, _stresses = _validated_prediction(prediction, structures)
    except Exception as exc:
        raise BatchPHSCError(
            f"batch predictor failed: {type(exc).__name__}: {exc}"
        ) from None
    return tuple(force.copy() for force in forces)


def _final_result(item: _PreparedPHSC) -> PHSCResult:
    dimension = 3 * len(item.base)
    if item.seen_columns.shape != (dimension,) or not item.seen_columns.all():
        raise BatchPHSCError(f"incomplete coordinate set for {item.sid}")
    force_call_count = 4 * dimension
    if force_call_count != 12 * len(item.base):
        raise BatchPHSCError(f"force-call count is not exact 12N for {item.sid}")
    if item.numerical_error is not None:
        return PHSCResult(
            status=PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            force_call_count=force_call_count,
            error=item.numerical_error,
        )
    try:
        spectral = analyze_hessian_pair(item.h_h, item.h_h2)
    except (PHSCValidationError, PHSCNumericalError) as exc:
        return PHSCResult(
            status=PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            force_call_count=force_call_count,
            error=f"PHSC spectral analysis failed: {exc}",
        )
    return PHSCResult(
        status=spectral.status,
        negative=spectral.negative,
        d_star=item.d_star,
        h=item.h,
        lambda_h=spectral.lambda_h,
        lambda_h2=spectral.lambda_h2,
        lambda_r=spectral.lambda_r,
        e_num=spectral.e_num,
        u_num=spectral.u_num,
        l_num=spectral.l_num,
        tau_alg=spectral.tau_alg,
        antisymmetric_norm_h=spectral.antisymmetric_norm_h,
        antisymmetric_norm_h2=spectral.antisymmetric_norm_h2,
        acoustic_residual_h=spectral.acoustic_residual_h,
        acoustic_residual_h2=spectral.acoustic_residual_h2,
        force_call_count=force_call_count,
    )


def evaluate_phsc_batch(
    sids: Sequence[str],
    structures: Sequence[Atoms],
    predictor: BatchForcePredictor,
    *,
    groups_per_call: int = FROZEN_GROUPS_PER_CALL,
) -> tuple[BatchPHSCResult, ...]:
    """Evaluate PHSC in complete streamed four-probe groups, sorted by sid."""

    if isinstance(sids, (str, bytes)) or isinstance(structures, (str, bytes)):
        raise BatchPHSCError("sids and structures must be aligned sequences")
    if len(sids) != len(structures):
        raise BatchPHSCError("sids and structures must have equal lengths")
    if any(type(sid) is not str or not sid for sid in sids):
        raise BatchPHSCError("sids must be nonempty exact strings")
    if len(set(sids)) != len(sids):
        raise BatchPHSCError("sids must be unique")
    if type(groups_per_call) is not int or groups_per_call <= 0:
        raise ValueError("groups_per_call must be a positive exact integer")

    ordered = sorted(zip(sids, structures, strict=True), key=lambda item: item[0])
    completed: dict[str, PHSCResult] = {}
    prepared: list[_PreparedPHSC] = []
    for sid, atoms in ordered:
        try:
            base, d_star = canonicalize_phsc_geometry(atoms)
        except PHSCValidationError as exc:
            completed[sid] = PHSCResult(
                status=PHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
                error=f"unsupported PHSC geometry: {exc}",
            )
            continue
        h = float(STEP_FRACTION * d_star)
        if not np.isfinite(h) or h <= 0.0:
            completed[sid] = PHSCResult(
                status=PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                error="PHSC step size must be positive and finite",
            )
            continue
        dimension = 3 * len(base)
        prepared.append(
            _PreparedPHSC(
                sid=sid,
                base=base,
                d_star=d_star,
                h=h,
                h_h=np.empty((dimension, dimension), dtype=np.float64),
                h_h2=np.empty((dimension, dimension), dtype=np.float64),
                seen_columns=np.zeros(dimension, dtype=bool),
            )
        )

    prepared = _preflight_probe_representability(prepared, completed)
    for chunk in _chunks(_probe_groups(prepared), groups_per_call):
        forces = _predict_group_chunk(predictor, chunk)
        if len(forces) != 4 * len(chunk):
            raise BatchPHSCError("predictor force output split a four-probe group")
        offset = 0
        for group in chunk:
            item = group.prepared
            coordinate = group.coordinate
            if item.seen_columns[coordinate]:
                raise BatchPHSCError(
                    f"duplicate coordinate {coordinate} for {item.sid}"
                )
            samples = forces[offset : offset + 4]
            offset += 4
            if len(samples) != 4:
                raise BatchPHSCError("predictor force output split a four-probe group")
            try:
                column_h, column_h2 = hessian_columns_from_force_samples(
                    *samples, h=item.h
                )
            except PHSCNumericalError as exc:
                item.numerical_error = (
                    item.numerical_error
                    or f"PHSC finite-difference construction failed: {exc}"
                )
            except PHSCValidationError as exc:
                raise BatchPHSCError(
                    f"validated predictor violated PHSC force contract: {exc}"
                ) from None
            else:
                dimension = 3 * len(item.base)
                if column_h.shape != (dimension,) or column_h2.shape != (dimension,):
                    raise BatchPHSCError("PHSC core returned a misaligned Hessian column")
                item.h_h[:, coordinate] = column_h
                item.h_h2[:, coordinate] = column_h2
            item.seen_columns[coordinate] = True
        if offset != len(forces):
            raise BatchPHSCError("unused predictor force output remains after grouping")

    for item in prepared:
        completed[item.sid] = _final_result(item)
    return tuple(BatchPHSCResult(sid, completed[sid]) for sid, _ in ordered)


def _result_row(
    record: Mapping[str, object],
    batch_result: BatchPHSCResult | None,
    *,
    natoms: int,
) -> dict[str, object]:
    if batch_result is None:
        return {
            "sid": record["sid"],
            "rk": record["rk"],
            "stage": record["stage"],
            "threshold_role": record["threshold_role"],
            "strict_x0_ok": False,
            "natoms": 0,
            "internal_dim": 0,
            "phsc_status": PHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY.value,
            "phsc_negative": None,
            **{column: np.nan for column in NUMERIC_DIAGNOSTIC_COLUMNS},
            "force_call_count": 0,
            "error": "nonstrict_x0",
        }
    result = batch_result.result
    internal_dim = 3 * natoms - 3 if natoms >= 2 else 0
    return {
        "sid": record["sid"],
        "rk": record["rk"],
        "stage": record["stage"],
        "threshold_role": record["threshold_role"],
        "strict_x0_ok": True,
        "natoms": natoms,
        "internal_dim": internal_dim,
        "phsc_status": result.status.value,
        "phsc_negative": result.negative,
        "d_star_angstrom": result.d_star,
        "h_angstrom": result.h,
        "lambda_h_ev_per_a2": result.lambda_h,
        "lambda_h2_ev_per_a2": result.lambda_h2,
        "lambda_r_ev_per_a2": result.lambda_r,
        "e_num_ev_per_a2": result.e_num,
        "u_num_ev_per_a2": result.u_num,
        "l_num_ev_per_a2": result.l_num,
        "tau_alg_ev_per_a2": result.tau_alg,
        "antisymmetric_norm_h_ev_per_a2": result.antisymmetric_norm_h,
        "antisymmetric_norm_h2_ev_per_a2": result.antisymmetric_norm_h2,
        "acoustic_residual_h_ev_per_a2": result.acoustic_residual_h,
        "acoustic_residual_h2_ev_per_a2": result.acoustic_residual_h2,
        "force_call_count": result.force_call_count,
        "error": result.error or "",
    }


def _strict_output_table(rows: Sequence[Mapping[str, object]]) -> Any:
    import pandas as pd

    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in (
        "sid",
        "rk",
        "stage",
        "threshold_role",
        "phsc_status",
        "error",
    ):
        table[column] = table[column].astype("string")
    table["strict_x0_ok"] = table["strict_x0_ok"].astype("bool")
    table["phsc_negative"] = table["phsc_negative"].astype("boolean")
    for column in ("natoms", "internal_dim", "force_call_count"):
        table[column] = table[column].astype("int64")
    for column in NUMERIC_DIAGNOSTIC_COLUMNS:
        table[column] = table[column].astype("float64")
    table = table.sort_values("sid", kind="stable").reset_index(drop=True)
    if table["sid"].duplicated().any():
        raise BatchPHSCError("output sid values must be unique")

    allowed = {status.value for status in PHSCStatus}
    if not set(table["phsc_status"].astype(str)).issubset(allowed):
        raise BatchPHSCError("output contains an unknown PHSC status")
    for row in table.itertuples(index=False):
        status = PHSCStatus(str(row.phsc_status))
        natoms = int(row.natoms)
        calls = int(row.force_call_count)
        strict = bool(row.strict_x0_ok)
        negative = row.phsc_negative
        diagnostics = np.asarray(
            [getattr(row, column) for column in NUMERIC_DIAGNOSTIC_COLUMNS],
            dtype=float,
        )
        if not strict:
            if (
                natoms != 0
                or int(row.internal_dim) != 0
                or calls != 0
                or status is not PHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY
                or negative is not pd.NA
                or str(row.error) != "nonstrict_x0"
                or not np.isnan(diagnostics).all()
            ):
                raise BatchPHSCError("nonstrict PHSC sentinel semantics mismatch")
            continue
        if natoms < 1 or int(row.internal_dim) != max(0, 3 * natoms - 3):
            raise BatchPHSCError("strict PHSC natoms/internal_dim mismatch")
        if status in _SUCCESS_STATUSES:
            if (
                natoms < 2
                or calls != 12 * natoms
                or negative is pd.NA
                or not np.isfinite(diagnostics).all()
                or str(row.error) != ""
            ):
                raise BatchPHSCError("successful PHSC row semantics mismatch")
            if bool(negative) != (status is PHSCStatus.RESOLVED_NEGATIVE):
                raise BatchPHSCError("PHSC negative flag differs from status")
        else:
            if negative is not pd.NA or not str(row.error):
                raise BatchPHSCError("abstained PHSC row semantics mismatch")
            if calls not in (0, 12 * natoms):
                raise BatchPHSCError("abstained PHSC row must have zero or exact 12N calls")
            if not np.isnan(diagnostics).all():
                raise BatchPHSCError("abstained PHSC diagnostics must be missing")
    return table


def _write_and_publish(
    table: Any,
    manifest: Mapping[str, object],
    output_dir: Path,
    *,
    verify_unchanged: Any,
) -> dict[str, object]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    try:
        table_path = staging / OUTPUT_NAME
        table.to_parquet(table_path, index=False)
        import pandas as pd

        reloaded = pd.read_parquet(table_path)
        try:
            pd.testing.assert_frame_equal(
                reloaded,
                table.reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise RuntimeError(
                "staged PHSC parquet failed exact value validation"
            ) from exc
        final_manifest = {
            **dict(manifest),
            "outputs_sha256": {OUTPUT_NAME: _sha256_file(table_path)},
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(final_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        verify_unchanged()
        _atomic_publish_directory_no_replace(staging, output_dir)
        return final_manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _is_canonical_cuda_device(device: str) -> bool:
    if not device.startswith("cuda:"):
        return False
    index = device.removeprefix("cuda:")
    return bool(index.isdigit() and str(int(index)) == index)


def _raw_frame_snapshot_from_geometry_manifest(
    geometry_manifest: Mapping[str, object],
) -> _InputSnapshot:
    inputs = geometry_manifest.get("inputs_sha256")
    if not isinstance(inputs, Mapping):
        raise ValueError("geometry-only manifest lacks input provenance")
    raw = inputs.get("raw_frames")
    if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
        raise ValueError("geometry-only raw-frame provenance is invalid")
    path = raw.get("path")
    digest = raw.get("sha256")
    if type(path) is not str or not path:
        raise ValueError("geometry-only raw-frame path is invalid")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("geometry-only raw-frame hash is invalid")
    # This proxy binds the upstream feature manifest to sanitizer provenance.
    # It deliberately does not open or hash the raw x0 archive.
    return _InputSnapshot(path=Path(path), sha256=digest, data=None)


def _require_frozen_next11_inputs(
    snapshots: Mapping[str, _InputSnapshot],
) -> None:
    observed = {
        "committee_features": snapshots["features"].sha256,
        "threshold_roles": snapshots["roles"].sha256,
        "geometry_only_frames": snapshots["frames"].sha256,
        "geometry_manifest": snapshots["geometry_manifest"].sha256,
        "feature_manifest": snapshots["feature_manifest"].sha256,
    }
    if observed != FROZEN_NEXT11_INPUT_SHA256:
        mismatched = sorted(
            role
            for role, expected in FROZEN_NEXT11_INPUT_SHA256.items()
            if observed.get(role) != expected
        )
        raise ValueError(
            "production inputs do not equal frozen next11 geometry-only identities: "
            f"{mismatched}"
        )


def _load_geometry_from_snapshots(
    *,
    frames_snapshot: _InputSnapshot,
    manifest_snapshot: _InputSnapshot,
    expected_sids: Sequence[str],
) -> tuple[list[str], list[Atoms]]:
    """Parse the exact initially hashed geometry bytes, never the live inputs."""

    if frames_snapshot.data is None or manifest_snapshot.data is None:
        raise RuntimeError("geometry snapshots must retain their initially hashed bytes")
    staging = Path(tempfile.mkdtemp(prefix=".next11-geometry-snapshot-"))
    try:
        archive_path = staging / "geometry_only_frames.zip"
        manifest_path = staging / "MANIFEST.json"
        archive_path.write_bytes(frames_snapshot.data)
        manifest_path.write_bytes(manifest_snapshot.data)
        return load_geometry_only_archive(
            archive_path=archive_path,
            manifest_path=manifest_path,
            expected_sids=expected_sids,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_label_free_features(
    *,
    features_path: Path,
    role_assignments_path: Path,
    frames_zip_path: Path,
    geometry_manifest_path: Path,
    feature_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    predictor: BatchForcePredictor | None = None,
    device: str = "cuda:0",
    model_batch_size: int = FROZEN_MODEL_BATCH_SIZE,
    groups_per_call: int = FROZEN_GROUPS_PER_CALL,
    engineering_smoke: bool = False,
) -> dict[str, object]:
    """Seal development-gate PHSC diagnostics without accepting label input."""

    target = Path(output_dir)
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    if type(model_batch_size) is not int or model_batch_size <= 0:
        raise ValueError("model_batch_size must be a positive exact integer")
    if type(groups_per_call) is not int or groups_per_call <= 0:
        raise ValueError("groups_per_call must be a positive exact integer")
    if type(engineering_smoke) is not bool:
        raise ValueError("engineering_smoke must be an exact boolean")
    if engineering_smoke and predictor is not None:
        raise ValueError("engineering CUDA smoke requires the builtin indexed predictor")
    device = str(device).strip().lower()
    if not device:
        raise ValueError("device must be a nonempty string")
    if predictor is None and (
        model_batch_size != FROZEN_MODEL_BATCH_SIZE
        or groups_per_call != FROZEN_GROUPS_PER_CALL
        or not _is_canonical_cuda_device(device)
    ):
        raise ValueError(
            "production PHSC requires cuda:N, model_batch_size=32, "
            "and groups_per_call=256"
        )

    paths = {
        "features": Path(features_path),
        "roles": Path(role_assignments_path),
        "frames": Path(frames_zip_path),
        "geometry_manifest": Path(geometry_manifest_path),
        "feature_manifest": Path(feature_manifest_path),
        "checkpoint": Path(checkpoint_path),
    }
    snapshots = {
        role: _snapshot(
            path,
            include_data=role
            in {
                "features",
                "roles",
                "frames",
                "geometry_manifest",
                "feature_manifest",
            },
        )
        for role, path in paths.items()
    }
    upstream_manifest = _strict_json_document(
        snapshots["feature_manifest"].data or b"", role="feature manifest"
    )
    geometry_manifest = _strict_json_document(
        snapshots["geometry_manifest"].data or b"", role="geometry-only manifest"
    )
    raw_frame_snapshot = _raw_frame_snapshot_from_geometry_manifest(
        geometry_manifest
    )
    _validate_upstream_manifest(
        upstream_manifest,
        features=snapshots["features"],
        frames=raw_frame_snapshot,
        checkpoint=snapshots["checkpoint"],
    )
    if predictor is None:
        _require_frozen_next11_inputs(snapshots)
        if snapshots["checkpoint"].sha256 != FROZEN_M5_SHA256:
            raise ValueError("production checkpoint does not equal frozen MatterSim 5M")

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        relative: repository_root / relative for relative in EXECUTED_SOURCE_RELATIVE
    }
    source_sha256 = {
        relative: _sha256_file(path) for relative, path in source_paths.items()
    }
    selected, feature_rows, role_rows = _selected_gate_rows(
        snapshots["features"].data or b"", snapshots["roles"].data or b""
    )
    expected_strict_sids = selected.loc[
        selected["strict_x0_ok"].astype(bool), "sid"
    ].tolist()
    strict_sids, structures = _load_geometry_from_snapshots(
        frames_snapshot=snapshots["frames"],
        manifest_snapshot=snapshots["geometry_manifest"],
        expected_sids=expected_strict_sids,
    )
    observed_formal_counts = {
        "feature_rows": feature_rows,
        "role_assignment_rows": role_rows,
        "selected_rows": len(selected),
        "strict_rows": len(strict_sids),
        "nonstrict_rows": len(selected) - len(strict_sids),
    }
    if predictor is None and observed_formal_counts != FORMAL_EXPECTED_COUNTS:
        raise ValueError("production cohort counts differ from the frozen PHSC gate")
    smoke_sids: tuple[str, ...] = ()
    if engineering_smoke:
        if len(strict_sids) < FROZEN_ENGINEERING_SMOKE_COUNT:
            raise ValueError("frozen engineering smoke cohort is incomplete")
        smoke_sids = tuple(strict_sids[:FROZEN_ENGINEERING_SMOKE_COUNT])
        retained = set(smoke_sids)
        selected = selected.loc[selected["sid"].isin(retained)].copy()
        selected = selected.sort_values("sid", kind="stable", ignore_index=True)
        paired = {
            sid: atoms for sid, atoms in zip(strict_sids, structures, strict=True)
        }
        strict_sids = list(smoke_sids)
        structures = [paired[sid] for sid in strict_sids]
    runtime = _runtime_identity(device)
    if predictor is None:
        if runtime.get("mattersim_version") != "1.2.3":
            raise RuntimeError("production runtime requires MatterSim 1.2.3")
        if runtime.get("cuda_available") is not True:
            raise RuntimeError("production PHSC requires available CUDA")
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
    batch_results = evaluate_phsc_batch(
        strict_sids,
        structures,
        counting_predictor,
        groups_per_call=groups_per_call,
    )
    elapsed = time.perf_counter() - started
    results_by_sid = {item.sid: item for item in batch_results}
    if len(results_by_sid) != len(batch_results):
        raise BatchPHSCError("batch PHSC results contain duplicate sid values")
    natoms_by_sid = {
        sid: len(atoms) for sid, atoms in zip(strict_sids, structures, strict=True)
    }
    rows: list[dict[str, object]] = []
    for record in selected.to_dict("records"):
        sid = str(record["sid"])
        if bool(record["strict_x0_ok"]):
            rows.append(
                _result_row(
                    record,
                    results_by_sid[sid],
                    natoms=natoms_by_sid[sid],
                )
            )
        else:
            rows.append(_result_row(record, None, natoms=0))
    table = _strict_output_table(rows)
    probe_evaluations = int(table["force_call_count"].sum())
    if any(size <= 0 or size % 4 != 0 for size in predictor_batch_sizes):
        raise BatchPHSCError("predictor call split a complete four-probe group")
    coordinate_groups = sum(predictor_batch_sizes) // 4
    if probe_evaluations != 4 * coordinate_groups:
        raise BatchPHSCError("probe telemetry differs from row exact-call counts")
    expected_predictor_calls = (
        math.ceil(coordinate_groups / groups_per_call) if coordinate_groups else 0
    )
    if len(predictor_batch_sizes) != expected_predictor_calls:
        raise BatchPHSCError("predictor-call count differs from frozen group chunks")

    if predictor is None:
        telemetry = _validated_builtin_telemetry(
            active_predictor,
            device=device,
            expected_evaluations=probe_evaluations,
        )
        expected_forward_calls = sum(
            math.ceil(size / model_batch_size) for size in predictor_batch_sizes
        )
        if int(telemetry["forward_calls"]) != expected_forward_calls:
            raise RuntimeError("MatterSim forward calls differ from frozen chunking")
        production_eligible = not engineering_smoke
        adapter = {
            "mode": adapter_mode,
            "index_alignment": "sid_indexed_exact_one_to_one",
            "index_alignment_verified": True,
            "device": device,
            "model_batch_size": model_batch_size,
            "groups_per_call": groups_per_call,
            "model_parameter_device": telemetry["model_parameter_device"],
            "result_tensor_devices": telemetry["result_tensor_devices"],
            "evaluations": telemetry["evaluations"],
        }
        forward_calls: int | None = int(telemetry["forward_calls"])
        peak_cuda_memory_bytes: int | None = int(
            telemetry["peak_cuda_memory_bytes"]
        )
    else:
        production_eligible = False
        adapter = {
            "mode": adapter_mode,
            "index_alignment": "injected_batch_force_predictor_declared_aligned",
            "index_alignment_verified": False,
            "device": device,
            "model_batch_size": model_batch_size,
            "groups_per_call": groups_per_call,
            "model_parameter_device": None,
            "result_tensor_devices": [],
            "evaluations": probe_evaluations,
        }
        forward_calls = None
        peak_cuda_memory_bytes = None

    def verify_unchanged() -> None:
        for role, snapshot in snapshots.items():
            if _sha256_file(snapshot.path) != snapshot.sha256:
                raise RuntimeError(f"input {role} changed after initial hash")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_sha256[relative]:
                raise RuntimeError(f"executed source {relative} changed after initial hash")

    statuses = table["phsc_status"].astype(str)
    strict = table["strict_x0_ok"].astype(bool)
    calls = table["force_call_count"].to_numpy(dtype=int)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "engineering_cuda_smoke" if engineering_smoke else "development_gate",
        "labels_opened": False,
        "input_isolation": {
            "geometry_only": True,
            "geometry_protocol": GEOMETRY_ONLY_PROTOCOL,
            "raw_x0_archive_opened": False,
            "endpoint_label_artifacts_opened": False,
        },
        "selection": (
            {
                "stage": "threshold_calibration",
                "threshold_role": "development_gate",
                "strategy": "lexicographically_first_strict_sid",
                "count": FROZEN_ENGINEERING_SMOKE_COUNT,
                "sids": list(smoke_sids),
            }
            if engineering_smoke
            else {
                "stage": "threshold_calibration",
                "threshold_role": "development_gate",
            }
        ),
        "adapter": adapter,
        "predictor_loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "production_protocol_eligible": production_eligible,
        "evidence_role": (
            "label_free_phsc_feature_generation"
            if production_eligible
            else (
                "engineering_cuda_smoke_only"
                if engineering_smoke
                else "testing_only_not_scientific_evidence"
            )
        ),
        "runtime": runtime,
        "inputs_sha256": {
            "committee_features": _hash_record(snapshots["features"]),
            "threshold_roles": _hash_record(snapshots["roles"]),
            "geometry_only_frames": _hash_record(snapshots["frames"]),
            "geometry_manifest": _hash_record(snapshots["geometry_manifest"]),
            "source_frames_provenance": _hash_record(raw_frame_snapshot),
            "feature_manifest": _hash_record(snapshots["feature_manifest"]),
            "checkpoint": _hash_record(snapshots["checkpoint"]),
        },
        "executed_source_sha256": source_sha256,
        "integrity": {"prepublish_rehash": "passed"},
        "feature_columns": list(OUTPUT_COLUMNS),
        "criterion": dict(CRITERION),
        "formal_expectations": {
            **FORMAL_EXPECTED_COUNTS,
            "model_batch_size": FROZEN_MODEL_BATCH_SIZE,
            "groups_per_call": FROZEN_GROUPS_PER_CALL,
            "device_contract": "canonical_cuda:N",
            "checkpoint_sha256": FROZEN_M5_SHA256,
            "geometry_protocol": GEOMETRY_ONLY_PROTOCOL,
            "geometry_only_frames_sha256": FROZEN_NEXT11_INPUT_SHA256[
                "geometry_only_frames"
            ],
            "geometry_manifest_sha256": FROZEN_NEXT11_INPUT_SHA256[
                "geometry_manifest"
            ],
        },
        "counts": {
            "feature_rows": feature_rows,
            "role_assignment_rows": role_rows,
            "selected_rows": len(table),
            "strict_rows": int(strict.sum()),
            "nonstrict_rows": int((~strict).sum()),
            "probe_eligible_rows": int(np.count_nonzero(calls > 0)),
            "resolved_negative_rows": int(
                statuses.eq(PHSCStatus.RESOLVED_NEGATIVE.value).sum()
            ),
            "resolved_nonnegative_rows": int(
                statuses.eq(PHSCStatus.RESOLVED_NONNEGATIVE.value).sum()
            ),
            "near_zero_or_inconsistent_rows": int(
                statuses.eq(PHSCStatus.NEAR_ZERO_OR_INCONSISTENT.value).sum()
            ),
            "abstained_rows": int(statuses.str.startswith("abstain_").sum()),
            "coordinate_groups": coordinate_groups,
            "probe_evaluations": probe_evaluations,
            "batch_predictor_calls": len(predictor_batch_sizes),
        },
        "execution": {
            "batch_predictor_calls": len(predictor_batch_sizes),
            "predictor_batch_sizes": [int(size) for size in predictor_batch_sizes],
            "max_predictor_batch_size": (
                int(max(predictor_batch_sizes)) if predictor_batch_sizes else 0
            ),
            "forward_calls": forward_calls,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "wall_time_seconds": elapsed,
        },
        "scientific_improvement_claim": False,
    }
    return _write_and_publish(
        table,
        manifest,
        target,
        verify_unchanged=verify_unchanged,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the one frozen production PHSC-v0 feature run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--role-assignments-path", type=Path, required=True)
    parser.add_argument("--frames-zip-path", type=Path, required=True)
    parser.add_argument("--geometry-manifest-path", type=Path, required=True)
    parser.add_argument("--feature-manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--engineering-smoke", action="store_true")
    arguments = parser.parse_args(argv)
    run_label_free_features(
        features_path=arguments.features_path,
        role_assignments_path=arguments.role_assignments_path,
        frames_zip_path=arguments.frames_zip_path,
        geometry_manifest_path=arguments.geometry_manifest_path,
        feature_manifest_path=arguments.feature_manifest_path,
        checkpoint_path=arguments.checkpoint_path,
        output_dir=arguments.output_dir,
        predictor=None,
        device=arguments.device,
        model_batch_size=FROZEN_MODEL_BATCH_SIZE,
        groups_per_call=FROZEN_GROUPS_PER_CALL,
        engineering_smoke=arguments.engineering_smoke,
    )
    return 0


__all__ = [
    "BatchPHSCError",
    "BatchPHSCResult",
    "CRITERION",
    "EXECUTED_SOURCE_RELATIVE",
    "FORMAL_EXPECTED_COUNTS",
    "FROZEN_ENGINEERING_SMOKE_COUNT",
    "FROZEN_NEXT11_INPUT_SHA256",
    "FROZEN_GROUPS_PER_CALL",
    "FROZEN_MODEL_BATCH_SIZE",
    "MANIFEST_NAME",
    "NUMERIC_DIAGNOSTIC_COLUMNS",
    "OUTPUT_COLUMNS",
    "OUTPUT_NAME",
    "PROTOCOL",
    "evaluate_phsc_batch",
    "main",
    "run_label_free_features",
]


if __name__ == "__main__":
    raise SystemExit(main())
