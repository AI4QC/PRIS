"""Label-free, fixed-protocol batched LRRC feature construction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any, Mapping, Protocol, Sequence
import zipfile

import numpy as np
import pandas as pd
from ase import Atoms
from ase.units import GPa

from src.next6_mattersim_baseline import frame_to_atoms
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)
from src.next9_lrrc import (
    LRRCResult,
    LRRCStatus,
    LRRCValidationError,
    STEP_FRACTION,
    evaluate_lrrc,
    median_nearest_neighbor_distance,
    translation_projected_direction,
)


PROTOCOL = "2026-08-01-next10-lrrc-mattersim-features-v1"
UPSTREAM_FEATURE_PROTOCOL = "2026-08-01-mattersim-dual-checkpoint-x0-v1"
FROZEN_M5_SHA256 = "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"
FROZEN_NEXT8_INPUT_SHA256 = {
    "committee_features": "65f0234010f17f43a96789bde7858bae038ffaa4aaa2130eaee163fd3245bc8c",
    "threshold_roles": "e6de5f5b5fc9545944043bda46e313fa2060833f1baa31dd93dcca12e4769602",
    "frames": "8c63e02932fcb0158c5d917702a4a863f01b0e0ec55fdddff426a24613e10457",
    "feature_manifest": "e59848270c0fd1693d6f7d579ee327aebf4f34399ee73d27eb2c97f947cab9dd",
}
OUTPUT_NAME = "lrrc_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
OUTPUT_COLUMNS = (
    "sid",
    "rk",
    "stage",
    "threshold_role",
    "strict_x0_ok",
    "natoms",
    "lrrc_status",
    "lrrc_negative",
    "d_star_angstrom",
    "h_angstrom",
    "kappa_h_ev_per_a2",
    "kappa_h2_ev_per_a2",
    "kappa_r_ev_per_a2",
    "error_proxy_ev_per_a2",
    "u_num_ev_per_a2",
    "force_call_count",
    "error",
)
_EXECUTED_SOURCE_RELATIVE = (
    "src/next10_lrrc_mattersim_features.py",
    "src/next9_lrrc.py",
    "src/next8_mattersim_committee_features.py",
    "src/next6_mattersim_baseline.py",
)


class BatchLRRCError(RuntimeError):
    """Raised when a batch LRRC run cannot preserve its complete cohort."""


@dataclass(frozen=True, slots=True)
class BatchPrediction:
    """One input-order-aligned batch of energies, forces, and stresses."""

    total_energies_ev: Sequence[object]
    forces_ev_per_a: Sequence[object]
    stresses_ev_per_a3: Sequence[object]


class BatchForcePredictor(Protocol):
    """Aligned batch-force interface implemented by the next7 adapter."""

    def __call__(self, structures: list[Atoms]) -> BatchPrediction:
        """Return one aligned :class:`BatchPrediction` per input structure."""


def _validated_prediction(
    prediction: object,
    structures: Sequence[Atoms],
) -> tuple[tuple[float, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    try:
        raw_energies = prediction.total_energies_ev
        raw_forces = prediction.forces_ev_per_a
        raw_stresses = prediction.stresses_ev_per_a3
    except AttributeError as exc:
        raise ValueError("prediction must satisfy the BatchPrediction contract") from exc
    expected = len(structures)
    if not all(len(values) == expected for values in (raw_energies, raw_forces, raw_stresses)):
        raise ValueError("prediction fields must align one-to-one with structures")
    energies: list[float] = []
    forces: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    for index, (atoms, energy, force, stress) in enumerate(
        zip(structures, raw_energies, raw_forces, raw_stresses, strict=True)
    ):
        energy_array = np.asarray(energy, dtype=float)
        if energy_array.shape != () or not np.isfinite(energy_array).all():
            raise ValueError(f"prediction {index} energy must be a finite scalar")
        force_array = np.asarray(force, dtype=float)
        if force_array.shape != (len(atoms), 3) or not np.isfinite(force_array).all():
            raise ValueError(
                f"prediction {index} forces must be finite with shape {(len(atoms), 3)}"
            )
        stress_array = np.asarray(stress, dtype=float)
        if stress_array.shape != (3, 3) or not np.isfinite(stress_array).all():
            raise ValueError(
                f"prediction {index} stress must be finite with shape (3, 3)"
            )
        energies.append(float(energy_array))
        forces.append(force_array.copy())
        stresses.append(stress_array.copy())
    return tuple(energies), tuple(forces), tuple(stresses)


class _IndexedMatterSimPredictor:
    """MatterSim 1.2.3 adapter that restores results by explicit input ordinal."""

    def __init__(
        self,
        *,
        potential: object,
        build_dataloader: Any,
        batch_to_dict: Any,
        torch_module: Any,
        device: str,
        batch_size: int,
    ) -> None:
        self._potential = potential
        self._build_dataloader = build_dataloader
        self._batch_to_dict = batch_to_dict
        self._torch = torch_module
        self._device = device
        self._batch_size = batch_size
        model_args = potential.model.model_args
        self._cutoff = float(model_args["cutoff"])
        self._threebody_cutoff = float(model_args["threebody_cutoff"])
        try:
            parameter = next(iter(potential.model.parameters()))
        except (AttributeError, StopIteration, TypeError) as exc:
            raise BatchLRRCError("MatterSim model has no parameter-device evidence") from exc
        self._model_parameter_device = str(parameter.device)
        self._result_tensor_devices: set[str] = set()
        self._forward_calls = 0
        self._evaluations = 0
        self._peak_cuda_memory_bytes = 0
        if device.lower().startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats(device)

    @property
    def telemetry(self) -> dict[str, object]:
        return {
            "model_parameter_device": self._model_parameter_device,
            "result_tensor_devices": sorted(self._result_tensor_devices),
            "forward_calls": self._forward_calls,
            "evaluations": self._evaluations,
            "peak_cuda_memory_bytes": self._peak_cuda_memory_bytes,
        }

    @staticmethod
    def _numpy(tensor: object) -> np.ndarray:
        value = tensor
        for method_name in ("detach", "cpu"):
            method = getattr(value, method_name, None)
            if callable(method):
                value = method()
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            value = numpy_method()
        return np.asarray(value)

    def __call__(self, structures: list[Atoms]) -> BatchPrediction:
        if not structures:
            return BatchPrediction((), (), ())
        loader = self._build_dataloader(
            structures,
            cutoff=self._cutoff,
            threebody_cutoff=self._threebody_cutoff,
            batch_size=self._batch_size,
            only_inference=True,
            shuffle=False,
        )
        try:
            dataset = loader.dataset
        except AttributeError as exc:
            raise BatchLRRCError("MatterSim loader has no indexable dataset") from exc
        if len(dataset) != len(structures):
            raise BatchLRRCError("MatterSim graph conversion changed cohort size")
        for ordinal, graph in enumerate(dataset):
            setattr(graph, "next10_input_ordinal", ordinal)

        energies: list[float | None] = [None] * len(structures)
        forces: list[np.ndarray | None] = [None] * len(structures)
        stresses: list[np.ndarray | None] = [None] * len(structures)
        seen: set[int] = set()
        self._potential.model.eval()
        for graph_batch in loader:
            try:
                ordinal_values = self._numpy(
                    graph_batch.next10_input_ordinal
                ).reshape(-1)
                natoms_values = self._numpy(graph_batch.num_atoms).reshape(-1)
            except Exception as exc:
                raise BatchLRRCError("MatterSim batch lacks aligned graph ordinals") from exc
            if not np.equal(ordinal_values, np.floor(ordinal_values)).all():
                raise BatchLRRCError("MatterSim input ordinals must be exact integers")
            ordinals = [int(value) for value in ordinal_values]
            try:
                natoms_numeric = np.asarray(natoms_values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise BatchLRRCError(
                    "MatterSim num_atoms must be finite exact integers"
                ) from exc
            if (
                not np.isfinite(natoms_numeric).all()
                or not np.equal(natoms_numeric, np.floor(natoms_numeric)).all()
                or (natoms_numeric <= 0).any()
            ):
                raise BatchLRRCError(
                    "MatterSim num_atoms must be finite exact integers"
                )
            natoms = [int(value) for value in natoms_numeric]
            if len(ordinals) != len(natoms):
                raise BatchLRRCError("MatterSim ordinal/natoms batch lengths differ")
            inputs = self._batch_to_dict(graph_batch, device=self._device)
            result = self._potential.forward(
                inputs,
                include_forces=True,
                include_stresses=True,
            )
            self._forward_calls += 1
            try:
                result_tensors = (
                    result["total_energy"],
                    result["forces"],
                    result["stresses"],
                )
                self._result_tensor_devices.update(
                    str(tensor.device) for tensor in result_tensors
                )
                batch_energies = self._numpy(result_tensors[0]).reshape(-1)
                batch_forces = self._numpy(result_tensors[1])
                batch_stresses = self._numpy(result_tensors[2])
            except Exception as exc:
                raise BatchLRRCError("MatterSim forward result is incomplete") from exc
            if (
                batch_energies.shape != (len(ordinals),)
                or batch_forces.shape != (sum(natoms), 3)
                or batch_stresses.shape != (len(ordinals), 3, 3)
            ):
                raise BatchLRRCError("MatterSim forward result shape mismatch")
            force_start = 0
            for batch_index, (ordinal, atom_count) in enumerate(
                zip(ordinals, natoms, strict=True)
            ):
                if ordinal < 0 or ordinal >= len(structures):
                    raise BatchLRRCError("MatterSim input ordinal is out of range")
                if ordinal in seen:
                    raise BatchLRRCError(f"duplicate input ordinal: {ordinal}")
                if atom_count != len(structures[ordinal]):
                    raise BatchLRRCError("MatterSim natoms differs from indexed input")
                force_end = force_start + atom_count
                energies[ordinal] = float(batch_energies[batch_index])
                forces[ordinal] = np.asarray(
                    batch_forces[force_start:force_end], dtype=float
                ).copy()
                stresses[ordinal] = (
                    np.asarray(batch_stresses[batch_index], dtype=float) * GPa
                )
                force_start = force_end
                seen.add(ordinal)
                self._evaluations += 1
        missing = sorted(set(range(len(structures))) - seen)
        if missing:
            raise BatchLRRCError(f"missing input ordinals: {missing}")
        if self._device.lower().startswith("cuda"):
            self._peak_cuda_memory_bytes = max(
                self._peak_cuda_memory_bytes,
                int(self._torch.cuda.max_memory_allocated(self._device)),
            )
        return BatchPrediction(
            total_energies_ev=[float(value) for value in energies],
            forces_ev_per_a=[np.asarray(value, dtype=float) for value in forces],
            stresses_ev_per_a3=[np.asarray(value, dtype=float) for value in stresses],
        )


@dataclass(frozen=True, slots=True)
class BatchLRRCResult:
    """One sid-aligned scalar LRRC outcome from the batched force passes."""

    sid: str
    result: LRRCResult
    force_call_count: int


@dataclass(frozen=True, slots=True)
class _InputSnapshot:
    path: Path
    sha256: str
    data: bytes | None


@dataclass(slots=True)
class _ReplayForceOracle:
    """Replay aligned forces only for the exact precomputed probe geometries."""

    expected_probes: tuple[Atoms, ...]
    force_values: tuple[np.ndarray, ...]
    _index: int = field(default=0, init=False)
    failure: str | None = field(default=None, init=False)

    @property
    def calls(self) -> int:
        return self._index

    def _reject(self, field_name: str) -> None:
        self.failure = f"replay probe {field_name} mismatch at call {self._index}"
        raise BatchLRRCError(self.failure)

    def __call__(self, probe: Atoms) -> np.ndarray:
        if self._index >= len(self.expected_probes):
            self.failure = "replay oracle received an unexpected extra call"
            raise BatchLRRCError(self.failure)
        expected = self.expected_probes[self._index]
        if not np.array_equal(probe.get_atomic_numbers(), expected.get_atomic_numbers()):
            self._reject("numbers")
        if not np.allclose(
            probe.get_positions(), expected.get_positions(), rtol=0.0, atol=1e-14
        ):
            self._reject("positions")
        if not np.allclose(
            np.asarray(probe.cell), np.asarray(expected.cell), rtol=0.0, atol=1e-14
        ):
            self._reject("cell")
        if not np.array_equal(np.asarray(probe.pbc), np.asarray(expected.pbc)):
            self._reject("pbc")
        forces = self.force_values[self._index]
        self._index += 1
        return forces.copy()


@dataclass(frozen=True, slots=True)
class _PreparedLRRC:
    sid: str
    base: Atoms
    probes: tuple[Atoms, ...]
    forces: tuple[np.ndarray, ...]


def _sanitize_structure(structure: Atoms) -> Atoms:
    if not isinstance(structure, Atoms):
        raise BatchLRRCError("structures must contain ase.Atoms objects")
    try:
        return Atoms(
            numbers=np.asarray(structure.get_atomic_numbers(), dtype=int).copy(),
            positions=np.asarray(structure.get_positions(), dtype=float).copy(),
            cell=np.asarray(structure.cell, dtype=float).copy(),
            pbc=np.asarray(structure.pbc, dtype=bool).copy(),
        )
    except Exception as exc:
        raise BatchLRRCError(
            f"could not sanitize structure: {type(exc).__name__}: {exc}"
        ) from None


def _predict_forces(
    predictor: BatchForcePredictor,
    structures: Sequence[Atoms],
    *,
    stage: str,
) -> tuple[np.ndarray, ...]:
    batch = [atoms.copy() for atoms in structures]
    try:
        prediction = predictor(batch)
    except Exception as exc:
        raise BatchLRRCError(
            f"batch predictor failed at {stage}: {type(exc).__name__}: {exc}"
        ) from None
    try:
        _energies, forces, _stresses = _validated_prediction(prediction, batch)
    except Exception as exc:
        raise BatchLRRCError(
            f"invalid batch prediction at {stage}: {type(exc).__name__}: {exc}"
        ) from None
    return tuple(force.copy() for force in forces)


def _replay_result(prepared: _PreparedLRRC) -> BatchLRRCResult:
    replay = _ReplayForceOracle(prepared.probes, prepared.forces)
    result = evaluate_lrrc(prepared.base, replay)
    if replay.failure is not None:
        raise BatchLRRCError(replay.failure)
    if replay.calls != len(prepared.probes):
        raise BatchLRRCError(
            f"replay consumed {replay.calls} calls; expected {len(prepared.probes)}"
        )
    if result.status is LRRCStatus.ABSTAIN_FORCE_FAILURE:
        raise BatchLRRCError("validated replay unexpectedly reported force failure")
    return BatchLRRCResult(
        sid=prepared.sid,
        result=result,
        force_call_count=replay.calls,
    )


def _terminal_result_after_base(
    sid: str,
    base: Atoms,
    forces: np.ndarray,
) -> BatchLRRCResult | None:
    """Return a scalar terminal result, or ``None`` when probes are required."""

    replay = _ReplayForceOracle((base,), (forces,))
    result = evaluate_lrrc(base, replay)
    if replay.failure is None:
        if replay.calls != 1:
            raise BatchLRRCError(
                f"post-base replay consumed {replay.calls} calls; expected 1"
            )
        if result.status is LRRCStatus.ABSTAIN_FORCE_FAILURE:
            raise BatchLRRCError("validated base replay reported force failure")
        return BatchLRRCResult(sid=sid, result=result, force_call_count=1)
    if replay.failure != "replay oracle received an unexpected extra call":
        raise BatchLRRCError(replay.failure)
    if result.status is not LRRCStatus.ABSTAIN_FORCE_FAILURE:
        raise BatchLRRCError("probe-request replay did not report force failure")
    return None


def evaluate_lrrc_batch(
    sids: Sequence[str],
    structures: Sequence[Atoms],
    predictor: BatchForcePredictor,
) -> tuple[BatchLRRCResult, ...]:
    """Evaluate LRRC in five fixed sid-aligned batches without scalar reimplementation."""

    if isinstance(sids, (str, bytes)) or isinstance(structures, (str, bytes)):
        raise BatchLRRCError("sids and structures must be aligned sequences")
    if len(sids) != len(structures):
        raise BatchLRRCError("sids and structures must have equal lengths")
    if any(type(sid) is not str or not sid for sid in sids):
        raise BatchLRRCError("sids must be nonempty exact strings")
    if len(set(sids)) != len(sids):
        raise BatchLRRCError("sids must be unique")

    ordered = sorted(
        (
            (sid, _sanitize_structure(structure))
            for sid, structure in zip(sids, structures, strict=True)
        ),
        key=lambda item: item[0],
    )
    completed: dict[str, BatchLRRCResult] = {}
    eligible: list[tuple[str, Atoms]] = []
    for sid, base in ordered:
        force_requested = False

        def mark_force_request(_probe: Atoms) -> np.ndarray:
            nonlocal force_requested
            force_requested = True
            raise RuntimeError("preforce geometry probe")

        preflight = evaluate_lrrc(base, mark_force_request)
        if force_requested:
            eligible.append((sid, base))
        else:
            completed[sid] = BatchLRRCResult(
                sid=sid,
                result=preflight,
                force_call_count=0,
            )
    if not eligible:
        return tuple(completed[sid] for sid, _atoms in ordered)

    eligible_sids = [sid for sid, _atoms in eligible]
    bases = [atoms for _sid, atoms in eligible]
    base_forces = _predict_forces(predictor, bases, stage="base")

    active: list[_PreparedLRRC] = []
    for sid, base, forces in zip(eligible_sids, bases, base_forces, strict=True):
        terminal = _terminal_result_after_base(sid, base, forces)
        if terminal is not None:
            completed[sid] = terminal
            continue
        direction = translation_projected_direction(forces)
        if direction is None:
            raise BatchLRRCError("scalar replay requested probes for stationary forces")
        try:
            d_star = median_nearest_neighbor_distance(base)
        except LRRCValidationError as exc:
            raise BatchLRRCError(
                f"scalar replay requested probes for unsupported geometry: {exc}"
            ) from None
        h = STEP_FRACTION * d_star
        if not np.isfinite(h) or h <= 0.0:
            raise BatchLRRCError("scalar replay requested probes with invalid step size")
        offsets = (h, -h, 0.5 * h, -0.5 * h)
        probes: list[Atoms] = [base]
        for offset in offsets:
            probe = base.copy()
            probe.set_positions(base.get_positions() + offset * direction)
            probes.append(probe)
        active.append(
            _PreparedLRRC(
                sid=sid,
                base=base,
                probes=tuple(probes),
                forces=(forces,),
            )
        )

    for probe_index, stage in enumerate(
        ("plus_h", "minus_h", "plus_h2", "minus_h2"), start=1
    ):
        if not active:
            break
        stage_forces = _predict_forces(
            predictor,
            [item.probes[probe_index] for item in active],
            stage=stage,
        )
        active = [
            _PreparedLRRC(
                sid=item.sid,
                base=item.base,
                probes=item.probes,
                forces=(*item.forces, force),
            )
            for item, force in zip(active, stage_forces, strict=True)
        ]

    for item in active:
        completed[item.sid] = _replay_result(item)
    return tuple(completed[sid] for sid, _atoms in ordered)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path: Path, *, include_data: bool) -> _InputSnapshot:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if include_data:
        data = resolved.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
    else:
        data = None
        digest = _sha256_file(resolved)
    return _InputSnapshot(path=resolved, sha256=digest, data=data)


def _strict_json_document(data: bytes, *, role: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{role} contains nonstandard JSON constant {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {role} document")
    return payload


def _hash_record(snapshot: _InputSnapshot) -> dict[str, str]:
    return {
        "path": str(snapshot.path.resolve()),
        "sha256": snapshot.sha256,
    }


def _validate_upstream_manifest(
    manifest: Mapping[str, object],
    *,
    features: _InputSnapshot,
    frames: _InputSnapshot,
    checkpoint: _InputSnapshot,
) -> None:
    if manifest.get("protocol") != UPSTREAM_FEATURE_PROTOCOL:
        raise ValueError("feature manifest protocol mismatch")
    if manifest.get("mode") != "development":
        raise ValueError("feature manifest mode must be development")
    if manifest.get("production_protocol_eligible") is not True:
        raise ValueError("feature manifest must be production-protocol eligible")
    outputs = manifest.get("outputs_sha256")
    if outputs != {features.path.name: features.sha256}:
        raise ValueError("feature manifest feature hash closure mismatch")
    inputs = manifest.get("inputs_sha256")
    if not isinstance(inputs, dict) or inputs.get("frames") != _hash_record(frames):
        raise ValueError("feature manifest frames hash/path mismatch")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict) or checkpoints.get("m5") != _hash_record(
        checkpoint
    ):
        raise ValueError("feature manifest checkpoint hash/path mismatch")
    loaded = manifest.get("predictor_loaded_checkpoint_sha256")
    if not isinstance(loaded, dict) or loaded.get("m5") != checkpoint.sha256:
        raise ValueError("feature manifest loaded checkpoint hash mismatch")


def _require_frozen_next8_inputs(
    snapshots: Mapping[str, _InputSnapshot],
) -> None:
    observed = {
        "committee_features": snapshots["features"].sha256,
        "threshold_roles": snapshots["roles"].sha256,
        "frames": snapshots["frames"].sha256,
        "feature_manifest": snapshots["feature_manifest"].sha256,
    }
    if observed != FROZEN_NEXT8_INPUT_SHA256:
        mismatched = sorted(
            role
            for role, expected in FROZEN_NEXT8_INPUT_SHA256.items()
            if observed.get(role) != expected
        )
        raise ValueError(
            f"production inputs do not equal frozen next8 sealed input identities: {mismatched}"
        )


def _exact_string_column(table: pd.DataFrame, column: str, *, role: str) -> None:
    if table[column].isna().any() or not table[column].map(
        lambda value: type(value) is str and bool(value)
    ).all():
        raise ValueError(f"{role} {column} must contain nonempty exact strings")


def _selected_gate_rows(
    features_data: bytes,
    roles_data: bytes,
) -> tuple[pd.DataFrame, int, int]:
    features = pd.read_parquet(io.BytesIO(features_data))
    roles = pd.read_parquet(io.BytesIO(roles_data))
    feature_required = {"sid", "rk", "stage", "strict_x0_ok"}
    role_required = {"sid", "rk", "stage", "threshold_role"}
    if missing := feature_required - set(features.columns):
        raise ValueError(f"features are missing columns: {sorted(missing)}")
    if missing := role_required - set(roles.columns):
        raise ValueError(f"role assignments are missing columns: {sorted(missing)}")
    for table, role, columns in (
        (features, "features", ("sid", "rk", "stage")),
        (roles, "role assignments", ("sid", "rk", "stage", "threshold_role")),
    ):
        for column in columns:
            _exact_string_column(table, column, role=role)
        if table["sid"].duplicated().any():
            raise ValueError(f"{role} sid values must be unique")
    if not features["strict_x0_ok"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("features strict_x0_ok must be exactly boolean")
    allowed_roles = {"threshold_fit", "development_gate"}
    if not set(roles["threshold_role"]).issubset(allowed_roles):
        raise ValueError("role assignments contain unsupported threshold_role")
    selected_roles = roles.loc[
        roles["threshold_role"].eq("development_gate"),
        ["sid", "rk", "stage", "threshold_role"],
    ].copy()
    if selected_roles.empty:
        raise ValueError("development_gate selection must be nonempty")
    selected = selected_roles.merge(
        features[["sid", "rk", "stage", "strict_x0_ok"]],
        on="sid",
        how="left",
        suffixes=("_role", "_feature"),
        validate="one_to_one",
        indicator=True,
    )
    if not selected["_merge"].eq("both").all():
        raise ValueError("development_gate sid coverage differs from features")
    if not selected["rk_role"].eq(selected["rk_feature"]).all():
        raise ValueError("development_gate rk values differ between inputs")
    if not selected["stage_role"].eq(selected["stage_feature"]).all():
        raise ValueError("development_gate stage values differ between inputs")
    if not selected["stage_role"].eq("threshold_calibration").all():
        raise ValueError("development_gate rows must be threshold_calibration")
    selected = selected.rename(
        columns={"rk_role": "rk", "stage_role": "stage"}
    ).drop(columns=["rk_feature", "stage_feature", "_merge"])
    selected = selected.sort_values("sid", kind="stable").reset_index(drop=True)
    return selected, len(features), len(roles)


def _strict_structures(
    selected: pd.DataFrame,
    frames_data: bytes,
) -> tuple[list[str], list[Atoms]]:
    strict_rows = selected.loc[selected["strict_x0_ok"].astype(bool)]
    sids: list[str] = []
    structures: list[Atoms] = []
    try:
        archive_context = zipfile.ZipFile(io.BytesIO(frames_data))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid x0 frame archive") from exc
    with archive_context as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            sid = Path(name).stem
            if sid in members:
                raise ValueError(f"frame archive has duplicate member stem sid: {sid}")
            members[sid] = name
        for record in strict_rows.to_dict("records"):
            sid = record["sid"]
            member = members.get(sid)
            if member is None:
                raise ValueError(f"strict development_gate sid lacks x0 frame: {sid}")
            try:
                atoms = frame_to_atoms(archive.read(member).decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"invalid strict x0 frame for sid {sid}") from exc
            numbers = np.asarray(atoms.get_atomic_numbers())
            positions = np.asarray(atoms.get_positions(), dtype=float)
            cell = np.asarray(atoms.cell, dtype=float)
            if (
                len(atoms) < 1
                or numbers.shape != (len(atoms),)
                or np.any(numbers <= 0)
                or positions.shape != (len(atoms), 3)
                or cell.shape != (3, 3)
                or not np.all(np.isfinite(positions))
                or not np.all(np.isfinite(cell))
            ):
                raise ValueError(f"invalid strict x0 geometry for sid {sid}")
            sids.append(sid)
            structures.append(_sanitize_structure(atoms))
    return sids, structures


def _result_row(
    record: Mapping[str, object],
    batch_result: BatchLRRCResult | None,
) -> dict[str, object]:
    if batch_result is None:
        return {
            "sid": record["sid"],
            "rk": record["rk"],
            "stage": record["stage"],
            "threshold_role": record["threshold_role"],
            "strict_x0_ok": False,
            "natoms": 0,
            "lrrc_status": LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY.value,
            "lrrc_negative": None,
            "d_star_angstrom": np.nan,
            "h_angstrom": np.nan,
            "kappa_h_ev_per_a2": np.nan,
            "kappa_h2_ev_per_a2": np.nan,
            "kappa_r_ev_per_a2": np.nan,
            "error_proxy_ev_per_a2": np.nan,
            "u_num_ev_per_a2": np.nan,
            "force_call_count": 0,
            "error": "nonstrict_x0",
        }
    result = batch_result.result
    return {
        "sid": record["sid"],
        "rk": record["rk"],
        "stage": record["stage"],
        "threshold_role": record["threshold_role"],
        "strict_x0_ok": True,
        "natoms": record["natoms"],
        "lrrc_status": result.status.value,
        "lrrc_negative": result.negative,
        "d_star_angstrom": result.d_star,
        "h_angstrom": result.h,
        "kappa_h_ev_per_a2": result.kappa_h,
        "kappa_h2_ev_per_a2": result.kappa_h2,
        "kappa_r_ev_per_a2": result.kappa_r,
        "error_proxy_ev_per_a2": result.error_proxy,
        "u_num_ev_per_a2": result.u_num,
        "force_call_count": batch_result.force_call_count,
        "error": result.error or "",
    }


def _strict_output_table(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in ("sid", "rk", "stage", "threshold_role", "lrrc_status", "error"):
        table[column] = table[column].astype("string")
    table["strict_x0_ok"] = table["strict_x0_ok"].astype("bool")
    table["lrrc_negative"] = table["lrrc_negative"].astype("boolean")
    table["natoms"] = table["natoms"].astype("int64")
    table["force_call_count"] = table["force_call_count"].astype("int64")
    for column in OUTPUT_COLUMNS[8:15]:
        table[column] = table[column].astype("float64")
    table = table.sort_values("sid", kind="stable").reset_index(drop=True)
    if table["sid"].duplicated().any():
        raise ValueError("output sid values must be unique")
    successful = table["lrrc_status"].eq(LRRCStatus.OK.value)
    if not np.isfinite(table.loc[successful, list(OUTPUT_COLUMNS[8:15])]).all().all():
        raise BatchLRRCError("successful LRRC rows must contain finite diagnostics")
    return table


def _runtime_identity(device: str) -> dict[str, object]:
    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    runtime: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "ase_version": version("ase"),
        "mattersim_version": version("mattersim"),
        "device": device,
    }
    try:
        import torch
    except Exception:
        runtime.update(
            {
                "torch_version": "unavailable",
                "cuda_available": "unavailable",
                "cuda_version": "unavailable",
                "gpu_name": "unavailable",
            }
        )
        return runtime

    runtime["torch_version"] = str(getattr(torch, "__version__", "unavailable"))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    runtime["cuda_version"] = (
        str(cuda_version) if cuda_version is not None else "unavailable"
    )
    try:
        cuda_available: bool | str = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = "unavailable"
    runtime["cuda_available"] = cuda_available
    if cuda_available is True:
        try:
            gpu_device = device if device.strip().lower().startswith("cuda") else None
            runtime["gpu_name"] = str(torch.cuda.get_device_name(gpu_device))
        except Exception:
            runtime["gpu_name"] = "unavailable"
    else:
        runtime["gpu_name"] = "unavailable"
    return runtime


def _write_and_publish(
    table: pd.DataFrame,
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
        reloaded = pd.read_parquet(table_path)
        if list(reloaded.columns) != list(OUTPUT_COLUMNS) or len(reloaded) != len(table):
            raise RuntimeError("staged LRRC parquet failed schema validation")
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


def _load_indexed_predictor_from_checkpoint(
    *,
    checkpoint_path: Path,
    expected_sha256: str,
    potential_class: Any,
    build_dataloader: Any,
    batch_to_dict: Any,
    torch_module: Any,
    device: str,
    batch_size: int,
) -> tuple[_IndexedMatterSimPredictor, str]:
    """Load MatterSim from an immutable anonymous fd sealed after one source read."""

    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    close_on_exec = getattr(os, "MFD_CLOEXEC", 0x0001)
    memfd_create = getattr(os, "memfd_create", None)
    if callable(memfd_create):
        file_descriptor = memfd_create(
            "next10-mattersim-checkpoint",
            flags=allow_sealing | close_on_exec,
        )
    else:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc_memfd_create = getattr(libc, "memfd_create", None)
        if libc_memfd_create is None:
            raise BatchLRRCError("immutable memfd checkpoint loading is unavailable")
        libc_memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        libc_memfd_create.restype = ctypes.c_int
        file_descriptor = int(
            libc_memfd_create(
                b"next10-mattersim-checkpoint",
                allow_sealing | close_on_exec,
            )
        )
        if file_descriptor < 0:
            error_number = ctypes.get_errno()
            raise BatchLRRCError(
                f"memfd_create failed: {os.strerror(error_number)}"
            )
    sealed = Path(f"/proc/self/fd/{file_descriptor}")
    digest = hashlib.sha256()
    try:
        os.fchmod(file_descriptor, 0o600)
        with source.open("rb") as input_stream:
            for chunk in iter(lambda: input_stream.read(1 << 20), b""):
                digest.update(chunk)
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(file_descriptor, remaining)
                    if written <= 0:
                        raise BatchLRRCError("checkpoint memfd write made no progress")
                    remaining = remaining[written:]
        os.fsync(file_descriptor)
        loaded_sha256 = digest.hexdigest()
        if loaded_sha256 != expected_sha256:
            raise BatchLRRCError("checkpoint streaming-copy hash mismatch")
        add_seals = getattr(fcntl, "F_ADD_SEALS", 1033)
        get_seals = getattr(fcntl, "F_GET_SEALS", 1034)
        required_seals = (
            getattr(fcntl, "F_SEAL_WRITE", 0x0008)
            | getattr(fcntl, "F_SEAL_GROW", 0x0004)
            | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
            | getattr(fcntl, "F_SEAL_SEAL", 0x0001)
        )
        fcntl.fcntl(file_descriptor, add_seals, required_seals)
        actual_seals = int(fcntl.fcntl(file_descriptor, get_seals))
        if actual_seals & required_seals != required_seals:
            raise BatchLRRCError("checkpoint memfd is not fully immutable")
        os.fchmod(file_descriptor, 0o400)
        if _sha256_file(sealed) != loaded_sha256:
            raise BatchLRRCError("sealed checkpoint hash mismatch before model load")
        try:
            potential = potential_class.from_checkpoint(
                str(sealed),
                device=device,
                load_training_state=False,
            )
        except Exception as exc:
            raise BatchLRRCError(
                f"MatterSim checkpoint load failed: {type(exc).__name__}: {exc}"
            ) from None
        if _sha256_file(sealed) != loaded_sha256:
            raise BatchLRRCError("sealed checkpoint changed during model load")
        predictor = _IndexedMatterSimPredictor(
            potential=potential,
            build_dataloader=build_dataloader,
            batch_to_dict=batch_to_dict,
            torch_module=torch_module,
            device=device,
            batch_size=batch_size,
        )
        return predictor, loaded_sha256
    finally:
        os.close(file_descriptor)


def _production_predictor(
    checkpoint_path: Path,
    *,
    device: str,
    batch_size: int,
) -> tuple[_IndexedMatterSimPredictor, str]:
    try:
        if importlib.metadata.version("mattersim") != "1.2.3":
            raise BatchLRRCError("production adapter requires MatterSim 1.2.3")
        import torch
        from mattersim.datasets.utils.build import build_dataloader
        from mattersim.forcefield import Potential
        from mattersim.forcefield.potential import batch_to_dict
    except BatchLRRCError:
        raise
    except Exception as exc:
        raise BatchLRRCError(
            f"could not import MatterSim 1.2.3 adapter: {type(exc).__name__}: {exc}"
        ) from None
    return _load_indexed_predictor_from_checkpoint(
        checkpoint_path=checkpoint_path,
        expected_sha256=FROZEN_M5_SHA256,
        potential_class=Potential,
        build_dataloader=build_dataloader,
        batch_to_dict=batch_to_dict,
        torch_module=torch,
        device=device,
        batch_size=batch_size,
    )


def _validated_builtin_telemetry(
    predictor: object,
    *,
    device: str,
    expected_evaluations: int,
) -> dict[str, object]:
    try:
        telemetry = dict(predictor.telemetry)
    except Exception as exc:
        raise RuntimeError("builtin indexed adapter lacks telemetry") from exc
    model_device = telemetry.get("model_parameter_device")
    result_devices = telemetry.get("result_tensor_devices")
    forward_calls = telemetry.get("forward_calls")
    evaluations = telemetry.get("evaluations")
    peak = telemetry.get("peak_cuda_memory_bytes")
    if (
        type(model_device) is not str
        or not isinstance(result_devices, list)
        or not result_devices
        or not all(type(value) is str and value for value in result_devices)
        or type(forward_calls) is not int
        or forward_calls <= 0
        or type(evaluations) is not int
        or evaluations != expected_evaluations
        or type(peak) is not int
        or peak < 0
    ):
        raise RuntimeError("builtin indexed adapter telemetry is incomplete")
    normalized = device.lower()
    if normalized == "cpu":
        if model_device != "cpu" or result_devices != ["cpu"] or peak != 0:
            raise RuntimeError("production CPU telemetry does not prove CPU execution")
    elif normalized == "cuda" or (
        normalized.startswith("cuda:")
        and normalized.removeprefix("cuda:").isdigit()
    ):
        def canonical_cuda(value: str) -> str | None:
            lowered = value.strip().lower()
            if lowered == "cuda":
                return "cuda:0"
            if lowered.startswith("cuda:") and lowered.removeprefix("cuda:").isdigit():
                return f"cuda:{int(lowered.removeprefix('cuda:'))}"
            return None

        model_canonical = canonical_cuda(model_device)
        result_canonical = [canonical_cuda(value) for value in result_devices]
        observed_devices = {model_canonical, *result_canonical}
        requested_canonical = canonical_cuda(normalized)
        if (
            None in observed_devices
            or len(observed_devices) != 1
            or (
                normalized != "cuda"
                and model_canonical != requested_canonical
            )
            or peak <= 0
        ):
            raise RuntimeError("production CUDA telemetry does not prove CUDA execution")
    else:
        raise ValueError("production device must be 'cpu', 'cuda', or 'cuda:N'")
    return telemetry


def run_label_free_features(
    *,
    features_path: Path,
    role_assignments_path: Path,
    frames_zip_path: Path,
    feature_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    predictor: BatchForcePredictor | None = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, object]:
    """Seal development-gate LRRC diagnostics without opening endpoint outcomes."""

    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive exact integer")
    device = str(device).strip()
    if not device:
        raise ValueError("device must be a nonempty string")
    input_paths = {
        "features": Path(features_path),
        "roles": Path(role_assignments_path),
        "frames": Path(frames_zip_path),
        "feature_manifest": Path(feature_manifest_path),
        "checkpoint": Path(checkpoint_path),
    }
    snapshots = {
        role: _snapshot(path, include_data=role != "checkpoint")
        for role, path in input_paths.items()
    }
    feature_manifest = _strict_json_document(
        snapshots["feature_manifest"].data or b"", role="feature manifest"
    )
    _validate_upstream_manifest(
        feature_manifest,
        features=snapshots["features"],
        frames=snapshots["frames"],
        checkpoint=snapshots["checkpoint"],
    )
    if predictor is None:
        _require_frozen_next8_inputs(snapshots)
        if snapshots["checkpoint"].sha256 != FROZEN_M5_SHA256:
            raise ValueError("production checkpoint does not equal frozen MatterSim 5M")

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        relative: repository_root / relative for relative in _EXECUTED_SOURCE_RELATIVE
    }
    source_sha256 = {
        relative: _sha256_file(path) for relative, path in source_paths.items()
    }
    selected, feature_rows, role_rows = _selected_gate_rows(
        snapshots["features"].data or b"", snapshots["roles"].data or b""
    )
    strict_sids, structures = _strict_structures(
        selected, snapshots["frames"].data or b""
    )
    runtime_identity = _runtime_identity(device)
    if predictor is None:
        if runtime_identity.get("mattersim_version") != "1.2.3":
            raise RuntimeError("production runtime requires MatterSim 1.2.3")
        if (
            device.lower().startswith("cuda")
            and runtime_identity.get("cuda_available") is not True
        ):
            raise RuntimeError("CUDA was requested but is unavailable at runtime")
        active_predictor, loaded_checkpoint_sha256 = _production_predictor(
            snapshots["checkpoint"].path,
            device=device,
            batch_size=batch_size,
        )
        adapter_mode = "builtin_indexed_mattersim"
    else:
        active_predictor = predictor
        adapter_mode = "injected_test_double"
        loaded_checkpoint_sha256 = None

    predictor_calls = 0

    def counting_predictor(batch: list[Atoms]) -> BatchPrediction:
        nonlocal predictor_calls
        predictor_calls += 1
        return active_predictor(batch)

    started = time.perf_counter()
    batch_results = evaluate_lrrc_batch(strict_sids, structures, counting_predictor)
    elapsed = time.perf_counter() - started
    results_by_sid = {item.sid: item for item in batch_results}
    natoms_by_sid = {
        sid: len(atoms) for sid, atoms in zip(strict_sids, structures, strict=True)
    }
    rows: list[dict[str, object]] = []
    for record in selected.to_dict("records"):
        if bool(record["strict_x0_ok"]):
            record["natoms"] = natoms_by_sid[record["sid"]]
            rows.append(_result_row(record, results_by_sid[record["sid"]]))
        else:
            rows.append(_result_row(record, None))
    table = _strict_output_table(rows)
    force_evaluations = int(table["force_call_count"].sum())
    if predictor is None:
        builtin_telemetry = _validated_builtin_telemetry(
            active_predictor,
            device=device,
            expected_evaluations=force_evaluations,
        )
        production_eligible = True
        adapter_manifest = {
            "mode": adapter_mode,
            "index_alignment": "sid_indexed_exact_one_to_one",
            "index_alignment_verified": True,
            "device": device,
            "batch_size": batch_size,
            "model_parameter_device": builtin_telemetry["model_parameter_device"],
            "result_tensor_devices": builtin_telemetry["result_tensor_devices"],
            "evaluations": builtin_telemetry["evaluations"],
        }
        forward_calls: int | None = int(builtin_telemetry["forward_calls"])
        peak_cuda_memory_bytes: int | None = int(
            builtin_telemetry["peak_cuda_memory_bytes"]
        )
    else:
        production_eligible = False
        adapter_manifest = {
            "mode": adapter_mode,
            "index_alignment": "injected_batch_force_predictor_declared_aligned",
            "index_alignment_verified": False,
            "device": device,
            "batch_size": batch_size,
            "model_parameter_device": None,
            "result_tensor_devices": [],
            "evaluations": force_evaluations,
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

    status = table["lrrc_status"]
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "development_gate",
        "labels_opened": False,
        "selection": {
            "stage": "threshold_calibration",
            "threshold_role": "development_gate",
        },
        "adapter": adapter_manifest,
        "predictor_loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "production_protocol_eligible": production_eligible,
        "evidence_role": (
            "label_free_lrrc_feature_generation"
            if production_eligible
            else "testing_only_not_scientific_evidence"
        ),
        "runtime": runtime_identity,
        "inputs_sha256": {
            "committee_features": _hash_record(snapshots["features"]),
            "threshold_roles": _hash_record(snapshots["roles"]),
            "frames": _hash_record(snapshots["frames"]),
            "feature_manifest": _hash_record(snapshots["feature_manifest"]),
            "checkpoint": _hash_record(snapshots["checkpoint"]),
        },
        "executed_source_sha256": source_sha256,
        "integrity": {"prepublish_rehash": "passed"},
        "feature_columns": list(OUTPUT_COLUMNS),
        "counts": {
            "feature_rows": feature_rows,
            "role_assignment_rows": role_rows,
            "selected_rows": len(table),
            "strict_rows": len(strict_sids),
            "nonstrict_rows": len(table) - len(strict_sids),
            "ok_rows": int(status.eq(LRRCStatus.OK.value).sum()),
            "stationary_rows": int(status.eq(LRRCStatus.STATIONARY_FALLBACK.value).sum()),
            "abstained_rows": int(status.str.startswith("abstain_").sum()),
            "batch_predictor_calls": predictor_calls,
            "force_evaluations": force_evaluations,
        },
        "execution": {
            "batch_predictor_calls": predictor_calls,
            "forward_calls": forward_calls,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "wall_time_seconds": elapsed,
        },
        "scientific_improvement_claim": False,
    }
    return _write_and_publish(
        table,
        manifest,
        output_dir,
        verify_unchanged=verify_unchanged,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for one label-free next10 LRRC feature publication."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--role-assignments-path", type=Path, required=True)
    parser.add_argument("--frames-zip-path", type=Path, required=True)
    parser.add_argument("--feature-manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    arguments = parser.parse_args(argv)
    run_label_free_features(
        features_path=arguments.features_path,
        role_assignments_path=arguments.role_assignments_path,
        frames_zip_path=arguments.frames_zip_path,
        feature_manifest_path=arguments.feature_manifest_path,
        checkpoint_path=arguments.checkpoint_path,
        output_dir=arguments.output_dir,
        predictor=None,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
