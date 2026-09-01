#!/usr/bin/env python3
"""Build label-free dual-checkpoint MatterSim x0 committee features."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
from numbers import Real
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Literal, Protocol
import zipfile

import numpy as np
import pandas as pd

from src.next6_mattersim_baseline import frame_to_atoms


DEVELOPMENT_STAGES = (
    "search_calibration",
    "formula_selection",
    "threshold_calibration",
)
TEST_STAGE = "test"
MODEL_KEYS = ("m1", "m5")
PROTOCOL = "2026-08-01-mattersim-dual-checkpoint-x0-v1"
OUTPUT_NAME = "mattersim_committee_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_CHECKPOINT_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "m1": "28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be",
        "m5": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5",
    }
)

_MFD_CLOEXEC = int(getattr(os, "MFD_CLOEXEC", 0x0001))
_MFD_ALLOW_SEALING = int(getattr(os, "MFD_ALLOW_SEALING", 0x0002))
_F_ADD_SEALS = int(getattr(fcntl, "F_ADD_SEALS", 1033))
_F_GET_SEALS = int(getattr(fcntl, "F_GET_SEALS", 1034))
_F_SEAL_SEAL = int(getattr(fcntl, "F_SEAL_SEAL", 0x0001))
_F_SEAL_SHRINK = int(getattr(fcntl, "F_SEAL_SHRINK", 0x0002))
_F_SEAL_GROW = int(getattr(fcntl, "F_SEAL_GROW", 0x0004))
_F_SEAL_WRITE = int(getattr(fcntl, "F_SEAL_WRITE", 0x0008))


@dataclass(frozen=True)
class ModelFeaturePrediction:
    """Label-free x0 scalars returned for one structure and one model."""

    energy_total_ev: float
    fmax_ev_per_a: float
    frms_ev_per_a: float


@dataclass(frozen=True)
class ModelEvaluationTelemetry:
    """Actual work reported by one model adapter for one batch.

    ``attempted_evaluations`` counts row-level model evaluations, including
    retries. ``retry_count`` counts the repeated row evaluations among them.
    ``successful_evaluations`` counts distinct requested rows that produced a
    usable final prediction, never successful retry attempts separately.
    ``forward_calls`` counts actual model forward invocations and may be lower
    than row evaluations because a forward can be batched.
    """

    requested_rows: int
    attempted_evaluations: int
    successful_evaluations: int
    forward_calls: int
    retry_count: int


@dataclass(frozen=True)
class CommitteePredictionBatch:
    """Predictions, loaded model identities, and measured adapter work."""

    predictions: Sequence[Mapping[str, ModelFeaturePrediction | None]]
    loaded_checkpoint_sha256: Mapping[str, str]
    telemetry: Mapping[str, ModelEvaluationTelemetry]


class CommitteePredictor(Protocol):
    """Injected, model-agnostic dual-checkpoint predictor contract."""

    def __call__(
        self, structures: Sequence[object]
    ) -> CommitteePredictionBatch: ...


class FatalCommitteePredictionError(RuntimeError):
    """An adapter failure after which inference must stop without publication."""


def _require_frozen_checkpoint_identity(
    checkpoint_sha256: Mapping[str, str],
) -> None:
    if set(checkpoint_sha256) != set(MODEL_KEYS) or set(
        FROZEN_CHECKPOINT_SHA256
    ) != set(MODEL_KEYS):
        raise FatalCommitteePredictionError(
            "frozen checkpoint identity must contain exactly m1 and m5"
        )
    for model in MODEL_KEYS:
        if checkpoint_sha256[model] != FROZEN_CHECKPOINT_SHA256[model]:
            raise FatalCommitteePredictionError(
                f"frozen checkpoint identity mismatch for {model}"
            )


def _load_mattersim_api() -> tuple[Any, Callable[..., Any]]:
    """Import MatterSim only when the real adapter is first used."""

    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield import Potential

    return Potential, build_dataloader


def _cuda_is_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_inference_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        return "cuda" if _cuda_is_available() else "cpu"
    if requested == "cpu":
        return requested
    if requested == "cuda" or (
        requested.startswith("cuda:")
        and requested.removeprefix("cuda:").isdigit()
    ):
        if not _cuda_is_available():
            raise FatalCommitteePredictionError(
                f"CUDA device requested but CUDA is unavailable: {device}"
            )
        return requested
    raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")


def _empty_cuda_cache() -> None:
    try:
        import torch

        if bool(torch.cuda.is_available()):
            torch.cuda.empty_cache()
    except Exception:
        return


def _positive_batch_size(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("batch_size must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("batch_size must be a positive integer")
    return normalized


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _create_memfd(name: str) -> int:
    flags = _MFD_CLOEXEC | _MFD_ALLOW_SEALING
    create = getattr(os, "memfd_create", None)
    if callable(create):
        return int(create(name, flags=flags))
    try:
        libc_create = ctypes.CDLL(None, use_errno=True).memfd_create
    except (AttributeError, OSError) as exc:
        raise FatalCommitteePredictionError(
            "checkpoint sealing requires Linux memfd support"
        ) from exc
    libc_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    libc_create.restype = ctypes.c_int
    ctypes.set_errno(0)
    fd = int(libc_create(os.fsencode(name), flags))
    if fd >= 0:
        return fd
    error_number = ctypes.get_errno() or errno.EIO
    raise OSError(error_number, os.strerror(error_number))


def _sealed_checkpoint_snapshot(
    path: Path, *, expected_sha256: str, model_key: str
) -> tuple[int, str]:
    """Copy a checkpoint into a sealed Linux memfd and hash loaded bytes."""

    if sys.platform != "linux":
        raise FatalCommitteePredictionError(
            "checkpoint sealing requires Linux memfd support"
        )
    try:
        fd = _create_memfd(f"mattersim-{model_key}")
    except OSError as exc:
        raise FatalCommitteePredictionError(
            f"checkpoint memfd creation failed for {model_key}"
        ) from exc
    try:
        try:
            with Path(path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    view = memoryview(chunk)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise OSError("short write to checkpoint memfd")
                        view = view[written:]
        except OSError as exc:
            raise FatalCommitteePredictionError(
                f"checkpoint snapshot copy failed for {model_key}"
            ) from exc

        seal_mask = (
            _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
        )
        try:
            fcntl.fcntl(fd, _F_ADD_SEALS, seal_mask)
            applied = int(fcntl.fcntl(fd, _F_GET_SEALS))
        except OSError as exc:
            raise FatalCommitteePredictionError(
                f"checkpoint sealing failed for {model_key}"
            ) from exc
        if applied & seal_mask != seal_mask:
            raise FatalCommitteePredictionError(
                f"checkpoint sealing incomplete for {model_key}"
            )
        actual_sha256 = _sha256_fd(fd)
        if actual_sha256 != expected_sha256:
            raise FatalCommitteePredictionError(
                f"checkpoint SHA-256 mismatch for {model_key}"
            )
        return fd, actual_sha256
    except BaseException:
        os.close(fd)
        raise


class _BatchPredictionError(RuntimeError):
    """A recoverable batch-level contract failure eligible for bisection."""


class _ForwardTelemetryTracker:
    def __init__(self) -> None:
        self._attempts: list[int] = []
        self._pending_indices: tuple[int, ...] | None = None
        self.forward_calls = 0
        self.retry_count = 0

    def start(self, requested_rows: int) -> None:
        self._attempts = [0] * requested_rows
        self._pending_indices = None
        self.forward_calls = 0
        self.retry_count = 0

    def set_pending(self, indices: Sequence[int]) -> None:
        if self._pending_indices is not None:
            raise FatalCommitteePredictionError(
                "overlapping MatterSim forward telemetry contexts"
            )
        self._pending_indices = tuple(indices)

    def clear_pending(self) -> None:
        self._pending_indices = None

    def record_forward(self) -> None:
        if self._pending_indices is None:
            raise FatalCommitteePredictionError(
                "MatterSim forward occurred outside a tracked batch"
            )
        self.forward_calls += 1
        for index in self._pending_indices:
            if self._attempts[index] > 0:
                self.retry_count += 1
            self._attempts[index] += 1

    def telemetry(
        self, *, requested_rows: int, successful_evaluations: int
    ) -> ModelEvaluationTelemetry:
        return ModelEvaluationTelemetry(
            requested_rows=requested_rows,
            attempted_evaluations=sum(self._attempts),
            successful_evaluations=successful_evaluations,
            forward_calls=self.forward_calls,
            retry_count=self.retry_count,
        )


_FORWARD_WRAPPER_ATTRIBUTE = "_next8_mattersim_forward_wrapper"


class _ForwardWrapperHandle:
    def __init__(
        self,
        model: object,
        original_forward: Callable[..., object],
        wrapped_forward: Callable[..., object],
        *,
        had_instance_forward: bool,
        original_instance_forward: object,
    ) -> None:
        self._model: object | None = model
        self._original_forward: Callable[..., object] | None = original_forward
        self._wrapped_forward: Callable[..., object] | None = wrapped_forward
        self._had_instance_forward = had_instance_forward
        self._original_instance_forward = original_instance_forward
        self._active = True

    def remove(self) -> None:
        if not self._active:
            return
        model = self._model
        try:
            if model is None:
                return
            marker = getattr(model, _FORWARD_WRAPPER_ATTRIBUTE, None)
            current_forward = getattr(model, "forward", None)
            if marker is self or current_forward is self._wrapped_forward:
                if self._had_instance_forward:
                    setattr(
                        model,
                        "forward",
                        self._original_instance_forward,
                    )
                else:
                    delattr(model, "forward")
            if marker is self:
                delattr(model, _FORWARD_WRAPPER_ATTRIBUTE)
        finally:
            self._active = False
            self._model = None
            self._original_forward = None
            self._wrapped_forward = None
            self._original_instance_forward = None


def _install_forward_wrapper(
    model: object, tracker: _ForwardTelemetryTracker
) -> _ForwardWrapperHandle:
    if getattr(model, _FORWARD_WRAPPER_ATTRIBUTE, None) is not None:
        raise FatalCommitteePredictionError(
            "MatterSim model forward wrapper is already installed"
        )
    original_forward = getattr(model, "forward", None)
    if not callable(original_forward):
        raise FatalCommitteePredictionError(
            "MatterSim model has no callable forward method"
        )

    def wrapped_forward(*args: object, **kwargs: object) -> object:
        tracker.record_forward()
        return original_forward(*args, **kwargs)

    instance_attributes = getattr(model, "__dict__", {})
    had_instance_forward = "forward" in instance_attributes
    original_instance_forward = instance_attributes.get("forward")
    handle = _ForwardWrapperHandle(
        model,
        original_forward,
        wrapped_forward,
        had_instance_forward=had_instance_forward,
        original_instance_forward=original_instance_forward,
    )
    try:
        setattr(model, "forward", wrapped_forward)
        setattr(model, _FORWARD_WRAPPER_ATTRIBUTE, handle)
    except Exception as exc:
        try:
            marker = getattr(model, _FORWARD_WRAPPER_ATTRIBUTE, None)
            if marker is handle:
                delattr(model, _FORWARD_WRAPPER_ATTRIBUTE)
            if had_instance_forward:
                setattr(model, "forward", original_instance_forward)
            else:
                delattr(model, "forward")
        except Exception:
            pass
        raise FatalCommitteePredictionError(
            "MatterSim model forward wrapper installation failed"
        ) from exc
    return handle


def _potential_max_z(potential: object, *, model_key: str) -> int:
    model = getattr(potential, "model", None)
    candidates = [
        getattr(model, "max_z", None),
        getattr(potential, "max_z", None),
    ]
    model_args = getattr(model, "model_args", None)
    if isinstance(model_args, Mapping):
        candidates.append(model_args.get("max_z"))
    for value in candidates:
        if isinstance(value, (bool, np.bool_)):
            continue
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if normalized > 0 and float(value) == normalized:
            return normalized
    raise FatalCommitteePredictionError(
        f"MatterSim checkpoint {model_key} does not expose a valid model.max_z"
    )


def _potential_model_args(
    potential: object, *, model_key: str
) -> tuple[str, float, float]:
    model_name = getattr(potential, "model_name", None)
    model = getattr(potential, "model", None)
    model_args = getattr(model, "model_args", None)
    if not isinstance(model_name, str) or not model_name:
        raise FatalCommitteePredictionError(
            f"MatterSim checkpoint {model_key} has no model_name"
        )
    if not isinstance(model_args, Mapping):
        raise FatalCommitteePredictionError(
            f"MatterSim checkpoint {model_key} has no model_args"
        )
    try:
        cutoff = float(model_args["cutoff"])
        threebody_cutoff = float(model_args["threebody_cutoff"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise FatalCommitteePredictionError(
            f"MatterSim checkpoint {model_key} has invalid cutoffs"
        ) from exc
    if not np.isfinite([cutoff, threebody_cutoff]).all():
        raise FatalCommitteePredictionError(
            f"MatterSim checkpoint {model_key} has nonfinite cutoffs"
        )
    return model_name, cutoff, threebody_cutoff


def _periodic_cell_is_nondegenerate(cell: np.ndarray, pbc: np.ndarray) -> bool:
    periodic_vectors = cell[pbc]
    if len(periodic_vectors) == 0:
        return True
    scale = max(float(np.max(np.abs(periodic_vectors))), 1.0)
    tolerance = np.finfo(float).eps * scale * 100.0
    return bool(
        np.linalg.matrix_rank(periodic_vectors, tol=tolerance)
        == len(periodic_vectors)
    )


def _structure_is_supported(structure: object, *, max_z: int) -> bool:
    try:
        natoms = int(len(structure))  # type: ignore[arg-type]
        if natoms <= 0:
            return False
        numbers = np.asarray(getattr(structure, "numbers"))
        positions = np.asarray(getattr(structure, "positions"), dtype=float)
        cell = np.asarray(getattr(structure, "cell"), dtype=float)
        pbc = np.asarray(getattr(structure, "pbc"), dtype=bool)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    if numbers.shape != (natoms,) or not np.issubdtype(numbers.dtype, np.number):
        return False
    try:
        numeric_numbers = np.asarray(numbers, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not np.all(np.isfinite(numeric_numbers))
        or not np.all(numeric_numbers == np.floor(numeric_numbers))
        or not np.all((numeric_numbers >= 1) & (numeric_numbers <= max_z))
    ):
        return False
    if positions.shape != (natoms, 3) or not np.all(np.isfinite(positions)):
        return False
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        return False
    if pbc.shape != (3,) or not _periodic_cell_is_nondegenerate(cell, pbc):
        return False
    return True


def _as_numpy(value: object) -> np.ndarray:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    return np.asarray(current)


def _normalize_energy_values(energies: object, *, expected: int) -> list[float]:
    try:
        array = _as_numpy(energies)
    except Exception as exc:
        raise _BatchPredictionError("energy output is not array-like") from exc
    if array.ndim == 0 and expected == 1:
        array = array.reshape(1)
    if array.size != expected:
        raise _BatchPredictionError("energy output length mismatch")
    try:
        flattened = np.asarray(array, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _BatchPredictionError("energy output is not numeric") from exc
    if len(flattened) != expected:
        raise _BatchPredictionError("energy output shape mismatch")
    return [float(value) for value in flattened]


def _normalize_force_values(
    forces: object, *, natoms: Sequence[int]
) -> list[np.ndarray]:
    expected = len(natoms)
    if isinstance(forces, (list, tuple)):
        if len(forces) != expected:
            raise _BatchPredictionError("force output length mismatch")
        try:
            return [np.asarray(_as_numpy(value), dtype=float) for value in forces]
        except (TypeError, ValueError, OverflowError) as exc:
            raise _BatchPredictionError("force output is not numeric") from exc
    try:
        array = np.asarray(_as_numpy(forces), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _BatchPredictionError("force output is not array-like") from exc
    total_atoms = sum(natoms)
    if array.shape == (total_atoms, 3):
        boundaries = np.cumsum(natoms[:-1], dtype=int)
        return list(np.split(array, boundaries, axis=0))
    if array.ndim == 3 and array.shape[0] == expected:
        return [array[index] for index in range(expected)]
    raise _BatchPredictionError("force output shape cannot be aligned to rows")


def _normalize_prediction_outputs(
    raw: object, structures: Sequence[object]
) -> list[ModelFeaturePrediction | None]:
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise _BatchPredictionError(
            "predict_properties must return energies, forces, stresses"
        )
    energies, forces, _stresses = raw
    natoms = [int(len(structure)) for structure in structures]  # type: ignore[arg-type]
    energy_values = _normalize_energy_values(energies, expected=len(structures))
    force_values = _normalize_force_values(forces, natoms=natoms)
    results: list[ModelFeaturePrediction | None] = []
    for energy, force, count in zip(
        energy_values, force_values, natoms, strict=True
    ):
        if (
            not np.isfinite(energy)
            or force.shape != (count, 3)
            or not np.all(np.isfinite(force))
        ):
            results.append(None)
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(force, axis=1)
            fmax = float(np.max(norms))
            frms = float(np.sqrt(np.mean(norms**2)))
        if not np.isfinite([fmax, frms]).all():
            results.append(None)
            continue
        results.append(
            ModelFeaturePrediction(
                energy_total_ev=float(energy),
                fmax_ev_per_a=fmax,
                frms_ev_per_a=frms,
            )
        )
    return results


def _is_out_of_memory_error(exc: BaseException) -> bool:
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text and any(
        token in text for token in ("cuda", "gpu", "accelerator", "hip")
    )


def _is_fatal_accelerator_error(exc: BaseException) -> bool:
    if _is_out_of_memory_error(exc):
        return False
    text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}".lower()
    return any(
        token in text
        for token in (
            "cuda",
            "accelerator",
            "device-side assert",
            "device side assert",
            "illegal memory access",
            "illegal access",
            "cublas",
            "cudnn",
        )
    )


class MatterSimCommitteePredictor:
    """Lazy, fixed-device MatterSim 1M/5M batch predictor."""

    def __init__(
        self,
        *,
        checkpoints: Mapping[str, Path],
        expected_checkpoint_sha256: Mapping[str, str],
        device: str,
        batch_size: int,
    ) -> None:
        if set(checkpoints) != set(MODEL_KEYS):
            raise FatalCommitteePredictionError(
                "checkpoint paths must contain exactly m1 and m5"
            )
        if set(expected_checkpoint_sha256) != set(MODEL_KEYS):
            raise FatalCommitteePredictionError(
                "checkpoint identities must contain exactly m1 and m5"
            )
        self._checkpoint_paths = {
            model: Path(checkpoints[model]) for model in MODEL_KEYS
        }
        self._expected_hashes = {
            model: str(expected_checkpoint_sha256[model])
            for model in MODEL_KEYS
        }
        _require_frozen_checkpoint_identity(self._expected_hashes)
        if len(set(self._expected_hashes.values())) != len(MODEL_KEYS):
            raise FatalCommitteePredictionError(
                "checkpoint identities must be distinct"
            )
        for model, digest in self._expected_hashes.items():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise FatalCommitteePredictionError(
                    f"checkpoint identity for {model} is not lowercase SHA-256"
                )
        self.device = _resolve_inference_device(device)
        self.batch_size = _positive_batch_size(batch_size)
        self._potentials: dict[str, object] = {}
        self._max_z: dict[str, int] = {}
        self._model_args: dict[str, tuple[str, float, float]] = {}
        self._trackers: dict[str, _ForwardTelemetryTracker] = {}
        self._forward_handles: list[_ForwardWrapperHandle] = []
        self._loaded_hashes: dict[str, str] = {}
        self._dataloader_builder: Callable[..., object] | None = None
        self._fatal_load_error: FatalCommitteePredictionError | None = None
        self._closed = False

    def close(self) -> None:
        """Deterministically release model wrappers and loaded model state."""

        if self._closed:
            return
        cleanup_errors: list[Exception] = []
        try:
            for handle in reversed(self._forward_handles):
                try:
                    handle.remove()
                except Exception as exc:
                    cleanup_errors.append(exc)
        finally:
            self._forward_handles.clear()
            self._potentials.clear()
            self._model_args.clear()
            self._max_z.clear()
            self._trackers.clear()
            self._loaded_hashes.clear()
            self._dataloader_builder = None
            self._fatal_load_error = None
            self._closed = True
        if cleanup_errors:
            raise FatalCommitteePredictionError(
                "MatterSim predictor cleanup failed"
            ) from cleanup_errors[0]

    def __enter__(self) -> MatterSimCommitteePredictor:
        if self._closed:
            raise FatalCommitteePredictionError(
                "MatterSim predictor is closed"
            )
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> Literal[False]:
        self.close()
        return False

    def _ensure_loaded(self) -> None:
        if self._closed:
            raise FatalCommitteePredictionError(
                "MatterSim predictor is closed"
            )
        if self._fatal_load_error is not None:
            raise self._fatal_load_error
        if self._potentials:
            return
        try:
            Potential, dataloader_builder = _load_mattersim_api()
            potentials: dict[str, object] = {}
            max_z: dict[str, int] = {}
            model_args: dict[str, tuple[str, float, float]] = {}
            trackers: dict[str, _ForwardTelemetryTracker] = {}
            loaded_hashes: dict[str, str] = {}
            forward_handles: list[_ForwardWrapperHandle] = []
            for model_key in MODEL_KEYS:
                fd, actual_hash = _sealed_checkpoint_snapshot(
                    self._checkpoint_paths[model_key],
                    expected_sha256=self._expected_hashes[model_key],
                    model_key=model_key,
                )
                try:
                    potential = Potential.from_checkpoint(
                        load_path=f"/proc/self/fd/{fd}",
                        device=self.device,
                        load_training_state=False,
                    )
                except Exception as exc:
                    raise FatalCommitteePredictionError(
                        f"checkpoint load failed for {model_key}"
                    ) from exc
                finally:
                    os.close(fd)
                tracker = _ForwardTelemetryTracker()
                model = getattr(potential, "model", None)
                forward_handle = _install_forward_wrapper(model, tracker)
                forward_handles.append(forward_handle)
                potentials[model_key] = potential
                max_z[model_key] = _potential_max_z(
                    potential, model_key=model_key
                )
                model_args[model_key] = _potential_model_args(
                    potential, model_key=model_key
                )
                trackers[model_key] = tracker
                loaded_hashes[model_key] = actual_hash
            if len(set(loaded_hashes.values())) != len(MODEL_KEYS):
                raise FatalCommitteePredictionError(
                    "loaded checkpoint identities must be distinct"
                )
        except FatalCommitteePredictionError as exc:
            for handle in locals().get("forward_handles", []):
                handle.remove()
            self._fatal_load_error = exc
            raise
        except Exception as exc:
            for handle in locals().get("forward_handles", []):
                handle.remove()
            wrapped = FatalCommitteePredictionError(
                "MatterSim checkpoint initialization failed"
            )
            self._fatal_load_error = wrapped
            raise wrapped from exc

        self._potentials = potentials
        self._max_z = max_z
        self._model_args = model_args
        self._trackers = trackers
        self._forward_handles = forward_handles
        self._loaded_hashes = loaded_hashes
        self._dataloader_builder = dataloader_builder

    def _recover_or_split(
        self,
        *,
        model_key: str,
        structures: Sequence[object],
        indices: tuple[int, ...],
        predictions: list[ModelFeaturePrediction | None],
        error: BaseException,
    ) -> None:
        if isinstance(error, FatalCommitteePredictionError):
            raise error
        if _is_out_of_memory_error(error):
            _empty_cuda_cache()
        elif _is_fatal_accelerator_error(error):
            raise FatalCommitteePredictionError(
                f"fatal accelerator error for {model_key}: {error}"
            ) from error
        if len(indices) <= 1:
            return
        midpoint = len(indices) // 2
        self._predict_indices(
            model_key=model_key,
            structures=structures,
            indices=indices[:midpoint],
            predictions=predictions,
        )
        self._predict_indices(
            model_key=model_key,
            structures=structures,
            indices=indices[midpoint:],
            predictions=predictions,
        )

    def _predict_indices(
        self,
        *,
        model_key: str,
        structures: Sequence[object],
        indices: tuple[int, ...],
        predictions: list[ModelFeaturePrediction | None],
    ) -> None:
        if not indices:
            return
        potential = self._potentials[model_key]
        tracker = self._trackers[model_key]
        model_name, cutoff, threebody_cutoff = self._model_args[model_key]
        chunk = [structures[index] for index in indices]
        try:
            assert self._dataloader_builder is not None
            loader = self._dataloader_builder(
                chunk,
                model_type=model_name,
                cutoff=cutoff,
                threebody_cutoff=threebody_cutoff,
                batch_size=len(chunk),
                only_inference=True,
                shuffle=False,
            )
        except Exception as exc:
            self._recover_or_split(
                model_key=model_key,
                structures=structures,
                indices=indices,
                predictions=predictions,
                error=exc,
            )
            return

        forward_calls_before = tracker.forward_calls
        tracker.set_pending(indices)
        prediction_error: Exception | None = None
        raw: object = None
        try:
            raw = potential.predict_properties(  # type: ignore[attr-defined]
                loader,
                include_forces=True,
                include_stresses=False,
            )
        except Exception as exc:
            prediction_error = exc
        finally:
            tracker.clear_pending()
        if prediction_error is not None:
            self._recover_or_split(
                model_key=model_key,
                structures=structures,
                indices=indices,
                predictions=predictions,
                error=prediction_error,
            )
            return
        if tracker.forward_calls == forward_calls_before:
            raise FatalCommitteePredictionError(
                f"MatterSim {model_key} returned without a tracked model forward"
            )
        try:
            normalized = _normalize_prediction_outputs(raw, chunk)
        except Exception as exc:
            self._recover_or_split(
                model_key=model_key,
                structures=structures,
                indices=indices,
                predictions=predictions,
                error=exc,
            )
            return
        for index, item in zip(indices, normalized, strict=True):
            predictions[index] = item

    def _predict_one_model(
        self, model_key: str, structures: Sequence[object]
    ) -> tuple[
        list[ModelFeaturePrediction | None], ModelEvaluationTelemetry
    ]:
        tracker = self._trackers[model_key]
        tracker.start(len(structures))
        predictions: list[ModelFeaturePrediction | None] = [None] * len(
            structures
        )
        valid_indices = [
            index
            for index, structure in enumerate(structures)
            if _structure_is_supported(
                structure, max_z=self._max_z[model_key]
            )
        ]
        for start in range(0, len(valid_indices), self.batch_size):
            self._predict_indices(
                model_key=model_key,
                structures=structures,
                indices=tuple(valid_indices[start : start + self.batch_size]),
                predictions=predictions,
            )
        successful = sum(item is not None for item in predictions)
        return predictions, tracker.telemetry(
            requested_rows=len(structures),
            successful_evaluations=successful,
        )

    def __call__(
        self, structures: Sequence[object]
    ) -> CommitteePredictionBatch:
        if self._closed:
            raise FatalCommitteePredictionError(
                "MatterSim predictor is closed"
            )
        self._ensure_loaded()
        per_model: dict[str, list[ModelFeaturePrediction | None]] = {}
        telemetry: dict[str, ModelEvaluationTelemetry] = {}
        for model_key in MODEL_KEYS:
            per_model[model_key], telemetry[model_key] = self._predict_one_model(
                model_key, structures
            )
        predictions = [
            {model: per_model[model][index] for model in MODEL_KEYS}
            for index in range(len(structures))
        ]
        return CommitteePredictionBatch(
            predictions=predictions,
            loaded_checkpoint_sha256=dict(self._loaded_hashes),
            telemetry=telemetry,
        )


_BUILTIN_ADAPTER_IMPLEMENTATION = MatterSimCommitteePredictor


_OUTPUT_COLUMNS = (
    "sid",
    "rk",
    "material",
    "stage",
    "strict_x0_ok",
    "feature_state",
    "committee_feature_ok",
    "committee_feature_error",
    "natoms",
    "m1_prediction_ok",
    "m1_prediction_error",
    "m1_energy_total_ev",
    "m1_energy_ev_per_atom",
    "m1_fmax_ev_per_a",
    "m1_frms_ev_per_a",
    "m5_prediction_ok",
    "m5_prediction_error",
    "m5_energy_total_ev",
    "m5_energy_ev_per_atom",
    "m5_fmax_ev_per_a",
    "m5_frms_ev_per_a",
)


def _output_dtypes() -> dict[str, str]:
    strings = {
        "sid",
        "rk",
        "material",
        "stage",
        "feature_state",
        "committee_feature_error",
        "m1_prediction_error",
        "m5_prediction_error",
    }
    booleans = {
        "strict_x0_ok",
        "committee_feature_ok",
        "m1_prediction_ok",
        "m5_prediction_ok",
    }
    return {
        column: (
            "string"
            if column in strings
            else "bool"
            if column in booleans
            else "int64"
            if column == "natoms"
            else "float64"
        )
        for column in _OUTPUT_COLUMNS
    }


def _validated_stages(
    mode: str, stages: Sequence[str]
) -> tuple[Literal["development", "test"], tuple[str, ...]]:
    if mode not in {"development", "test"}:
        raise ValueError("mode must be 'development' or 'test'")
    if isinstance(stages, (str, bytes)):
        raise ValueError("stages must be an explicit sequence")
    selected = tuple(stages)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("stages must be nonempty and unique")
    if mode == "development":
        unknown = sorted(set(selected) - set(DEVELOPMENT_STAGES))
        if unknown:
            raise ValueError(
                "development stages contain unsupported or test values: "
                f"{unknown}"
            )
        return "development", selected
    if selected != (TEST_STAGE,):
        raise ValueError("test mode requires stages=('test',) with no mixing")
    return "test", selected


def _normalize_key_table(data: pd.DataFrame, *, role: str) -> pd.DataFrame:
    missing = {"sid", "rk"} - set(data.columns)
    if missing:
        raise ValueError(f"{role} is missing key columns: {sorted(missing)}")
    out = data.copy()
    if out[["sid", "rk"]].isna().any().any():
        raise ValueError(f"{role} sid/rk keys must be nonmissing")
    if not out["sid"].map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{role} sid keys must be exact strings")
    if not out["rk"].map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{role} rk keys must be exact strings")
    if out["sid"].duplicated().any() or out.duplicated(["sid", "rk"]).any():
        raise ValueError(f"{role} has duplicate sid/key values; sid must be unique")
    return out


def _failure_metrics(reason: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for model in MODEL_KEYS:
        result.update(
            {
                f"{model}_prediction_ok": False,
                f"{model}_prediction_error": reason,
                f"{model}_energy_total_ev": np.nan,
                f"{model}_energy_ev_per_atom": np.nan,
                f"{model}_fmax_ev_per_a": np.nan,
                f"{model}_frms_ev_per_a": np.nan,
            }
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_implementation_identity(
    target: object,
) -> tuple[dict[str, object], Path | None, str | None]:
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    source_path: Path | None = None
    source_sha256: str | None = None
    try:
        source_value = inspect.getsourcefile(target) or inspect.getfile(target)
        candidate = Path(source_value).resolve()
        if candidate.is_file():
            source_path = candidate
            source_sha256 = _sha256_file(candidate)
    except (OSError, TypeError):
        source_path = None
        source_sha256 = None
    identity: dict[str, object] = {
        "module": module if isinstance(module, str) else None,
        "qualname": qualname if isinstance(qualname, str) else None,
        "source_path": None if source_path is None else str(source_path),
        "source_sha256": source_sha256,
        "source_hash_verified": source_sha256 is not None,
    }
    return identity, source_path, source_sha256


def _implementation_identity(
    predictor: CommitteePredictor,
) -> tuple[dict[str, object], Path | None, str | None]:
    target: object
    if inspect.isfunction(predictor) or inspect.ismethod(predictor):
        target = predictor
    else:
        target = type(predictor)
    return _target_implementation_identity(target)


(
    _expected_builtin_identity,
    _EXPECTED_BUILTIN_SOURCE_PATH,
    _EXPECTED_BUILTIN_SOURCE_SHA256,
) = _target_implementation_identity(_BUILTIN_ADAPTER_IMPLEMENTATION)
_EXPECTED_BUILTIN_IDENTITY: Mapping[str, object] = MappingProxyType(
    _expected_builtin_identity
)
del _expected_builtin_identity


def _validated_builtin_implementation(
    predictor: CommitteePredictor,
) -> tuple[dict[str, object], Path, str]:
    if type(predictor) is not _BUILTIN_ADAPTER_IMPLEMENTATION:
        raise FatalCommitteePredictionError(
            "built-in MatterSim adapter must have the exact built-in identity"
        )
    identity, source_path, source_sha256 = _implementation_identity(predictor)
    if (
        source_path is None
        or source_sha256 is None
        or _EXPECTED_BUILTIN_SOURCE_PATH is None
        or _EXPECTED_BUILTIN_SOURCE_SHA256 is None
        or identity != dict(_EXPECTED_BUILTIN_IDENTITY)
        or source_path != _EXPECTED_BUILTIN_SOURCE_PATH
        or source_sha256 != _EXPECTED_BUILTIN_SOURCE_SHA256
    ):
        raise FatalCommitteePredictionError(
            "built-in MatterSim adapter implementation identity mismatch"
        )
    return identity, source_path, source_sha256


def _cuda_device_index(device: str) -> int | None:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        suffix = device.removeprefix("cuda:")
        if suffix.isdigit():
            return int(suffix)
    return None


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
    device: str, *, collect_cuda_peak: bool
) -> tuple[dict[str, object], int | None]:
    try:
        mattersim_version = importlib.metadata.version("mattersim")
    except importlib.metadata.PackageNotFoundError:
        mattersim_version = "unknown"
    runtime: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "gpu_name": None,
        "mattersim_version": mattersim_version,
        "device": str(device),
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
        index = _cuda_device_index(device)
        if available and index is not None:
            runtime["gpu_name"] = str(torch.cuda.get_device_name(index))
            if collect_cuda_peak:
                peak = int(torch.cuda.max_memory_allocated(index))
    except Exception:
        pass
    return runtime, peak


def _abstain_row(base: Mapping[str, object], reason: str) -> dict[str, object]:
    return {
        **base,
        "feature_state": "ABSTAIN",
        "committee_feature_ok": False,
        "committee_feature_error": reason,
        "natoms": 0,
        **_failure_metrics(reason),
    }


def _prediction_row(
    base: Mapping[str, object],
    natoms: int,
    prediction: Mapping[str, ModelFeaturePrediction | None],
) -> dict[str, object]:
    if set(prediction) != set(MODEL_KEYS):
        raise ValueError("prediction must contain exactly m1 and m5")
    output: dict[str, object] = {
        **base,
        "natoms": natoms,
    }
    successful_models = 0
    for model in MODEL_KEYS:
        item = prediction[model]
        if item is None:
            output.update(
                {
                    f"{model}_prediction_ok": False,
                    f"{model}_prediction_error": "model_failed",
                    f"{model}_energy_total_ev": np.nan,
                    f"{model}_energy_ev_per_atom": np.nan,
                    f"{model}_fmax_ev_per_a": np.nan,
                    f"{model}_frms_ev_per_a": np.nan,
                }
            )
            continue
        if not isinstance(item, ModelFeaturePrediction):
            raise ValueError(
                "prediction values must be ModelFeaturePrediction or None"
            )
        successful_models += 1
        energy = float(item.energy_total_ev)
        output.update(
            {
                f"{model}_prediction_ok": True,
                f"{model}_prediction_error": "",
                f"{model}_energy_total_ev": energy,
                f"{model}_energy_ev_per_atom": energy / natoms,
                f"{model}_fmax_ev_per_a": float(item.fmax_ev_per_a),
                f"{model}_frms_ev_per_a": float(item.frms_ev_per_a),
            }
        )
    if successful_models == len(MODEL_KEYS):
        output.update(
            {
                "feature_state": "READY",
                "committee_feature_ok": True,
                "committee_feature_error": "",
            }
        )
    else:
        output.update(
            {
                "feature_state": "ABSTAIN",
                "committee_feature_ok": False,
                "committee_feature_error": (
                    "predictor_failed"
                    if successful_models == 0
                    else "partial_predictor_failure"
                ),
            }
        )
    return output


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"telemetry {field} must be a nonnegative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"telemetry {field} must be a nonnegative integer")
    return normalized


def _validate_prediction_scalars(item: ModelFeaturePrediction) -> None:
    values = {
        "energy_total_ev": item.energy_total_ev,
        "fmax_ev_per_a": item.fmax_ev_per_a,
        "frms_ev_per_a": item.frms_ev_per_a,
    }
    normalized: dict[str, float] = {}
    for field, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(
                f"prediction scalar {field} must be a real non-boolean number"
            )
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"prediction scalar {field} must be finite")
        normalized[field] = numeric
    for field in ("fmax_ev_per_a", "frms_ev_per_a"):
        if normalized[field] < 0.0:
            raise ValueError(
                f"prediction force scalar {field} must be nonnegative"
            )


def _validate_prediction_batch(
    batch: object,
    *,
    requested_rows: int,
    expected_checkpoint_sha256: Mapping[str, str],
) -> tuple[
    list[Mapping[str, ModelFeaturePrediction | None]],
    dict[str, str],
    dict[str, dict[str, object]],
]:
    if not isinstance(batch, CommitteePredictionBatch):
        raise ValueError("predictor must return CommitteePredictionBatch")

    loaded = batch.loaded_checkpoint_sha256
    if not isinstance(loaded, Mapping) or set(loaded) != set(MODEL_KEYS):
        raise ValueError(
            "loaded checkpoint identity must contain exactly m1 and m5"
        )
    loaded_hashes = {model: loaded[model] for model in MODEL_KEYS}
    if not all(isinstance(value, str) for value in loaded_hashes.values()):
        raise ValueError("loaded checkpoint identity values must be SHA-256 strings")
    if len(set(loaded_hashes.values())) != len(MODEL_KEYS):
        raise ValueError("loaded checkpoint identities must be distinct")
    for model in MODEL_KEYS:
        if loaded_hashes[model] != expected_checkpoint_sha256[model]:
            raise ValueError(
                f"loaded checkpoint identity mismatch for model {model}"
            )

    predictions = list(batch.predictions)
    if len(predictions) != requested_rows:
        raise ValueError("predictor output length mismatch")
    successful_by_model = {model: 0 for model in MODEL_KEYS}
    for prediction in predictions:
        if not isinstance(prediction, Mapping) or set(prediction) != set(
            MODEL_KEYS
        ):
            raise ValueError("prediction must contain exactly m1 and m5")
        for model in MODEL_KEYS:
            item = prediction[model]
            if item is not None and not isinstance(item, ModelFeaturePrediction):
                raise ValueError(
                    "prediction values must be ModelFeaturePrediction or None"
                )
            if item is not None:
                _validate_prediction_scalars(item)
            successful_by_model[model] += int(item is not None)

    if not isinstance(batch.telemetry, Mapping) or set(batch.telemetry) != set(
        MODEL_KEYS
    ):
        raise ValueError("telemetry must contain exactly m1 and m5")
    serialized: dict[str, dict[str, object]] = {}
    for model in MODEL_KEYS:
        item = batch.telemetry[model]
        if not isinstance(item, ModelEvaluationTelemetry):
            raise ValueError(
                "telemetry values must be ModelEvaluationTelemetry"
            )
        values = {
            field: _nonnegative_integer(getattr(item, field), field=field)
            for field in (
                "requested_rows",
                "attempted_evaluations",
                "successful_evaluations",
                "forward_calls",
                "retry_count",
            )
        }
        if values["requested_rows"] != requested_rows:
            raise ValueError("telemetry requested_rows mismatch")
        attempted = values["attempted_evaluations"]
        successful = values["successful_evaluations"]
        forward_calls = values["forward_calls"]
        retries = values["retry_count"]
        initial_attempts = attempted - retries
        if (
            retries > attempted
            or initial_attempts > requested_rows
            or (retries > 0 and initial_attempts == 0)
        ):
            raise ValueError("telemetry retry/attempt counts are inconsistent")
        if successful > initial_attempts or successful > requested_rows:
            raise ValueError("telemetry success/attempt counts are inconsistent")
        if forward_calls > attempted or (attempted > 0 and forward_calls == 0):
            raise ValueError("telemetry forward/attempt counts are inconsistent")
        if successful != successful_by_model[model]:
            raise ValueError(
                "telemetry successful_evaluations mismatch predictions"
            )
        serialized[model] = {"reported": True, **values}
    return predictions, loaded_hashes, serialized


def _unreported_telemetry(requested_rows: int) -> dict[str, dict[str, object]]:
    return {
        model: {
            "reported": False,
            "requested_rows": requested_rows,
            "attempted_evaluations": None,
            "successful_evaluations": None,
            "forward_calls": None,
            "retry_count": None,
        }
        for model in MODEL_KEYS
    }


def _atomic_publish_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically publish ``source`` without ever replacing ``target``."""

    unsupported_errno = getattr(errno, "ENOTSUP", errno.EINVAL)
    if sys.platform != "linux":
        raise OSError(
            unsupported_errno,
            "atomic no-replace directory publication is unsupported",
            str(target),
        )
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise OSError(
            unsupported_errno,
            "atomic no-replace directory publication is unsupported",
            str(target),
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            f"refusing to overwrite existing output directory: {target}",
            str(target),
        )
    if error_number in {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "ENOTSUP", -1),
        getattr(errno, "EOPNOTSUPP", -1),
    }:
        raise OSError(
            error_number,
            "atomic no-replace directory publication is unsupported",
            str(target),
        )
    raise OSError(error_number, os.strerror(error_number), str(target))


def _write_and_publish(
    table: pd.DataFrame,
    manifest: Mapping[str, object],
    output_dir: Path,
    *,
    verify_unchanged: Callable[[], None],
) -> dict[str, object]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    try:
        table_path = staging_dir / OUTPUT_NAME
        table.to_parquet(table_path, index=False)
        final_manifest = {
            **dict(manifest),
            "outputs_sha256": {OUTPUT_NAME: _sha256_file(table_path)},
        }
        (staging_dir / MANIFEST_NAME).write_text(
            json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_unchanged()
        _atomic_publish_directory_no_replace(staging_dir, output_dir)
        return final_manifest
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def run_committee_features(
    frames_path: Path,
    metadata_path: Path,
    stage_assignments_path: Path,
    output_dir: Path,
    *,
    checkpoints: Mapping[str, Path],
    stages: Sequence[str],
    mode: str,
    predictor: CommitteePredictor | None = None,
    device: str = "cuda",
    batch_size: int = 32,
    allow_injected_predictor_for_testing: bool = False,
) -> dict[str, object]:
    """Create features, owning only the built-in adapter it constructs."""

    if predictor is not None and allow_injected_predictor_for_testing is not True:
        raise ValueError(
            "injected predictor requires explicit testing-only authorization"
        )
    validated_batch_size = _positive_batch_size(batch_size)
    requested_device = str(device)
    if predictor is not None:
        resolved_device = str(getattr(predictor, "device", requested_device))
        return _run_committee_features_impl(
            frames_path,
            metadata_path,
            stage_assignments_path,
            output_dir,
            checkpoints=checkpoints,
            stages=stages,
            mode=mode,
            predictor=predictor,
            device_requested=requested_device,
            device_resolved=resolved_device,
            batch_size=validated_batch_size,
            adapter_mode="injected_test_double",
            implementation_provenance=None,
            production_protocol_eligible=False,
            finalize_before_publish=None,
        )

    if set(checkpoints) != set(MODEL_KEYS):
        raise ValueError("checkpoints must contain exactly m1 and m5")
    checkpoint_paths = {key: Path(checkpoints[key]) for key in MODEL_KEYS}
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha256 = {
        model: _sha256_file(path) for model, path in checkpoint_paths.items()
    }
    if len(set(checkpoint_sha256.values())) != len(MODEL_KEYS):
        raise ValueError("declared checkpoint identities must be distinct")
    _require_frozen_checkpoint_identity(checkpoint_sha256)
    owned_predictor = MatterSimCommitteePredictor(
        checkpoints=checkpoint_paths,
        expected_checkpoint_sha256=checkpoint_sha256,
        device=requested_device,
        batch_size=validated_batch_size,
    )
    close_attempted = False

    def close_owned_predictor() -> None:
        nonlocal close_attempted
        if close_attempted:
            return
        close_attempted = True
        close = getattr(owned_predictor, "close", None)
        if not callable(close):
            raise FatalCommitteePredictionError(
                "built-in MatterSim adapter has no close method"
            )
        try:
            close()
        except FatalCommitteePredictionError:
            raise
        except Exception as exc:
            raise FatalCommitteePredictionError(
                "built-in MatterSim adapter close failed"
            ) from exc

    try:
        implementation_provenance = _validated_builtin_implementation(
            owned_predictor
        )
        resolved_device = str(
            getattr(owned_predictor, "device", requested_device)
        )
        return _run_committee_features_impl(
            frames_path,
            metadata_path,
            stage_assignments_path,
            output_dir,
            checkpoints=checkpoint_paths,
            stages=stages,
            mode=mode,
            predictor=owned_predictor,
            device_requested=requested_device,
            device_resolved=resolved_device,
            batch_size=validated_batch_size,
            adapter_mode="builtin_mattersim",
            implementation_provenance=implementation_provenance,
            production_protocol_eligible=True,
            finalize_before_publish=close_owned_predictor,
        )
    except BaseException as primary_error:
        if not close_attempted:
            try:
                close_owned_predictor()
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
        raise
    finally:
        if not close_attempted:
            close_owned_predictor()


def _run_committee_features_impl(
    frames_path: Path,
    metadata_path: Path,
    stage_assignments_path: Path,
    output_dir: Path,
    *,
    checkpoints: Mapping[str, Path],
    stages: Sequence[str],
    mode: str,
    predictor: CommitteePredictor,
    device_requested: str,
    device_resolved: str,
    batch_size: int,
    adapter_mode: Literal["builtin_mattersim", "injected_test_double"],
    implementation_provenance: tuple[
        dict[str, object], Path, str
    ]
    | None,
    production_protocol_eligible: bool,
    finalize_before_publish: Callable[[], None] | None,
) -> dict[str, object]:
    """Implementation shared by owned production and injected test adapters."""

    validated_mode, selected_stages = _validated_stages(mode, stages)
    frames_path = Path(frames_path)
    metadata_path = Path(metadata_path)
    stage_assignments_path = Path(stage_assignments_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )
    if set(checkpoints) != set(MODEL_KEYS):
        raise ValueError("checkpoints must contain exactly m1 and m5")
    validated_batch_size = _positive_batch_size(batch_size)
    checkpoint_paths = {key: Path(checkpoints[key]) for key in MODEL_KEYS}
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    initial_checkpoint_sha256 = {
        model: _sha256_file(path) for model, path in checkpoint_paths.items()
    }
    if len(set(initial_checkpoint_sha256.values())) != len(MODEL_KEYS):
        raise ValueError("declared checkpoint identities must be distinct")
    _require_frozen_checkpoint_identity(initial_checkpoint_sha256)
    checkpoints_manifest = {
        model: {
            "path": str(path.resolve()),
            "sha256": initial_checkpoint_sha256[model],
        }
        for model, path in checkpoint_paths.items()
    }
    repository_root = Path(__file__).resolve().parents[1]
    executed_source_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("next6_mattersim_baseline.py"),
        Path(__file__).resolve().with_name("next6_wbm_features.py"),
    )
    initial_executed_source_sha256 = {
        path.relative_to(repository_root).as_posix(): _sha256_file(path)
        for path in executed_source_paths
    }
    input_paths = {
        "frames": frames_path,
        "metadata": metadata_path,
        "stage_assignments": stage_assignments_path,
    }
    inputs_sha256 = {
        role: {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
        for role, path in input_paths.items()
    }
    if production_protocol_eligible:
        if (
            adapter_mode != "builtin_mattersim"
            or implementation_provenance is None
        ):
            raise FatalCommitteePredictionError(
                "production eligibility requires verified built-in provenance"
            )
        (
            implementation_identity,
            implementation_source_path,
            implementation_source_sha256,
        ) = implementation_provenance
    else:
        if implementation_provenance is not None:
            raise ValueError(
                "nonproduction adapter cannot claim built-in provenance"
            )
        (
            implementation_identity,
            implementation_source_path,
            implementation_source_sha256,
        ) = _implementation_identity(predictor)
    inference_device = str(device_resolved)

    metadata = _normalize_key_table(
        pd.read_parquet(
            metadata_path,
            columns=("sid", "rk", "material", "strict_x0_ok"),
        ),
        role="metadata",
    )
    stage_table = _normalize_key_table(
        pd.read_parquet(
            stage_assignments_path, columns=("sid", "rk", "stage")
        ),
        role="stage assignments",
    )
    if set(metadata["sid"]) != set(stage_table["sid"]):
        raise ValueError("metadata and stage-assignment sid sets differ")
    if stage_table["stage"].isna().any():
        raise ValueError("stage assignments must be nonmissing")
    if not stage_table["stage"].map(
        lambda value: isinstance(value, str)
    ).all():
        raise ValueError("stage assignments must be exact strings")
    if not set(stage_table["stage"]).issubset(
        {*DEVELOPMENT_STAGES, TEST_STAGE}
    ):
        raise ValueError("stage assignments contain unsupported stages")

    joined = metadata.merge(
        stage_table,
        on="sid",
        how="inner",
        suffixes=("_metadata", "_stage"),
        validate="one_to_one",
        sort=False,
    )
    if not joined["rk_metadata"].eq(joined["rk_stage"]).all():
        raise ValueError("metadata and stage-assignment rk values differ")
    joined = joined.rename(columns={"rk_metadata": "rk"}).drop(
        columns="rk_stage"
    )
    selected = joined.loc[joined["stage"].isin(selected_stages)].copy()

    with zipfile.ZipFile(frames_path) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            sid = Path(name).stem
            if sid in members:
                raise ValueError(
                    "initial-frame archive has duplicate member/stem sid: "
                    f"{sid}"
                )
            members[sid] = name

        rows: list[dict[str, object]] = []
        prediction_bases: list[dict[str, object]] = []
        structures: list[object] = []
        for record in selected.to_dict("records"):
            strict_value = record["strict_x0_ok"]
            strict_ok = isinstance(strict_value, (bool, np.bool_)) and bool(
                strict_value
            )
            base = {
                "sid": record["sid"],
                "rk": record["rk"],
                "material": str(record["material"]),
                "stage": record["stage"],
                "strict_x0_ok": strict_ok,
            }
            if not strict_ok:
                rows.append(_abstain_row(base, "nonstrict_x0"))
                continue
            member = members.get(record["sid"])
            if member is None:
                rows.append(_abstain_row(base, "unsupported_initial_frame"))
                continue
            try:
                frame_text = archive.read(member).decode("utf-8")
                atoms = frame_to_atoms(frame_text)
                if len(atoms) <= 0:
                    raise ValueError("empty frame")
            except Exception:
                rows.append(_abstain_row(base, "unsupported_initial_frame"))
                continue
            prediction_bases.append(base)
            structures.append(atoms)

    predictor_calls = 0
    prediction_wall_time = 0.0
    cuda_tracking_started = False
    loaded_checkpoint_sha256: dict[str, str | None] = {
        model: None for model in MODEL_KEYS
    }
    model_telemetry = _unreported_telemetry(len(structures))
    if structures:
        predictor_calls = 1
        cuda_tracking_started = _reset_peak_cuda_memory(inference_device)
        prediction_started = time.perf_counter()
        try:
            batch = predictor(structures)
        except FatalCommitteePredictionError:
            raise
        except Exception:
            predicted_rows = [
                _abstain_row(base, "predictor_failed")
                for base in prediction_bases
            ]
        else:
            predictions, validated_loaded, model_telemetry = (
                _validate_prediction_batch(
                    batch,
                    requested_rows=len(structures),
                    expected_checkpoint_sha256=initial_checkpoint_sha256,
                )
            )
            loaded_checkpoint_sha256 = validated_loaded
            predicted_rows = [
                _prediction_row(base, len(atoms), prediction)
                for base, atoms, prediction in zip(
                    prediction_bases, structures, predictions, strict=True
                )
            ]
        prediction_wall_time = time.perf_counter() - prediction_started
        rows.extend(predicted_rows)

    table = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS).astype(
        _output_dtypes()
    )
    table = table.sort_values("sid", kind="stable").reset_index(drop=True)
    if len(table) != len(selected) or table["sid"].duplicated().any():
        raise ValueError("output must contain exactly one row per selected sid")

    runtime, peak_cuda_memory = _runtime_metadata(
        inference_device, collect_cuda_peak=cuda_tracking_started
    )

    def verify_unchanged() -> None:
        expected_paths = [
            *(
                (f"input {role}", path, inputs_sha256[role]["sha256"])
                for role, path in input_paths.items()
            ),
            *(
                (
                    f"checkpoint {model}",
                    checkpoint_paths[model],
                    initial_checkpoint_sha256[model],
                )
                for model in MODEL_KEYS
            ),
            *(
                (
                    f"executed source {relative}",
                    repository_root / relative,
                    expected,
                )
                for relative, expected in initial_executed_source_sha256.items()
            ),
        ]
        if (
            implementation_source_path is not None
            and implementation_source_sha256 is not None
        ):
            expected_paths.append(
                (
                    "predictor implementation",
                    implementation_source_path,
                    implementation_source_sha256,
                )
            )
        for role, path, expected in expected_paths:
            try:
                current = _sha256_file(Path(path))
            except OSError as exc:
                raise RuntimeError(
                    f"{role} changed after initial hash"
                ) from exc
            if current != expected:
                raise RuntimeError(f"{role} changed after initial hash")

    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": validated_mode,
        "stages": list(selected_stages),
        "device": inference_device,
        "adapter": {
            "mode": adapter_mode,
            "batch_size": validated_batch_size,
            "device_requested": str(device_requested),
            "device_resolved": inference_device,
            "implementation": implementation_identity,
        },
        "production_protocol_eligible": production_protocol_eligible,
        "evidence_role": (
            "protocol_feature_generation"
            if production_protocol_eligible
            else "testing_only_not_scientific_evidence"
        ),
        "runtime": runtime,
        "checkpoints": checkpoints_manifest,
        "predictor_loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "inputs_sha256": inputs_sha256,
        "executed_source_sha256": initial_executed_source_sha256,
        "integrity": {"prepublish_rehash": "passed"},
        "counts": {
            "input_rows": len(metadata),
            "stage_assignment_rows": len(stage_table),
            "selected_rows": len(selected),
            "strict_rows": int(table["strict_x0_ok"].sum()),
            "nonstrict_rows": int((~table["strict_x0_ok"]).sum()),
            "successful_rows": int(table["committee_feature_ok"].sum()),
            "abstained_rows": int((~table["committee_feature_ok"]).sum()),
            "predictor_calls": predictor_calls,
            "prediction_rows_requested": len(structures),
            "model_telemetry": model_telemetry,
        },
        "execution": {
            "predictor_calls": predictor_calls,
            "prediction_wall_time_seconds": prediction_wall_time,
            "peak_cuda_memory_bytes": peak_cuda_memory,
        },
    }
    if finalize_before_publish is not None:
        finalize_before_publish()
    return _write_and_publish(
        table,
        manifest,
        output_dir,
        verify_unchanged=verify_unchanged,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run label-free MatterSim committee feature generation from the CLI."""

    parser = argparse.ArgumentParser(
        description="Build label-free MatterSim 1M/5M x0 committee features."
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("development", "test"), required=True
    )
    args = parser.parse_args(argv)
    selected_stages = (
        DEVELOPMENT_STAGES if args.mode == "development" else (TEST_STAGE,)
    )
    run_committee_features(
        args.frames,
        args.metadata,
        args.stages,
        args.output,
        checkpoints={"m1": args.m1, "m5": args.m5},
        stages=selected_stages,
        mode=args.mode,
        predictor=None,
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


__all__ = [
    "CommitteePredictionBatch",
    "CommitteePredictor",
    "DEVELOPMENT_STAGES",
    "FatalCommitteePredictionError",
    "FROZEN_CHECKPOINT_SHA256",
    "MatterSimCommitteePredictor",
    "MODEL_KEYS",
    "ModelEvaluationTelemetry",
    "ModelFeaturePrediction",
    "main",
    "run_committee_features",
]


if __name__ == "__main__":
    raise SystemExit(main())
