"""Direct mixed-mode energy confirmation for ACSC-v0 candidates.

ACSC-DIRECT-v0 evaluates actual structures along the minimum mode of the
Richardson-extrapolated coupled Hessian.  It is a second numerical route through
the same MLIP, not independent DFT evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from ase import Atoms
from scipy.linalg import expm

from .next11_phsc import (
    canonicalize_phsc_geometry,
    classify_phsc_state,
    helmert_internal_basis,
)
from .next12_chsc import strain_basis


DIRECT_VERSION = "ACSC-DIRECT-v0"
DIRECT_STEP = 2**-8
TAU_MULTIPLIER = 64.0


class DirectValidationError(ValueError):
    """Raised when direct confirmation receives invalid input."""


class DirectNumericalError(RuntimeError):
    """Raised when finite direct-confirmation input cannot be analyzed."""


class DirectStatus(str, Enum):
    """Frozen one-dimensional two-scale states."""

    RESOLVED_NEGATIVE = "resolved_negative"
    RESOLVED_NONNEGATIVE = "resolved_nonnegative"
    NEAR_ZERO_OR_INCONSISTENT = "near_zero_or_inconsistent"


@dataclass(frozen=True, slots=True)
class MinimumMode:
    """Canonical minimum eigenmode of the Richardson coupled Hessian."""

    lambda_r: float
    spectral_gap: float
    vector: np.ndarray


@dataclass(frozen=True, slots=True)
class DirectCurvatureResult:
    """Immutable direct mixed-mode two-scale energy-curvature result."""

    status: DirectStatus
    negative: bool
    h: float
    q_h: float
    q_h2: float
    q_r: float
    e_num: float
    u_num: float
    l_num: float
    tau_alg: float
    energy_call_count: int = 5


def _exception_text(exc: Exception) -> str:
    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DirectValidationError(
            f"{name} is not a floating-point array: {_exception_text(exc)}"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise DirectValidationError(f"{name} must contain only finite values")
    return array


def _positive_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DirectValidationError(f"{name} must be a finite positive scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DirectValidationError(
            f"{name} must be a finite positive scalar: {_exception_text(exc)}"
        ) from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise DirectValidationError(f"{name} must be a finite positive scalar")
    return scalar


def _coupled_pair(first: Any, second: Any) -> tuple[np.ndarray, np.ndarray]:
    matrices = tuple(
        _finite_array(value, name=name)
        for value, name in ((first, "K_h"), (second, "K_h2"))
    )
    if any(matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] for matrix in matrices):
        raise DirectValidationError("K_h and K_h2 must be square")
    if matrices[0].shape != matrices[1].shape:
        raise DirectValidationError("K_h and K_h2 must have the same shape")
    dimension = matrices[0].shape[0]
    if dimension < 9 or (dimension - 3) % 3 != 0:
        raise DirectValidationError("coupled dimension must be 3N+3 for N >= 2")
    return matrices[0], matrices[1]


def minimum_richardson_mode(k_h: Any, k_h2: Any) -> MinimumMode:
    """Return the frozen-sign minimum eigenvector of ``(4*K_h2-K_h)/3``."""

    first, second = _coupled_pair(k_h, k_h2)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            symmetric_h = 0.5 * (first + first.T)
            symmetric_h2 = 0.5 * (second + second.T)
            richardson = (4.0 * symmetric_h2 - symmetric_h) / 3.0
            richardson = 0.5 * (richardson + richardson.T)
            eigenvalues, eigenvectors = np.linalg.eigh(richardson)
            vector = np.asarray(eigenvectors[:, 0], dtype=np.float64).copy()
            norm = float(np.linalg.norm(vector))
            vector /= norm
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                vector *= -1.0
            lambda_r = float(eigenvalues[0])
            spectral_gap = float(eigenvalues[1] - eigenvalues[0])
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        raise DirectNumericalError(
            f"minimum coupled mode failed: {_exception_text(exc)}"
        ) from exc
    if (
        not np.isfinite(lambda_r)
        or not np.isfinite(spectral_gap)
        or spectral_gap < 0.0
        or not np.all(np.isfinite(vector))
        or not np.isclose(np.linalg.norm(vector), 1.0, rtol=0.0, atol=2e-15)
    ):
        raise DirectNumericalError("minimum coupled mode diagnostics must be finite")
    vector.setflags(write=False)
    return MinimumMode(lambda_r=lambda_r, spectral_gap=spectral_gap, vector=vector)


def mixed_mode_probe(atoms: Atoms, mode: Any, *, amplitude: float) -> Atoms:
    """Construct one exact ``f*A(t) + d_star*Q*(t*z)`` mixed-mode probe."""

    try:
        base, d_star = canonicalize_phsc_geometry(atoms)
    except Exception as exc:
        raise DirectValidationError(f"unsupported direct geometry: {exc}") from exc
    vector = _finite_array(mode, name="mode")
    internal_dim = 3 * len(base) - 3
    if vector.shape != (internal_dim + 6,):
        raise DirectValidationError(
            f"mode must have shape {(internal_dim + 6,)} for this geometry"
        )
    norm = float(np.linalg.norm(vector))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-12):
        raise DirectValidationError("mode must have unit Euclidean norm")
    if isinstance(amplitude, (bool, np.bool_)):
        raise DirectValidationError("amplitude must be a finite nonzero scalar")
    try:
        step = float(amplitude)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DirectValidationError(
            f"amplitude must be a finite nonzero scalar: {_exception_text(exc)}"
        ) from exc
    if not np.isfinite(step) or step == 0.0:
        raise DirectValidationError("amplitude must be a finite nonzero scalar")

    z = vector[:internal_dim]
    eta = vector[internal_dim:]
    q = helmert_internal_basis(len(base))
    generator = np.einsum("a,aij->ij", eta, strain_basis())
    reference_cell = np.asarray(base.cell.array, dtype=np.float64)
    fractions = np.asarray(base.get_scaled_positions(wrap=False), dtype=np.float64)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            new_cell = reference_cell @ expm(step * generator).T
            affine = fractions @ new_cell
            internal = d_star * step * (q @ z).reshape(len(base), 3)
            positions = affine + internal
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError, ValueError) as exc:
        raise DirectNumericalError(
            f"mixed-mode geometry construction failed: {_exception_text(exc)}"
        ) from exc
    if not (np.all(np.isfinite(new_cell)) and np.all(np.isfinite(positions))):
        raise DirectNumericalError("mixed-mode geometry must contain only finite values")
    probe = base.copy()
    probe.set_cell(new_cell, scale_atoms=False)
    probe.set_positions(positions)
    probe.set_pbc((True, True, True))
    return probe


def _energy_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DirectValidationError(f"{name} must be a finite scalar")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DirectValidationError(
            f"{name} must be a finite scalar: {_exception_text(exc)}"
        ) from exc
    if array.shape != () or not np.isfinite(array.item()):
        raise DirectValidationError(f"{name} must be a finite scalar")
    return float(array.item())


def direct_curvature_from_energies(
    center: Any,
    plus_h: Any,
    minus_h: Any,
    plus_h2: Any,
    minus_h2: Any,
    *,
    n_atoms: int,
    h: float = DIRECT_STEP,
) -> DirectCurvatureResult:
    """Classify direct mixed-mode curvature from five aligned energies."""

    if isinstance(n_atoms, (bool, np.bool_)) or not isinstance(
        n_atoms, (int, np.integer)
    ) or int(n_atoms) < 2:
        raise DirectValidationError("n_atoms must be an integer >= 2")
    count = int(n_atoms)
    step = _positive_scalar(h, name="h")
    values = tuple(
        _energy_scalar(value, name=name)
        for value, name in (
            (center, "center"),
            (plus_h, "plus_h"),
            (minus_h, "minus_h"),
            (plus_h2, "plus_h2"),
            (minus_h2, "minus_h2"),
        )
    )
    e0, ep, em, ep2, em2 = values
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            q_h = float((ep - 2.0 * e0 + em) / (count * step**2))
            half = step / 2.0
            q_h2 = float((ep2 - 2.0 * e0 + em2) / (count * half**2))
            q_r = float((4.0 * q_h2 - q_h) / 3.0)
            e_num = float(abs((q_h2 - q_h) / 3.0))
            u_num = float(q_r + e_num)
            l_num = float(q_r - e_num)
            tau_alg = float(
                TAU_MULTIPLIER
                * np.finfo(np.float64).eps
                * max(1.0, abs(q_h), abs(q_h2), abs(q_r))
            )
    except (FloatingPointError, OverflowError) as exc:
        raise DirectNumericalError(
            f"direct curvature calculation failed: {_exception_text(exc)}"
        ) from exc
    diagnostics = (q_h, q_h2, q_r, e_num, u_num, l_num, tau_alg)
    if not all(np.isfinite(value) for value in diagnostics):
        raise DirectNumericalError("direct curvature diagnostics must all be finite")
    state = classify_phsc_state(q_h, q_h2, u_num, l_num, tau_alg)
    status = DirectStatus(state.value)
    return DirectCurvatureResult(
        status=status,
        negative=status is DirectStatus.RESOLVED_NEGATIVE,
        h=step,
        q_h=q_h,
        q_h2=q_h2,
        q_r=q_r,
        e_num=e_num,
        u_num=u_num,
        l_num=l_num,
        tau_alg=tau_alg,
    )
