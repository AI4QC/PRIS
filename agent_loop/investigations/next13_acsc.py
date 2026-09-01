"""Numerical core for the additive ACSC-v0 coupled-Hessian diagnostic.

ACSC-v0 combines the PHSC-v0 internal atomic Hessian, the CHSC-v0 homogeneous
strain Hessian, and their force-response cross block in dimensionless
generalized coordinates.  It is label-free candidate rejection evidence, not a
certificate of DFT stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .next11_phsc import classify_phsc_state, helmert_internal_basis


ACSC_VERSION = "ACSC-v0"
STRAIN_DIMENSION = 6
TAU_MULTIPLIER = 64.0


class ACSCValidationError(ValueError):
    """Raised when a direct ACSC utility receives invalid input."""


class ACSCNumericalError(RuntimeError):
    """Raised when finite, shape-valid ACSC input cannot be analyzed."""


class ACSCStatus(str, Enum):
    """Frozen ACSC-v0 success and abstention states."""

    RESOLVED_NEGATIVE = "resolved_negative"
    RESOLVED_NONNEGATIVE = "resolved_nonnegative"
    NEAR_ZERO_OR_INCONSISTENT = "near_zero_or_inconsistent"
    ABSTAIN_UNSUPPORTED_GEOMETRY = "abstain_unsupported_geometry"
    ABSTAIN_PREDICTION_FAILURE = "abstain_prediction_failure"
    ABSTAIN_NUMERICAL_FAILURE = "abstain_numerical_failure"


@dataclass(frozen=True, slots=True)
class ACSCSpectralResult:
    """Immutable result of the two-scale coupled spectral analysis."""

    status: ACSCStatus
    negative: bool
    lambda_h: float
    lambda_h2: float
    lambda_r: float
    e_num: float
    u_num: float
    l_num: float
    tau_alg: float
    antisymmetric_norm_h: float
    antisymmetric_norm_h2: float


@dataclass(frozen=True, slots=True)
class ACSCResult:
    """Immutable end-to-end ACSC result; failures explicitly abstain."""

    status: ACSCStatus
    negative: bool | None = None
    coupling_only_negative: bool | None = None
    d_star: float | None = None
    lambda_h: float | None = None
    lambda_h2: float | None = None
    lambda_r: float | None = None
    e_num: float | None = None
    u_num: float | None = None
    l_num: float | None = None
    tau_alg: float | None = None
    antisymmetric_norm_h: float | None = None
    antisymmetric_norm_h2: float | None = None
    prediction_evaluation_count: int = 0
    error: str | None = None


def _exception_text(exc: Exception) -> str:
    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _float_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ACSCValidationError(
            f"{name} is not a floating-point array: {_exception_text(exc)}"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise ACSCValidationError(f"{name} must contain only finite values")
    return array


def _positive_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ACSCValidationError(f"{name} must be a finite positive scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ACSCValidationError(
            f"{name} must be a finite positive scalar: {_exception_text(exc)}"
        ) from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ACSCValidationError(f"{name} must be a finite positive scalar")
    return scalar


def cross_hessians_from_strain_forces(
    force_plus_h: Any,
    force_minus_h: Any,
    force_plus_h2: Any,
    force_minus_h2: Any,
    *,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``d2E/(dR deta)`` blocks at strain steps ``h`` and ``h/2``.

    Every sample has shape ``(6, N, 3)``.  Axis zero follows the six axial
    directions in the frozen CHSC strain basis.  Since Cartesian force is
    ``-dE/dR``, the cross block is ``-dF/deta``.
    """

    step = _positive_scalar(h, name="h")
    samples = tuple(
        _float_array(value, name=name)
        for value, name in (
            (force_plus_h, "force_plus_h"),
            (force_minus_h, "force_minus_h"),
            (force_plus_h2, "force_plus_h2"),
            (force_minus_h2, "force_minus_h2"),
        )
    )
    expected_shape = samples[0].shape
    if (
        len(expected_shape) != 3
        or expected_shape[0] != STRAIN_DIMENSION
        or expected_shape[1] < 2
        or expected_shape[2] != 3
    ):
        raise ACSCValidationError("strain force samples must have shape (6, N, 3), N >= 2")
    if any(sample.shape != expected_shape for sample in samples[1:]):
        raise ACSCValidationError("all four strain force samples must have the same shape")
    plus_h, minus_h, plus_h2, minus_h2 = samples
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            cross_h = -((plus_h - minus_h) / (2.0 * step)).reshape(6, -1).T
            cross_h2 = -((plus_h2 - minus_h2) / step).reshape(6, -1).T
    except (FloatingPointError, OverflowError) as exc:
        raise ACSCNumericalError(
            f"cross-Hessian finite difference failed: {_exception_text(exc)}"
        ) from exc
    if not (np.all(np.isfinite(cross_h)) and np.all(np.isfinite(cross_h2))):
        raise ACSCNumericalError("cross-Hessian finite differences must be finite")
    return cross_h, cross_h2


def _validated_blocks(
    atomic_hessian: Any, strain_hessian: Any, cross_hessian: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    atomic = _float_array(atomic_hessian, name="atomic_hessian")
    strain = _float_array(strain_hessian, name="strain_hessian")
    cross = _float_array(cross_hessian, name="cross_hessian")
    if atomic.ndim != 2 or atomic.shape[0] != atomic.shape[1]:
        raise ACSCValidationError("atomic_hessian must be square")
    if atomic.shape[0] < 6 or atomic.shape[0] % 3 != 0:
        raise ACSCValidationError("atomic_hessian dimension must be 3N for N >= 2")
    n_atoms = atomic.shape[0] // 3
    if strain.shape != (STRAIN_DIMENSION, STRAIN_DIMENSION):
        raise ACSCValidationError("strain_hessian must have shape (6, 6)")
    if cross.shape != (3 * n_atoms, STRAIN_DIMENSION):
        raise ACSCValidationError(
            f"cross_hessian must have shape {(3 * n_atoms, STRAIN_DIMENSION)}"
        )
    return atomic, strain, cross, n_atoms


def scaled_internal_coupled_hessian(
    atomic_hessian: Any,
    strain_hessian: Any,
    cross_hessian: Any,
    *,
    d_star: float,
) -> np.ndarray:
    """Build the dimensionless internal atomic–strain Hessian in eV/atom.

    ``strain_hessian`` is the already per-atom CHSC block.  Atomic translation
    modes are removed with the same deterministic Helmert basis used by PHSC.
    """

    atomic, strain, cross, n_atoms = _validated_blocks(
        atomic_hessian, strain_hessian, cross_hessian
    )
    distance = _positive_scalar(d_star, name="d_star")
    q = helmert_internal_basis(n_atoms)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            atomic_symmetric = 0.5 * (atomic + atomic.T)
            strain_symmetric = 0.5 * (strain + strain.T)
            atomic_internal = (distance**2 / float(n_atoms)) * (
                q.T @ atomic_symmetric @ q
            )
            cross_internal = (distance / float(n_atoms)) * (q.T @ cross)
            coupled = np.block(
                [
                    [atomic_internal, cross_internal],
                    [cross_internal.T, strain_symmetric],
                ]
            )
            coupled = 0.5 * (coupled + coupled.T)
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        raise ACSCNumericalError(
            f"coupled-Hessian construction failed: {_exception_text(exc)}"
        ) from exc
    if not np.all(np.isfinite(coupled)):
        raise ACSCNumericalError("coupled Hessian must contain only finite values")
    return coupled


def _validated_coupled_pair(first: Any, second: Any) -> tuple[np.ndarray, np.ndarray]:
    matrices = tuple(
        _float_array(value, name=name)
        for value, name in ((first, "K_h"), (second, "K_h2"))
    )
    if any(matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] for matrix in matrices):
        raise ACSCValidationError("K_h and K_h2 must be square")
    if matrices[0].shape != matrices[1].shape:
        raise ACSCValidationError("K_h and K_h2 must have the same shape")
    dimension = matrices[0].shape[0]
    if dimension < 9 or (dimension - 3) % 3 != 0:
        raise ACSCValidationError("coupled dimension must be 3N+3 for N >= 2")
    return matrices[0], matrices[1]


def analyze_coupled_hessian_pair(k_h: Any, k_h2: Any) -> ACSCSpectralResult:
    """Apply the frozen two-scale rule to coupled dimensionless Hessians."""

    raw_h, raw_h2 = _validated_coupled_pair(k_h, k_h2)
    dimension = raw_h.shape[0]
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            antisymmetric_h = 0.5 * (raw_h - raw_h.T)
            antisymmetric_h2 = 0.5 * (raw_h2 - raw_h2.T)
            symmetric_h = 0.5 * (raw_h + raw_h.T)
            symmetric_h2 = 0.5 * (raw_h2 + raw_h2.T)
            richardson = (4.0 * symmetric_h2 - symmetric_h) / 3.0
            richardson = 0.5 * (richardson + richardson.T)
            difference = (symmetric_h2 - symmetric_h) / 3.0

            norm_h = float(np.linalg.norm(symmetric_h, ord=2))
            norm_h2 = float(np.linalg.norm(symmetric_h2, ord=2))
            norm_r = float(np.linalg.norm(richardson, ord=2))
            e_num = float(np.linalg.norm(difference, ord=2))
            lambda_h = float(np.linalg.eigvalsh(symmetric_h)[0])
            lambda_h2 = float(np.linalg.eigvalsh(symmetric_h2)[0])
            lambda_r = float(np.linalg.eigvalsh(richardson)[0])
            u_num = float(lambda_r + e_num)
            l_num = float(lambda_r - e_num)
            tau_alg = float(
                TAU_MULTIPLIER
                * dimension
                * np.finfo(np.float64).eps
                * max(1.0, norm_h, norm_h2, norm_r)
            )
            antisymmetric_norm_h = float(np.linalg.norm(antisymmetric_h, ord=2))
            antisymmetric_norm_h2 = float(np.linalg.norm(antisymmetric_h2, ord=2))
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        raise ACSCNumericalError(
            f"ACSC spectral analysis failed: {_exception_text(exc)}"
        ) from exc
    diagnostics = (
        norm_h,
        norm_h2,
        norm_r,
        e_num,
        lambda_h,
        lambda_h2,
        lambda_r,
        u_num,
        l_num,
        tau_alg,
        antisymmetric_norm_h,
        antisymmetric_norm_h2,
    )
    if not all(np.isfinite(value) for value in diagnostics):
        raise ACSCNumericalError("ACSC spectral diagnostics must all be finite")

    state = classify_phsc_state(lambda_h, lambda_h2, u_num, l_num, tau_alg)
    status = ACSCStatus(state.value)
    return ACSCSpectralResult(
        status=status,
        negative=status is ACSCStatus.RESOLVED_NEGATIVE,
        lambda_h=lambda_h,
        lambda_h2=lambda_h2,
        lambda_r=lambda_r,
        e_num=e_num,
        u_num=u_num,
        l_num=l_num,
        tau_alg=tau_alg,
        antisymmetric_norm_h=antisymmetric_norm_h,
        antisymmetric_norm_h2=antisymmetric_norm_h2,
    )


def analyze_acsc_blocks(
    atomic_h: Any,
    atomic_h2: Any,
    strain_h: Any,
    strain_h2: Any,
    cross_h: Any,
    cross_h2: Any,
    *,
    d_star: float,
) -> ACSCSpectralResult:
    """Build both generalized Hessians and apply the ACSC-v0 spectral rule."""

    k_h = scaled_internal_coupled_hessian(
        atomic_h, strain_h, cross_h, d_star=d_star
    )
    k_h2 = scaled_internal_coupled_hessian(
        atomic_h2, strain_h2, cross_h2, d_star=d_star
    )
    return analyze_coupled_hessian_pair(k_h, k_h2)
