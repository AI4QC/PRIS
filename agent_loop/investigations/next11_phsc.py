"""Deterministic numerical core for the PHSC-v0 engineering diagnostic.

PHSC-v0 is a fixed-cell, Gamma-point, MLIP Hessian diagnostic.  Its two-scale
quantities are numerical-consistency proxies, not confidence bounds or formal
error bounds, and its outcome is not a certificate of DFT stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
from ase import Atoms


PHSC_VERSION = "PHSC-v0"
STEP_FRACTION = 2**-8
TAU_MULTIPLIER = 64.0


class PHSCValidationError(ValueError):
    """Raised when a direct PHSC utility receives invalid input."""


class PHSCNumericalError(RuntimeError):
    """Raised when finite, shape-valid PHSC input cannot be analyzed."""


class PHSCStatus(str, Enum):
    """Stable PHSC-v0 success and abstention codes."""

    RESOLVED_NEGATIVE = "resolved_negative"
    RESOLVED_NONNEGATIVE = "resolved_nonnegative"
    NEAR_ZERO_OR_INCONSISTENT = "near_zero_or_inconsistent"
    ABSTAIN_UNSUPPORTED_GEOMETRY = "abstain_unsupported_geometry"
    ABSTAIN_FORCE_FAILURE = "abstain_force_failure"
    ABSTAIN_INVALID_FORCE = "abstain_invalid_force"
    ABSTAIN_NUMERICAL_FAILURE = "abstain_numerical_failure"


@dataclass(frozen=True, slots=True)
class PHSCSpectralResult:
    """Immutable result of the shared two-matrix PHSC spectral analysis."""

    status: PHSCStatus
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
    acoustic_residual_h: float
    acoustic_residual_h2: float


@dataclass(frozen=True, slots=True)
class PHSCResult:
    """Immutable end-to-end result; failed evaluations explicitly abstain."""

    status: PHSCStatus
    negative: bool | None = None
    d_star: float | None = None
    h: float | None = None
    lambda_h: float | None = None
    lambda_h2: float | None = None
    lambda_r: float | None = None
    e_num: float | None = None
    u_num: float | None = None
    l_num: float | None = None
    tau_alg: float | None = None
    antisymmetric_norm_h: float | None = None
    antisymmetric_norm_h2: float | None = None
    acoustic_residual_h: float | None = None
    acoustic_residual_h2: float | None = None
    force_call_count: int = 0
    error: str | None = None


ForceOracle = Callable[[Atoms], Any]


class _ForceOracleFailure(RuntimeError):
    """Internal marker for an exception raised by a force oracle."""


class _InvalidForceOutput(ValueError):
    """Internal marker for a malformed force-oracle result."""


def _exception_text(exc: Exception) -> str:
    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _exact_atom_count(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise PHSCValidationError("atom count must be an integer")
    count = int(value)
    if count < 2:
        raise PHSCValidationError("PHSC requires at least two atoms")
    return count


def helmert_internal_basis(n_atoms: int) -> np.ndarray:
    """Return the frozen deterministic ``kron(C_N, I_3)`` internal basis.

    ``C_N`` is the normalized Helmert contrast matrix with positive leading
    entries and one negative trailing entry in each ordered contrast column.
    The returned shape is ``(3N, 3N-3)`` for atom-major Cartesian vectors.
    """

    count = _exact_atom_count(n_atoms)
    contrasts = np.zeros((count, count - 1), dtype=np.float64)
    for column in range(count - 1):
        leading = column + 1
        denominator = np.sqrt(float(leading * (leading + 1)))
        contrasts[:leading, column] = 1.0 / denominator
        contrasts[leading, column] = -float(leading) / denominator
    return np.kron(contrasts, np.eye(3, dtype=np.float64))


def _wrapped_fractional_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Map finite Cartesian positions into the half-open fractional unit cell."""

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            scaled = np.linalg.solve(cell.T, positions.T).T
            wrapped = scaled - np.floor(scaled)
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        raise PHSCValidationError(
            f"could not convert positions to fractional coordinates: {_exception_text(exc)}"
        ) from exc
    if not np.all(np.isfinite(wrapped)):
        raise PHSCValidationError("wrapped fractional positions must be finite")
    # Floating-point subtraction can exceptionally round a half-open endpoint
    # to 1.0.  Canonicalize that representation explicitly and deterministically.
    wrapped = np.where(wrapped >= 1.0, 0.0, wrapped)
    wrapped = np.where(wrapped < 0.0, wrapped + 1.0, wrapped)
    if not (np.all(wrapped >= 0.0) and np.all(wrapped < 1.0)):
        raise PHSCValidationError("fractional-coordinate wrapping failed")
    return np.asarray(wrapped, dtype=np.float64)


def canonicalize_phsc_geometry(atoms: Atoms) -> tuple[Atoms, float]:
    """Validate PHSC geometry, wrap it into the cell, and return ``d_star``.

    Geometry must contain at least two atoms, have strict three-dimensional
    periodicity and a finite nonsingular cell, and contain no MIC-coincident
    pair.  ``d_star`` is the median of per-atom nearest positive MIC distances.
    """

    if not isinstance(atoms, Atoms):
        raise PHSCValidationError("atoms must be an ase.Atoms instance")
    count = _exact_atom_count(len(atoms))
    try:
        pbc = np.asarray(atoms.pbc)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
    except Exception as exc:
        raise PHSCValidationError(
            f"could not read atomic geometry: {_exception_text(exc)}"
        ) from exc
    if pbc.shape != (3,) or not np.array_equal(pbc, np.ones(3, dtype=bool)):
        raise PHSCValidationError("PHSC geometry must be periodic in exactly three dimensions")
    if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
        raise PHSCValidationError(
            f"atomic positions must be finite with shape {(count, 3)}"
        )
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise PHSCValidationError("cell must be a finite 3 by 3 matrix")
    try:
        rank = int(np.linalg.matrix_rank(cell))
        determinant = float(np.linalg.det(cell))
    except np.linalg.LinAlgError as exc:
        raise PHSCValidationError(f"could not validate cell: {_exception_text(exc)}") from exc
    if rank != 3 or not np.isfinite(determinant) or determinant == 0.0:
        raise PHSCValidationError("cell must be nonsingular")

    wrapped = _wrapped_fractional_positions(positions, cell)
    try:
        canonical = atoms.copy()
        canonical.set_scaled_positions(wrapped)
        canonical.set_pbc((True, True, True))
        distances = np.asarray(canonical.get_all_distances(mic=True), dtype=np.float64)
    except Exception as exc:
        raise PHSCValidationError(
            f"minimum-image geometry is unavailable: {_exception_text(exc)}"
        ) from exc
    if distances.shape != (count, count) or not np.all(np.isfinite(distances)):
        raise PHSCValidationError("MIC distance matrix must be finite with shape (N, N)")
    off_diagonal = ~np.eye(count, dtype=bool)
    pair_distances = distances[off_diagonal]
    if pair_distances.size == 0 or np.any(pair_distances <= 0.0):
        raise PHSCValidationError("PHSC geometry contains a MIC-coincident atom pair")
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    nearest = np.min(masked, axis=1)
    d_star = float(np.median(nearest))
    if not np.isfinite(d_star) or d_star <= 0.0:
        raise PHSCValidationError("median nearest-neighbor MIC distance must be positive")
    return canonical, d_star


def _validated_force_sample(sample: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(sample, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PHSCValidationError(
            f"{name} is not a floating-point array: {_exception_text(exc)}"
        ) from exc
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != 3:
        raise PHSCValidationError(f"{name} must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(array)):
        raise PHSCValidationError(f"{name} must contain only finite values")
    return array


def hessian_columns_from_force_samples(
    force_plus_h: Any,
    force_minus_h: Any,
    force_plus_h2: Any,
    force_minus_h2: Any,
    *,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one column of each raw Hessian from the frozen four force samples."""

    try:
        step = float(h)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PHSCValidationError(
            f"h must be a finite positive scalar: {_exception_text(exc)}"
        ) from exc
    if not np.isfinite(step) or step <= 0.0:
        raise PHSCValidationError("h must be a finite positive scalar")
    samples = tuple(
        _validated_force_sample(value, name=name)
        for value, name in (
            (force_plus_h, "force_plus_h"),
            (force_minus_h, "force_minus_h"),
            (force_plus_h2, "force_plus_h2"),
            (force_minus_h2, "force_minus_h2"),
        )
    )
    expected_shape = samples[0].shape
    if any(sample.shape != expected_shape for sample in samples[1:]):
        raise PHSCValidationError("all four force samples must have the same shape")
    plus_h, minus_h, plus_h2, minus_h2 = samples
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            column_h = -((plus_h - minus_h) / (2.0 * step)).reshape(-1)
            column_h2 = -((plus_h2 - minus_h2) / step).reshape(-1)
    except (FloatingPointError, OverflowError) as exc:
        raise PHSCNumericalError(
            f"finite-difference column calculation failed: {_exception_text(exc)}"
        ) from exc
    if not (np.all(np.isfinite(column_h)) and np.all(np.isfinite(column_h2))):
        raise PHSCNumericalError("finite-difference Hessian columns must be finite")
    return column_h, column_h2


def _validated_hessian_pair(h_h: Any, h_h2: Any) -> tuple[np.ndarray, np.ndarray, int]:
    matrices: list[np.ndarray] = []
    for value, name in ((h_h, "H_h"), (h_h2, "H_h2")):
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise PHSCValidationError(
                f"{name} is not a floating-point matrix: {_exception_text(exc)}"
            ) from exc
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise PHSCValidationError(f"{name} must be square")
        if not np.all(np.isfinite(matrix)):
            raise PHSCValidationError(f"{name} must contain only finite values")
        matrices.append(matrix)
    if matrices[0].shape != matrices[1].shape:
        raise PHSCValidationError("H_h and H_h2 must have the same shape")
    dimension = matrices[0].shape[0]
    if dimension < 6 or dimension % 3 != 0:
        raise PHSCValidationError("Hessian dimension must be 3N for N >= 2")
    return matrices[0], matrices[1], dimension // 3


def classify_phsc_state(
    lambda_h: float,
    lambda_h2: float,
    u_num: float,
    l_num: float,
    tau_alg: float,
) -> PHSCStatus:
    """Apply the frozen strict PHSC-v0 sign comparisons.

    Equality with either ``+tau_alg`` or ``-tau_alg`` is intentionally
    unresolved.  This function is the sole implementation of the three-state
    spectral classification used by :func:`analyze_hessian_pair`.
    """

    try:
        values = tuple(
            float(value) for value in (lambda_h, lambda_h2, u_num, l_num, tau_alg)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise PHSCValidationError(
            f"spectral state inputs must be finite scalars: {_exception_text(exc)}"
        ) from exc
    observed_lambda_h, observed_lambda_h2, observed_u, observed_l, tolerance = values
    if not all(np.isfinite(value) for value in values):
        raise PHSCValidationError("spectral state inputs must be finite scalars")
    if tolerance < 0.0:
        raise PHSCValidationError("tau_alg must be nonnegative")
    if (
        observed_lambda_h < -tolerance
        and observed_lambda_h2 < -tolerance
        and observed_u < -tolerance
    ):
        return PHSCStatus.RESOLVED_NEGATIVE
    if (
        observed_lambda_h > tolerance
        and observed_lambda_h2 > tolerance
        and observed_l > tolerance
    ):
        return PHSCStatus.RESOLVED_NONNEGATIVE
    return PHSCStatus.NEAR_ZERO_OR_INCONSISTENT


def analyze_hessian_pair(h_h: Any, h_h2: Any) -> PHSCSpectralResult:
    """Apply the sole PHSC-v0 symmetrization, projection, and spectral rule.

    This function is the shared implementation for scalar and batched runners.
    Antisymmetry and acoustic residuals are recorded as diagnostics only.  The
    projected two-scale operator difference is the only decision proxy.
    """

    raw_h, raw_h2, n_atoms = _validated_hessian_pair(h_h, h_h2)
    q = helmert_internal_basis(n_atoms)
    translation = np.kron(
        np.ones((n_atoms, 1), dtype=np.float64) / np.sqrt(float(n_atoms)),
        np.eye(3, dtype=np.float64),
    )
    internal_dimension = 3 * n_atoms - 3
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            antisymmetric_h = 0.5 * (raw_h - raw_h.T)
            antisymmetric_h2 = 0.5 * (raw_h2 - raw_h2.T)
            symmetric_h = 0.5 * (raw_h + raw_h.T)
            symmetric_h2 = 0.5 * (raw_h2 + raw_h2.T)
            a_h = q.T @ symmetric_h @ q
            a_h2 = q.T @ symmetric_h2 @ q
            # Matrix multiplication can leave last-bit upper/lower-triangle
            # differences even though the mathematical operators are
            # symmetric.  Freeze one explicitly symmetric input for both the
            # spectral norm and eigensolver.
            a_h = 0.5 * (a_h + a_h.T)
            a_h2 = 0.5 * (a_h2 + a_h2.T)
            a_r = (4.0 * a_h2 - a_h) / 3.0
            a_r = 0.5 * (a_r + a_r.T)
            difference = (a_h2 - a_h) / 3.0

            norm_h = float(np.linalg.norm(a_h, ord=2))
            norm_h2 = float(np.linalg.norm(a_h2, ord=2))
            norm_r = float(np.linalg.norm(a_r, ord=2))
            e_num = float(np.linalg.norm(difference, ord=2))
            lambda_h = float(np.linalg.eigvalsh(a_h)[0])
            lambda_h2 = float(np.linalg.eigvalsh(a_h2)[0])
            lambda_r = float(np.linalg.eigvalsh(a_r)[0])
            u_num = float(lambda_r + e_num)
            l_num = float(lambda_r - e_num)
            tau_alg = float(
                TAU_MULTIPLIER
                * internal_dimension
                * np.finfo(np.float64).eps
                * max(1.0, norm_h, norm_h2, norm_r)
            )
            antisymmetric_norm_h = float(np.linalg.norm(antisymmetric_h, ord=2))
            antisymmetric_norm_h2 = float(np.linalg.norm(antisymmetric_h2, ord=2))
            acoustic_residual_h = float(
                np.linalg.norm(symmetric_h @ translation, ord=2)
            )
            acoustic_residual_h2 = float(
                np.linalg.norm(symmetric_h2 @ translation, ord=2)
            )
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        raise PHSCNumericalError(f"PHSC spectral analysis failed: {_exception_text(exc)}") from exc

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
        acoustic_residual_h,
        acoustic_residual_h2,
    )
    if not all(np.isfinite(value) for value in diagnostics):
        raise PHSCNumericalError("PHSC spectral diagnostics must all be finite")

    status = classify_phsc_state(lambda_h, lambda_h2, u_num, l_num, tau_alg)
    negative = status is PHSCStatus.RESOLVED_NEGATIVE
    return PHSCSpectralResult(
        status=status,
        negative=negative,
        lambda_h=lambda_h,
        lambda_h2=lambda_h2,
        lambda_r=lambda_r,
        e_num=e_num,
        u_num=u_num,
        l_num=l_num,
        tau_alg=tau_alg,
        antisymmetric_norm_h=antisymmetric_norm_h,
        antisymmetric_norm_h2=antisymmetric_norm_h2,
        acoustic_residual_h=acoustic_residual_h,
        acoustic_residual_h2=acoustic_residual_h2,
    )


def _checked_force_call(force_oracle: ForceOracle, probe: Atoms, *, stage: str) -> np.ndarray:
    try:
        raw = force_oracle(probe.copy())
    except Exception as exc:
        raise _ForceOracleFailure(
            f"force oracle failed at {stage}: {_exception_text(exc)}"
        ) from None
    try:
        forces = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _InvalidForceOutput(
            f"force output at {stage} is not a floating-point array: {_exception_text(exc)}"
        ) from None
    expected_shape = (len(probe), 3)
    if forces.shape != expected_shape:
        raise _InvalidForceOutput(
            f"force output at {stage} must have shape {expected_shape}; got {forces.shape}"
        )
    if not np.all(np.isfinite(forces)):
        raise _InvalidForceOutput(f"force output at {stage} must contain only finite values")
    return forces


def _perturbed_probe(base: Atoms, flat_coordinate: int, displacement: float) -> Atoms:
    cell = np.asarray(base.cell.array, dtype=np.float64)
    positions = np.asarray(base.get_positions(), dtype=np.float64).copy()
    flat = positions.reshape(-1)
    original_coordinate = float(flat[flat_coordinate])
    try:
        with np.errstate(over="raise", invalid="raise"):
            displaced_coordinate = float(original_coordinate + displacement)
    except (FloatingPointError, OverflowError) as exc:
        raise PHSCNumericalError(
            f"perturbed Cartesian coordinate is invalid: {_exception_text(exc)}"
        ) from exc
    if displaced_coordinate == original_coordinate:
        raise PHSCNumericalError(
            "target Cartesian-coordinate displacement is not representable at "
            "the base-coordinate magnitude"
        )
    flat[flat_coordinate] = displaced_coordinate
    if not np.all(np.isfinite(positions)):
        raise PHSCNumericalError("perturbed Cartesian positions must be finite")
    try:
        wrapped = _wrapped_fractional_positions(positions, cell)
        probe = base.copy()
        probe.set_scaled_positions(wrapped)
    except PHSCValidationError as exc:
        raise PHSCNumericalError(str(exc)) from exc
    except Exception as exc:
        raise PHSCNumericalError(
            f"could not construct perturbed geometry: {_exception_text(exc)}"
        ) from exc
    target_atom = flat_coordinate // 3
    probe_positions = np.asarray(probe.get_positions(), dtype=np.float64)
    base_positions = np.asarray(base.get_positions(), dtype=np.float64)
    if np.array_equal(probe_positions[target_atom], base_positions[target_atom]):
        raise PHSCNumericalError(
            "wrapped target-coordinate probe is cell-equivalent to the base geometry"
        )
    return probe


def phsc_probe_group(base: Atoms, coordinate: int, h: float) -> tuple[Atoms, ...]:
    """Return the frozen four wrapped probes for one Cartesian coordinate.

    The order is exactly ``(+h, -h, +h/2, -h/2)``.  This public helper is the
    single geometry-ordering implementation shared by scalar and batch PHSC
    runners.  Its input is canonicalized under the full PHSC geometry contract
    before any displacement is applied.
    """

    canonical, _ = canonicalize_phsc_geometry(base)
    if isinstance(coordinate, (bool, np.bool_)) or not isinstance(
        coordinate, (int, np.integer)
    ):
        raise PHSCValidationError("coordinate must be an integer")
    flat_coordinate = int(coordinate)
    dimension = 3 * len(canonical)
    if flat_coordinate < 0 or flat_coordinate >= dimension:
        raise PHSCValidationError(
            f"coordinate must lie in [0, {dimension}); got {flat_coordinate}"
        )
    try:
        step = float(h)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PHSCValidationError(
            f"h must be a finite positive scalar: {_exception_text(exc)}"
        ) from exc
    if not np.isfinite(step) or step <= 0.0:
        raise PHSCValidationError("h must be a finite positive scalar")
    return tuple(
        _perturbed_probe(canonical, flat_coordinate, displacement)
        for displacement in (step, -step, 0.5 * step, -0.5 * step)
    )


def _abstain(status: PHSCStatus, error: str, force_call_count: int = 0) -> PHSCResult:
    return PHSCResult(status=status, force_call_count=force_call_count, error=error)


def evaluate_phsc(atoms: Atoms, force_oracle: ForceOracle) -> PHSCResult:
    """Evaluate PHSC-v0 with exactly ``12N`` force calls on every success.

    Unsupported geometry, oracle exceptions, malformed force output, and
    numerical failure are explicit abstentions.  None is converted into
    negative-mode evidence.
    """

    try:
        base, d_star = canonicalize_phsc_geometry(atoms)
    except PHSCValidationError as exc:
        return _abstain(
            PHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            f"unsupported PHSC geometry: {exc}",
        )
    h = float(STEP_FRACTION * d_star)
    if not np.isfinite(h) or h <= 0.0:
        return _abstain(
            PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            "PHSC step size must be positive and finite",
        )

    dimension = 3 * len(base)
    h_h = np.empty((dimension, dimension), dtype=np.float64)
    h_h2 = np.empty((dimension, dimension), dtype=np.float64)
    force_call_count = 0
    try:
        # Freeze all coordinate probes before the first oracle call so that an
        # unrepresentable late coordinate cannot leave a partially evaluated
        # structure with a misleading feature status.
        probe_groups = tuple(
            phsc_probe_group(base, column, h) for column in range(dimension)
        )
    except (PHSCNumericalError, PHSCValidationError) as exc:
        return _abstain(
            PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            f"PHSC probe construction failed before oracle evaluation: {exc}",
        )
    except Exception as exc:
        return _abstain(
            PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            "unexpected PHSC probe construction failure before oracle evaluation: "
            f"{_exception_text(exc)}",
        )
    try:
        for column, probes in enumerate(probe_groups):
            samples: list[np.ndarray] = []
            for probe, stage in zip(
                probes,
                (
                    f"coordinate_{column}:x+h",
                    f"coordinate_{column}:x-h",
                    f"coordinate_{column}:x+h/2",
                    f"coordinate_{column}:x-h/2",
                ),
                strict=True,
            ):
                force_call_count += 1
                samples.append(_checked_force_call(force_oracle, probe, stage=stage))
            column_h, column_h2 = hessian_columns_from_force_samples(*samples, h=h)
            h_h[:, column] = column_h
            h_h2[:, column] = column_h2
    except _ForceOracleFailure as exc:
        return _abstain(PHSCStatus.ABSTAIN_FORCE_FAILURE, str(exc), force_call_count)
    except _InvalidForceOutput as exc:
        return _abstain(PHSCStatus.ABSTAIN_INVALID_FORCE, str(exc), force_call_count)
    except (PHSCNumericalError, PHSCValidationError) as exc:
        return _abstain(
            PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            f"PHSC finite-difference construction failed: {exc}",
            force_call_count,
        )
    except Exception as exc:
        return _abstain(
            PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            f"unexpected PHSC finite-difference failure: {_exception_text(exc)}",
            force_call_count,
        )

    try:
        spectral = analyze_hessian_pair(h_h, h_h2)
    except (PHSCValidationError, PHSCNumericalError) as exc:
        return _abstain(
            PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            f"PHSC spectral analysis failed: {exc}",
            force_call_count,
        )
    except Exception as exc:
        return _abstain(
            PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            f"unexpected PHSC spectral failure: {_exception_text(exc)}",
            force_call_count,
        )
    return PHSCResult(
        status=spectral.status,
        negative=spectral.negative,
        d_star=d_star,
        h=h,
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


__all__ = [
    "PHSCNumericalError",
    "PHSCResult",
    "PHSCSpectralResult",
    "PHSCStatus",
    "PHSCValidationError",
    "PHSC_VERSION",
    "STEP_FRACTION",
    "TAU_MULTIPLIER",
    "analyze_hessian_pair",
    "canonicalize_phsc_geometry",
    "classify_phsc_state",
    "evaluate_phsc",
    "helmert_internal_basis",
    "hessian_columns_from_force_samples",
    "phsc_probe_group",
]
