"""Deterministic numerical core for the CHSC-v0 engineering diagnostic.

CHSC-v0 probes homogeneous cell-strain curvature at fixed fractional atomic
coordinates.  Its MatterSim use and two-scale consistency interval are candidate
label-free rejection evidence, not a certificate of DFT stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
from ase import Atoms
from scipy.linalg import expm


CHSC_VERSION = "CHSC-v0"
STEP_STRAIN = 2**-7
STRAIN_DIMENSION = 6
TAU_MULTIPLIER = 64.0


class CHSCValidationError(ValueError):
    """Raised when a direct CHSC utility receives invalid input."""


class CHSCNumericalError(RuntimeError):
    """Raised when finite, shape-valid CHSC input cannot be analyzed."""


class CHSCStatus(str, Enum):
    """Frozen CHSC-v0 success and abstention codes."""

    RESOLVED_NEGATIVE = "resolved_negative"
    RESOLVED_NONNEGATIVE = "resolved_nonnegative"
    NEAR_ZERO_OR_INCONSISTENT = "near_zero_or_inconsistent"
    ABSTAIN_UNSUPPORTED_GEOMETRY = "abstain_unsupported_geometry"
    ABSTAIN_ENERGY_FAILURE = "abstain_energy_failure"
    ABSTAIN_INVALID_ENERGY = "abstain_invalid_energy"
    ABSTAIN_NUMERICAL_FAILURE = "abstain_numerical_failure"


@dataclass(frozen=True, slots=True)
class CHSCSpectralResult:
    """Immutable result of the two-matrix CHSC spectral analysis."""

    status: CHSCStatus
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
class CHSCResult:
    """Immutable end-to-end result; failed evaluations explicitly abstain."""

    status: CHSCStatus
    negative: bool | None = None
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
    energy_call_count: int = 0
    error: str | None = None


EnergyOracle = Callable[[Atoms], Any]


class _EnergyOracleFailure(RuntimeError):
    """Internal marker for an exception raised by an energy oracle."""


class _InvalidEnergyOutput(ValueError):
    """Internal marker for a malformed energy-oracle result."""


def _exception_text(exc: Exception) -> str:
    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def strain_basis() -> np.ndarray:
    """Return the frozen Frobenius-orthonormal symmetric strain basis."""

    basis = np.zeros((STRAIN_DIMENSION, 3, 3), dtype=np.float64)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    basis[2, 2, 2] = 1.0
    shear = 1.0 / np.sqrt(2.0)
    basis[3, 1, 2] = basis[3, 2, 1] = shear
    basis[4, 0, 2] = basis[4, 2, 0] = shear
    basis[5, 0, 1] = basis[5, 1, 0] = shear
    return basis


def direction_set() -> np.ndarray:
    """Return the 21 fixed unit directions used to reconstruct a 6x6 Hessian."""

    directions = [row.copy() for row in np.eye(STRAIN_DIMENSION, dtype=np.float64)]
    amplitude = 1.0 / np.sqrt(2.0)
    for first in range(STRAIN_DIMENSION):
        for second in range(first + 1, STRAIN_DIMENSION):
            direction = np.zeros(STRAIN_DIMENSION, dtype=np.float64)
            direction[first] = amplitude
            direction[second] = amplitude
            directions.append(direction)
    return np.stack(directions, axis=0)


def _validated_geometry(atoms: Atoms) -> Atoms:
    if not isinstance(atoms, Atoms):
        raise CHSCValidationError("atoms must be an ase.Atoms instance")
    count = len(atoms)
    if isinstance(count, bool) or count < 1:
        raise CHSCValidationError("CHSC requires at least one atom")
    try:
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        pbc = np.asarray(atoms.pbc)
    except Exception as exc:
        raise CHSCValidationError(
            f"could not read atomic geometry: {_exception_text(exc)}"
        ) from exc
    if pbc.shape != (3,) or not np.array_equal(pbc, np.ones(3, dtype=bool)):
        raise CHSCValidationError("CHSC geometry must be periodic in exactly three dimensions")
    if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
        raise CHSCValidationError(f"atomic positions must be finite with shape {(count, 3)}")
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise CHSCValidationError("cell must be a finite 3 by 3 matrix")
    try:
        determinant = float(np.linalg.det(cell))
        rank = int(np.linalg.matrix_rank(cell))
        scaled = np.linalg.solve(cell.T, positions.T).T
    except np.linalg.LinAlgError as exc:
        raise CHSCValidationError(f"could not validate cell: {_exception_text(exc)}") from exc
    if rank != 3 or not np.isfinite(determinant) or determinant == 0.0:
        raise CHSCValidationError("cell must be nonsingular")
    if not np.all(np.isfinite(scaled)):
        raise CHSCValidationError("fractional coordinates must be finite")
    canonical = atoms.copy()
    wrapped = scaled - np.floor(scaled)
    wrapped = np.where(wrapped >= 1.0, 0.0, wrapped)
    canonical.set_scaled_positions(wrapped)
    canonical.set_pbc((True, True, True))
    return canonical


def _validated_direction(direction: Any) -> np.ndarray:
    try:
        vector = np.asarray(direction, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CHSCValidationError(
            f"strain direction is not a floating-point vector: {_exception_text(exc)}"
        ) from exc
    if vector.shape != (STRAIN_DIMENSION,):
        raise CHSCValidationError("strain direction must have shape (6,)")
    if not np.all(np.isfinite(vector)):
        raise CHSCValidationError("strain direction must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, rtol=0.0, atol=1e-12):
        raise CHSCValidationError("strain direction must have Euclidean norm one")
    return vector


def _validated_step(step: Any) -> float:
    try:
        value = float(step)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CHSCValidationError(
            f"strain step must be a finite positive scalar: {_exception_text(exc)}"
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise CHSCValidationError("strain step must be a finite positive scalar")
    return value


def deform_cell(atoms: Atoms, direction: Any, step: float) -> Atoms:
    """Apply one exponential homogeneous strain while preserving fractions."""

    base = _validated_geometry(atoms)
    vector = _validated_direction(direction)
    displacement = _validated_step(step)
    cell = np.asarray(base.cell.array, dtype=np.float64)
    scaled = np.asarray(base.get_scaled_positions(wrap=False), dtype=np.float64)
    generator = np.einsum("a,aij->ij", vector, strain_basis())
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            deformation = expm(displacement * generator)
            new_cell = cell @ deformation.T
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise CHSCNumericalError(
            f"cell deformation failed: {_exception_text(exc)}"
        ) from exc
    if not np.all(np.isfinite(new_cell)):
        raise CHSCNumericalError("deformed cell must contain only finite values")
    probe = base.copy()
    probe.set_cell(new_cell, scale_atoms=False)
    probe.set_scaled_positions(scaled)
    return probe


def directional_curvatures_to_hessian(curvatures: Any) -> np.ndarray:
    """Reconstruct a symmetric 6x6 Hessian from the frozen 21 directions."""

    try:
        values = np.asarray(curvatures, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CHSCValidationError(
            f"directional curvatures are not a floating-point vector: {_exception_text(exc)}"
        ) from exc
    if values.shape != (21,):
        raise CHSCValidationError("directional curvatures must have shape (21,)")
    if not np.all(np.isfinite(values)):
        raise CHSCValidationError("directional curvatures must contain only finite values")
    hessian = np.zeros((STRAIN_DIMENSION, STRAIN_DIMENSION), dtype=np.float64)
    hessian[np.diag_indices(STRAIN_DIMENSION)] = values[:STRAIN_DIMENSION]
    cursor = STRAIN_DIMENSION
    for first in range(STRAIN_DIMENSION):
        for second in range(first + 1, STRAIN_DIMENSION):
            off_diagonal = values[cursor] - 0.5 * (
                hessian[first, first] + hessian[second, second]
            )
            hessian[first, second] = off_diagonal
            hessian[second, first] = off_diagonal
            cursor += 1
    return hessian


def _validated_hessian_pair(first: Any, second: Any) -> tuple[np.ndarray, np.ndarray]:
    matrices: list[np.ndarray] = []
    for value, name in ((first, "H_h"), (second, "H_h2")):
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CHSCValidationError(
                f"{name} is not a floating-point matrix: {_exception_text(exc)}"
            ) from exc
        if matrix.shape != (STRAIN_DIMENSION, STRAIN_DIMENSION):
            raise CHSCValidationError(f"{name} must have shape (6, 6)")
        if not np.all(np.isfinite(matrix)):
            raise CHSCValidationError(f"{name} must contain only finite values")
        matrices.append(matrix)
    return matrices[0], matrices[1]


def analyze_strain_hessian_pair(h_h: Any, h_h2: Any) -> CHSCSpectralResult:
    """Apply the frozen two-scale CHSC-v0 spectral rule."""

    raw_h, raw_h2 = _validated_hessian_pair(h_h, h_h2)
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
                * STRAIN_DIMENSION
                * np.finfo(np.float64).eps
                * max(1.0, norm_h, norm_h2, norm_r)
            )
            antisymmetric_norm_h = float(np.linalg.norm(antisymmetric_h, ord=2))
            antisymmetric_norm_h2 = float(np.linalg.norm(antisymmetric_h2, ord=2))
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        raise CHSCNumericalError(
            f"CHSC spectral analysis failed: {_exception_text(exc)}"
        ) from exc
    diagnostics = (
        lambda_h,
        lambda_h2,
        lambda_r,
        e_num,
        u_num,
        l_num,
        tau_alg,
        antisymmetric_norm_h,
        antisymmetric_norm_h2,
    )
    if not all(np.isfinite(value) for value in diagnostics):
        raise CHSCNumericalError("CHSC spectral diagnostics must all be finite")

    from .next11_phsc import classify_phsc_state

    state = classify_phsc_state(lambda_h, lambda_h2, u_num, l_num, tau_alg)
    status = CHSCStatus(state.value)
    return CHSCSpectralResult(
        status=status,
        negative=status is CHSCStatus.RESOLVED_NEGATIVE,
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


def _checked_energy_call(energy_oracle: EnergyOracle, probe: Atoms, *, stage: str) -> float:
    try:
        raw = energy_oracle(probe.copy())
    except Exception as exc:
        raise _EnergyOracleFailure(
            f"energy oracle failed at {stage}: {_exception_text(exc)}"
        ) from None
    if isinstance(raw, (bool, np.bool_)):
        raise _InvalidEnergyOutput(f"energy output at {stage} must be a finite scalar")
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _InvalidEnergyOutput(
            f"energy output at {stage} is not a floating-point scalar: {_exception_text(exc)}"
        ) from None
    if array.shape != () or not np.isfinite(array.item()):
        raise _InvalidEnergyOutput(f"energy output at {stage} must be a finite scalar")
    return float(array.item())


def evaluate_chsc(atoms: Atoms, energy_oracle: EnergyOracle) -> CHSCResult:
    """Evaluate CHSC-v0 with one center and 84 deterministic strain probes."""

    if not callable(energy_oracle):
        raise CHSCValidationError("energy_oracle must be callable")
    try:
        base = _validated_geometry(atoms)
    except CHSCValidationError as exc:
        return CHSCResult(
            status=CHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            error=_exception_text(exc),
        )

    calls = 0

    def evaluate(probe: Atoms, stage: str) -> float:
        nonlocal calls
        calls += 1
        return _checked_energy_call(energy_oracle, probe, stage=stage)

    try:
        center = evaluate(base, "center")
        energies_h = np.empty((21, 2), dtype=np.float64)
        energies_h2 = np.empty((21, 2), dtype=np.float64)
        directions = direction_set()
        for index, direction in enumerate(directions):
            energies_h[index, 0] = evaluate(
                deform_cell(base, direction, STEP_STRAIN), f"direction_{index:02d}_plus_h"
            )
            energies_h[index, 1] = evaluate(
                deform_cell(base, -direction, STEP_STRAIN), f"direction_{index:02d}_minus_h"
            )
            energies_h2[index, 0] = evaluate(
                deform_cell(base, direction, STEP_STRAIN / 2.0),
                f"direction_{index:02d}_plus_h2",
            )
            energies_h2[index, 1] = evaluate(
                deform_cell(base, -direction, STEP_STRAIN / 2.0),
                f"direction_{index:02d}_minus_h2",
            )
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            scale = float(len(base))
            curvatures_h = (energies_h[:, 0] - 2.0 * center + energies_h[:, 1]) / (
                scale * STEP_STRAIN**2
            )
            curvatures_h2 = (
                energies_h2[:, 0] - 2.0 * center + energies_h2[:, 1]
            ) / (scale * (STEP_STRAIN / 2.0) ** 2)
        hessian_h = directional_curvatures_to_hessian(curvatures_h)
        hessian_h2 = directional_curvatures_to_hessian(curvatures_h2)
        spectral = analyze_strain_hessian_pair(hessian_h, hessian_h2)
    except _EnergyOracleFailure as exc:
        return CHSCResult(
            status=CHSCStatus.ABSTAIN_ENERGY_FAILURE,
            h=STEP_STRAIN,
            energy_call_count=calls,
            error=str(exc),
        )
    except _InvalidEnergyOutput as exc:
        return CHSCResult(
            status=CHSCStatus.ABSTAIN_INVALID_ENERGY,
            h=STEP_STRAIN,
            energy_call_count=calls,
            error=str(exc),
        )
    except (CHSCValidationError, CHSCNumericalError, FloatingPointError, OverflowError) as exc:
        return CHSCResult(
            status=CHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
            h=STEP_STRAIN,
            energy_call_count=calls,
            error=_exception_text(exc),
        )

    return CHSCResult(
        status=spectral.status,
        negative=spectral.negative,
        h=STEP_STRAIN,
        lambda_h=spectral.lambda_h,
        lambda_h2=spectral.lambda_h2,
        lambda_r=spectral.lambda_r,
        e_num=spectral.e_num,
        u_num=spectral.u_num,
        l_num=spectral.l_num,
        tau_alg=spectral.tau_alg,
        antisymmetric_norm_h=spectral.antisymmetric_norm_h,
        antisymmetric_norm_h2=spectral.antisymmetric_norm_h2,
        energy_call_count=calls,
    )
