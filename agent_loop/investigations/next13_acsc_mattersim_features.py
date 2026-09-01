"""Combined label-free MatterSim probes for ACSC-v0.

The combined evaluator preserves the frozen PHSC-v0 and CHSC-v0 probes.  Forces
already returned for the six axial CHSC strain directions are reused to build
the atomic–strain cross block; ACSC adds no probe structures beyond the union of
PHSC and CHSC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms

from src.next10_lrrc_mattersim_features import (
    BatchForcePredictor,
    _validated_prediction,
)
from src.next11_phsc import (
    PHSCNumericalError,
    PHSCResult,
    PHSCStatus,
    PHSCValidationError,
    STEP_FRACTION,
    analyze_hessian_pair,
    canonicalize_phsc_geometry,
    hessian_columns_from_force_samples,
    phsc_probe_group,
)
from src.next12_chsc import (
    CHSCNumericalError,
    CHSCResult,
    CHSCStatus,
    CHSCValidationError,
)
from src.next12_chsc_mattersim_features import _probe_group as _chsc_probe_group
from src.next12_chsc_mattersim_features import _result_from_energies
from src.next13_acsc import (
    ACSCNumericalError,
    ACSCResult,
    ACSCStatus,
    ACSCValidationError,
    analyze_acsc_blocks,
    cross_hessians_from_strain_forces,
)


FROZEN_STRUCTURES_PER_CALL = 4


class BatchACSCError(RuntimeError):
    """Raised when a batch ACSC run cannot preserve complete aligned groups."""


@dataclass(frozen=True, slots=True)
class BatchACSCResult:
    """One sid-aligned combined PHSC, CHSC, and ACSC result."""

    sid: str
    phsc: PHSCResult
    chsc: CHSCResult
    acsc: ACSCResult
    cross_h: np.ndarray
    cross_h2: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedACSC:
    sid: str
    base: Atoms
    d_star: float
    h_atomic: float
    probes: tuple[Atoms, ...]


def _abstained_result(
    sid: str,
    *,
    acsc_status: ACSCStatus,
    phsc_status: PHSCStatus,
    chsc_status: CHSCStatus,
    error: str,
) -> BatchACSCResult:
    return BatchACSCResult(
        sid=sid,
        phsc=PHSCResult(status=phsc_status, error=error),
        chsc=CHSCResult(status=chsc_status, error=error),
        acsc=ACSCResult(status=acsc_status, error=error),
        cross_h=np.full((0, 6), np.nan),
        cross_h2=np.full((0, 6), np.nan),
    )


def _prepare(sid: str, atoms: Atoms) -> _PreparedACSC:
    base, d_star = canonicalize_phsc_geometry(atoms)
    h_atomic = float(STEP_FRACTION * d_star)
    if not np.isfinite(h_atomic) or h_atomic <= 0.0:
        raise PHSCNumericalError("PHSC atomic step must be positive and finite")
    atomic_probes = tuple(
        probe
        for coordinate in range(3 * len(base))
        for probe in phsc_probe_group(base, coordinate, h_atomic)
    )
    expected_atomic = 12 * len(base)
    if len(atomic_probes) != expected_atomic:
        raise BatchACSCError("PHSC probe set is incomplete")
    strain_probes = _chsc_probe_group(base)
    if len(strain_probes) != 85:
        raise BatchACSCError("CHSC probe set is incomplete")
    return _PreparedACSC(
        sid=sid,
        base=base,
        d_star=d_star,
        h_atomic=h_atomic,
        probes=atomic_probes + strain_probes,
    )


def _phsc_result(
    h_h: np.ndarray, h_h2: np.ndarray, *, d_star: float, h: float, n_atoms: int
) -> PHSCResult:
    spectral = analyze_hessian_pair(h_h, h_h2)
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
        force_call_count=12 * n_atoms,
    )


def _acsc_result(
    spectral: object,
    *,
    d_star: float,
    evaluations: int,
    phsc_status: PHSCStatus,
    chsc_status: CHSCStatus,
) -> ACSCResult:
    # Keep this conversion explicit so abstention and incremental-gate semantics
    # cannot be inferred from truthiness or a nullable table column.
    from src.next13_acsc import ACSCSpectralResult

    if not isinstance(spectral, ACSCSpectralResult):
        raise BatchACSCError("ACSC core returned an unexpected result type")
    coupling_only = bool(
        phsc_status is PHSCStatus.RESOLVED_NONNEGATIVE
        and chsc_status is CHSCStatus.RESOLVED_NONNEGATIVE
        and spectral.status is ACSCStatus.RESOLVED_NEGATIVE
    )
    return ACSCResult(
        status=spectral.status,
        negative=spectral.negative,
        coupling_only_negative=coupling_only,
        d_star=d_star,
        lambda_h=spectral.lambda_h,
        lambda_h2=spectral.lambda_h2,
        lambda_r=spectral.lambda_r,
        e_num=spectral.e_num,
        u_num=spectral.u_num,
        l_num=spectral.l_num,
        tau_alg=spectral.tau_alg,
        antisymmetric_norm_h=spectral.antisymmetric_norm_h,
        antisymmetric_norm_h2=spectral.antisymmetric_norm_h2,
        prediction_evaluation_count=evaluations,
    )


def _parse_prediction(
    item: _PreparedACSC,
    energies: Sequence[float],
    forces: Sequence[np.ndarray],
) -> BatchACSCResult:
    n_atoms = len(item.base)
    atomic_count = 12 * n_atoms
    expected_count = atomic_count + 85
    if len(energies) != expected_count or len(forces) != expected_count:
        raise BatchACSCError("one ACSC group has an incomplete prediction")

    dimension = 3 * n_atoms
    atomic_h = np.empty((dimension, dimension), dtype=np.float64)
    atomic_h2 = np.empty((dimension, dimension), dtype=np.float64)
    for coordinate in range(dimension):
        offset = 4 * coordinate
        column_h, column_h2 = hessian_columns_from_force_samples(
            *forces[offset : offset + 4], h=item.h_atomic
        )
        atomic_h[:, coordinate] = column_h
        atomic_h2[:, coordinate] = column_h2
    phsc = _phsc_result(
        atomic_h,
        atomic_h2,
        d_star=item.d_star,
        h=item.h_atomic,
        n_atoms=n_atoms,
    )

    strain_energies = energies[atomic_count:]
    chsc, strain_h, strain_h2 = _result_from_energies(strain_energies, n_atoms)
    strain_forces = forces[atomic_count:]
    axial = tuple(
        np.stack([strain_forces[1 + 4 * axis + offset] for axis in range(6)])
        for offset in range(4)
    )
    cross_h, cross_h2 = cross_hessians_from_strain_forces(
        *axial,
        h=2**-7,
    )
    spectral = analyze_acsc_blocks(
        atomic_h,
        atomic_h2,
        strain_h,
        strain_h2,
        cross_h,
        cross_h2,
        d_star=item.d_star,
    )
    acsc = _acsc_result(
        spectral,
        d_star=item.d_star,
        evaluations=expected_count,
        phsc_status=phsc.status,
        chsc_status=chsc.status,
    )
    return BatchACSCResult(
        sid=item.sid,
        phsc=phsc,
        chsc=chsc,
        acsc=acsc,
        cross_h=cross_h,
        cross_h2=cross_h2,
    )


def evaluate_acsc_batch(
    sids: Sequence[str],
    structures: Sequence[Atoms],
    predictor: BatchForcePredictor,
    *,
    structures_per_call: int = FROZEN_STRUCTURES_PER_CALL,
) -> tuple[BatchACSCResult, ...]:
    """Evaluate complete combined groups, sorted by sid, without split probes."""

    if isinstance(sids, (str, bytes)) or isinstance(structures, (str, bytes)):
        raise BatchACSCError("sids and structures must be aligned sequences")
    if len(sids) != len(structures):
        raise BatchACSCError("sids and structures must have equal lengths")
    if any(type(sid) is not str or not sid for sid in sids):
        raise BatchACSCError("sids must be nonempty exact strings")
    if len(set(sids)) != len(sids):
        raise BatchACSCError("sids must be unique")
    if type(structures_per_call) is not int or structures_per_call <= 0:
        raise ValueError("structures_per_call must be a positive exact integer")
    if not callable(predictor):
        raise ValueError("predictor must be callable")

    ordered = sorted(zip(sids, structures, strict=True), key=lambda pair: pair[0])
    prepared: list[_PreparedACSC] = []
    completed: dict[str, BatchACSCResult] = {}
    # Preflight every probe before the first predictor call. A late
    # unrepresentable displacement must not leave a partially evaluated row.
    for sid, atoms in ordered:
        try:
            prepared.append(_prepare(sid, atoms))
        except (PHSCValidationError, CHSCValidationError) as exc:
            completed[sid] = _abstained_result(
                sid,
                acsc_status=ACSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
                phsc_status=PHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
                chsc_status=CHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
                error=f"unsupported ACSC geometry: {exc}",
            )
        except (PHSCNumericalError, CHSCNumericalError, ACSCNumericalError) as exc:
            completed[sid] = _abstained_result(
                sid,
                acsc_status=ACSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                phsc_status=PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                chsc_status=CHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                error=f"ACSC probe construction failed: {exc}",
            )

    for start in range(0, len(prepared), structures_per_call):
        chunk = prepared[start : start + structures_per_call]
        flat = [probe for item in chunk for probe in item.probes]
        try:
            prediction = predictor(flat)
            energies, forces, _stresses = _validated_prediction(prediction, flat)
        except Exception as exc:
            raise BatchACSCError(
                f"batch predictor failed: {type(exc).__name__}: {exc}"
            ) from None
        offset = 0
        for item in chunk:
            count = len(item.probes)
            try:
                completed[item.sid] = _parse_prediction(
                    item,
                    energies[offset : offset + count],
                    forces[offset : offset + count],
                )
            except (PHSCValidationError, CHSCValidationError, ACSCValidationError) as exc:
                raise BatchACSCError(
                    f"validated prediction violated ACSC contract for {item.sid}: {exc}"
                ) from None
            except (PHSCNumericalError, CHSCNumericalError, ACSCNumericalError) as exc:
                completed[item.sid] = _abstained_result(
                    item.sid,
                    acsc_status=ACSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                    phsc_status=PHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                    chsc_status=CHSCStatus.ABSTAIN_NUMERICAL_FAILURE,
                    error=f"ACSC numerical analysis failed: {exc}",
                )
            offset += count
        if offset != len(flat):
            raise BatchACSCError("unused predictor output remains after ACSC grouping")
    if set(completed) != {sid for sid, _atoms in ordered}:
        raise BatchACSCError("batch ACSC did not produce one result per sid")
    return tuple(completed[sid] for sid, _atoms in ordered)
