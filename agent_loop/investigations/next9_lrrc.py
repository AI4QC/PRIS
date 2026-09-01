"""Deterministic LRRC-v0 numerical diagnostics and decision utilities.

The module provides engineering diagnostics only.  Its numerical proxy is not a
confidence bound and must not be interpreted as formal or physical certainty.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any, Callable

import numpy as np
from ase import Atoms


LRRC_VERSION = "LRRC-v0"
STEP_FRACTION = 2**-8
FORCE_RMS_FLOOR = 1e-12


class LRRCValidationError(ValueError):
    """Raised when a direct LRRC utility receives invalid numerical input."""


class LRRCStatus(str, Enum):
    """Stable outcome codes for an LRRC evaluation."""

    OK = "ok"
    STATIONARY_FALLBACK = "stationary_fallback"
    ABSTAIN_UNSUPPORTED_GEOMETRY = "abstain_unsupported_geometry"
    ABSTAIN_FORCE_FAILURE = "abstain_force_failure"
    ABSTAIN_INVALID_FORCE = "abstain_invalid_force"
    ABSTAIN_NUMERICAL_FAILURE = "abstain_numerical_failure"


class Decision(str, Enum):
    """Stable downstream decision values."""

    KEEP = "keep"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class LRRCResult:
    """Immutable LRRC outcome; ``u_num`` is a numerical proxy, not a bound."""

    status: LRRCStatus
    negative: bool | None = None
    d_star: float | None = None
    h: float | None = None
    kappa_h: float | None = None
    kappa_h2: float | None = None
    kappa_r: float | None = None
    error_proxy: float | None = None
    u_num: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class QuotaCRCRow:
    """Immutable quota-CRC input/output row in deterministic input order."""

    row_id: str
    group_id: str
    score: float
    decision: Decision
    supported: bool = True


class QuotaCRCValidationError(ValueError):
    """Raised when quota-CRC rows violate their deterministic input contract."""


ForceOracle = Callable[[Atoms], Any]


class _ForceOracleFailure(RuntimeError):
    """Internal marker for an exception raised by a force oracle."""


class _InvalidForceOutput(ValueError):
    """Internal marker for invalid force-oracle output."""


def translation_projected_direction(forces: Any) -> np.ndarray | None:
    """Return the mean-zero, unit per-atom-RMS direction of finite ``(N, 3)`` forces.

    Inputs that cannot represent a nonempty finite floating-point ``(N, 3)``
    array raise :class:`LRRCValidationError`.  A translation-projected RMS at or
    below :data:`FORCE_RMS_FLOOR` returns ``None`` deterministically.
    """

    try:
        array = np.asarray(forces, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LRRCValidationError(f"forces are not a floating-point array: {exc}") from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 3:
        raise LRRCValidationError(
            f"forces must have shape (N, 3) with N >= 1; got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise LRRCValidationError("forces must contain only finite values")

    magnitude = float(np.max(np.abs(array)))
    if magnitude == 0.0:
        return None
    scaled = array / magnitude
    projected_scaled = scaled - np.mean(scaled, axis=0, keepdims=True)
    scaled_rms = float(
        np.sqrt(np.mean(np.sum(projected_scaled * projected_scaled, axis=1)))
    )
    if scaled_rms <= FORCE_RMS_FLOOR / magnitude:
        return None
    return projected_scaled / scaled_rms


def median_nearest_neighbor_distance(atoms: Atoms) -> float:
    """Return the median per-atom nearest positive finite MIC distance.

    ``atoms`` must be an ASE :class:`~ase.Atoms` object with at least two atoms
    and a geometry for which minimum-image distances are defined.  Invalid
    geometry or the absence of a positive finite neighbor raises
    :class:`LRRCValidationError`.
    """

    if not isinstance(atoms, Atoms):
        raise LRRCValidationError("atoms must be an ase.Atoms instance")
    if len(atoms) < 2:
        raise LRRCValidationError("atoms must contain at least two atoms")
    if not np.all(np.isfinite(atoms.get_positions())):
        raise LRRCValidationError("atomic positions must contain only finite values")

    try:
        distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    except Exception as exc:
        raise LRRCValidationError(f"minimum-image distances are unavailable: {exc}") from exc
    if distances.shape != (len(atoms), len(atoms)):
        raise LRRCValidationError(
            "minimum-image distance matrix has an unexpected shape"
        )

    nearest: list[float] = []
    for row in distances:
        positive = row[np.isfinite(row) & (row > 0.0)]
        if positive.size == 0:
            raise LRRCValidationError(
                "each atom must have a positive finite neighbor distance"
            )
        nearest.append(float(np.min(positive)))

    result = float(np.median(np.asarray(nearest, dtype=float)))
    if not np.isfinite(result) or result <= 0.0:
        raise LRRCValidationError("median nearest-neighbor distance must be positive and finite")
    return result


def _exception_text(exc: Exception) -> str:
    """Format an exception diagnostic without serializing a traceback."""

    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _checked_force_call(
    force_oracle: ForceOracle,
    probe: Atoms,
    *,
    stage: str,
) -> np.ndarray:
    try:
        raw_forces = force_oracle(probe.copy())
    except Exception as exc:
        raise _ForceOracleFailure(
            f"force oracle failed at {stage}: {_exception_text(exc)}"
        ) from None

    try:
        forces = np.asarray(raw_forces, dtype=float)
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
        raise _InvalidForceOutput(
            f"force output at {stage} must contain only finite values"
        )
    return forces


def _abstain(status: LRRCStatus, error: str) -> LRRCResult:
    return LRRCResult(status=status, error=error)


def lrrc_negative_gate(kappa_h: float, kappa_h2: float, u_num: float) -> bool:
    """Return true only when both curvatures and the numerical proxy are negative."""

    return bool(kappa_h < 0.0 and kappa_h2 < 0.0 and u_num < 0.0)


def evaluate_lrrc(atoms: Atoms, force_oracle: ForceOracle) -> LRRCResult:
    """Evaluate LRRC-v0 with one base and, when nonstationary, four force calls.

    Oracle exceptions, invalid outputs, unsupported geometry, and numerical
    failures return explicit abstention codes.  They do not raise through this
    evaluator.  The negative result is a local numerical diagnostic along the
    force-derived direction; it is not a confidence bound or certainty claim.
    """

    if not isinstance(atoms, Atoms):
        return _abstain(
            LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            "atoms must be an ase.Atoms instance",
        )
    if len(atoms) < 2:
        return _abstain(
            LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            "LRRC geometry must contain at least two atoms",
        )
    try:
        base = atoms.copy()
        base_positions = np.asarray(base.get_positions(), dtype=float)
    except Exception as exc:
        return _abstain(
            LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            f"could not copy atomic geometry: {_exception_text(exc)}",
        )
    if base_positions.shape != (len(base), 3) or not np.all(np.isfinite(base_positions)):
        return _abstain(
            LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            "atomic positions must have shape (N, 3) and contain only finite values",
        )

    try:
        base_forces = _checked_force_call(force_oracle, base, stage="x")
    except _ForceOracleFailure as exc:
        return _abstain(LRRCStatus.ABSTAIN_FORCE_FAILURE, str(exc))
    except _InvalidForceOutput as exc:
        return _abstain(LRRCStatus.ABSTAIN_INVALID_FORCE, str(exc))

    direction = translation_projected_direction(base_forces)
    if direction is None:
        return LRRCResult(status=LRRCStatus.STATIONARY_FALLBACK)

    try:
        d_star = median_nearest_neighbor_distance(base)
    except LRRCValidationError as exc:
        return _abstain(
            LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            f"unsupported LRRC geometry: {exc}",
        )
    h = float(STEP_FRACTION * d_star)
    if not np.isfinite(h) or h <= 0.0:
        return _abstain(
            LRRCStatus.ABSTAIN_NUMERICAL_FAILURE,
            "LRRC step size must be positive and finite",
        )

    force_samples: list[np.ndarray] = []
    displacements = (h, -h, 0.5 * h, -0.5 * h)
    stage_names = ("x+h*u", "x-h*u", "x+h/2*u", "x-h/2*u")
    try:
        for displacement, stage in zip(displacements, stage_names, strict=True):
            probe = base.copy()
            positions = base_positions + displacement * direction
            if not np.all(np.isfinite(positions)):
                return _abstain(
                    LRRCStatus.ABSTAIN_NUMERICAL_FAILURE,
                    f"perturbed positions at {stage} are nonfinite",
                )
            probe.set_positions(positions)
            force_samples.append(_checked_force_call(force_oracle, probe, stage=stage))
    except _ForceOracleFailure as exc:
        return _abstain(LRRCStatus.ABSTAIN_FORCE_FAILURE, str(exc))
    except _InvalidForceOutput as exc:
        return _abstain(LRRCStatus.ABSTAIN_INVALID_FORCE, str(exc))
    except Exception as exc:
        return _abstain(
            LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY,
            f"could not construct perturbed geometry: {_exception_text(exc)}",
        )

    force_plus_h, force_minus_h, force_plus_h2, force_minus_h2 = force_samples
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            kappa_h = -float(
                np.sum(direction * ((force_plus_h - force_minus_h) / (2.0 * h)))
                / len(base)
            )
            kappa_h2 = -float(
                np.sum(direction * ((force_plus_h2 - force_minus_h2) / h))
                / len(base)
            )
            kappa_r = float((4.0 * kappa_h2 - kappa_h) / 3.0)
            error_proxy = float(abs(kappa_h2 - kappa_h) / 3.0)
            u_num = float(kappa_r + error_proxy)
    except (FloatingPointError, OverflowError) as exc:
        return _abstain(
            LRRCStatus.ABSTAIN_NUMERICAL_FAILURE,
            f"LRRC curvature calculation failed: {_exception_text(exc)}",
        )

    diagnostics = (kappa_h, kappa_h2, kappa_r, error_proxy, u_num)
    if not all(np.isfinite(value) for value in diagnostics):
        return _abstain(
            LRRCStatus.ABSTAIN_NUMERICAL_FAILURE,
            "LRRC curvature diagnostics must be finite",
        )
    negative = lrrc_negative_gate(kappa_h, kappa_h2, u_num)
    return LRRCResult(
        status=LRRCStatus.OK,
        negative=negative,
        d_star=d_star,
        h=h,
        kappa_h=kappa_h,
        kappa_h2=kappa_h2,
        kappa_r=kappa_r,
        error_proxy=error_proxy,
        u_num=u_num,
    )


def compose_decision(baseline: Decision, lrrc: LRRCResult) -> Decision:
    """Compose baseline and LRRC decisions with explicit failure abstention."""

    if baseline is Decision.ABSTAIN:
        return Decision.ABSTAIN
    if not isinstance(lrrc, LRRCResult):
        return Decision.ABSTAIN
    if lrrc.status is LRRCStatus.OK:
        if lrrc.negative is True:
            return Decision.REJECT
        if lrrc.negative is False:
            return baseline
        return Decision.ABSTAIN
    if lrrc.status is LRRCStatus.STATIONARY_FALLBACK:
        return baseline
    if lrrc.status.name.startswith("ABSTAIN_"):
        return Decision.ABSTAIN
    return Decision.ABSTAIN


def quota_crc(rows: tuple[QuotaCRCRow, ...] | list[QuotaCRCRow]) -> tuple[QuotaCRCRow, ...]:
    """Apply deterministic per-group quota promotion without adding rejections.

    For each group, eligible rows are supported, non-abstaining rows with finite
    scores.  With ``n`` eligible rows, every eligible row at or below the
    ``ceil(sqrt(n))``-th smallest score becomes ``KEEP`` (including all ties).
    Unsupported and abstaining rows remain unchanged.
    """

    frozen_rows = tuple(rows)
    seen_ids: set[str] = set()
    groups: dict[str, list[tuple[int, float]]] = {}
    for index, row in enumerate(frozen_rows):
        if not isinstance(row, QuotaCRCRow):
            raise QuotaCRCValidationError(
                f"row at index {index} must be a QuotaCRCRow"
            )
        if not isinstance(row.row_id, str) or not row.row_id:
            raise QuotaCRCValidationError("row_id must be a nonempty string")
        if row.row_id in seen_ids:
            raise QuotaCRCValidationError(f"duplicate row_id: {row.row_id}")
        seen_ids.add(row.row_id)
        if not isinstance(row.group_id, str) or not row.group_id:
            raise QuotaCRCValidationError("group_id must be a nonempty string")
        if not isinstance(row.decision, Decision):
            raise QuotaCRCValidationError(
                f"row {row.row_id} has an invalid input decision"
            )
        if not isinstance(row.supported, bool):
            raise QuotaCRCValidationError(
                f"row {row.row_id} supported flag must be boolean"
            )
        if not row.supported or row.decision is Decision.ABSTAIN:
            continue
        try:
            score = float(row.score)
        except (TypeError, ValueError) as exc:
            raise QuotaCRCValidationError(
                f"eligible row {row.row_id} score must be finite"
            ) from exc
        if not math.isfinite(score):
            raise QuotaCRCValidationError(
                f"eligible row {row.row_id} score must be finite"
            )
        groups.setdefault(row.group_id, []).append((index, score))

    decisions = list(frozen_rows)
    for eligible in groups.values():
        k = math.ceil(math.sqrt(len(eligible)))
        threshold = sorted(score for _, score in eligible)[k - 1]
        for index, score in eligible:
            if score <= threshold:
                decisions[index] = replace(decisions[index], decision=Decision.KEEP)
    return tuple(decisions)
