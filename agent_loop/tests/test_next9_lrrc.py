"""Contract tests for the LRRC-v0 engineering diagnostic."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ase import Atoms


def _base_atoms() -> Atoms:
    return Atoms(
        "H3",
        positions=[[0.4, 0.8, 1.1], [1.8, 0.6, 0.7], [3.2, 1.7, 0.9]],
        cell=[7.0, 8.0, 9.0],
        pbc=True,
    )


def _pair_quadratic_forces(atoms: Atoms, curvature: float = 1.0) -> np.ndarray:
    """Forces from curvature/2 times the sum of squared MIC pair distances."""

    pair_vectors = atoms.get_all_distances(mic=True, vector=True)
    return curvature * np.sum(pair_vectors, axis=1)


class _RecordingOracle:
    def __init__(self, function):
        self.function = function
        self.positions: list[np.ndarray] = []

    def __call__(self, atoms: Atoms) -> np.ndarray:
        self.positions.append(atoms.get_positions().copy())
        return self.function(atoms)


def test_frozen_constants_and_translation_projected_direction() -> None:
    from src.next9_lrrc import (
        FORCE_RMS_FLOOR,
        LRRC_VERSION,
        STEP_FRACTION,
        translation_projected_direction,
    )

    forces = np.array([[2.0, -1.0, 4.0], [-1.0, 3.0, 0.0], [5.0, 2.0, -2.0]])
    direction = translation_projected_direction(forces)

    assert LRRC_VERSION == "LRRC-v0"
    assert STEP_FRACTION == 2**-8
    assert FORCE_RMS_FLOOR == 1e-12
    assert direction is not None
    np.testing.assert_allclose(direction.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.sum(direction * direction, axis=1))),
        1.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    "forces",
    [
        np.zeros((2, 2)),
        np.zeros((2, 3, 1)),
        np.zeros((0, 3)),
        np.array([[0.0, np.nan, 0.0]]),
        np.array([[0.0, np.inf, 0.0]]),
    ],
)
def test_translation_projected_direction_rejects_invalid_arrays(forces: np.ndarray) -> None:
    from src.next9_lrrc import LRRCValidationError, translation_projected_direction

    with pytest.raises(LRRCValidationError):
        translation_projected_direction(forces)


def test_translation_projected_direction_returns_none_at_floor() -> None:
    from src.next9_lrrc import translation_projected_direction

    assert translation_projected_direction(np.ones((4, 3))) is None
    almost_uniform = np.zeros((2, 3))
    almost_uniform[1, 0] = 1e-12
    assert translation_projected_direction(almost_uniform) is None


def test_translation_projected_direction_handles_all_finite_float_magnitudes() -> None:
    from src.next9_lrrc import translation_projected_direction

    largest = np.finfo(float).max
    forces = np.array(
        [[largest, -largest, 0.0], [-largest, largest, 0.0], [largest, largest, 0.0]]
    )

    direction = translation_projected_direction(forces)

    assert direction is not None
    assert np.all(np.isfinite(direction))
    np.testing.assert_allclose(direction.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.sum(direction * direction, axis=1))),
        1.0,
        atol=1e-15,
    )


def test_median_nearest_neighbor_distance_is_mic_and_order_invariant() -> None:
    from src.next9_lrrc import median_nearest_neighbor_distance

    atoms = Atoms(
        "H3",
        positions=[[0.2, 0.2, 0.2], [2.2, 0.2, 0.2], [5.7, 0.2, 0.2]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    wrapped = atoms.copy()
    wrapped.positions[0] += wrapped.cell[0]
    permuted = atoms[[2, 0, 1]]

    expected = 0.5
    assert median_nearest_neighbor_distance(atoms) == pytest.approx(expected)
    assert median_nearest_neighbor_distance(wrapped) == pytest.approx(expected)
    assert median_nearest_neighbor_distance(permuted) == pytest.approx(expected)


@pytest.mark.parametrize(
    "atoms",
    [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H2", positions=np.zeros((2, 3))),
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]),
    ],
)
def test_median_nearest_neighbor_distance_rejects_invalid_geometry(atoms: Atoms) -> None:
    from src.next9_lrrc import LRRCValidationError, median_nearest_neighbor_distance

    with pytest.raises(LRRCValidationError):
        median_nearest_neighbor_distance(atoms)


def test_positive_pair_quadratic_has_analytic_curvature_and_exact_call_points() -> None:
    from src.next9_lrrc import LRRCStatus, STEP_FRACTION, evaluate_lrrc

    atoms = _base_atoms()
    oracle = _RecordingOracle(_pair_quadratic_forces)
    result = evaluate_lrrc(atoms, oracle)

    assert result.status is LRRCStatus.OK
    assert result.negative is False
    assert result.kappa_h == pytest.approx(3.0, abs=1e-11)
    assert result.kappa_h2 == pytest.approx(3.0, abs=1e-11)
    assert result.kappa_r == pytest.approx(3.0, abs=1e-11)
    assert result.error_proxy == pytest.approx(0.0, abs=1e-11)
    assert result.u_num == pytest.approx(3.0, abs=1e-11)
    assert result.h == pytest.approx(STEP_FRACTION * result.d_star)
    assert result.error is None
    assert len(oracle.positions) == 5

    base = atoms.get_positions()
    direction = (oracle.positions[1] - base) / result.h
    np.testing.assert_allclose(oracle.positions[0], base)
    np.testing.assert_allclose(oracle.positions[1], base + result.h * direction)
    np.testing.assert_allclose(oracle.positions[2], base - result.h * direction)
    np.testing.assert_allclose(oracle.positions[3], base + 0.5 * result.h * direction)
    np.testing.assert_allclose(oracle.positions[4], base - 0.5 * result.h * direction)
    np.testing.assert_allclose(direction.mean(axis=0), 0.0, atol=1e-14)


def test_inverted_pair_quadratic_is_negative_only_when_all_proxies_are_negative() -> None:
    from src.next9_lrrc import LRRCStatus, evaluate_lrrc

    result = evaluate_lrrc(
        _base_atoms(), lambda atoms: _pair_quadratic_forces(atoms, curvature=-2.0)
    )

    assert result.status is LRRCStatus.OK
    assert result.kappa_h == pytest.approx(-6.0, abs=1e-11)
    assert result.kappa_h2 == pytest.approx(-6.0, abs=1e-11)
    assert result.u_num == pytest.approx(-6.0, abs=1e-11)
    assert result.negative is True


@pytest.mark.parametrize(
    ("kappa_h", "kappa_h2", "u_num", "expected"),
    [
        (-1.0, -1.0, -1.0, True),
        (-1.0, -1.0, 1.0, False),
        (-1.0, 1.0, -1.0, False),
        (-1.0, 1.0, 1.0, False),
        (1.0, -1.0, -1.0, False),
        (1.0, -1.0, 1.0, False),
        (1.0, 1.0, -1.0, False),
        (1.0, 1.0, 1.0, False),
        (0.0, -1.0, -1.0, False),
        (-1.0, 0.0, -1.0, False),
        (-1.0, -1.0, 0.0, False),
    ],
)
def test_lrrc_negative_gate_requires_all_three_strictly_negative(
    kappa_h: float,
    kappa_h2: float,
    u_num: float,
    expected: bool,
) -> None:
    from src.next9_lrrc import lrrc_negative_gate

    assert lrrc_negative_gate(kappa_h, kappa_h2, u_num) is expected


def test_lrrc_is_invariant_to_rigid_and_index_representations() -> None:
    from src.next9_lrrc import LRRCStatus, evaluate_lrrc

    base = _base_atoms()
    translated = base.copy()
    translated.translate([1.3, -0.4, 0.7])

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = base.copy()
    rotated.set_positions(base.get_positions() @ rotation.T)
    rotated.set_cell(np.asarray(base.cell) @ rotation.T)

    permuted = base[[2, 0, 1]]
    wrapped = base.copy()
    wrapped.positions[1] += wrapped.cell[0] - wrapped.cell[1]

    reference = evaluate_lrrc(base, _pair_quadratic_forces)
    assert reference.status is LRRCStatus.OK
    for transformed in (translated, rotated, permuted, wrapped):
        observed = evaluate_lrrc(transformed, _pair_quadratic_forces)
        assert observed.status is LRRCStatus.OK
        assert observed.negative is reference.negative
        assert observed.d_star == pytest.approx(reference.d_star, abs=1e-12)
        assert observed.kappa_h == pytest.approx(reference.kappa_h, abs=1e-10)
        assert observed.kappa_h2 == pytest.approx(reference.kappa_h2, abs=1e-10)
        assert observed.kappa_r == pytest.approx(reference.kappa_r, abs=1e-10)


def test_exact_zero_force_saddle_uses_documented_stationary_fallback() -> None:
    from src.next9_lrrc import LRRCStatus, evaluate_lrrc

    atoms = _base_atoms()
    reference = atoms.get_positions().copy()

    def saddle_forces(probe: Atoms) -> np.ndarray:
        displacement = probe.get_positions() - reference
        forces = np.zeros_like(displacement)
        forces[:, 0] = displacement[:, 0]
        forces[:, 1] = -displacement[:, 1]
        return forces

    oracle = _RecordingOracle(saddle_forces)
    result = evaluate_lrrc(atoms, oracle)

    assert result.status is LRRCStatus.STATIONARY_FALLBACK
    assert result.negative is None
    assert result.kappa_h is None
    assert result.u_num is None
    assert len(oracle.positions) == 1


def test_evaluator_has_stable_failure_codes_and_no_traceback_text() -> None:
    from src.next9_lrrc import LRRCStatus, evaluate_lrrc

    atoms = _base_atoms()

    def exploding_oracle(_atoms: Atoms) -> np.ndarray:
        raise RuntimeError("oracle exploded")

    exception = evaluate_lrrc(atoms, exploding_oracle)
    wrong_shape = evaluate_lrrc(atoms, lambda _atoms: np.zeros((2, 3)))
    nonfinite = evaluate_lrrc(
        atoms, lambda _atoms: np.full((len(atoms), 3), np.nan)
    )
    unsupported = evaluate_lrrc(
        Atoms("H2", positions=np.zeros((2, 3))),
        lambda _atoms: np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )

    assert exception.status is LRRCStatus.ABSTAIN_FORCE_FAILURE
    assert exception.error is not None
    assert "RuntimeError: oracle exploded" in exception.error
    assert "Traceback" not in exception.error
    assert wrong_shape.status is LRRCStatus.ABSTAIN_INVALID_FORCE
    assert nonfinite.status is LRRCStatus.ABSTAIN_INVALID_FORCE
    assert unsupported.status is LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY
    assert exception.negative is None
    assert wrong_shape.negative is None


def test_perturbed_oracle_failure_abstains_after_preserving_diagnostic() -> None:
    from src.next9_lrrc import LRRCStatus, evaluate_lrrc

    calls = 0

    def fails_on_third_call(atoms: Atoms) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ArithmeticError("third-call failure")
        return _pair_quadratic_forces(atoms)

    result = evaluate_lrrc(_base_atoms(), fails_on_third_call)

    assert calls == 3
    assert result.status is LRRCStatus.ABSTAIN_FORCE_FAILURE
    assert result.error is not None
    assert "ArithmeticError: third-call failure" in result.error


def test_decision_composition_is_fail_open_and_preserves_baseline_fallbacks() -> None:
    from src.next9_lrrc import Decision, LRRCResult, LRRCStatus, compose_decision

    positive = LRRCResult(status=LRRCStatus.OK, negative=False)
    negative = LRRCResult(status=LRRCStatus.OK, negative=True)
    stationary = LRRCResult(status=LRRCStatus.STATIONARY_FALLBACK)
    failed = LRRCResult(status=LRRCStatus.ABSTAIN_FORCE_FAILURE, error="failed")

    assert compose_decision(Decision.KEEP, positive) is Decision.KEEP
    assert compose_decision(Decision.REJECT, positive) is Decision.REJECT
    assert compose_decision(Decision.KEEP, negative) is Decision.REJECT
    assert compose_decision(Decision.REJECT, negative) is Decision.REJECT
    assert compose_decision(Decision.KEEP, stationary) is Decision.KEEP
    assert compose_decision(Decision.REJECT, stationary) is Decision.REJECT
    assert compose_decision(Decision.KEEP, failed) is Decision.ABSTAIN
    assert compose_decision(Decision.REJECT, failed) is Decision.ABSTAIN
    assert compose_decision(Decision.ABSTAIN, positive) is Decision.ABSTAIN
    assert compose_decision(Decision.ABSTAIN, negative) is Decision.ABSTAIN


def test_public_records_are_frozen_and_status_values_are_stable() -> None:
    from src.next9_lrrc import (
        Decision,
        LRRCResult,
        LRRCStatus,
        QuotaCRCRow,
    )

    result = LRRCResult(status=LRRCStatus.OK, negative=False)
    row = QuotaCRCRow("row-1", "group-1", 0.5, Decision.REJECT)

    with pytest.raises(FrozenInstanceError):
        result.negative = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        row.score = 0.1  # type: ignore[misc]
    assert LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY.value == "abstain_unsupported_geometry"
    assert LRRCStatus.ABSTAIN_FORCE_FAILURE.value == "abstain_force_failure"
    assert LRRCStatus.ABSTAIN_INVALID_FORCE.value == "abstain_invalid_force"
    assert [decision.value for decision in Decision] == ["keep", "reject", "abstain"]


def test_quota_crc_promotes_kth_threshold_with_all_ties_and_preserves_order() -> None:
    from src.next9_lrrc import Decision, QuotaCRCRow, quota_crc

    rows = (
        QuotaCRCRow("a", "g", 0.1, Decision.REJECT),
        QuotaCRCRow("b", "g", 0.2, Decision.REJECT),
        QuotaCRCRow("c", "g", 0.2, Decision.REJECT),
        QuotaCRCRow("d", "g", 0.2, Decision.KEEP),
        QuotaCRCRow("e", "g", 0.5, Decision.REJECT),
        QuotaCRCRow("f", "g", 0.6, Decision.KEEP),
        QuotaCRCRow("g", "g", 0.0, Decision.ABSTAIN),
        QuotaCRCRow("h", "g", np.nan, Decision.ABSTAIN),
        QuotaCRCRow("i", "unsupported", np.nan, Decision.REJECT, supported=False),
        QuotaCRCRow("j", "single", 4.0, Decision.REJECT),
    )

    output = quota_crc(rows)

    assert [row.row_id for row in output] == [row.row_id for row in rows]
    assert [row.decision for row in output] == [
        Decision.KEEP,
        Decision.KEEP,
        Decision.KEEP,
        Decision.KEEP,
        Decision.REJECT,
        Decision.KEEP,
        Decision.ABSTAIN,
        Decision.ABSTAIN,
        Decision.REJECT,
        Decision.KEEP,
    ]
    assert np.isnan(output[7].score)
    assert np.isnan(output[8].score)

    input_rejections = {row.row_id for row in rows if row.decision is Decision.REJECT}
    output_rejections = {row.row_id for row in output if row.decision is Decision.REJECT}
    assert output_rejections <= input_rejections
    assert rows[0].decision is Decision.REJECT


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
def test_quota_crc_rejects_nonfinite_scores_on_eligible_rows(score: float) -> None:
    from src.next9_lrrc import (
        Decision,
        QuotaCRCRow,
        QuotaCRCValidationError,
        quota_crc,
    )

    with pytest.raises(QuotaCRCValidationError, match="finite"):
        quota_crc((QuotaCRCRow("row", "group", score, Decision.REJECT),))


def test_quota_crc_rejects_duplicate_row_ids() -> None:
    from src.next9_lrrc import (
        Decision,
        QuotaCRCRow,
        QuotaCRCValidationError,
        quota_crc,
    )

    rows = (
        QuotaCRCRow("same", "a", 0.1, Decision.REJECT),
        QuotaCRCRow("same", "b", 0.2, Decision.KEEP),
    )
    with pytest.raises(QuotaCRCValidationError, match="duplicate row_id"):
        quota_crc(rows)
