"""Contract tests for the label-free CHSC-v0 numerical core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ase import Atoms
from scipy.linalg import logm


def _base_atoms() -> Atoms:
    return Atoms(
        "Si2O4",
        scaled_positions=[
            [0.11, 0.23, 0.37],
            [0.57, 0.61, 0.79],
            [0.31, 0.73, 0.17],
            [0.83, 0.19, 0.43],
            [0.47, 0.41, 0.89],
            [0.71, 0.87, 0.59],
        ],
        cell=[[6.2, 0.0, 0.0], [0.7, 7.1, 0.0], [0.3, 0.5, 8.4]],
        pbc=True,
    )


def _strain_quadratic_oracle(atoms: Atoms, hessian: np.ndarray):
    from src.next12_chsc import strain_basis

    reference_cell = np.asarray(atoms.cell.array, dtype=np.float64)
    basis = strain_basis()
    n_atoms = len(atoms)

    def oracle(probe: Atoms) -> float:
        relative = np.linalg.solve(reference_cell, probe.cell.array)
        strain = np.real_if_close(logm(relative.T), tol=1000)
        assert not np.iscomplexobj(strain)
        coordinates = np.einsum("aij,ij->a", basis, strain)
        return float(0.5 * n_atoms * coordinates @ hessian @ coordinates)

    return oracle


def test_frozen_constants_basis_and_direction_order() -> None:
    from src.next12_chsc import (
        CHSC_VERSION,
        STEP_STRAIN,
        direction_set,
        strain_basis,
    )

    basis = strain_basis()
    directions = direction_set()

    assert CHSC_VERSION == "CHSC-v0"
    assert STEP_STRAIN == 2**-7
    assert basis.shape == (6, 3, 3)
    np.testing.assert_allclose(
        np.einsum("aij,bij->ab", basis, basis), np.eye(6), atol=2e-15
    )
    np.testing.assert_allclose(basis, np.swapaxes(basis, 1, 2), atol=0.0)

    assert directions.shape == (21, 6)
    np.testing.assert_allclose(directions[:6], np.eye(6), atol=0.0)
    expected_pairs = [
        (i, j) for i in range(6) for j in range(i + 1, 6)
    ]
    for row, (i, j) in zip(directions[6:], expected_pairs, strict=True):
        expected = np.zeros(6)
        expected[[i, j]] = 1 / np.sqrt(2)
        np.testing.assert_allclose(row, expected, atol=0.0)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_directional_curvatures_reconstruct_exact_symmetric_hessian() -> None:
    from src.next12_chsc import direction_set, directional_curvatures_to_hessian

    raw = np.arange(36, dtype=float).reshape(6, 6) / 13.0
    expected = 0.5 * (raw + raw.T) - 1.75 * np.eye(6)
    directions = direction_set()
    curvatures = np.einsum("di,ij,dj->d", directions, expected, directions)

    observed = directional_curvatures_to_hessian(curvatures)

    np.testing.assert_allclose(observed, expected, atol=2e-15)


@pytest.mark.parametrize(
    "curvatures",
    [
        np.zeros(20),
        np.zeros(22),
        np.zeros((21, 1)),
        np.full(21, np.nan),
        ["bad"] * 21,
    ],
)
def test_hessian_reconstruction_rejects_invalid_curvatures(curvatures: object) -> None:
    from src.next12_chsc import CHSCValidationError, directional_curvatures_to_hessian

    with pytest.raises(CHSCValidationError):
        directional_curvatures_to_hessian(curvatures)


def test_deformation_is_exponential_affine_and_fractional_coordinates_are_fixed() -> None:
    from scipy.linalg import expm

    from src.next12_chsc import deform_cell, strain_basis

    atoms = _base_atoms()
    direction = np.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.6])
    direction /= np.linalg.norm(direction)
    step = 2**-7

    deformed = deform_cell(atoms, direction, step)

    generator = np.einsum("a,aij->ij", direction, strain_basis())
    expected_cell = atoms.cell.array @ expm(step * generator).T
    np.testing.assert_allclose(deformed.cell.array, expected_cell, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        deformed.get_scaled_positions(wrap=False),
        atoms.get_scaled_positions(wrap=False),
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_array_equal(deformed.pbc, atoms.pbc)
    np.testing.assert_allclose(atoms.cell.array, _base_atoms().cell.array, atol=0.0)


@pytest.mark.parametrize(
    ("atoms", "direction", "step"),
    [
        (_base_atoms(), np.zeros(6), 2**-7),
        (_base_atoms(), np.ones(5), 2**-7),
        (_base_atoms(), np.full(6, np.nan), 2**-7),
        (_base_atoms(), np.eye(6)[0], 0.0),
        (_base_atoms(), np.eye(6)[0], np.nan),
        (Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=False), np.eye(6)[0], 2**-7),
    ],
)
def test_deformation_rejects_unsupported_geometry_direction_or_step(
    atoms: Atoms, direction: object, step: float
) -> None:
    from src.next12_chsc import CHSCValidationError, deform_cell

    with pytest.raises(CHSCValidationError):
        deform_cell(atoms, direction, step)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("eigenvalues", "expected_status", "expected_negative"),
    [
        ([-3.0, 1.0, 2.0, 3.0, 4.0, 5.0], "resolved_negative", True),
        ([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], "resolved_nonnegative", False),
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "near_zero_or_inconsistent", False),
    ],
)
def test_exact_strain_spectra_have_three_frozen_success_states(
    eigenvalues: list[float], expected_status: str, expected_negative: bool
) -> None:
    from src.next12_chsc import CHSCStatus, analyze_strain_hessian_pair

    matrix = np.diag(eigenvalues)
    result = analyze_strain_hessian_pair(matrix, matrix)

    assert result.status is CHSCStatus(expected_status)
    assert result.negative is expected_negative
    assert result.lambda_h == pytest.approx(min(eigenvalues), abs=2e-14)
    assert result.lambda_h2 == pytest.approx(min(eigenvalues), abs=2e-14)
    assert result.lambda_r == pytest.approx(min(eigenvalues), abs=2e-14)
    assert result.e_num == pytest.approx(0.0, abs=2e-14)
    assert result.tau_alg == pytest.approx(
        64 * 6 * np.finfo(np.float64).eps * max(1.0, max(abs(x) for x in eigenvalues))
    )


def test_two_scale_inconsistency_and_interval_crossing_abstain() -> None:
    from src.next12_chsc import CHSCStatus, analyze_strain_hessian_pair

    sign_flip = analyze_strain_hessian_pair(
        np.diag([-2.0, 1, 1, 1, 1, 1]),
        np.diag([2.0, 1, 1, 1, 1, 1]),
    )
    interval_crossing = analyze_strain_hessian_pair(
        np.diag([-10.0, 20, 20, 20, 20, 20]),
        np.diag([-1.0, 20, 20, 20, 20, 20]),
    )

    assert sign_flip.status is CHSCStatus.NEAR_ZERO_OR_INCONSISTENT
    assert sign_flip.lambda_h < 0.0 < sign_flip.lambda_h2
    assert interval_crossing.lambda_h < 0.0
    assert interval_crossing.lambda_h2 < 0.0
    assert interval_crossing.u_num > 0.0
    assert interval_crossing.status is CHSCStatus.NEAR_ZERO_OR_INCONSISTENT


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (np.eye(5), np.eye(5)),
        (np.eye(7), np.eye(7)),
        (np.eye(6), np.eye(5)),
        (np.zeros((6, 5)), np.zeros((6, 5))),
        (np.full((6, 6), np.nan), np.eye(6)),
    ],
)
def test_spectral_analysis_rejects_invalid_matrix_pairs(first: object, second: object) -> None:
    from src.next12_chsc import CHSCValidationError, analyze_strain_hessian_pair

    with pytest.raises(CHSCValidationError):
        analyze_strain_hessian_pair(first, second)


def test_end_to_end_quadratic_oracle_uses_exactly_85_energy_calls() -> None:
    from src.next12_chsc import CHSCStatus, evaluate_chsc

    atoms = _base_atoms()
    hessian = np.diag([-1.25, 2.0, 3.0, 4.0, 5.0, 6.0])
    oracle = _strain_quadratic_oracle(atoms, hessian)
    calls = 0

    def recording_oracle(probe: Atoms) -> float:
        nonlocal calls
        calls += 1
        return oracle(probe)

    result = evaluate_chsc(atoms, recording_oracle)

    assert calls == 85
    assert result.energy_call_count == 85
    assert result.status is CHSCStatus.RESOLVED_NEGATIVE
    assert result.negative is True
    assert result.h == 2**-7
    assert result.lambda_h == pytest.approx(-1.25, abs=2e-9)
    assert result.lambda_h2 == pytest.approx(-1.25, abs=2e-8)
    assert result.error is None
    with pytest.raises(FrozenInstanceError):
        result.status = CHSCStatus.RESOLVED_NONNEGATIVE  # type: ignore[misc]


def test_oracle_exception_and_invalid_energy_are_explicit_abstentions() -> None:
    from src.next12_chsc import CHSCStatus, evaluate_chsc

    failed = evaluate_chsc(
        _base_atoms(),
        lambda atoms: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    invalid = evaluate_chsc(_base_atoms(), lambda atoms: np.nan)

    assert failed.status is CHSCStatus.ABSTAIN_ENERGY_FAILURE
    assert failed.negative is None
    assert failed.energy_call_count == 1
    assert failed.error is not None and "boom" in failed.error
    assert invalid.status is CHSCStatus.ABSTAIN_INVALID_ENERGY
    assert invalid.negative is None
    assert invalid.energy_call_count == 1


def test_unsupported_geometry_abstains_before_oracle_access() -> None:
    from src.next12_chsc import CHSCStatus, evaluate_chsc

    calls = 0

    def oracle(atoms: Atoms) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=np.eye(3), pbc=False)
    result = evaluate_chsc(atoms, oracle)

    assert calls == 0
    assert result.energy_call_count == 0
    assert result.status is CHSCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY
    assert result.negative is None
