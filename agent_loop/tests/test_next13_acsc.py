"""Contract tests for the additive ACSC-v0 coupled-Hessian diagnostic."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_frozen_identity_and_dimensionless_block_scaling() -> None:
    from src.next11_phsc import helmert_internal_basis
    from src.next13_acsc import ACSC_VERSION, scaled_internal_coupled_hessian

    n_atoms = 3
    q = helmert_internal_basis(n_atoms)
    atomic = q @ np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) @ q.T
    cell = np.diag([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    cross = q @ np.arange(36, dtype=float).reshape(6, 6) / 17.0

    observed = scaled_internal_coupled_hessian(
        atomic, cell, cross, d_star=2.5
    )

    expected_atomic = (2.5**2 / n_atoms) * (q.T @ atomic @ q)
    expected_cross = (2.5 / n_atoms) * (q.T @ cross)
    expected = np.block(
        [[expected_atomic, expected_cross], [expected_cross.T, cell]]
    )
    assert ACSC_VERSION == "ACSC-v0"
    assert observed.shape == (12, 12)
    np.testing.assert_allclose(observed, expected, atol=3e-14)


def test_cross_hessian_uses_six_axial_strain_force_responses() -> None:
    from src.next13_acsc import cross_hessians_from_strain_forces

    n_atoms = 2
    cross = np.arange(36, dtype=float).reshape(6, 6) / 9.0 - 1.0
    h = 0.125
    base_force = np.arange(6, dtype=float).reshape(2, 3) / 11.0

    def force(eta: float, axis: int) -> np.ndarray:
        return base_force - (eta * cross[:, axis]).reshape(n_atoms, 3)

    plus_h = np.stack([force(h, axis) for axis in range(6)])
    minus_h = np.stack([force(-h, axis) for axis in range(6)])
    plus_h2 = np.stack([force(h / 2.0, axis) for axis in range(6)])
    minus_h2 = np.stack([force(-h / 2.0, axis) for axis in range(6)])

    observed_h, observed_h2 = cross_hessians_from_strain_forces(
        plus_h, minus_h, plus_h2, minus_h2, h=h
    )

    np.testing.assert_allclose(observed_h, cross, atol=3e-15)
    np.testing.assert_allclose(observed_h2, cross, atol=3e-15)


def test_coupling_detects_saddle_missed_by_both_positive_diagonal_blocks() -> None:
    from src.next11_phsc import helmert_internal_basis
    from src.next13_acsc import ACSCStatus, analyze_acsc_blocks

    n_atoms = 2
    q = helmert_internal_basis(n_atoms)
    atomic_internal = np.diag([1.0, 2.0, 3.0])
    atomic = q @ atomic_internal @ q.T
    cell = np.diag([1.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    cross_internal = np.zeros((3, 6))
    cross_internal[0, 0] = 2.0
    cross = q @ cross_internal

    result = analyze_acsc_blocks(
        atomic,
        atomic,
        cell,
        cell,
        cross,
        cross,
        d_star=np.sqrt(float(n_atoms)),
    )

    assert np.linalg.eigvalsh(atomic_internal)[0] > 0.0
    assert np.linalg.eigvalsh(cell)[0] > 0.0
    assert result.status is ACSCStatus.RESOLVED_NEGATIVE
    assert result.negative is True
    # The dimensionless cross entry is d*/N * 2 = sqrt(2), so the active
    # 2x2 block has eigenvalues 1 +/- sqrt(2).
    assert result.lambda_r == pytest.approx(1.0 - np.sqrt(2.0), abs=2e-14)
    assert result.e_num == pytest.approx(0.0, abs=2e-14)


def test_pure_atomic_and_cell_subspaces_are_exact_principal_blocks() -> None:
    from src.next11_phsc import helmert_internal_basis
    from src.next13_acsc import scaled_internal_coupled_hessian

    n_atoms = 4
    q = helmert_internal_basis(n_atoms)
    raw = np.arange((3 * n_atoms) ** 2, dtype=float).reshape(3 * n_atoms, -1)
    atomic = 0.5 * (raw + raw.T) / 31.0
    cell_raw = np.arange(36, dtype=float).reshape(6, 6)
    cell = 0.5 * (cell_raw + cell_raw.T) / 19.0
    cross = np.arange(18 * n_atoms, dtype=float).reshape(3 * n_atoms, 6) / 23.0
    d_star = 1.7

    coupled = scaled_internal_coupled_hessian(
        atomic, cell, cross, d_star=d_star
    )
    internal_dim = 3 * n_atoms - 3

    np.testing.assert_allclose(
        coupled[:internal_dim, :internal_dim],
        (d_star**2 / n_atoms) * q.T @ (0.5 * (atomic + atomic.T)) @ q,
        atol=2e-14,
    )
    np.testing.assert_allclose(coupled[internal_dim:, internal_dim:], cell)


def test_two_scale_sign_flip_and_error_interval_are_unresolved() -> None:
    from src.next13_acsc import ACSCStatus, analyze_coupled_hessian_pair

    sign_flip = analyze_coupled_hessian_pair(
        np.diag([-2.0] + [1.0] * 8), np.diag([2.0] + [1.0] * 8)
    )
    interval_crossing = analyze_coupled_hessian_pair(
        np.diag([-10.0] + [20.0] * 8), np.diag([-1.0] + [20.0] * 8)
    )

    assert sign_flip.status is ACSCStatus.NEAR_ZERO_OR_INCONSISTENT
    assert interval_crossing.lambda_h < 0.0
    assert interval_crossing.lambda_h2 < 0.0
    assert interval_crossing.u_num > 0.0
    assert interval_crossing.status is ACSCStatus.NEAR_ZERO_OR_INCONSISTENT


@pytest.mark.parametrize(
    ("atomic", "cell", "cross", "d_star"),
    [
        (np.eye(5), np.eye(6), np.zeros((6, 6)), 1.0),
        (np.eye(6), np.eye(5), np.zeros((6, 6)), 1.0),
        (np.eye(6), np.eye(6), np.zeros((5, 6)), 1.0),
        (np.full((6, 6), np.nan), np.eye(6), np.zeros((6, 6)), 1.0),
        (np.eye(6), np.eye(6), np.zeros((6, 6)), 0.0),
        (np.eye(6), np.eye(6), np.zeros((6, 6)), np.nan),
    ],
)
def test_block_builder_rejects_invalid_inputs(
    atomic: object, cell: object, cross: object, d_star: float
) -> None:
    from src.next13_acsc import ACSCValidationError, scaled_internal_coupled_hessian

    with pytest.raises(ACSCValidationError):
        scaled_internal_coupled_hessian(atomic, cell, cross, d_star=d_star)


@pytest.mark.parametrize(
    "samples,h",
    [
        ((np.zeros((5, 2, 3)),) * 4, 1.0),
        ((np.zeros((6, 1, 3)),) * 4, 1.0),
        ((np.zeros((6, 2, 3)), np.zeros((6, 3, 3)), np.zeros((6, 2, 3)), np.zeros((6, 2, 3))), 1.0),
        ((np.full((6, 2, 3), np.nan),) * 4, 1.0),
        ((np.zeros((6, 2, 3)),) * 4, 0.0),
    ],
)
def test_cross_force_builder_rejects_invalid_inputs(samples: tuple, h: float) -> None:
    from src.next13_acsc import ACSCValidationError, cross_hessians_from_strain_forces

    with pytest.raises(ACSCValidationError):
        cross_hessians_from_strain_forces(*samples, h=h)


def test_results_are_finite_and_immutable() -> None:
    from src.next13_acsc import analyze_coupled_hessian_pair

    result = analyze_coupled_hessian_pair(np.eye(9), np.eye(9))

    assert all(
        np.isfinite(value)
        for value in (
            result.lambda_h,
            result.lambda_h2,
            result.lambda_r,
            result.e_num,
            result.u_num,
            result.l_num,
            result.tau_alg,
            result.antisymmetric_norm_h,
            result.antisymmetric_norm_h2,
        )
    )
    with pytest.raises(FrozenInstanceError):
        result.lambda_r = 0.0  # type: ignore[misc]
