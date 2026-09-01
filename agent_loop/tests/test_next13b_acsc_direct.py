"""Contracts for direct mixed-mode energy confirmation of ACSC-v0."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ase import Atoms
from scipy.linalg import expm


def _atoms() -> Atoms:
    return Atoms(
        "H3",
        scaled_positions=[[0.12, 0.23, 0.34], [0.48, 0.57, 0.69], [0.81, 0.16, 0.73]],
        cell=[[7.0, 0.0, 0.0], [0.4, 8.0, 0.0], [0.2, 0.3, 9.0]],
        pbc=True,
    )


def test_minimum_richardson_mode_has_frozen_sign_and_unit_norm() -> None:
    from src.next13b_acsc_direct import minimum_richardson_mode

    first = np.diag([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    second = first.copy()
    rotation = np.eye(9)
    angle = np.pi / 7.0
    rotation[0, 4] = -np.sin(angle)
    rotation[4, 0] = np.sin(angle)
    rotation[0, 0] = rotation[4, 4] = np.cos(angle)
    rotated = rotation @ first @ rotation.T

    result = minimum_richardson_mode(rotated, rotated)

    assert result.lambda_r == pytest.approx(-1.0, abs=2e-14)
    assert result.spectral_gap == pytest.approx(3.0, abs=2e-14)
    assert np.linalg.norm(result.vector) == pytest.approx(1.0, abs=2e-15)
    pivot = int(np.argmax(np.abs(result.vector)))
    assert result.vector[pivot] > 0.0


def test_mixed_probe_follows_exact_affine_plus_internal_path() -> None:
    from src.next11_phsc import canonicalize_phsc_geometry, helmert_internal_basis
    from src.next12_chsc import strain_basis
    from src.next13b_acsc_direct import mixed_mode_probe

    base, d_star = canonicalize_phsc_geometry(_atoms())
    q = helmert_internal_basis(len(base))
    internal_dim = q.shape[1]
    mode = np.zeros(internal_dim + 6)
    mode[1] = 0.6
    mode[internal_dim + 3] = 0.8
    amplitude = 2**-8

    probe = mixed_mode_probe(base, mode, amplitude=amplitude)

    eta = mode[internal_dim:]
    generator = np.einsum("a,aij->ij", eta, strain_basis())
    expected_cell = base.cell.array @ expm(amplitude * generator).T
    expected_affine = base.get_scaled_positions(wrap=False) @ expected_cell
    expected_internal = d_star * amplitude * (q @ mode[:internal_dim]).reshape(-1, 3)
    np.testing.assert_allclose(probe.cell.array, expected_cell, atol=2e-15)
    np.testing.assert_allclose(
        probe.get_positions(), expected_affine + expected_internal, atol=3e-15
    )
    np.testing.assert_allclose(base.cell.array, _atoms().cell.array, atol=0.0)


def test_direct_quadratic_curvature_recovers_selected_negative_mode() -> None:
    from src.next13b_acsc_direct import (
        DIRECT_STEP,
        DirectStatus,
        direct_curvature_from_energies,
    )

    n_atoms = 4
    curvature = -1.75
    center = -10.0

    def energy(t: float) -> float:
        return center + 0.5 * n_atoms * curvature * t**2

    result = direct_curvature_from_energies(
        center,
        energy(DIRECT_STEP),
        energy(-DIRECT_STEP),
        energy(DIRECT_STEP / 2.0),
        energy(-DIRECT_STEP / 2.0),
        n_atoms=n_atoms,
        h=DIRECT_STEP,
    )

    assert result.status is DirectStatus.RESOLVED_NEGATIVE
    assert result.negative is True
    assert result.q_h == pytest.approx(curvature, abs=2e-10)
    assert result.q_h2 == pytest.approx(curvature, abs=2e-10)
    assert result.q_r == pytest.approx(curvature, abs=2e-10)
    assert result.e_num == pytest.approx(0.0, abs=2e-10)


def test_direct_two_scale_sign_flip_is_unresolved() -> None:
    from src.next13b_acsc_direct import DirectStatus, direct_curvature_from_energies

    result = direct_curvature_from_energies(
        0.0,
        -0.5,
        -0.5,
        0.125,
        0.125,
        n_atoms=2,
        h=1.0,
    )

    assert result.q_h < 0.0 < result.q_h2
    assert result.status is DirectStatus.NEAR_ZERO_OR_INCONSISTENT
    assert result.negative is False


@pytest.mark.parametrize(
    ("matrix", "second"),
    [
        (np.eye(8), np.eye(8)),
        (np.eye(9), np.eye(12)),
        (np.zeros((9, 8)), np.zeros((9, 8))),
        (np.full((9, 9), np.nan), np.eye(9)),
    ],
)
def test_mode_builder_rejects_invalid_matrix_pairs(matrix: object, second: object) -> None:
    from src.next13b_acsc_direct import DirectValidationError, minimum_richardson_mode

    with pytest.raises(DirectValidationError):
        minimum_richardson_mode(matrix, second)


@pytest.mark.parametrize(
    ("mode", "amplitude"),
    [
        (np.ones(11), 2**-8),
        (np.zeros(12), 2**-8),
        (np.full(12, np.nan), 2**-8),
        (np.ones(12) / np.sqrt(12), 0.0),
        (np.ones(12) / np.sqrt(12), np.nan),
    ],
)
def test_mixed_probe_rejects_invalid_mode_or_amplitude(
    mode: object, amplitude: float
) -> None:
    from src.next13b_acsc_direct import DirectValidationError, mixed_mode_probe

    with pytest.raises(DirectValidationError):
        mixed_mode_probe(_atoms(), mode, amplitude=amplitude)


def test_results_are_immutable() -> None:
    from src.next13b_acsc_direct import minimum_richardson_mode

    result = minimum_richardson_mode(np.eye(9), np.eye(9))
    with pytest.raises(FrozenInstanceError):
        result.lambda_r = 0.0  # type: ignore[misc]
