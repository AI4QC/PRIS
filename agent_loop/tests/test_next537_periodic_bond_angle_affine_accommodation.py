from __future__ import annotations

import math

from ase import Atoms
from ase.build import bulk
import numpy as np

from src.next537_periodic_bond_angle_affine_accommodation import (
    BOUNDARY_FLAGS,
    FEATURE_NAMES,
    _project_affine_columns,
    compute_periodic_bond_angle_affine_accommodation,
    periodic_bond_angle_affine_accommodation,
)


def _cubic_vectors() -> np.ndarray:
    return np.eye(3, dtype=float)


def test_pure_kernel_separates_fully_constrained_cell_from_free_strain() -> None:
    rigid = periodic_bond_angle_affine_accommodation(
        n_sites=1,
        endpoints=np.asarray([[0, 0], [0, 0], [0, 0]]),
        vectors=_cubic_vectors(),
    )
    assert rigid.supported
    assert rigid.direct_rank == 6
    assert rigid.features[FEATURE_NAMES[0]] == 0.0
    np.testing.assert_allclose(rigid.generalized_eigenvalues, np.ones(6), atol=1e-10)

    flexible = periodic_bond_angle_affine_accommodation(
        n_sites=1,
        endpoints=np.asarray([[0, 0]]),
        vectors=np.asarray([[1.0, 0.0, 0.0]]),
    )
    assert flexible.supported
    assert flexible.direct_rank < 6
    assert flexible.features[FEATURE_NAMES[0]] == 1.0


def test_pure_kernel_is_scale_rotation_order_and_replication_invariant() -> None:
    endpoints = np.asarray([[0, 0], [0, 0], [0, 0]])
    vectors = _cubic_vectors()
    reference = periodic_bond_angle_affine_accommodation(
        n_sites=1, endpoints=endpoints, vectors=vectors
    )
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    variants = (
        (1, endpoints[::-1], vectors[::-1]),
        (1, endpoints, 7.3 * vectors),
        (1, endpoints, vectors @ rotation.T),
        (
            2,
            np.asarray([[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]),
            np.vstack([vectors, vectors]),
        ),
    )
    for n_sites, pair, displacement in variants:
        result = periodic_bond_angle_affine_accommodation(
            n_sites=n_sites, endpoints=pair, vectors=displacement
        )
        assert result.supported
        assert result.features == reference.features


def test_pure_kernel_fails_closed_on_malformed_population() -> None:
    malformed = periodic_bond_angle_affine_accommodation(
        n_sites=2,
        endpoints=np.asarray([[0, 2]]),
        vectors=np.asarray([[1.0, 0.0, 0.0]]),
    )
    assert not malformed.supported
    assert "endpoint" in str(malformed.failure_reason)
    assert malformed.features == {}


def test_rank_deficient_projection_is_atomic_column_permutation_invariant() -> None:
    rng = np.random.default_rng(537)
    basis = rng.normal(size=(90, 25))
    coefficients = rng.normal(size=(25, 42))
    atomic = basis @ coefficients
    atomic[:, 31:] = atomic[:, :11]
    affine = rng.normal(size=(90, 6))
    reference_residual, reference_rank = _project_affine_columns(atomic, affine)
    for seed in range(6):
        permutation = np.random.default_rng(seed).permutation(atomic.shape[1])
        residual, rank = _project_affine_columns(atomic[:, permutation], affine)
        assert rank == reference_rank == 25
        np.testing.assert_allclose(
            residual.T @ residual,
            reference_residual.T @ reference_residual,
            rtol=1e-11,
            atol=1e-11,
        )
        assert np.linalg.norm(atomic[:, permutation].T @ residual) <= 1e-10 * (
            np.linalg.norm(atomic) * np.linalg.norm(residual)
        )


def test_raw_wrapper_is_representation_invariant_and_geometry_only() -> None:
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    reference = compute_periodic_bond_angle_affine_accommodation(atoms)
    assert reference.supported, reference.failure_reason
    value = reference.features[FEATURE_NAMES[0]]
    assert math.isfinite(value) and 0.0 <= value <= 1.0

    translated = atoms.copy()
    translated.positions += np.asarray([0.37, -1.2, 0.81])
    permuted = atoms[np.asarray([3, 0, 7, 2, 6, 1, 5, 4])]
    supercell = atoms.repeat((2, 1, 1))
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = atoms.copy()
    rotated.positions = rotated.positions @ rotation.T
    rotated.cell = np.asarray(rotated.cell) @ rotation.T
    for variant in (translated, permuted, supercell, rotated):
        result = compute_periodic_bond_angle_affine_accommodation(variant)
        assert result.supported, result.failure_reason
        assert abs(result.features[FEATURE_NAMES[0]] - value) <= 1e-6

    contaminated = atoms.copy()
    contaminated.info["energy"] = -1.0
    result = compute_periodic_bond_angle_affine_accommodation(contaminated)
    assert not result.supported
    assert "geometry-only" in str(result.failure_reason)


def test_boundary_and_schema_are_frozen_zero_dft() -> None:
    assert FEATURE_NAMES == ("pbaaa_periodic_bond_angle_affine_accommodation",)
    assert BOUNDARY_FLAGS == {
        "dft_calculation_executed": False,
        "dft_values_used": False,
        "relaxed_structures_used": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_or_virtual_coordinate_relaxation_executed": False,
    }
