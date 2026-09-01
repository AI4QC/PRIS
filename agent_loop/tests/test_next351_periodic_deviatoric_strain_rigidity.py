from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next351_periodic_deviatoric_strain_rigidity as n


def _spanning_directions() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def test_frozen_schema_is_one_protected_high_feature() -> None:
    assert n.PROTOCOL == "2026-08-13-next351-periodic-deviatoric-strain-rigidity-v1"
    assert n.FEATURE_NAMES == ("pdsr_deviatoric_retention_floor",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    basis = n.deviatoric_strain_basis()
    assert basis.shape == (5, 3, 3)
    assert np.einsum("aij,bij->ab", basis, basis) == pytest.approx(np.eye(5))
    assert np.trace(basis, axis1=1, axis2=2) == pytest.approx(np.zeros(5))


def test_self_image_framework_has_exact_unit_retention() -> None:
    vectors = _spanning_directions()
    result = n.periodic_deviatoric_strain_retention(
        n_sites=1,
        endpoints=np.zeros((len(vectors), 2), dtype=int),
        displacements=vectors,
        weights=np.ones(len(vectors)),
    )
    assert result.retention_floor == pytest.approx(1.0)
    assert result.generalized_eigenvalues == pytest.approx((1.0,) * 5)
    assert result.maximum_orthogonality_residual == 0.0


def test_independent_leaf_hinges_cancel_every_deviatoric_strain() -> None:
    vectors = _spanning_directions()
    result = n.periodic_deviatoric_strain_retention(
        n_sites=1 + len(vectors),
        endpoints=np.column_stack((np.zeros(len(vectors), dtype=int), np.arange(1, 7))),
        displacements=vectors,
        weights=np.arange(1.0, 7.0),
    )
    assert result.retention_floor == pytest.approx(0.0, abs=1.0e-12)
    assert result.generalized_eigenvalues == pytest.approx((0.0,) * 5, abs=1.0e-12)


def test_kernel_is_orientation_order_weight_scale_and_length_scale_invariant() -> None:
    vectors = np.vstack((_spanning_directions(), [[1.0, 1.0, 1.0]]))
    endpoints = np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3, 0]])
    weights = np.arange(1.0, 8.0)
    reference = n.periodic_deviatoric_strain_retention(
        n_sites=4, endpoints=endpoints, displacements=vectors, weights=weights
    )
    order = np.asarray([6, 2, 4, 0, 5, 1, 3])
    changed = n.periodic_deviatoric_strain_retention(
        n_sites=4,
        endpoints=endpoints[order, ::-1],
        displacements=-17.0 * vectors[order],
        weights=23.0 * weights[order],
    )
    assert changed.retention_floor == pytest.approx(reference.retention_floor, abs=1.0e-10)
    assert changed.generalized_eigenvalues == pytest.approx(
        reference.generalized_eigenvalues, abs=1.0e-10
    )


def test_primitive_self_images_equal_explicit_two_copy_cover() -> None:
    vectors = _spanning_directions()
    primitive = n.periodic_deviatoric_strain_retention(
        n_sites=1,
        endpoints=np.zeros((len(vectors), 2), dtype=int),
        displacements=vectors,
        weights=np.arange(1.0, 7.0),
    )
    cover = n.periodic_deviatoric_strain_retention(
        n_sites=2,
        endpoints=np.tile([[0, 1], [1, 0]], (len(vectors), 1)),
        displacements=np.repeat(vectors, 2, axis=0),
        weights=np.repeat(np.arange(1.0, 7.0), 2),
    )
    assert cover.retention_floor == pytest.approx(primitive.retention_floor, abs=1.0e-12)
    assert cover.generalized_eigenvalues == pytest.approx(
        primitive.generalized_eigenvalues, abs=1.0e-12
    )


def test_kernel_refuses_unspanned_strain_and_invalid_edges() -> None:
    with pytest.raises(ValueError, match="affine deviatoric Gram"):
        n.periodic_deviatoric_strain_retention(
            n_sites=2,
            endpoints=[[0, 1]] * 5,
            displacements=[[1.0, 0.0, 0.0]] * 5,
            weights=[1.0] * 5,
        )
    with pytest.raises(ValueError, match="edge population"):
        n.periodic_deviatoric_strain_retention(
            n_sites=2,
            endpoints=[[0.0, 0.5]],
            displacements=[[1.0, 0.0, 0.0]],
            weights=[1.0],
        )


def _feature(atoms: Atoms) -> float:
    result = n.compute_pdsr_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_high_symmetry_unspanned_frameworks_abstain_but_distorted_is_finite() -> None:
    for atoms in (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
    ):
        result = n.compute_pdsr_features(atoms)
        assert result.supported is False
        assert "affine deviatoric Gram" in str(result.failure_reason)

    result = n.compute_pdsr_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.edge_count >= 5
    assert result.maximum_orthogonality_residual <= n.ORTHOGONALITY_TOLERANCE
    assert 0.0 <= result.features[n.FEATURE_NAMES[0]] <= 1.0


def test_geometry_equivalences_preserve_pdsr() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int) @ atoms.cell.array,
        scale_atoms=False,
    ); rebased.wrap()
    replicated = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, replicated):
        assert _feature(equivalent) == pytest.approx(reference, abs=1.0e-8)


def test_geometry_boundary_fails_closed() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    with_calculator = atoms.copy(); with_calculator.calc = Calculator()
    with_metadata = atoms.copy(); with_metadata.info["outcome"] = 1
    with_array = atoms.copy(); with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy(); nonperiodic.pbc = False
    nonfinite = atoms.copy(); nonfinite.positions[0, 0] = np.nan
    for changed in (with_calculator, with_metadata, with_array, nonperiodic, nonfinite):
        result = n.compute_pdsr_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_boundary_and_future_builder_interface_are_exact() -> None:
    row = n.compute_pdsr_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("pdsr_")) == (
        "pdsr_deviatoric_retention_floor", "pdsr_supported", "pdsr_failure",
        "pdsr_site_count", "pdsr_edge_count", "pdsr_maximum_orthogonality_residual",
        "pdsr_affine_minimum_eigenvalue",
    )
    assert row["pdsr_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
    parameters = tuple(inspect.signature(n.compute_pdsr_features).parameters)
    assert parameters == ("atoms",)
