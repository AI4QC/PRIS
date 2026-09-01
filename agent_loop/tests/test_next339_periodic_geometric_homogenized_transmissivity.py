from __future__ import annotations

import inspect

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next339_periodic_geometric_homogenized_transmissivity as n


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def test_frozen_schema_is_one_protected_high_floor() -> None:
    assert n.PROTOCOL == "2026-08-13-next339-periodic-geometric-homogenized-transmissivity-v1"
    assert n.FEATURE_NAMES == ("pght_affine_retention_floor",)
    assert n.FEATURE_DIRECTIONS == {"pght_affine_retention_floor": "protected_high"}


def test_one_site_bravais_network_has_unit_affine_retention() -> None:
    result = n.periodic_homogenized_retention(
        n_sites=1,
        endpoints=np.zeros((3, 2), dtype=int),
        displacements=np.eye(3),
        conductances=np.asarray([2.0, 3.0, 5.0]),
        volume=7.0,
    )
    np.testing.assert_allclose(result.generalized_eigenvalues, np.ones(3), atol=1.0e-12)
    assert result.affine_retention_floor == pytest.approx(1.0)
    assert result.maximum_corrector_residual <= 1.0e-12


def test_two_site_internal_corrector_has_analytic_sixteen_twenty_fifths_retention() -> None:
    endpoints = np.asarray([[0, 1], [0, 1], [0, 0], [0, 0]], dtype=int)
    displacements = np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    result = n.periodic_homogenized_retention(
        n_sites=2,
        endpoints=endpoints,
        displacements=displacements,
        conductances=np.asarray([1.0, 4.0, 1.0, 1.0]),
        volume=1.0,
    )
    np.testing.assert_allclose(result.generalized_eigenvalues, [0.64, 1.0, 1.0], atol=1.0e-12)
    assert result.affine_retention_floor == pytest.approx(0.64)


def test_kernel_is_edge_orientation_order_and_gauge_invariant() -> None:
    endpoints = np.asarray([[0, 1], [0, 1], [0, 0], [1, 1]], dtype=int)
    displacements = np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    conductances = np.asarray([1.0, 4.0, 1.0, 1.0])
    reference = n.periodic_homogenized_retention(
        n_sites=2, endpoints=endpoints, displacements=displacements,
        conductances=conductances, volume=1.0,
    )
    order = np.asarray([3, 1, 0, 2])
    reversed_endpoints = endpoints[order].copy()
    reversed_endpoints[:2] = reversed_endpoints[:2, ::-1]
    reversed_displacements = displacements[order].copy()
    reversed_displacements[:2] *= -1.0
    changed = n.periodic_homogenized_retention(
        n_sites=2, endpoints=reversed_endpoints, displacements=reversed_displacements,
        conductances=conductances[order], volume=1.0,
    )
    np.testing.assert_allclose(changed.generalized_eigenvalues, reference.generalized_eigenvalues)
    np.testing.assert_allclose(changed.homogenized_tensor, reference.homogenized_tensor)


def test_kernel_refuses_rank_deficient_affine_tensor_and_invalid_edges() -> None:
    with pytest.raises(ValueError, match="affine tensor is not positive definite"):
        n.periodic_homogenized_retention(
            n_sites=1,
            endpoints=np.zeros((2, 2), dtype=int),
            displacements=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            conductances=np.ones(2),
            volume=1.0,
        )
    with pytest.raises(ValueError, match="edge population differs"):
        n.periodic_homogenized_retention(
            n_sites=2,
            endpoints=np.asarray([[0, 2]]),
            displacements=np.asarray([[1.0, 0.0, 0.0]]),
            conductances=np.ones(1),
            volume=1.0,
        )


def _feature(atoms: Atoms) -> float:
    result = n.compute_pght_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_standard_and_distorted_crystals_have_finite_pght() -> None:
    for atoms in (
        bulk("NaCl", "rocksalt", a=5.64, cubic=True),
        bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
        bulk("ZnS", "zincblende", a=5.41, cubic=True),
        _distorted_nacl(),
    ):
        result = n.compute_pght_features(atoms)
        assert result.supported, result.failure_reason
        assert result.site_count == len(atoms)
        assert result.edge_count >= 3
        assert result.maximum_reciprocal_area_relative_error <= n.RECIPROCAL_AREA_RELATIVE_TOLERANCE
        assert result.maximum_corrector_residual <= n.CORRECTOR_RESIDUAL_TOLERANCE
        assert 0.0 < result.features[n.FEATURE_NAMES[0]] <= 1.0


def test_geometry_equivalences_preserve_pght() -> None:
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
        result = n.compute_pght_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_boundary_and_builder_interface_are_exact() -> None:
    row = n.compute_pght_row(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert tuple(name for name in row if name.startswith("pght_")) == (
        "pght_affine_retention_floor", "pght_supported", "pght_failure",
        "pght_site_count", "pght_edge_count", "pght_minimum_facet_area",
        "pght_maximum_reciprocal_area_relative_error",
        "pght_maximum_corrector_residual",
        "pght_volume_tiling_relative_error",
    )
    assert row["pght_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
    parameters = tuple(inspect.signature(n.build_cross_source_pght_features).parameters)
    assert parameters == (
        "scigen_cohort_dir", "wyformer_cohort_dir", "design_path",
        "probe_result_path", "output_dir", "workers", "require_formal_inputs",
    )
    assert not any(
        token in name for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )
