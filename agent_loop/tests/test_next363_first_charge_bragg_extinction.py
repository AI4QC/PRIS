from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next363_first_charge_bragg_extinction as n


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
    assert n.PROTOCOL == "2026-08-13-next363-first-charge-bragg-extinction-v1"
    assert n.DESIGN_SHA256 == "8184c6866d9f1f62aa61342b7d3ce39c87051e7b34393884c490fce6fa0568e9"
    assert n.FEATURE_NAMES == ("fcbe_first_charge_bragg_wavenumber",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert n.INTENSITY_FLOOR == 1.0e-12
    assert n.DIMENSIONLESS_CUTOFF == 18.0


def test_simple_alternating_basis_has_analytic_first_peak() -> None:
    lattice = np.eye(3)
    fractional = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result = n.first_charge_bragg_wavenumber(lattice, fractional, [1.0, -1.0])
    expected = 2.0 * np.pi / np.cbrt(2.0)
    assert result.first_wavenumber == pytest.approx(expected, abs=1.0e-12)
    assert result.first_intensity == pytest.approx(1.0)
    assert set(map(tuple, result.first_integer_vectors)) == {(-1, 0, 0), (1, 0, 0)}


def test_scale_charge_amplitude_translation_order_and_rotation_are_invariant() -> None:
    lattice = np.asarray([[3.1, 0.0, 0.0], [0.3, 2.7, 0.0], [0.2, 0.4, 3.5]])
    fractional = np.asarray([[0.1, 0.2, 0.3], [0.62, 0.19, 0.31], [0.33, 0.71, 0.91]])
    charge = np.asarray([2.0, -1.0, -1.0])
    reference = n.first_charge_bragg_wavenumber(lattice, fractional, charge).first_wavenumber
    assert n.first_charge_bragg_wavenumber(7.0 * lattice, fractional, charge).first_wavenumber == pytest.approx(reference)
    assert n.first_charge_bragg_wavenumber(lattice, fractional, 9.0 * charge).first_wavenumber == pytest.approx(reference)
    assert n.first_charge_bragg_wavenumber(lattice, fractional + [0.17, 0.29, 0.41], charge).first_wavenumber == pytest.approx(reference)
    assert n.first_charge_bragg_wavenumber(lattice, fractional[[2, 0, 1]], charge[[2, 0, 1]]).first_wavenumber == pytest.approx(reference)
    angle = np.deg2rad(31.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    assert n.first_charge_bragg_wavenumber(lattice @ rotation.T, fractional, charge).first_wavenumber == pytest.approx(reference)


def test_numerical_extinction_floor_ignores_subthreshold_mode() -> None:
    lattice = np.eye(3)
    fractional = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.5, 0.0]])
    exactly = np.asarray([1.0, 1.0, -2.0])
    baseline = n.first_charge_bragg_wavenumber(lattice, fractional, exactly)
    perturbed = exactly + np.asarray([1.0e-8, -1.0e-8, 0.0])
    guarded = n.first_charge_bragg_wavenumber(lattice, fractional, perturbed)
    assert guarded.first_wavenumber == pytest.approx(baseline.first_wavenumber)


def _feature(atoms: Atoms) -> float:
    result = n.compute_fcbe_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_distorted_crystal_has_finite_feature_and_charge_policy() -> None:
    result = n.compute_fcbe_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.reciprocal_vector_count >= result.nonextinct_vector_count >= 2
    assert result.first_peak_multiplicity >= 2
    assert result.first_intensity >= n.INTENSITY_FLOOR
    assert result.valence_policy in {
        "integer_oxidation_state", "fractional_oxidation_state", "electronegativity_partition"
    }
    assert 0.0 < result.features[n.FEATURE_NAMES[0]] <= n.DIMENSIONLESS_CUTOFF


def test_structure_equivalences_preserve_feature() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int) @ atoms.cell.array,
        scale_atoms=False,
    ); rebased.wrap()
    diagonal_supercell = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, diagonal_supercell):
        assert _feature(equivalent) == pytest.approx(reference, abs=1.0e-8)


def test_geometry_boundary_fails_closed() -> None:
    atoms = _distorted_nacl()
    with_calculator = atoms.copy(); with_calculator.calc = Calculator()
    with_metadata = atoms.copy(); with_metadata.info["outcome"] = 1
    with_array = atoms.copy(); with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy(); nonperiodic.pbc = False
    nonfinite = atoms.copy(); nonfinite.positions[0, 0] = np.nan
    for changed in (with_calculator, with_metadata, with_array, nonperiodic, nonfinite):
        result = n.compute_fcbe_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_boundary_flags_are_exact() -> None:
    row = n.compute_fcbe_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("fcbe_")) == (
        "fcbe_first_charge_bragg_wavenumber", "fcbe_supported", "fcbe_failure",
        "fcbe_site_count", "fcbe_reciprocal_vector_count",
        "fcbe_nonextinct_vector_count", "fcbe_first_peak_multiplicity",
        "fcbe_first_intensity", "fcbe_valence_policy",
    )
    assert row["fcbe_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
