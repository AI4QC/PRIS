from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next520_madelung_chemical_potential_equalization as n


def test_frozen_schema_direction_atomic_table_and_boundary() -> None:
    assert n.PROTOCOL == "2026-08-13-next520-madelung-chemical-potential-equalization-v1"
    assert n.DESIGN_SHA256 == "9a3109ea4db49f0e4199eca651538fe561ea7df429137337a3d6291eddd8f660"
    assert n.ATOMIC_TABLE_SHA256 == "b11669f8ccb0a9fe7647d9026ecbd30ee15ded7c464df828820a15768556d0aa"
    assert n.FEATURE_NAMES == ("mcpe_madelung_chemical_potential_equalization",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_exact_equal_and_unequal_chemical_potential_cases() -> None:
    equal = n.madelung_chemical_potential_equalization(
        charges=(1.0, -1.0),
        ionization_energies=(5.0, 7.0),
        electron_affinities=(1.0, 3.0),
        site_potentials=(-1.0, 3.0),
    )
    assert equal.supported, equal.failure_reason
    np.testing.assert_allclose(equal.electronegativities, [3.0, 5.0])
    np.testing.assert_allclose(equal.hardnesses, [4.0, 4.0])
    np.testing.assert_allclose(equal.chemical_potentials, [6.0, 4.0])
    # d = (8, 12), ordered 2x2 discrepancy sum = 2*(2/20).
    assert equal.pair_count == 4
    assert equal.mean_normalized_discrepancy == pytest.approx(0.05)
    assert equal.features[n.FEATURE_NAMES[0]] == pytest.approx(0.95)

    stationary = n.madelung_chemical_potential_equalization(
        charges=(1.0, -1.0),
        ionization_energies=(5.0, 7.0),
        electron_affinities=(1.0, 3.0),
        site_potentials=(-1.0, 5.0),
    )
    assert stationary.chemical_potential_spread == pytest.approx(0.0)
    assert stationary.features[n.FEATURE_NAMES[0]] == 1.0


def test_kernel_is_site_order_and_exact_replication_invariant() -> None:
    charges = np.asarray((2.0, 1.0, -1.0, -2.0))
    ionization = np.asarray((6.0, 5.0, 10.0, 12.0))
    affinity = np.asarray((1.0, 0.5, 2.0, 3.0))
    potential = np.asarray((-5.0, -2.0, 3.0, 8.0))
    reference = n.madelung_chemical_potential_equalization(
        charges=charges,
        ionization_energies=ionization,
        electron_affinities=affinity,
        site_potentials=potential,
    )
    order = np.asarray((2, 0, 3, 1))
    for result in (
        n.madelung_chemical_potential_equalization(
            charges=charges[order],
            ionization_energies=ionization[order],
            electron_affinities=affinity[order],
            site_potentials=potential[order],
        ),
        n.madelung_chemical_potential_equalization(
            charges=np.tile(charges, 3),
            ionization_energies=np.tile(ionization, 3),
            electron_affinities=np.tile(affinity, 3),
            site_potentials=np.tile(potential, 3),
        ),
    ):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-10
        )


@pytest.mark.parametrize(
    ("charges", "ionization", "affinity", "potentials"),
    (
        ((), (), (), ()),
        ((1.0,), (5.0,), (1.0,), (-1.0,)),
        ((1.0, -1.0), (5.0,), (1.0, 2.0), (-1.0, 1.0)),
        ((1.0, -1.0), (5.0, 6.0), (1.0,), (-1.0, 1.0)),
        ((1.0, -1.0), (5.0, 6.0), (1.0, 2.0), (-1.0,)),
        ((1.0, -0.5), (5.0, 6.0), (1.0, 2.0), (-1.0, 1.0)),
        ((1.0, 0.0, -1.0), (5.0, 6.0, 7.0), (1.0, 2.0, 3.0), (-1.0, 0.0, 1.0)),
        ((1.0, -1.0), (5.0, np.nan), (1.0, 2.0), (-1.0, 1.0)),
        ((1.0, -1.0), (5.0, 1.0), (1.0, 2.0), (-1.0, 1.0)),
    ),
)
def test_malformed_inputs_fail_closed(
    charges: object, ionization: object, affinity: object, potentials: object
) -> None:
    result = n.madelung_chemical_potential_equalization(
        charges=charges,
        ionization_energies=ionization,
        electron_affinities=affinity,
        site_potentials=potentials,
    )
    assert not result.supported
    assert result.features == {}


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        [[5.64, 0, 0], [0.27, 5.77, 0], [0.18, 0.31, 5.53]],
        scale_atoms=True,
    )
    atoms.positions[1] += [0.08, -0.04, 0.06]
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_mcpe_features(atoms)
    assert result.supported, result.failure_reason
    assert result.ewald_identity_relative_error <= 1.0e-10
    return float(result.features[n.FEATURE_NAMES[0]])


def test_periodic_equivalents_and_geometry_firewall() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.17, 0.29, 0.42])
    translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy()
    rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    for equivalent in (rotated, translated, permuted, rebased, atoms.repeat((2, 1, 1))):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)

    bad = atoms.copy()
    bad.calc = Calculator()
    result = n.compute_mcpe_features(bad)
    assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_mcpe_row(atoms)
    assert row["mcpe_supported"] is True
    assert tuple(key for key in row if key.startswith("mcpe_")) == (
        "mcpe_madelung_chemical_potential_equalization",
        "mcpe_supported",
        "mcpe_failure",
        "mcpe_site_count",
        "mcpe_pair_count",
        "mcpe_mean_normalized_discrepancy",
        "mcpe_maximum_normalized_discrepancy",
        "mcpe_chemical_potential_spread",
        "mcpe_electronegativity_spread",
        "mcpe_hardness_spread",
        "mcpe_ewald_identity_relative_error",
        "mcpe_valence_policy",
        "mcpe_atomic_table_sha256",
    )


def test_electronegativity_partition_is_intensive_under_supercell_replication() -> None:
    atoms = Atoms(
        "DyAc",
        positions=((0.0, 0.0, 0.0), (1.4, 1.7, 2.1)),
        cell=((3.8, 0.0, 0.0), (-1.9, 3.3, 0.0), (0.1, 0.3, 5.9)),
        pbc=True,
    )
    unit = n.compute_mcpe_features(atoms)
    repeated = n.compute_mcpe_features(atoms.repeat((2, 1, 1)))
    assert unit.supported and repeated.supported
    assert unit.valence_policy == repeated.valence_policy == "electronegativity_partition"
    assert repeated.features[n.FEATURE_NAMES[0]] == pytest.approx(
        unit.features[n.FEATURE_NAMES[0]], abs=1e-8
    )
