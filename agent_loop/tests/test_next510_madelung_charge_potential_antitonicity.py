from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next510_madelung_charge_potential_antitonicity as n


def test_frozen_schema_direction_and_boundary() -> None:
    assert n.PROTOCOL == "2026-08-13-next510-madelung-charge-potential-antitonicity-v1"
    assert n.FEATURE_NAMES == ("mcpa_madelung_charge_potential_antitonicity",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_exact_pair_margins_and_reversed_order() -> None:
    ordered = n.madelung_charge_potential_antitonicity(
        charges=(2.0, 1.0, -1.0, -2.0),
        site_potentials=(-4.0, -2.0, 2.0, 4.0),
    )
    reversed_order = n.madelung_charge_potential_antitonicity(
        charges=(2.0, 1.0, -1.0, -2.0),
        site_potentials=(4.0, 2.0, -2.0, -4.0),
    )
    assert ordered.supported and reversed_order.supported
    assert ordered.pair_count == 6
    assert ordered.minimum_exchange_margin == pytest.approx(1.0 / 3.0)
    assert ordered.mean_exchange_margin == pytest.approx(7.0 / 9.0)
    assert ordered.maximum_exchange_margin == 1.0
    assert ordered.features[n.FEATURE_NAMES[0]] == pytest.approx(8.0 / 9.0)
    assert reversed_order.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0 / 9.0)


def test_two_site_and_zero_potential_limits() -> None:
    stable = n.madelung_charge_potential_antitonicity(
        charges=(1.0, -1.0), site_potentials=(-5.0, 5.0)
    )
    unstable = n.madelung_charge_potential_antitonicity(
        charges=(1.0, -1.0), site_potentials=(5.0, -5.0)
    )
    zero = n.madelung_charge_potential_antitonicity(
        charges=(1.0, -1.0), site_potentials=(0.0, 0.0)
    )
    assert stable.features[n.FEATURE_NAMES[0]] == 1.0
    assert unstable.features[n.FEATURE_NAMES[0]] == 0.0
    assert zero.features[n.FEATURE_NAMES[0]] == 0.5


def test_kernel_is_charge_scale_order_and_replication_invariant() -> None:
    charge = np.asarray((2.0, 1.0, -1.0, -2.0))
    potential = np.asarray((-4.0, -2.0, 2.0, 4.0))
    reference = n.madelung_charge_potential_antitonicity(
        charges=charge, site_potentials=potential
    )
    order = np.asarray((2, 0, 3, 1))
    candidates = (
        n.madelung_charge_potential_antitonicity(
            charges=charge * 7.3, site_potentials=potential * 7.3
        ),
        n.madelung_charge_potential_antitonicity(
            charges=-charge, site_potentials=-potential
        ),
        n.madelung_charge_potential_antitonicity(
            charges=charge[order], site_potentials=potential[order]
        ),
        n.madelung_charge_potential_antitonicity(
            charges=np.tile(charge, 3), site_potentials=np.tile(potential, 3)
        ),
    )
    for result in candidates:
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-10
        )


@pytest.mark.parametrize(
    ("charges", "potentials"),
    (
        ((), ()),
        ((1.0,), (-1.0,)),
        ((1.0, -1.0), (-1.0,)),
        ((1.0,), (-1.0, 1.0)),
        ((1.0, -1.0), (-1.0, np.nan)),
        ((1.0, -0.5), (-1.0, 1.0)),
        ((1.0, 0.0, -1.0), (-1.0, 0.0, 1.0)),
        ((1.0, 1.0), (-1.0, -1.0)),
    ),
)
def test_malformed_inputs_fail_closed(charges: object, potentials: object) -> None:
    result = n.madelung_charge_potential_antitonicity(
        charges=charges, site_potentials=potentials
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
    result = n.compute_mcpa_features(atoms)
    assert result.supported, result.failure_reason
    assert result.ewald_identity_relative_error <= 1.0e-10
    return float(result.features[n.FEATURE_NAMES[0]])


def test_periodic_equivalents_geometry_scaling_and_firewall() -> None:
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
    scaled = atoms.copy()
    scaled.set_cell(atoms.cell.array * 1.7, scale_atoms=True)
    for equivalent in (
        rotated,
        translated,
        permuted,
        rebased,
        scaled,
        atoms.repeat((2, 1, 1)),
    ):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)

    bad = atoms.copy()
    bad.calc = Calculator()
    result = n.compute_mcpa_features(bad)
    assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_mcpa_row(atoms)
    assert row["mcpa_supported"] is True
    assert tuple(key for key in row if key.startswith("mcpa_")) == (
        "mcpa_madelung_charge_potential_antitonicity",
        "mcpa_supported",
        "mcpa_failure",
        "mcpa_site_count",
        "mcpa_pair_count",
        "mcpa_minimum_exchange_margin",
        "mcpa_mean_exchange_margin",
        "mcpa_maximum_exchange_margin",
        "mcpa_ewald_identity_relative_error",
        "mcpa_valence_policy",
    )
