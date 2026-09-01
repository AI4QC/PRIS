from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next505_madelung_valence_class_homogeneity as n


def test_frozen_schema_direction_and_boundary() -> None:
    assert n.PROTOCOL == "2026-08-13-next505-madelung-valence-class-homogeneity-v1"
    assert n.FEATURE_NAMES == ("mvch_madelung_valence_class_homogeneity",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_exact_anova_identity_and_perfect_classes() -> None:
    exact = n.madelung_valence_class_homogeneity(
        site_energies=(0.0, 1.0, 2.0, 4.0),
        atomic_numbers=(11, 11, 17, 17),
        charges=(1.0, 1.0, -1.0, -1.0),
    )
    # Within SSE = 0.5 + 2 = 2.5; total SSE = 8.75.
    assert exact.supported, exact.failure_reason
    assert exact.within_sum_squares == pytest.approx(2.5)
    assert exact.total_sum_squares == pytest.approx(8.75)
    assert exact.features[n.FEATURE_NAMES[0]] == pytest.approx(5.0 / 7.0)

    perfect = n.madelung_valence_class_homogeneity(
        site_energies=(-2.0, -2.0, -3.0, -3.0),
        atomic_numbers=(11, 11, 17, 17),
        charges=(1.0, 1.0, -1.0, -1.0),
    )
    constant = n.madelung_valence_class_homogeneity(
        site_energies=(-2.0, -2.0, -2.0, -2.0),
        atomic_numbers=(11, 11, 17, 17),
        charges=(1.0, 1.0, -1.0, -1.0),
    )
    assert perfect.features[n.FEATURE_NAMES[0]] == 1.0
    assert constant.features[n.FEATURE_NAMES[0]] == 1.0


def test_joint_element_valence_classes_are_not_element_only() -> None:
    split = n.madelung_valence_class_homogeneity(
        site_energies=(-1.0, -4.0, -2.0, -2.0),
        atomic_numbers=(26, 26, 8, 8),
        charges=(2.0, 3.0, -2.5, -2.5),
    )
    merged = n.madelung_valence_class_homogeneity(
        site_energies=(-1.0, -4.0, -2.0, -2.0),
        atomic_numbers=(26, 26, 8, 8),
        charges=(2.5, 2.5, -2.5, -2.5),
    )
    assert split.supported and merged.supported
    assert split.features[n.FEATURE_NAMES[0]] == 1.0
    assert merged.features[n.FEATURE_NAMES[0]] < 1.0


def test_kernel_is_charge_scale_order_and_replication_invariant() -> None:
    energy = np.asarray((0.0, 1.0, 2.0, 4.0))
    numbers = np.asarray((11, 11, 17, 17))
    charge = np.asarray((1.0, 1.0, -1.0, -1.0))
    reference = n.madelung_valence_class_homogeneity(
        site_energies=energy, atomic_numbers=numbers, charges=charge
    )
    scaled = n.madelung_valence_class_homogeneity(
        site_energies=energy * 7.3**2,
        atomic_numbers=numbers,
        charges=charge * 7.3,
    )
    order = np.asarray((2, 0, 3, 1))
    permuted = n.madelung_valence_class_homogeneity(
        site_energies=energy[order],
        atomic_numbers=numbers[order],
        charges=charge[order],
    )
    replicated = n.madelung_valence_class_homogeneity(
        site_energies=np.tile(energy, 3),
        atomic_numbers=np.tile(numbers, 3),
        charges=np.tile(charge, 3),
    )
    for result in (reference, scaled, permuted, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-10
        )


@pytest.mark.parametrize(
    ("energies", "numbers", "charges"),
    (
        ((), (), ()),
        ((1.0,), (11,), (1.0,)),
        ((1.0, 2.0), (11,), (1.0, -1.0)),
        ((1.0, 2.0), (11, 17), (1.0,)),
        ((1.0, np.nan), (11, 17), (1.0, -1.0)),
        ((1.0, 2.0), (0, 17), (1.0, -1.0)),
        ((1.0, 2.0), (11, 17), (1.0, -0.5)),
        ((1.0, 2.0), (11, 17), (1.0, 0.0)),
    ),
)
def test_malformed_inputs_fail_closed(
    energies: object, numbers: object, charges: object
) -> None:
    result = n.madelung_valence_class_homogeneity(
        site_energies=energies, atomic_numbers=numbers, charges=charges
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
    result = n.compute_mvch_features(atoms)
    assert result.supported, result.failure_reason
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
    result = n.compute_mvch_features(bad)
    assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_mvch_row(atoms)
    assert row["mvch_supported"] is True
    assert tuple(key for key in row if key.startswith("mvch_")) == (
        "mvch_madelung_valence_class_homogeneity",
        "mvch_supported",
        "mvch_failure",
        "mvch_site_count",
        "mvch_class_count",
        "mvch_repeated_class_site_fraction",
        "mvch_within_sum_squares",
        "mvch_total_sum_squares",
        "mvch_valence_policy",
    )
