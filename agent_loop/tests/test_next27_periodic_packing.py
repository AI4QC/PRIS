"""Contracts for the NEXT27 exact periodic intermolecular contact law."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest


def _two_molecules(cell: float = 7.0) -> Atoms:
    return Atoms(
        [6, 1, 1, 8, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [1.05, 0.0, 0.0],
            [0.0, 1.05, 0.0],
            [2.5, 2.5, 2.5],
            [3.45, 2.5, 2.5],
        ],
        cell=np.eye(3) * cell,
        pbc=True,
    )


def test_periodic_path_exclusion_removes_exact_one_to_four_tuple_only() -> None:
    from src.next27_periodic_packing import periodic_nonbonded_contacts

    atoms = Atoms(
        [6, 6, 6, 6, 8],
        positions=[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [4.2, 0, 0], [7.4, 0, 0]],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    contacts = periodic_nonbonded_contacts(atoms)
    identities = {(i, j, shift) for i, j, shift, _q in contacts}

    assert (0, 3, (0, 0, 0)) not in identities
    assert (3, 4, (0, 0, 0)) in identities


def test_single_atom_periodic_images_are_real_nonbonded_contacts() -> None:
    from ase.data import vdw_radii
    from src.next27_periodic_packing import periodic_nonbonded_contacts

    atoms = Atoms([6], positions=[[0, 0, 0]], cell=np.eye(3) * 3.0, pbc=True)
    contacts = periodic_nonbonded_contacts(atoms)
    ratios = [row[3] for row in contacts]

    assert ratios
    assert min(ratios) == pytest.approx(3.0 / (2.0 * vdw_radii[6]))


def test_periodic_pressure_is_translation_and_permutation_invariant() -> None:
    from src.next27_periodic_packing import NEXT27_FEATURE_COLUMNS, compute_periodic_features

    atoms = _two_molecules()
    expected = compute_periodic_features(atoms)
    transformed = atoms.copy()
    transformed.positions += transformed.cell.array[0] - transformed.cell.array[1]
    transformed = transformed[[4, 1, 3, 0, 2]]
    observed = compute_periodic_features(transformed)

    assert tuple(expected) == NEXT27_FEATURE_COLUMNS
    for name in NEXT27_FEATURE_COLUMNS:
        assert observed[name] == pytest.approx(expected[name], rel=1e-12, abs=1e-12)


def test_cell_compression_monotonically_increases_overlap_pressure() -> None:
    from src.next27_periodic_packing import compute_periodic_features

    loose = _two_molecules(8.0)
    dense = loose.copy()
    dense.set_cell(loose.cell.array * 0.70, scale_atoms=True)
    a = compute_periodic_features(loose)
    b = compute_periodic_features(dense)

    assert b["periodic_overlap2_pa"] > a["periodic_overlap2_pa"]
    assert b["periodic_repulsion12_pa"] > a["periodic_repulsion12_pa"]
    assert b["periodic_contact_coord100"] >= a["periodic_contact_coord100"]


def test_feature_schema_is_strictly_no_dft() -> None:
    from src.next27_periodic_packing import NEXT27_FEATURE_COLUMNS

    forbidden = ("energy", "force", "stress", "relax", "dft", "label", "endpoint", "mlip")
    assert not [name for name in NEXT27_FEATURE_COLUMNS if any(token in name for token in forbidden)]
