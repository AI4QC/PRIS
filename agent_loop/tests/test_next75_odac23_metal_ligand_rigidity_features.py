from __future__ import annotations

import math

from ase import Atoms
from ase.build import bulk

from src.next75_odac23_metal_ligand_rigidity_features import (
    METAL_LIGAND_RIGIDITY_FEATURE_NAMES,
    compute_metal_ligand_rigidity_features,
)


def test_metal_ligand_rigidity_is_finite_and_supercell_invariant() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    primitive = compute_metal_ligand_rigidity_features(atoms)
    repeated = compute_metal_ligand_rigidity_features(atoms.repeat((2, 1, 1)))

    assert primitive.supported
    assert repeated.supported
    assert tuple(primitive.features) == METAL_LIGAND_RIGIDITY_FEATURE_NAMES
    for name in METAL_LIGAND_RIGIDITY_FEATURE_NAMES:
        assert math.isfinite(primitive.features[name])
        assert math.isclose(
            primitive.features[name], repeated.features[name], rel_tol=1e-7, abs_tol=1e-9
        )


def test_metal_ligand_rigidity_fails_open_without_metal() -> None:
    atoms = Atoms("C2", positions=[[0, 0, 0], [1.4, 0, 0]], cell=[8, 8, 8], pbc=True)
    result = compute_metal_ligand_rigidity_features(atoms)

    assert not result.supported
    assert result.features == {}
