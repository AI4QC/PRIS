from __future__ import annotations

import math

from ase import Atoms
from ase.build import bulk

from src.next69_odac23_local_bond_valence_features import (
    BOND_VALENCE_FEATURE_NAMES,
    compute_odac23_local_bond_valence_features,
)


def test_local_bond_valence_features_are_finite_and_intensive_for_nacl() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    primitive = compute_odac23_local_bond_valence_features(atoms)
    repeated = compute_odac23_local_bond_valence_features(atoms.repeat((2, 1, 1)))

    assert primitive.any_supported
    assert repeated.any_supported
    for mode in ("crystalnn", "voronoi"):
        if primitive.mode_supported[mode] and repeated.mode_supported[mode]:
            for name in BOND_VALENCE_FEATURE_NAMES:
                left = primitive.features[f"{mode}_{name}"]
                right = repeated.features[f"{mode}_{name}"]
                assert math.isfinite(left)
                assert math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-9)


def test_local_bond_valence_features_fail_open_without_charge_partition() -> None:
    atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    result = compute_odac23_local_bond_valence_features(atoms)

    assert not result.any_supported
    assert not any(result.mode_supported.values())
    assert all(math.isnan(value) for value in result.features.values())
