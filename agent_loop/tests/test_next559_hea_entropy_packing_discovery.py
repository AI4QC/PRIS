from __future__ import annotations

from ase import Atoms
import numpy as np

from src.next559_hea_entropy_packing_discovery import (
    composition_entropy,
    entropy_packing_union,
)


def test_composition_entropy_uses_atomic_fractions() -> None:
    atoms = Atoms("Fe2Ni2", positions=np.zeros((4, 3)), cell=[4, 4, 4], pbc=True)
    assert composition_entropy(atoms) == np.log(2.0)


def test_entropy_packing_union_is_coefficient_free() -> None:
    np.testing.assert_allclose(
        entropy_packing_union(np.array([0.2, 0.8]), np.array([0.5, 0.4])),
        [0.6, 0.88],
    )
