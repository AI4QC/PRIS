from __future__ import annotations

import numpy as np
from ase import Atoms

from src.next546_lisi_analytic_mechanism_search import (
    primitive_geometry_features,
    symmetric_pair_score,
)


def test_primitive_geometry_features_are_translation_and_order_invariant() -> None:
    atoms = Atoms(
        symbols=["Li", "Si", "Li"],
        scaled_positions=[[0, 0, 0], [0.4, 0.4, 0.4], [0.8, 0.1, 0.2]],
        cell=[[5, 0, 0], [0.2, 5.5, 0], [0.1, 0.3, 6]],
        pbc=True,
    )
    moved = atoms.copy()
    moved.translate([1.2, -0.7, 0.4])
    permuted = moved[[2, 0, 1]]

    expected = primitive_geometry_features(atoms)
    actual = primitive_geometry_features(permuted)

    assert expected.keys() == actual.keys()
    np.testing.assert_allclose(list(expected.values()), list(actual.values()), rtol=0, atol=1e-12)


def test_symmetric_pair_scores_are_frozen() -> None:
    u = np.array([0.2, 0.8])
    v = np.array([0.5, 0.4])

    np.testing.assert_allclose(symmetric_pair_score(u, v, "mean"), [0.35, 0.6])
    np.testing.assert_allclose(symmetric_pair_score(u, v, "maximum"), [0.5, 0.8])
    np.testing.assert_allclose(symmetric_pair_score(u, v, "union"), [0.6, 0.88])
    np.testing.assert_allclose(symmetric_pair_score(u, v, "concurrence"), [0.2, 0.4])
