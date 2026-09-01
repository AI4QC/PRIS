"""Contracts for deterministic raw-x0 local-motif coherence features."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure


def test_motif_features_are_finite_and_use_fixed_schema() -> None:
    from src.next46_motif_coherence_features import (
        FEATURE_NAMES,
        compute_motif_coherence_features,
    )

    atoms = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    ).to_ase_atoms()
    result = compute_motif_coherence_features(atoms)
    assert result.supported is True
    assert tuple(result.features) == FEATURE_NAMES
    assert np.isfinite(list(result.features.values())).all()
    assert 0.0 <= result.features["motif_cn_dominance_mean"] <= 1.0
    assert result.features["motif_same_element_dispersion_rms"] == 0.0


def test_motif_aggregates_are_invariant_to_site_permutation() -> None:
    from src.next46_motif_coherence_features import compute_motif_coherence_features

    structure = Structure(
        Lattice.cubic(5.5),
        ["Li", "Li", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    first = compute_motif_coherence_features(structure.to_ase_atoms())
    order = [2, 0, 3, 1]
    permuted = Structure(
        structure.lattice,
        [structure[index].specie for index in order],
        [structure[index].frac_coords for index in order],
    )
    second = compute_motif_coherence_features(permuted.to_ase_atoms())
    assert first.supported and second.supported
    assert np.allclose(
        list(first.features.values()),
        list(second.features.values()),
        rtol=1e-10,
        atol=1e-12,
    )


def test_motif_feature_names_do_not_cross_execution_boundary() -> None:
    from src.next46_motif_coherence_features import FEATURE_NAMES

    assert not any(
        token in name.lower()
        for name in FEATURE_NAMES
        for token in ("dft", "energy", "force", "stress", "relax", "model")
    )
