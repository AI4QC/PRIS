from __future__ import annotations

import math

from ase import Atoms
from ase.build import bulk
import numpy as np

from src.next80_periodic_repulsive_load_resolvability import (
    PRLR_FEATURE_NAMES,
    compute_periodic_repulsive_load_resolvability,
    repulsive_load_resolvability_features,
)


def test_collinear_bar_chain_exactly_resolves_compressive_contact() -> None:
    result = repulsive_load_resolvability_features(
        n_sites=3,
        covalent_endpoints=np.asarray([[0, 1], [1, 2]]),
        covalent_vectors=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        contact_endpoints=np.asarray([[0, 2]]),
        contact_vectors=np.asarray([[2.0, 0.0, 0.0]]),
        contact_weights=np.asarray([3.0]),
        characteristic_length=1.0,
    )

    assert result.supported
    assert tuple(result.features) == PRLR_FEATURE_NAMES
    assert result.features["prlr_residual_fraction"] < 1e-10
    assert result.features["prlr_atomic_residual_fraction"] < 1e-10
    assert result.features["prlr_cell_residual_fraction"] < 1e-10
    assert result.features["prlr_risk"] < 1e-10


def test_perpendicular_contact_load_is_not_resolved_by_collinear_bars() -> None:
    result = repulsive_load_resolvability_features(
        n_sites=3,
        covalent_endpoints=np.asarray([[0, 1], [1, 2]]),
        covalent_vectors=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        contact_endpoints=np.asarray([[0, 2]]),
        contact_vectors=np.asarray([[0.0, 2.0, 0.0]]),
        contact_weights=np.asarray([3.0]),
        characteristic_length=1.0,
    )

    assert result.supported
    assert result.features["prlr_residual_fraction"] > 0.9
    assert result.features["prlr_risk"] > 0.0


def test_kernel_is_invariant_to_uniform_length_scaling() -> None:
    kwargs = {
        "n_sites": 3,
        "covalent_endpoints": np.asarray([[0, 1], [1, 2]]),
        "covalent_vectors": np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "contact_endpoints": np.asarray([[0, 2]]),
        "contact_vectors": np.asarray([[0.0, 2.0, 0.0]]),
        "contact_weights": np.asarray([3.0]),
        "characteristic_length": 1.0,
    }
    original = repulsive_load_resolvability_features(**kwargs)
    scaled = repulsive_load_resolvability_features(
        **{
            **kwargs,
            "covalent_vectors": 2.0 * kwargs["covalent_vectors"],
            "contact_vectors": 2.0 * kwargs["contact_vectors"],
            "characteristic_length": 2.0,
        }
    )

    assert original.supported and scaled.supported
    for name in PRLR_FEATURE_NAMES:
        assert math.isclose(
            original.features[name], scaled.features[name], rel_tol=1e-10, abs_tol=1e-12
        ), name


def test_real_structure_features_are_finite_and_supercell_invariant() -> None:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    primitive = compute_periodic_repulsive_load_resolvability(atoms)
    repeated = compute_periodic_repulsive_load_resolvability(atoms.repeat((2, 1, 1)))

    assert primitive.supported and repeated.supported
    assert tuple(primitive.features) == PRLR_FEATURE_NAMES
    for name in PRLR_FEATURE_NAMES:
        assert math.isfinite(primitive.features[name])
        assert math.isclose(
            primitive.features[name], repeated.features[name], rel_tol=2e-6, abs_tol=2e-8
        ), name


def test_structure_without_covalent_framework_fails_open() -> None:
    atoms = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    result = compute_periodic_repulsive_load_resolvability(atoms)

    assert not result.supported
    assert result.features == {}
