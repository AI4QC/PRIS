from __future__ import annotations

from ase import Atoms
import numpy as np

from src.next49_framework_topology import (
    FRAMEWORK_FEATURE_NAMES,
    _environment_versions,
    compute_framework_topology_features,
)


def _carbon_chain() -> Atoms:
    return Atoms(
        "C",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([1.40, 8.0, 8.0]),
        pbc=True,
    )


def test_translation_rank_separates_discrete_chain_and_three_dimensional_net() -> None:
    molecule = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    chain = _carbon_chain()
    net = Atoms(
        "C",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.diag([1.40, 1.40, 1.40]),
        pbc=True,
    )

    discrete = compute_framework_topology_features(molecule)
    one_d = compute_framework_topology_features(chain)
    three_d = compute_framework_topology_features(net)

    assert discrete.supported and one_d.supported and three_d.supported
    assert discrete.features["periodic_dimension_max"] == 0.0
    assert discrete.features["periodic_framework_fraction"] == 0.0
    assert one_d.features["periodic_dimension_max"] == 1.0
    assert one_d.features["periodic_framework_fraction"] == 1.0
    assert three_d.features["periodic_dimension_max"] == 3.0
    assert three_d.features["periodic_framework_fraction"] == 1.0


def test_framework_features_are_translation_permutation_and_supercell_invariant() -> None:
    primitive = _carbon_chain()
    translated = primitive.copy()
    translated.positions += [0.37, -0.22, 0.19]
    supercell = primitive.repeat((2, 1, 1))

    base = compute_framework_topology_features(primitive)
    shifted = compute_framework_topology_features(translated)
    repeated = compute_framework_topology_features(supercell)

    assert base.supported and shifted.supported and repeated.supported
    for name in FRAMEWORK_FEATURE_NAMES:
        assert np.isclose(base.features[name], shifted.features[name], atol=1e-10)
        assert np.isclose(base.features[name], repeated.features[name], atol=1e-10)


def test_metal_vector_imbalance_detects_missing_coordination_direction() -> None:
    balanced = Atoms(
        symbols=["Cu", "O", "O", "O", "O"],
        positions=[
            [5.0, 5.0, 5.0],
            [6.9, 5.0, 5.0],
            [3.1, 5.0, 5.0],
            [5.0, 6.9, 5.0],
            [5.0, 3.1, 5.0],
        ],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    missing = balanced[[0, 1, 2, 3]]

    symmetric = compute_framework_topology_features(balanced)
    asymmetric = compute_framework_topology_features(missing)

    assert symmetric.supported and asymmetric.supported
    assert symmetric.features["metal_vector_imbalance_q95"] < 1e-10
    assert asymmetric.features["metal_vector_imbalance_q95"] > 0.30


def test_geometry_only_boundary_fails_open_and_schema_has_no_endpoint_tokens() -> None:
    invalid = _carbon_chain()
    invalid.info["source"] = "not geometry only"
    result = compute_framework_topology_features(invalid)

    assert not result.supported
    assert result.failure_reason is not None
    assert result.features == {}
    forbidden = ("energy", "force", "stress", "relax", "dft", "label", "target")
    assert not any(token in name for name in FRAMEWORK_FEATURE_NAMES for token in forbidden)


def test_environment_versions_are_explicitly_frozen_for_rebuilds() -> None:
    versions = _environment_versions()
    assert tuple(versions) == ("ase", "numpy", "pandas", "pymatgen")
    assert all(isinstance(value, str) and value for value in versions.values())
