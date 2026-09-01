from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next395_periodic_bond_valence_tensor_isotropy as n


def _axis_system(weights: object, planar: bool = False):
    directions = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
    if not planar:
        directions += ((0, 0, 1), (0, 0, -1))
    vectors = np.asarray(directions, dtype=float)
    endpoints = np.tile(np.asarray(((0, 1),), dtype=int), (len(vectors), 1))
    return n.bond_valence_tensor_isotropy(
        n_sites=2,
        endpoints=endpoints,
        vectors=vectors,
        bond_valences=np.asarray(weights, dtype=float),
    )


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray([[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_pbvti_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_and_direction_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next395-periodic-bond-valence-tensor-isotropy-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("pbvti_bond_valence_tensor_isotropy_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}


def test_isotropic_axis_system_is_one_and_planar_system_is_zero() -> None:
    isotropic = _axis_system(np.ones(6))
    planar = _axis_system(np.ones(4), planar=True)
    assert isotropic.supported and planar.supported
    assert isotropic.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0, abs=1e-14)
    assert planar.features[n.FEATURE_NAMES[0]] == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(isotropic.site_isotropy, np.ones(2), atol=1e-14)


def test_directional_valence_redistribution_strictly_lowers_isotropy() -> None:
    equal = _axis_system([1, 1, 1, 1, 1, 1])
    mild = _axis_system([1.3, 1.3, 1, 1, 0.7, 0.7])
    strong = _axis_system([1.8, 1.8, 1, 1, 0.2, 0.2])
    assert (
        equal.features[n.FEATURE_NAMES[0]]
        > mild.features[n.FEATURE_NAMES[0]]
        > strong.features[n.FEATURE_NAMES[0]]
    )


def test_kernel_is_scale_order_rotation_and_disjoint_replication_invariant() -> None:
    vectors = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=float,
    )
    endpoints = np.tile(np.asarray(((0, 1),), dtype=int), (6, 1))
    weights = np.asarray((1.4, 1.1, 0.8, 1.2, 0.6, 0.9))
    reference = n.bond_valence_tensor_isotropy(
        n_sites=2, endpoints=endpoints, vectors=vectors, bond_valences=weights
    )
    order = np.asarray((4, 1, 5, 0, 3, 2))
    rotation = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=float)
    reordered = n.bond_valence_tensor_isotropy(
        n_sites=2,
        endpoints=endpoints[order],
        vectors=3.7 * vectors[order] @ rotation.T,
        bond_valences=11.0 * weights[order],
    )
    replicated = n.bond_valence_tensor_isotropy(
        n_sites=4,
        endpoints=np.vstack((endpoints, endpoints + 2)),
        vectors=np.vstack((vectors, vectors)),
        bond_valences=np.concatenate((weights, weights)),
    )
    for result in (reference, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-14
        )


@pytest.mark.parametrize(
    ("n_sites", "endpoints", "vectors", "weights"),
    (
        (1, [[0, 0]], [[1, 0, 0]], [1]),
        (2, [[0, 2]], [[1, 0, 0]], [1]),
        (3, [[0, 1]], [[1, 0, 0]], [1]),
        (2, [[0, 1]], [[0, 0, 0]], [1]),
        (2, [[0, 1]], [[1, 0, 0]], [0]),
        (2, [[0, 1]], [[1, 0]], [1]),
    ),
)
def test_malformed_or_isolated_inputs_fail_closed(
    n_sites: int, endpoints: object, vectors: object, weights: object
) -> None:
    result = n.bond_valence_tensor_isotropy(
        n_sites=n_sites,
        endpoints=np.asarray(endpoints),
        vectors=np.asarray(vectors),
        bond_valences=np.asarray(weights),
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_crystal_is_supported_with_bounded_diagnostics() -> None:
    result = n.compute_pbvti_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.edge_count >= result.site_count
    assert result.minimum_degree >= 1
    assert result.maximum_degree >= result.minimum_degree
    assert result.valence_policy in {
        "integer_oxidation_state",
        "fractional_oxidation_state",
        "electronegativity_partition",
    }
    assert 0 <= result.parameter_exact_fraction <= 1
    assert 0 <= result.parameter_generic_fraction <= 1
    assert 0 <= result.features[n.FEATURE_NAMES[0]] <= 1


def test_structure_equivalences_preserve_feature() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.173, 0.291, 0.419])
    translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy()
    rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
        @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    supercell = atoms.repeat((2, 1, 1))
    for equivalent in (rotated, translated, permuted, rebased, supercell):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)


def test_geometry_boundary_fails_closed() -> None:
    atoms = _distorted_nacl()
    with_calculator = atoms.copy()
    with_calculator.calc = Calculator()
    with_metadata = atoms.copy()
    with_metadata.info["outcome"] = 1
    with_array = atoms.copy()
    with_array.new_array("energy", np.zeros(len(with_array)))
    nonperiodic = atoms.copy()
    nonperiodic.pbc = False
    nonfinite = atoms.copy()
    nonfinite.positions[0, 0] = np.nan
    for changed in (
        with_calculator,
        with_metadata,
        with_array,
        nonperiodic,
        nonfinite,
    ):
        result = n.compute_pbvti_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_no_dft_flags_are_exact() -> None:
    row = n.compute_pbvti_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("pbvti_")) == (
        "pbvti_bond_valence_tensor_isotropy_q10",
        "pbvti_supported",
        "pbvti_failure",
        "pbvti_site_count",
        "pbvti_edge_count",
        "pbvti_minimum_degree",
        "pbvti_maximum_degree",
        "pbvti_valence_policy",
        "pbvti_parameter_exact_fraction",
        "pbvti_parameter_generic_fraction",
    )
    assert row["pbvti_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
