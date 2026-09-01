from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next399_zachara_bond_valence_vector_closure as n


def _star(vectors: object, strengths: object):
    vector = np.asarray(vectors, dtype=float)
    endpoints = np.column_stack(
        (np.zeros(len(vector), dtype=int), np.arange(1, len(vector) + 1))
    )
    return n.zachara_vector_closure(
        site_valences=np.ones(len(vector) + 1),
        endpoints=endpoints,
        vectors=vector,
        bond_valences=np.asarray(strengths, dtype=float),
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
    result = n.compute_zbvvc_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_and_direction_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next399-zachara-bond-valence-vector-closure-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("zbvvc_zachara_vector_closure_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}


def test_unequal_collinear_two_coordinate_center_has_exact_closure() -> None:
    result = _star(((1, 0, 0), (-1, 0, 0)), (1.0, 4.0))
    assert result.supported, result.failure_reason
    assert result.site_closure[0] == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(result.site_closure[1:], 0.0, atol=1e-14)


def test_equal_trigonal_center_closes_and_angular_strain_lowers_closure() -> None:
    root3 = np.sqrt(3.0)
    ideal = _star(((1, 0, 0), (-0.5, root3 / 2, 0), (-0.5, -root3 / 2, 0)), (1, 1, 1))
    strained = _star(((1, 0, 0), (-0.3, root3 / 2, 0), (-0.5, -root3 / 2, 0)), (1, 1, 1))
    assert ideal.site_closure[0] == pytest.approx(1.0, abs=1e-14)
    assert 0.0 < strained.site_closure[0] < ideal.site_closure[0]


def test_kernel_is_strength_charge_vector_scale_order_rotation_and_replication_invariant() -> None:
    vectors = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=float,
    )
    endpoints = np.tile(np.asarray(((0, 1),), dtype=int), (6, 1))
    strengths = np.asarray((1.4, 1.1, 0.8, 1.2, 0.6, 0.9))
    reference = n.zachara_vector_closure(
        site_valences=np.asarray((2.0, 3.0)),
        endpoints=endpoints,
        vectors=vectors,
        bond_valences=strengths,
    )
    order = np.asarray((4, 1, 5, 0, 3, 2))
    rotation = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=float)
    transformed = n.zachara_vector_closure(
        site_valences=np.asarray((14.0, 21.0)),
        endpoints=endpoints[order],
        vectors=3.7 * vectors[order] @ rotation.T,
        bond_valences=11.0 * strengths[order],
    )
    replicated = n.zachara_vector_closure(
        site_valences=np.asarray((2.0, 3.0, 2.0, 3.0)),
        endpoints=np.vstack((endpoints, endpoints + 2)),
        vectors=np.vstack((vectors, vectors)),
        bond_valences=np.concatenate((strengths, strengths)),
    )
    for result in (reference, transformed, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-14
        )


@pytest.mark.parametrize(
    ("valences", "endpoints", "vectors", "strengths"),
    (
        ([1], [[0, 0]], [[1, 0, 0]], [1]),
        ([1, 1], [[0, 2]], [[1, 0, 0]], [1]),
        ([1, 1, 1], [[0, 1]], [[1, 0, 0]], [1]),
        ([1, 1], [[0, 1]], [[0, 0, 0]], [1]),
        ([1, 1], [[0, 1]], [[1, 0, 0]], [0]),
        ([1, 1], [[0, 1]], [[1, 0]], [1]),
        ([1, 0], [[0, 1]], [[1, 0, 0]], [1]),
    ),
)
def test_malformed_or_isolated_inputs_fail_closed(
    valences: object, endpoints: object, vectors: object, strengths: object
) -> None:
    result = n.zachara_vector_closure(
        site_valences=np.asarray(valences),
        endpoints=np.asarray(endpoints),
        vectors=np.asarray(vectors),
        bond_valences=np.asarray(strengths),
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_crystal_is_supported_with_bounded_diagnostics() -> None:
    result = n.compute_zbvvc_features(_distorted_nacl())
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
        result = n.compute_zbvvc_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_no_dft_flags_are_exact() -> None:
    row = n.compute_zbvvc_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("zbvvc_")) == (
        "zbvvc_zachara_vector_closure_q10",
        "zbvvc_supported",
        "zbvvc_failure",
        "zbvvc_site_count",
        "zbvvc_edge_count",
        "zbvvc_minimum_degree",
        "zbvvc_maximum_degree",
        "zbvvc_valence_policy",
        "zbvvc_parameter_exact_fraction",
        "zbvvc_parameter_generic_fraction",
    )
    assert row["zbvvc_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
