from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next403_zachara_over_linear_vector_gain as n


def _star(vectors: object, strengths: object):
    vector = np.asarray(vectors, dtype=float)
    endpoints = np.column_stack(
        (np.zeros(len(vector), dtype=int), np.arange(1, len(vector) + 1))
    )
    return n.zachara_over_linear_vector_gain(
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
    result = n.compute_zbvvg_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_and_direction_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next403-zachara-over-linear-vector-gain-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("zbvvg_zachara_over_linear_gain_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}


def test_unequal_collinear_center_has_exact_zachara_gain() -> None:
    result = _star(((1, 0, 0), (-1, 0, 0)), (1.0, 4.0))
    assert result.supported, result.failure_reason
    assert result.site_linear_closure[0] == pytest.approx(0.4, abs=1e-14)
    assert result.site_zachara_closure[0] == pytest.approx(1.0, abs=1e-14)
    assert result.site_gain[0] == pytest.approx(0.8, abs=1e-14)


def test_equal_collinear_and_trigonal_centers_have_neutral_gain() -> None:
    root3 = np.sqrt(3.0)
    linear = _star(((1, 0, 0), (-1, 0, 0)), (1, 1))
    trigonal = _star(
        ((1, 0, 0), (-0.5, root3 / 2, 0), (-0.5, -root3 / 2, 0)),
        (1, 1, 1),
    )
    assert linear.site_gain[0] == pytest.approx(0.5, abs=1e-14)
    assert trigonal.site_gain[0] == pytest.approx(0.5, abs=1e-14)


def test_kernel_is_scale_charge_order_rotation_and_replication_invariant() -> None:
    vectors = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=float,
    )
    endpoints = np.tile(np.asarray(((0, 1),), dtype=int), (6, 1))
    strengths = np.asarray((1.4, 1.1, 0.8, 1.2, 0.6, 0.9))
    reference = n.zachara_over_linear_vector_gain(
        site_valences=np.asarray((2.0, 3.0)), endpoints=endpoints,
        vectors=vectors, bond_valences=strengths,
    )
    order = np.asarray((4, 1, 5, 0, 3, 2))
    rotation = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=float)
    transformed = n.zachara_over_linear_vector_gain(
        site_valences=np.asarray((14.0, 21.0)), endpoints=endpoints[order],
        vectors=3.7 * vectors[order] @ rotation.T,
        bond_valences=11.0 * strengths[order],
    )
    replicated = n.zachara_over_linear_vector_gain(
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
    result = n.zachara_over_linear_vector_gain(
        site_valences=np.asarray(valences), endpoints=np.asarray(endpoints),
        vectors=np.asarray(vectors), bond_valences=np.asarray(strengths),
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_crystal_and_equivalents_are_supported_and_invariant() -> None:
    atoms = _distorted_nacl()
    result = n.compute_zbvvg_features(atoms)
    assert result.supported, result.failure_reason
    assert result.site_count == len(atoms)
    assert result.minimum_degree >= 1
    assert 0 <= result.features[n.FEATURE_NAMES[0]] <= 1
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy(); rebased.set_cell(np.asarray([[1,1,0],[0,1,0],[0,0,1]]) @ atoms.cell.array, scale_atoms=False); rebased.wrap()
    for equivalent in (rotated, translated, permuted, rebased, atoms.repeat((2,1,1))):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)


def test_geometry_boundary_and_row_schema_are_exact() -> None:
    atoms = _distorted_nacl()
    changed = []
    item = atoms.copy(); item.calc = Calculator(); changed.append(item)
    item = atoms.copy(); item.info["outcome"] = 1; changed.append(item)
    item = atoms.copy(); item.new_array("energy", np.zeros(len(item))); changed.append(item)
    item = atoms.copy(); item.pbc = False; changed.append(item)
    item = atoms.copy(); item.positions[0, 0] = np.nan; changed.append(item)
    for item in changed:
        result = n.compute_zbvvg_features(item)
        assert not result.supported
        assert "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_zbvvg_row(atoms)
    assert tuple(name for name in row if name.startswith("zbvvg_")) == (
        "zbvvg_zachara_over_linear_gain_q10", "zbvvg_supported", "zbvvg_failure",
        "zbvvg_site_count", "zbvvg_edge_count", "zbvvg_minimum_degree",
        "zbvvg_maximum_degree", "zbvvg_valence_policy",
        "zbvvg_parameter_exact_fraction", "zbvvg_parameter_generic_fraction",
    )
    assert row["zbvvg_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
