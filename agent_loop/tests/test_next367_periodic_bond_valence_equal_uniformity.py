from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next367_periodic_bond_valence_equal_uniformity as n


def _star(values: object):
    return n.equal_valence_uniformity_features(
        n_sites=4,
        endpoints=np.asarray(((0, 1), (0, 2), (0, 3)), dtype=int),
        bond_valences=np.asarray(values, dtype=float),
    )


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        np.asarray(
            [[5.64, 0.0, 0.0], [0.27, 5.77, 0.0], [0.18, 0.31, 5.53]]
        ),
        scale_atoms=True,
    )
    atoms.positions[1] += np.asarray([0.08, -0.04, 0.06])
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_pbveu_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_and_direction_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next367-periodic-bond-valence-equal-uniformity-v1"
    assert n.DESIGN_SHA256 == (
        "c63b1042315a6df72a7368de31921f2f8e10cce67aa1e408a581bb5bd197132c"
    )
    assert n.FEATURE_NAMES == ("pbveu_equal_valence_uniformity_q10",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}


def test_equal_star_is_exactly_uniform_and_unequal_star_is_not() -> None:
    equal = _star([2.0, 2.0, 2.0])
    unequal = _star([1.0, 2.0, 3.0])
    assert equal.supported and unequal.supported
    assert equal.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0, abs=1.0e-14)
    assert 0.0 < unequal.features[n.FEATURE_NAMES[0]] < 1.0
    np.testing.assert_allclose(equal.site_uniformities, np.ones(4), atol=1.0e-14)


def test_mean_preserving_valence_transfer_strictly_lowers_uniformity() -> None:
    equal = _star([2.0, 2.0, 2.0])
    mild = _star([1.8, 2.0, 2.2])
    strong = _star([1.0, 2.0, 3.0])
    assert (
        equal.features[n.FEATURE_NAMES[0]]
        > mild.features[n.FEATURE_NAMES[0]]
        > strong.features[n.FEATURE_NAMES[0]]
    )


def test_kernel_is_strength_scale_edge_order_and_replication_invariant() -> None:
    endpoints = np.asarray(((0, 2), (0, 3), (1, 2), (1, 3)), dtype=int)
    values = np.asarray((1.4, 1.0, 0.8, 1.2), dtype=float)
    reference = n.equal_valence_uniformity_features(
        n_sites=4, endpoints=endpoints, bond_valences=values
    )
    order = np.asarray((2, 0, 3, 1))
    reordered = n.equal_valence_uniformity_features(
        n_sites=4,
        endpoints=endpoints[order],
        bond_valences=7.0 * values[order],
    )
    replicated = n.equal_valence_uniformity_features(
        n_sites=8,
        endpoints=np.vstack((endpoints, endpoints + 4)),
        bond_valences=np.concatenate((values, values)),
    )
    for result in (reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1.0e-14
        )


@pytest.mark.parametrize(
    ("n_sites", "endpoints", "values"),
    (
        (1, np.asarray(((0, 0),)), np.asarray((1.0,))),
        (3, np.asarray(((0, 3),)), np.asarray((1.0,))),
        (3, np.asarray(((0, 1),)), np.asarray((1.0, 2.0))),
        (3, np.asarray(((0, 1), (1, 2))), np.asarray((0.0, 1.0))),
        (3, np.asarray(((0, 1),)), np.asarray((1.0,))),
    ),
)
def test_malformed_or_isolated_kernel_inputs_fail_closed(
    n_sites: int, endpoints: np.ndarray, values: np.ndarray
) -> None:
    result = n.equal_valence_uniformity_features(
        n_sites=n_sites, endpoints=endpoints, bond_valences=values
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_crystal_is_supported_with_bounded_diagnostics() -> None:
    result = n.compute_pbveu_features(_distorted_nacl())
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
    assert 0.0 <= result.parameter_exact_fraction <= 1.0
    assert 0.0 <= result.parameter_generic_fraction <= 1.0
    assert 0.0 < result.features[n.FEATURE_NAMES[0]] <= 1.0


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
        assert _feature(equivalent) == pytest.approx(reference, abs=1.0e-8)


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
        result = n.compute_pbveu_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_no_dft_boundary_flags_are_exact() -> None:
    row = n.compute_pbveu_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("pbveu_")) == (
        "pbveu_equal_valence_uniformity_q10",
        "pbveu_supported",
        "pbveu_failure",
        "pbveu_site_count",
        "pbveu_edge_count",
        "pbveu_minimum_degree",
        "pbveu_maximum_degree",
        "pbveu_valence_policy",
        "pbveu_parameter_exact_fraction",
        "pbveu_parameter_generic_fraction",
    )
    assert row["pbveu_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
