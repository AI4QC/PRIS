from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next371_periodic_chardi_return_consistency as n


def _balanced_square():
    return n.chardi_return_consistency(
        charges=np.asarray((1.0, 1.0, -1.0, -1.0)),
        endpoints=np.asarray(((0, 2), (0, 3), (1, 2), (1, 3))),
        distances=np.ones(4),
    )


def _unbalanced_square():
    return n.chardi_return_consistency(
        charges=np.asarray((1.0, 1.0, -1.0, -1.0)),
        endpoints=np.asarray(((0, 2), (0, 3), (1, 2), (1, 3))),
        distances=np.asarray((1.0, 1.55, 1.0, 1.13)),
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
    result = n.compute_pchardi_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_and_direction_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next371-periodic-chardi-return-consistency-v1"
    assert n.DESIGN_SHA256 == (
        "4517c5a01f65293665c7029322cafa1878767248628a72ec33d9b035187014fa"
    )
    assert n.FEATURE_NAMES == ("pchardi_cation_return_mapd",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_low"}


def test_balanced_square_returns_exact_formal_cation_charge() -> None:
    result = _balanced_square()
    assert result.supported, result.failure_reason
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(result.anion_received_charges, [1.0, 1.0])
    np.testing.assert_allclose(result.cation_returned_charges, [1.0, 1.0])


def test_asymmetric_forward_distributions_produce_positive_return_mapd() -> None:
    balanced = _balanced_square()
    unbalanced = _unbalanced_square()
    assert unbalanced.supported, unbalanced.failure_reason
    assert unbalanced.features[n.FEATURE_NAMES[0]] > balanced.features[n.FEATURE_NAMES[0]]
    assert not np.allclose(unbalanced.anion_received_charges, [1.0, 1.0])
    assert not np.allclose(unbalanced.cation_returned_charges, [1.0, 1.0])


def test_kernel_is_distance_scale_edge_order_and_replication_invariant() -> None:
    charges = np.asarray((1.0, 1.0, -1.0, -1.0))
    endpoints = np.asarray(((0, 2), (0, 3), (1, 2), (1, 3)))
    distances = np.asarray((1.0, 1.55, 1.0, 1.13))
    reference = n.chardi_return_consistency(
        charges=charges, endpoints=endpoints, distances=distances
    )
    order = np.asarray((2, 0, 3, 1))
    reordered = n.chardi_return_consistency(
        charges=charges,
        endpoints=endpoints[order],
        distances=7.0 * distances[order],
    )
    replicated = n.chardi_return_consistency(
        charges=np.concatenate((charges, charges)),
        endpoints=np.vstack((endpoints, endpoints + 4)),
        distances=np.concatenate((distances, distances)),
    )
    for result in (reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1.0e-14
        )


@pytest.mark.parametrize(
    ("charges", "endpoints", "distances"),
    (
        ([1.0, -1.0], [[0, 0]], [1.0]),
        ([1.0, -1.0], [[0, 2]], [1.0]),
        ([1.0, -1.0], [[0, 1]], [0.0]),
        ([1.0, -1.0], [[0, 1]], [1.0, 2.0]),
        ([1.0, 1.0, -1.0], [[0, 2]], [1.0]),
        ([1.0, 0.0, -1.0], [[0, 2]], [1.0]),
    ),
)
def test_malformed_or_chemically_incomplete_kernel_inputs_fail_closed(
    charges: object, endpoints: object, distances: object
) -> None:
    result = n.chardi_return_consistency(
        charges=np.asarray(charges),
        endpoints=np.asarray(endpoints),
        distances=np.asarray(distances),
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_single_anion_crystal_is_supported() -> None:
    result = n.compute_pchardi_features(_distorted_nacl())
    assert result.supported, result.failure_reason
    assert result.site_count == len(_distorted_nacl())
    assert result.edge_count >= result.site_count
    assert result.cation_count > 0 and result.anion_count > 0
    assert result.iterations >= 1
    assert result.maximum_mean_distance_residual <= n.MEAN_DISTANCE_TOLERANCE
    assert result.valence_policy in {
        "integer_oxidation_state",
        "fractional_oxidation_state",
        "electronegativity_partition",
    }
    assert result.anion_species == "Cl"
    assert result.features[n.FEATURE_NAMES[0]] >= 0.0


def test_multiple_anion_species_abstain() -> None:
    atoms = Atoms(
        symbols=["Na", "Cl", "F"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        cell=np.eye(3) * 6.0,
        pbc=True,
    )
    result = n.compute_pchardi_features(atoms)
    assert result.supported is False
    assert "one negative-valence chemical species" in str(result.failure_reason)


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
    for changed in (with_calculator, with_metadata, with_array, nonperiodic, nonfinite):
        result = n.compute_pchardi_features(changed)
        assert result.supported is False
        assert "geometry-only Atoms" in str(result.failure_reason)


def test_row_schema_and_no_dft_boundary_flags_are_exact() -> None:
    row = n.compute_pchardi_row(_distorted_nacl())
    assert tuple(name for name in row if name.startswith("pchardi_")) == (
        "pchardi_cation_return_mapd",
        "pchardi_supported",
        "pchardi_failure",
        "pchardi_site_count",
        "pchardi_edge_count",
        "pchardi_cation_count",
        "pchardi_anion_count",
        "pchardi_iterations",
        "pchardi_maximum_mean_distance_residual",
        "pchardi_valence_policy",
        "pchardi_anion_species",
    )
    assert row["pchardi_supported"] is True
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())
