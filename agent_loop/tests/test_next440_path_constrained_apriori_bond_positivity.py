from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next440_path_constrained_apriori_bond_positivity as n


def test_frozen_schema_direction_and_boundary() -> None:
    assert n.PROTOCOL == "2026-08-13-next440-path-constrained-apriori-bond-positivity-v1"
    assert n.DESIGN_SHA256 == "b3a49a6a5f50c42e843b479551458b0a8a6c3e46252a20e64b241a5368f75153"
    assert n.FEATURE_NAMES == ("pcabp_path_constrained_bond_positivity",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_balanced_complete_graph_has_only_forward_bonds() -> None:
    result = n.path_constrained_apriori_bond_positivity(
        charges=(1.0, 1.0, -1.0, -1.0),
        endpoints=((0, 2), (0, 3), (1, 2), (1, 3)),
    )
    assert result.supported, result.failure_reason
    assert result.feasible
    assert result.edge_strengths == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert result.negative_strength_mass == pytest.approx(0.0)
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0)
    assert result.maximum_equality_residual <= n.SOLVE_RESIDUAL_TOLERANCE
    assert result.maximum_path_residual <= n.SOLVE_RESIDUAL_TOLERANCE


def test_unique_path_constrained_field_detects_a_reversed_bond() -> None:
    result = n.path_constrained_apriori_bond_positivity(
        charges=(0.25, 0.5, -0.25, -0.5),
        endpoints=((0, 2), (0, 3), (1, 2)),
    )
    assert result.supported, result.failure_reason
    assert result.feasible
    assert result.edge_strengths == pytest.approx((-0.25, 0.5, 0.5), abs=1e-12)
    assert result.positive_strength_mass == pytest.approx(1.0)
    assert result.negative_strength_mass == pytest.approx(0.25)
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(0.8)


def test_component_charge_obstruction_and_isolation_are_supported_zero() -> None:
    obstructed = n.path_constrained_apriori_bond_positivity(
        charges=(1.0, 0.5, -0.5, -1.0),
        endpoints=((0, 2), (1, 3)),
    )
    isolated = n.path_constrained_apriori_bond_positivity(
        charges=(1.0, 1.0, -1.0, -1.0),
        endpoints=((0, 2),),
    )
    for result in (obstructed, isolated):
        assert result.supported, result.failure_reason
        assert result.feasible is False
        assert result.features[n.FEATURE_NAMES[0]] == 0.0


def test_kernel_is_charge_edge_order_and_exact_replication_invariant() -> None:
    charges = (0.25, 0.5, -0.25, -0.5)
    endpoints = ((0, 2), (0, 3), (1, 2))
    reference = n.path_constrained_apriori_bond_positivity(
        charges=charges, endpoints=endpoints
    )
    scaled = n.path_constrained_apriori_bond_positivity(
        charges=tuple(7.3 * value for value in charges), endpoints=endpoints
    )
    reordered = n.path_constrained_apriori_bond_positivity(
        charges=charges, endpoints=tuple(reversed(endpoints))
    )
    replicated = n.path_constrained_apriori_bond_positivity(
        charges=charges + charges,
        endpoints=endpoints
        + tuple((left + 4, right + 4) for left, right in endpoints),
    )
    for result in (reference, scaled, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-10
        )


@pytest.mark.parametrize(
    ("charges", "endpoints"),
    (
        ((), ()),
        ((1.0, -0.5), ((0, 1),)),
        ((1.0, -1.0), ((1, 0),)),
        ((1.0, -1.0), ((0, 2),)),
        ((1.0, np.nan), ((0, 1),)),
    ),
)
def test_malformed_inputs_fail_closed(charges: object, endpoints: object) -> None:
    result = n.path_constrained_apriori_bond_positivity(
        charges=charges, endpoints=endpoints
    )
    assert not result.supported
    assert result.features == {}


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell(
        [[5.64, 0, 0], [0.27, 5.77, 0], [0.18, 0.31, 5.53]],
        scale_atoms=True,
    )
    atoms.positions[1] += [0.08, -0.04, 0.06]
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_pcabp_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_periodic_equivalents_and_geometry_firewall() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy()
    rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy()
    translated.translate([0.17, 0.29, 0.42])
    translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 7, 4, 2, 5]]
    rebased = atoms.copy()
    rebased.set_cell(
        np.asarray([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell.array,
        scale_atoms=False,
    )
    rebased.wrap()
    for equivalent in (
        rotated,
        translated,
        permuted,
        rebased,
        atoms.repeat((2, 1, 1)),
    ):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)

    bad = atoms.copy()
    bad.calc = Calculator()
    result = n.compute_pcabp_features(bad)
    assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_pcabp_row(atoms)
    assert row["pcabp_supported"] is True
    assert tuple(key for key in row if key.startswith("pcabp_")) == (
        "pcabp_path_constrained_bond_positivity",
        "pcabp_supported",
        "pcabp_failure",
        "pcabp_feasible",
        "pcabp_site_count",
        "pcabp_edge_count",
        "pcabp_positive_strength_mass",
        "pcabp_negative_strength_mass",
        "pcabp_maximum_equality_residual",
        "pcabp_maximum_path_residual",
        "pcabp_valence_policy",
    )
