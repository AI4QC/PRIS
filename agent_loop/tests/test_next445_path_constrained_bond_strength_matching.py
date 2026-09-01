from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next445_path_constrained_bond_strength_matching as n


def test_frozen_schema_direction_and_boundary() -> None:
    assert n.PROTOCOL == "2026-08-13-next445-path-constrained-bond-strength-matching-v1"
    assert n.DESIGN_SHA256 == "e7c91c51167a4c4653bfc8a0eb9ee7cfc25bacb7d7f1300c20f384f477da80b6"
    assert n.FEATURE_NAMES == ("pcabsm_path_constrained_bond_strength_matching",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_complete_balanced_graph_matches_endpoint_reference_exactly() -> None:
    result = n.path_constrained_bond_strength_matching(
        charges=(1.0, 1.0, -1.0, -1.0),
        endpoints=((0, 2), (0, 3), (1, 2), (1, 3)),
    )
    assert result.supported, result.failure_reason
    assert result.feasible
    assert result.edge_strengths == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert result.reference_strengths == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert result.normalized_mismatch == pytest.approx(0.0)
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0)


def test_reversed_path_edge_has_maximal_local_mismatch() -> None:
    result = n.path_constrained_bond_strength_matching(
        charges=(0.25, 0.5, -0.25, -0.5),
        endpoints=((0, 2), (0, 3), (1, 2)),
    )
    assert result.supported, result.failure_reason
    assert result.feasible
    assert result.edge_strengths == pytest.approx((-0.25, 0.5, 0.5), abs=1e-12)
    assert all(value > 0.0 for value in result.reference_strengths)
    assert 0.0 < result.normalized_mismatch < 1.0
    assert 0.0 < result.features[n.FEATURE_NAMES[0]] < 1.0


def test_obstructed_graph_is_supported_zero() -> None:
    result = n.path_constrained_bond_strength_matching(
        charges=(1.0, 0.5, -0.5, -1.0),
        endpoints=((0, 2), (1, 3)),
    )
    assert result.supported, result.failure_reason
    assert result.feasible is False
    assert result.features[n.FEATURE_NAMES[0]] == 0.0


def test_kernel_is_charge_edge_order_and_exact_replication_invariant() -> None:
    charges = (0.25, 0.5, -0.25, -0.5)
    endpoints = ((0, 2), (0, 3), (1, 2))
    reference = n.path_constrained_bond_strength_matching(
        charges=charges, endpoints=endpoints
    )
    scaled = n.path_constrained_bond_strength_matching(
        charges=tuple(7.3 * value for value in charges), endpoints=endpoints
    )
    reordered = n.path_constrained_bond_strength_matching(
        charges=charges, endpoints=tuple(reversed(endpoints))
    )
    replicated = n.path_constrained_bond_strength_matching(
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
    result = n.path_constrained_bond_strength_matching(
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
    result = n.compute_pcabsm_features(atoms)
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
    result = n.compute_pcabsm_features(bad)
    assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_pcabsm_row(atoms)
    assert row["pcabsm_supported"] is True
    assert tuple(key for key in row if key.startswith("pcabsm_")) == (
        "pcabsm_path_constrained_bond_strength_matching",
        "pcabsm_supported",
        "pcabsm_failure",
        "pcabsm_feasible",
        "pcabsm_site_count",
        "pcabsm_edge_count",
        "pcabsm_normalized_mismatch",
        "pcabsm_maximum_equality_residual",
        "pcabsm_maximum_path_residual",
        "pcabsm_valence_policy",
    )
