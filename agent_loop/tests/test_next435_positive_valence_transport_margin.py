from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next435_positive_valence_transport_margin as n


def test_frozen_schema_direction_and_boundary() -> None:
    assert n.PROTOCOL == "2026-08-13-next435-positive-valence-transport-margin-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("pvtm_positive_transport_margin",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_complete_graph_has_unit_raw_margin_and_half_bounded_feature() -> None:
    result = n.positive_valence_transport_margin(
        charges=(1.0, 1.0, -1.0, -1.0),
        endpoints=((0, 2), (0, 3), (1, 2), (1, 3)),
    )
    assert result.supported, result.failure_reason
    assert result.feasible
    assert result.raw_margin == pytest.approx(1.0)
    assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(0.5)
    assert result.maximum_equality_residual <= n.LP_RESIDUAL_TOLERANCE


def test_boundary_and_infeasible_graphs_are_supported_zero_violations() -> None:
    boundary = n.positive_valence_transport_margin(
        charges=(1.0, 1.0, -1.0, -1.0),
        endpoints=((0, 2), (0, 3), (1, 3)),
    )
    infeasible = n.positive_valence_transport_margin(
        charges=(1.5, 0.5, -1.0, -1.0),
        endpoints=((0, 2), (1, 3)),
    )
    for result in (boundary, infeasible):
        assert result.supported, result.failure_reason
        assert result.raw_margin == 0.0
        assert result.features[n.FEATURE_NAMES[0]] == 0.0
    assert boundary.feasible is True
    assert infeasible.feasible is False


def test_kernel_is_charge_edge_order_and_exact_replication_invariant() -> None:
    charges = (1.0, 1.0, -0.5, -1.5)
    endpoints = ((0, 2), (0, 3), (1, 3))
    reference = n.positive_valence_transport_margin(charges=charges, endpoints=endpoints)
    scaled = n.positive_valence_transport_margin(
        charges=tuple(7.3 * value for value in charges), endpoints=endpoints
    )
    reordered = n.positive_valence_transport_margin(
        charges=charges, endpoints=tuple(reversed(endpoints))
    )
    replicated = n.positive_valence_transport_margin(
        charges=charges + charges,
        endpoints=endpoints + tuple((left + 4, right + 4) for left, right in endpoints),
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
    result = n.positive_valence_transport_margin(charges=charges, endpoints=endpoints)
    assert not result.supported
    assert result.features == {}


def _distorted_nacl() -> Atoms:
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.set_cell([[5.64,0,0],[0.27,5.77,0],[0.18,0.31,5.53]], scale_atoms=True)
    atoms.positions[1] += [0.08, -0.04, 0.06]
    atoms.wrap()
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_pvtm_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_periodic_equivalents_and_geometry_firewall() -> None:
    atoms = _distorted_nacl()
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([.17,.29,.42]); translated.wrap()
    permuted = atoms[[3,0,6,1,7,4,2,5]]
    rebased = atoms.copy(); rebased.set_cell(np.asarray([[1,1,0],[0,1,0],[0,0,1]]) @ atoms.cell.array, scale_atoms=False); rebased.wrap()
    for equivalent in (rotated, translated, permuted, rebased, atoms.repeat((2,1,1))):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)
    bad = atoms.copy(); bad.calc = Calculator()
    result = n.compute_pvtm_features(bad)
    assert not result.supported and "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_pvtm_row(atoms)
    assert row["pvtm_supported"] is True
    assert tuple(key for key in row if key.startswith("pvtm_")) == (
        "pvtm_positive_transport_margin", "pvtm_supported", "pvtm_failure",
        "pvtm_feasible", "pvtm_site_count", "pvtm_edge_count",
        "pvtm_raw_margin", "pvtm_maximum_equality_residual",
        "pvtm_minimum_lower_bound_residual", "pvtm_valence_policy",
    )
