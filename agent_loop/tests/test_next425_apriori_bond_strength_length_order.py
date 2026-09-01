from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next425_apriori_bond_strength_length_order as n


def _transport_example(distances: tuple[float, float, float]):
    return n.apriori_length_order_protection(
        charges=(1.0, 1.0, -0.5, -1.5),
        endpoints=((0, 2), (0, 3), (1, 3)),
        distances=distances,
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
    result = n.compute_aprbs_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_direction_and_boundary_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next425-apriori-bond-strength-length-order-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("aprbs_length_order_protection",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert n.MARGINAL_TOLERANCE == 1.0e-10
    assert n.MAXIMUM_ITERATIONS == 20_000
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_maximum_entropy_field_matches_unique_analytic_transport() -> None:
    result = _transport_example((2.0, 2.0, 1.0))
    assert result.supported, result.failure_reason
    assert result.edge_strengths == pytest.approx((0.5, 0.5, 1.0), abs=1e-10)
    assert result.maximum_marginal_residual <= n.MARGINAL_TOLERANCE
    assert result.iterations > 0


def test_kernel_distinguishes_agreeing_reversed_and_uninformative_order() -> None:
    agreeing = _transport_example((2.0, 2.0, 1.0))
    reversed_order = _transport_example((1.0, 1.0, 2.0))
    tied = _transport_example((1.0, 1.0, 1.0))
    assert agreeing.supported and reversed_order.supported and tied.supported
    assert agreeing.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0)
    assert reversed_order.features[n.FEATURE_NAMES[0]] == pytest.approx(0.0)
    assert tied.features[n.FEATURE_NAMES[0]] == pytest.approx(0.5)
    assert agreeing.informative_weight > 0.0
    assert agreeing.violation_weight == pytest.approx(0.0)
    assert reversed_order.violation_weight == pytest.approx(
        reversed_order.informative_weight
    )


def test_kernel_is_charge_length_edge_order_and_replication_invariant() -> None:
    charges = (1.0, 1.0, -0.5, -1.5)
    endpoints = ((0, 2), (0, 3), (1, 3))
    distances = (1.7, 2.2, 1.1)
    reference = n.apriori_length_order_protection(
        charges=charges, endpoints=endpoints, distances=distances
    )
    scaled = n.apriori_length_order_protection(
        charges=tuple(7.3 * value for value in charges),
        endpoints=endpoints,
        distances=tuple(4.2 * value for value in distances),
    )
    order = (2, 0, 1)
    reordered = n.apriori_length_order_protection(
        charges=charges,
        endpoints=tuple(endpoints[index] for index in order),
        distances=tuple(distances[index] for index in order),
    )
    replicated = n.apriori_length_order_protection(
        charges=charges + charges,
        endpoints=endpoints + tuple((left + 4, right + 4) for left, right in endpoints),
        distances=distances + distances,
    )
    for result in (reference, scaled, reordered, replicated):
        assert result.supported, result.failure_reason
        assert result.features[n.FEATURE_NAMES[0]] == pytest.approx(
            reference.features[n.FEATURE_NAMES[0]], abs=1e-10
        )


@pytest.mark.parametrize(
    ("charges", "endpoints", "distances"),
    (
        ((), (), ()),
        ((1.0, -1.0), (), ()),
        ((1.0, -0.5), ((0, 1),), (1.0,)),
        ((1.0, -1.0), ((1, 0),), (1.0,)),
        ((1.0, -1.0), ((0, 1),), (0.0,)),
        ((1.0, -1.0), ((0, 1),), (np.nan,)),
        ((1.0, 1.0, -1.0, -1.0), ((0, 2), (1, 2), (1, 3)), (1, 1, 1)),
    ),
)
def test_malformed_or_infeasible_kernel_inputs_fail_closed(
    charges: object, endpoints: object, distances: object
) -> None:
    result = n.apriori_length_order_protection(
        charges=charges, endpoints=endpoints, distances=distances
    )
    assert result.supported is False
    assert result.features == {}


def test_distorted_periodic_crystal_and_equivalents_are_supported() -> None:
    atoms = _distorted_nacl()
    result = n.compute_aprbs_features(atoms)
    assert result.supported, result.failure_reason
    assert result.site_count == len(atoms)
    assert result.edge_count >= len(atoms) // 2
    assert result.pair_count > 0
    assert 0.0 <= result.features[n.FEATURE_NAMES[0]] <= 1.0
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
        result = n.compute_aprbs_features(item)
        assert not result.supported
        assert "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_aprbs_row(atoms)
    assert tuple(name for name in row if name.startswith("aprbs_")) == (
        "aprbs_length_order_protection",
        "aprbs_supported",
        "aprbs_failure",
        "aprbs_site_count",
        "aprbs_edge_count",
        "aprbs_pair_count",
        "aprbs_informative_weight",
        "aprbs_violation_weight",
        "aprbs_maximum_marginal_residual",
        "aprbs_iterations",
        "aprbs_valence_policy",
    )
    assert row["aprbs_supported"] is True
