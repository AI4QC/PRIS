from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

import src.next430_endpoint_strength_length_order as n


def _example(distances: tuple[float, float, float]):
    return n.endpoint_strength_length_order_protection(
        charges=(3.0, 1.0, -1.0, -3.0),
        endpoints=((0, 2), (0, 3), (1, 3)),
        distances=distances,
    )


def _distorted_spinel() -> Atoms:
    atoms = Atoms(
        "MgAl2O4",
        scaled_positions=[
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25),
            (0.18, 0.18, 0.18), (0.82, 0.82, 0.82), (0.32, 0.68, 0.68),
            (0.68, 0.32, 0.32),
        ],
        cell=[[8.1, 0.0, 0.0], [0.2, 8.0, 0.0], [0.1, 0.3, 8.2]],
        pbc=True,
    )
    return atoms


def _feature(atoms: Atoms) -> float:
    result = n.compute_ecslo_features(atoms)
    assert result.supported, result.failure_reason
    return float(result.features[n.FEATURE_NAMES[0]])


def test_frozen_protocol_schema_direction_and_boundary_are_exact() -> None:
    assert n.PROTOCOL == "2026-08-13-next430-endpoint-strength-length-order-v1"
    assert len(n.DESIGN_SHA256) == 64
    assert n.FEATURE_NAMES == ("ecslo_endpoint_strength_length_order_protection",)
    assert n.FEATURE_DIRECTIONS == {n.FEATURE_NAMES[0]: "protected_high"}
    assert all(value is False for value in n.BOUNDARY_FLAGS.values())


def test_endpoint_strengths_and_order_examples_are_exact() -> None:
    agreeing = _example((2.0, 1.0, 2.0))
    reversed_order = _example((1.0, 2.0, 1.0))
    tied = _example((1.0, 1.0, 1.0))
    expected = (math_sqrt(1.5), 1.5, math_sqrt(1.5))
    assert agreeing.supported and reversed_order.supported and tied.supported
    assert agreeing.edge_strengths == pytest.approx(expected)
    assert agreeing.features[n.FEATURE_NAMES[0]] == pytest.approx(1.0)
    assert reversed_order.features[n.FEATURE_NAMES[0]] == pytest.approx(0.0)
    assert tied.features[n.FEATURE_NAMES[0]] == pytest.approx(0.5)


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def test_kernel_is_charge_length_edge_order_and_replication_invariant() -> None:
    charges = (3.0, 1.0, -1.0, -3.0)
    endpoints = ((0, 2), (0, 3), (1, 3))
    distances = (1.7, 2.2, 1.1)
    reference = n.endpoint_strength_length_order_protection(
        charges=charges, endpoints=endpoints, distances=distances
    )
    scaled = n.endpoint_strength_length_order_protection(
        charges=tuple(7.3 * value for value in charges),
        endpoints=endpoints,
        distances=tuple(4.2 * value for value in distances),
    )
    order = (2, 0, 1)
    reordered = n.endpoint_strength_length_order_protection(
        charges=charges,
        endpoints=tuple(endpoints[index] for index in order),
        distances=tuple(distances[index] for index in order),
    )
    replicated = n.endpoint_strength_length_order_protection(
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
        ((1.0, 1.0, -1.0, -1.0), ((0, 2), (1, 2)), (1.0, 1.0)),
    ),
)
def test_malformed_kernel_inputs_fail_closed(
    charges: object, endpoints: object, distances: object
) -> None:
    result = n.endpoint_strength_length_order_protection(
        charges=charges, endpoints=endpoints, distances=distances
    )
    assert result.supported is False
    assert result.features == {}


def test_periodic_crystal_and_equivalents_are_supported() -> None:
    atoms = _distorted_spinel()
    result = n.compute_ecslo_features(atoms)
    assert result.supported, result.failure_reason
    assert result.site_count == len(atoms)
    assert result.edge_count >= 1
    assert result.pair_count > 0
    reference = _feature(atoms)
    rotated = atoms.copy(); rotated.rotate(31.0, "z", rotate_cell=True)
    translated = atoms.copy(); translated.translate([0.173, 0.291, 0.419]); translated.wrap()
    permuted = atoms[[3, 0, 6, 1, 5, 4, 2]]
    rebased = atoms.copy(); rebased.set_cell(np.asarray([[1,1,0],[0,1,0],[0,0,1]]) @ atoms.cell.array, scale_atoms=False); rebased.wrap()
    for equivalent in (rotated, translated, permuted, rebased, atoms.repeat((2,1,1))):
        assert _feature(equivalent) == pytest.approx(reference, abs=1e-8)


def test_geometry_boundary_and_row_schema_are_exact() -> None:
    atoms = _distorted_spinel()
    changed = []
    item = atoms.copy(); item.calc = Calculator(); changed.append(item)
    item = atoms.copy(); item.info["outcome"] = 1; changed.append(item)
    item = atoms.copy(); item.new_array("energy", np.zeros(len(item))); changed.append(item)
    item = atoms.copy(); item.pbc = False; changed.append(item)
    for item in changed:
        result = n.compute_ecslo_features(item)
        assert not result.supported
        assert "geometry-only Atoms" in str(result.failure_reason)
    row = n.compute_ecslo_row(atoms)
    assert tuple(name for name in row if name.startswith("ecslo_")) == (
        "ecslo_endpoint_strength_length_order_protection",
        "ecslo_supported", "ecslo_failure", "ecslo_site_count",
        "ecslo_edge_count", "ecslo_pair_count", "ecslo_informative_weight",
        "ecslo_violation_weight", "ecslo_valence_policy",
    )
    assert row["ecslo_supported"] is True
