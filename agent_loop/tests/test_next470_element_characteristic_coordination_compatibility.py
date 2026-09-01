from __future__ import annotations

import math

import numpy as np
import pytest

import src.next460_characteristic_lewis_anion_matching as n460
import src.next470_element_characteristic_coordination_compatibility as n470


def test_characteristic_cn_sets_are_frozen_from_same_asset() -> None:
    assert n470.ASSET_SHA256 == n460.ASSET_SHA256
    assert n470.CHARACTERISTIC_CN_BY_ELEMENT["Fe"] == (5.68,)
    assert n470.CHARACTERISTIC_CN_BY_ELEMENT["Mn"] == (4.0, 5.85, 5.9, 5.98)
    assert n470.CHARACTERISTIC_CN_BY_ELEMENT["Ce"] == (8.8, 9.4)
    assert "H" not in n470.CHARACTERISTIC_CN_BY_ELEMENT


def test_manual_nearest_characteristic_cn_score() -> None:
    result = n470.element_characteristic_coordination_compatibility(
        charges=(1.0, 1.0, -1.0, -1.0),
        symbols=("Na", "Mn", "Cl", "O"),
        endpoints=((0, 2), (0, 3), (1, 2), (1, 3), (1, 3), (1, 2)),
    )
    # CN(Na)=2 nearest 6.31; CN(Mn)=4 exactly matches a printed state.
    expected = 1 - 4.31 / (2 + 6.31 + 4 + 4)
    assert result.supported and result.feasible
    assert result.coordination == (2, 4)
    assert result.nearest_characteristic_cn == pytest.approx((6.31, 4.0))
    assert result.features[n470.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)


def test_charge_magnitude_does_not_change_sign_only_formula() -> None:
    integer = n470.element_characteristic_coordination_compatibility(
        charges=(2.0, -2.0), symbols=("Fe", "O"), endpoints=((0, 1),) * 6
    )
    fractional = n470.element_characteristic_coordination_compatibility(
        charges=(0.37, -0.37), symbols=("Fe", "O"), endpoints=((0, 1),) * 6
    )
    assert integer.supported and fractional.supported
    assert integer.features == fractional.features


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    charge = (0.4, 0.6, -1.0)
    symbol = ("Fe", "Ce", "O")
    edges = ((0, 2),) * 6 + ((1, 2),) * 9
    base = n470.element_characteristic_coordination_compatibility(
        charges=charge, symbols=symbol, endpoints=edges
    )
    reordered = n470.element_characteristic_coordination_compatibility(
        charges=charge, symbols=symbol, endpoints=tuple(reversed(edges))
    )
    permuted = n470.element_characteristic_coordination_compatibility(
        charges=(-1.0, 0.6, 0.4),
        symbols=("O", "Ce", "Fe"),
        endpoints=((2, 0),) * 6 + ((1, 0),) * 9,
    )
    replicated = n470.element_characteristic_coordination_compatibility(
        charges=charge * 2,
        symbols=symbol * 2,
        endpoints=edges + tuple((i + 3, j + 3) for i, j in edges),
    )
    results = (base, reordered, permuted, replicated)
    assert all(item.supported for item in results)
    values = np.asarray([item.features[n470.FEATURE_NAMES[0]] for item in results])
    assert np.max(np.abs(values - values[0])) <= 1e-10


@pytest.mark.parametrize(
    ("charges", "symbols", "endpoints", "needle"),
    [
        ((1.0, -1.0), ("H", "Cl"), ((0, 1),), "lookup"),
        ((1.0, -1.0, 0.0), ("Na", "Cl", "He"), ((0, 1),), "charged"),
        ((1.0, -1.0), ("Na", "Cl"), (), "contact"),
        ((1.0, 1.0, -2.0), ("Na", "Na", "O"), ((0, 2),), "isolated"),
    ],
)
def test_unsupported_inputs_fail_closed(charges, symbols, endpoints, needle) -> None:
    result = n470.element_characteristic_coordination_compatibility(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    assert not result.supported
    assert needle in str(result.failure_reason).lower()


def test_boundary_is_identical_and_contains_no_dft() -> None:
    assert n470.BOUNDARY_FLAGS == n460.BOUNDARY_FLAGS
    assert all(value is False for value in n470.BOUNDARY_FLAGS.values())
    assert math.isfinite(n470.CHARACTERISTIC_CN_BY_ELEMENT["Na"][0])
