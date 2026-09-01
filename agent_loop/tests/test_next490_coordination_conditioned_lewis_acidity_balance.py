from __future__ import annotations

import math

import numpy as np
import pytest

import src.next460_characteristic_lewis_anion_matching as n460
import src.next490_coordination_conditioned_lewis_acidity_balance as n490


def test_appendix3_states_are_frozen_without_hydrogen() -> None:
    assert n490.ASSET_SHA256 == n460.ASSET_SHA256
    assert n490.CHARACTERISTIC_STATES_BY_ELEMENT["Fe"] == ((2, 5.68, 0.352), (3, 5.68, 0.528))
    assert n490.CHARACTERISTIC_STATES_BY_ELEMENT["Mn"][-3:] == (
        (5, 4.0, 1.25),
        (6, 4.0, 1.5),
        (7, 4.0, 1.75),
    )
    assert "H" not in n490.CHARACTERISTIC_STATES_BY_ELEMENT


def test_na_cl_like_sixfold_network_has_manual_balance() -> None:
    charges = (1.0,) * 6 + (-1.0,) * 6
    symbols = ("Na",) * 6 + ("Cl",) * 6
    endpoints = tuple((i, 6 + j) for i in range(6) for j in range(6))
    result = n490.coordination_conditioned_lewis_acidity_balance(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    received = 6.0 * 0.159
    expected = 1.0 - abs(received - 1.0) / (received + 1.0)
    assert result.supported and result.feasible
    assert result.cation_coordination == (6,) * 6
    assert result.nearest_characteristic_cn == pytest.approx((6.31,) * 6)
    assert result.received_lower == pytest.approx((received,) * 6)
    assert result.received_upper == pytest.approx((received,) * 6)
    assert result.features[n490.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)


def test_tied_nearest_states_form_an_interval_instead_of_a_choice() -> None:
    result = n490.coordination_conditioned_lewis_acidity_balance(
        charges=(4.0, -2.0, -2.0),
        symbols=("Mn", "O", "O"),
        endpoints=((0, 1), (0, 1), (0, 2), (0, 2)),
    )
    assert result.supported
    assert result.nearest_characteristic_cn == pytest.approx((4.0,))
    assert result.cation_acidity_lower == pytest.approx((1.25,))
    assert result.cation_acidity_upper == pytest.approx((1.75,))
    assert result.received_lower == pytest.approx((2.5, 2.5))
    assert result.received_upper == pytest.approx((3.5, 3.5))
    expected = 1.0 - 1.0 / 9.0
    assert result.features[n490.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)


def test_electronegativity_fallback_restores_reduced_formula_scale_only() -> None:
    base = n490._representation_invariant_charges(
        (1.0, -1.0), ("Au", "Yb"), "electronegativity_partition"
    )
    supercell = n490._representation_invariant_charges(
        (0.5, 0.5, -0.5, -0.5),
        ("Au", "Au", "Yb", "Yb"),
        "electronegativity_partition",
    )
    enumerated = n490._representation_invariant_charges(
        (1.0, 1.0, -1.0, -1.0),
        ("Na", "Na", "Cl", "Cl"),
        "oxidation_state_enumeration",
    )
    assert base == pytest.approx((1.0, -1.0))
    assert supercell == pytest.approx((1.0, 1.0, -1.0, -1.0))
    assert enumerated == pytest.approx((1.0, 1.0, -1.0, -1.0))


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    charge = (1.0, 1.0, -2.0)
    symbol = ("Na", "Na", "O")
    edges = ((0, 2),) * 6 + ((1, 2),) * 6
    base = n490.coordination_conditioned_lewis_acidity_balance(
        charges=charge, symbols=symbol, endpoints=edges
    )
    reordered = n490.coordination_conditioned_lewis_acidity_balance(
        charges=charge, symbols=symbol, endpoints=tuple(reversed(edges))
    )
    permuted = n490.coordination_conditioned_lewis_acidity_balance(
        charges=(-2.0, 1.0, 1.0),
        symbols=("O", "Na", "Na"),
        endpoints=((1, 0),) * 6 + ((2, 0),) * 6,
    )
    replicated = n490.coordination_conditioned_lewis_acidity_balance(
        charges=charge * 2,
        symbols=symbol * 2,
        endpoints=edges + tuple((i + 3, j + 3) for i, j in edges),
    )
    results = (base, reordered, permuted, replicated)
    assert all(item.supported for item in results)
    values = np.asarray([item.features[n490.FEATURE_NAMES[0]] for item in results])
    assert np.max(np.abs(values - values[0])) <= 1e-10


@pytest.mark.parametrize(
    ("charges", "symbols", "endpoints", "needle"),
    [
        ((1.0, -1.0), ("H", "Cl"), ((0, 1),), "lookup"),
        ((1.0, -1.0, 0.0), ("Na", "Cl", "He"), ((0, 1),), "charged"),
        ((1.0, -1.0), ("Na", "Cl"), (), "contact"),
        ((1.0, 1.0, -2.0), ("Na", "Na", "O"), ((0, 2),), "isolated"),
        ((1.0, -2.0), ("Na", "O"), ((0, 1),), "neutral"),
    ],
)
def test_unsupported_inputs_fail_closed(charges, symbols, endpoints, needle) -> None:
    result = n490.coordination_conditioned_lewis_acidity_balance(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    assert not result.supported
    assert needle in str(result.failure_reason).lower()


def test_boundary_is_identical_and_contains_no_dft() -> None:
    assert n490.BOUNDARY_FLAGS == n460.BOUNDARY_FLAGS
    assert all(value is False for value in n490.BOUNDARY_FLAGS.values())
    assert math.isfinite(n490.CHARACTERISTIC_STATES_BY_ELEMENT["Na"][0][2])
