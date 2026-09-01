from __future__ import annotations

import numpy as np
import pytest

import src.next470_element_characteristic_coordination_compatibility as n470
import src.next475_characteristic_coordination_bottleneck as n475


def test_table_policy_reuses_eccc_and_adds_only_unambiguous_h_cn() -> None:
    assert n475.ASSET_SHA256 == n470.ASSET_SHA256
    assert n475.CHARACTERISTIC_CN_BY_ELEMENT["Fe"] == (5.68,)
    assert n475.CHARACTERISTIC_CN_BY_ELEMENT["H"] == (2.03,)


def test_manual_bottleneck_is_worst_local_compatibility() -> None:
    result = n475.characteristic_coordination_bottleneck(
        charges=(0.4, 0.6, -1.0),
        symbols=("Na", "Mn", "O"),
        endpoints=((0, 2),) * 2 + ((1, 2),) * 4,
    )
    na = 1 - abs(2 - 6.31) / (2 + 6.31)
    mn = 1.0
    assert result.supported and result.feasible
    assert result.site_compatibility == pytest.approx((na, mn))
    assert result.features[n475.FEATURE_NAMES[0]] == pytest.approx(na, abs=5e-11)


def test_unlisted_element_is_conservative_zero_not_unsupported() -> None:
    result = n475.characteristic_coordination_bottleneck(
        charges=(1.0, -1.0), symbols=("Ac", "O"), endpoints=((0, 1),) * 6
    )
    assert result.supported and result.feasible
    assert result.missing_element_count == 1
    assert result.features[n475.FEATURE_NAMES[0]] == 0.0


def test_hydrogen_uses_cn_but_no_ambiguous_acidity() -> None:
    result = n475.characteristic_coordination_bottleneck(
        charges=(1.0, -1.0), symbols=("H", "O"), endpoints=((0, 1),) * 2
    )
    expected = 1 - 0.03 / 4.03
    assert result.supported
    assert result.nearest_characteristic_cn == pytest.approx((2.03,))
    assert result.features[n475.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)


def test_charge_magnitude_does_not_change_sign_only_formula() -> None:
    integer = n475.characteristic_coordination_bottleneck(
        charges=(2.0, -2.0), symbols=("Fe", "O"), endpoints=((0, 1),) * 6
    )
    fractional = n475.characteristic_coordination_bottleneck(
        charges=(0.37, -0.37), symbols=("Fe", "O"), endpoints=((0, 1),) * 6
    )
    assert integer.supported and fractional.supported
    assert integer.features == fractional.features


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    charge = (0.4, 0.6, -1.0)
    symbol = ("Fe", "Ce", "O")
    edges = ((0, 2),) * 6 + ((1, 2),) * 9
    base = n475.characteristic_coordination_bottleneck(charges=charge, symbols=symbol, endpoints=edges)
    reordered = n475.characteristic_coordination_bottleneck(charges=charge, symbols=symbol, endpoints=tuple(reversed(edges)))
    permuted = n475.characteristic_coordination_bottleneck(
        charges=(-1.0, 0.6, 0.4), symbols=("O", "Ce", "Fe"), endpoints=((2, 0),) * 6 + ((1, 0),) * 9
    )
    replicated = n475.characteristic_coordination_bottleneck(
        charges=charge * 2, symbols=symbol * 2, endpoints=edges + tuple((i + 3, j + 3) for i, j in edges)
    )
    results = (base, reordered, permuted, replicated)
    assert all(item.supported for item in results)
    values = np.asarray([item.features[n475.FEATURE_NAMES[0]] for item in results])
    assert np.max(np.abs(values - values[0])) <= 1e-10


@pytest.mark.parametrize(
    ("charges", "symbols", "endpoints", "needle"),
    [
        ((1.0, -1.0, 0.0), ("Na", "Cl", "He"), ((0, 1),), "charged"),
        ((1.0, -1.0), ("Na", "Cl"), (), "contact"),
        ((1.0, 1.0, -2.0), ("Na", "Na", "O"), ((0, 2),), "isolated"),
    ],
)
def test_unsupported_inputs_fail_closed(charges, symbols, endpoints, needle) -> None:
    result = n475.characteristic_coordination_bottleneck(charges=charges, symbols=symbols, endpoints=endpoints)
    assert not result.supported
    assert needle in str(result.failure_reason).lower()


def test_boundary_is_identical_and_no_dft() -> None:
    assert n475.BOUNDARY_FLAGS == n470.BOUNDARY_FLAGS
    assert all(value is False for value in n475.BOUNDARY_FLAGS.values())
