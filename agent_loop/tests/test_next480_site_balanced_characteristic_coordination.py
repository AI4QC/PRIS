from __future__ import annotations

import numpy as np
import pytest

import src.next475_characteristic_coordination_bottleneck as n475
import src.next480_site_balanced_characteristic_coordination as n480


def test_manual_score_is_equal_weight_mean_of_local_compatibilities() -> None:
    result = n480.site_balanced_characteristic_coordination(
        charges=(0.4, 0.6, -1.0), symbols=("Na", "Mn", "O"),
        endpoints=((0, 2),) * 2 + ((1, 2),) * 4,
    )
    na = 1 - abs(2 - 6.31) / (2 + 6.31)
    expected = (na + 1.0) / 2
    assert result.supported and result.feasible
    assert result.site_compatibility == pytest.approx((na, 1.0))
    assert result.features[n480.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)


def test_unlisted_element_contributes_zero_one_site_vote() -> None:
    result = n480.site_balanced_characteristic_coordination(
        charges=(0.5, 0.5, -1.0), symbols=("Ac", "Mn", "O"),
        endpoints=((0, 2),) * 6 + ((1, 2),) * 4,
    )
    assert result.supported and result.missing_element_count == 1
    assert result.features[n480.FEATURE_NAMES[0]] == 0.5


def test_differs_analytically_from_pooled_eccc_and_bottleneck() -> None:
    kwargs = {
        "charges": (0.4, 0.6, -1.0),
        "symbols": ("Na", "Mn", "O"),
        "endpoints": ((0, 2),) * 2 + ((1, 2),) * 4,
    }
    mean = n480.site_balanced_characteristic_coordination(**kwargs)
    minimum = n475.characteristic_coordination_bottleneck(**kwargs)
    assert mean.supported and minimum.supported
    assert mean.features[n480.FEATURE_NAMES[0]] != minimum.features[n475.FEATURE_NAMES[0]]


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    charge = (0.4, 0.6, -1.0)
    symbol = ("Fe", "Ce", "O")
    edges = ((0, 2),) * 6 + ((1, 2),) * 9
    base = n480.site_balanced_characteristic_coordination(charges=charge, symbols=symbol, endpoints=edges)
    reordered = n480.site_balanced_characteristic_coordination(charges=charge, symbols=symbol, endpoints=tuple(reversed(edges)))
    permuted = n480.site_balanced_characteristic_coordination(
        charges=(-1.0, 0.6, 0.4), symbols=("O", "Ce", "Fe"), endpoints=((2, 0),) * 6 + ((1, 0),) * 9
    )
    replicated = n480.site_balanced_characteristic_coordination(
        charges=charge * 2, symbols=symbol * 2, endpoints=edges + tuple((i + 3, j + 3) for i, j in edges)
    )
    results = (base, reordered, permuted, replicated)
    assert all(item.supported for item in results)
    values = np.asarray([item.features[n480.FEATURE_NAMES[0]] for item in results])
    assert np.max(np.abs(values - values[0])) <= 1e-10


def test_boundary_and_asset_are_inherited_without_dft() -> None:
    assert n480.ASSET_SHA256 == n475.ASSET_SHA256
    assert n480.BOUNDARY_FLAGS == n475.BOUNDARY_FLAGS
    assert all(value is False for value in n480.BOUNDARY_FLAGS.values())
