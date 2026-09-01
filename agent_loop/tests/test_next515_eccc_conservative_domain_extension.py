from __future__ import annotations

import math

import numpy as np
import pytest

import src.next470_element_characteristic_coordination_compatibility as n470
import src.next515_eccc_conservative_domain_extension as n515


def test_shared_domain_is_exactly_next470() -> None:
    kwargs = {
        "charges": (1.0, 1.0, -2.0),
        "symbols": ("Na", "Na", "O"),
        "endpoints": ((0, 2),) * 6 + ((1, 2),) * 6,
    }
    base = n470.element_characteristic_coordination_compatibility(**kwargs)
    extended = n515.eccc_conservative_domain_extension(**kwargs)
    assert base.supported and extended.supported
    assert extended.unknown_cation_count == 0
    assert extended.features[n515.FEATURE_NAMES[0]] == base.features[n470.FEATURE_NAMES[0]]


def test_any_unknown_cation_produces_physical_zero() -> None:
    result = n515.eccc_conservative_domain_extension(
        charges=(1.0, 1.0, -1.0, -1.0),
        symbols=("Na", "Ac", "O", "Cl"),
        endpoints=((0, 2),) * 6 + ((1, 3),) * 8,
    )
    assert result.supported and not result.feasible
    assert result.unknown_cation_count == 1
    assert result.features[n515.FEATURE_NAMES[0]] == 0.0


def test_conservative_extension_never_exceeds_ignore_unknown_diagnostic() -> None:
    kwargs = {
        "charges": (1.0, 1.0, -2.0),
        "symbols": ("Na", "Ac", "O"),
        "endpoints": ((0, 2),) * 6 + ((1, 2),) * 8,
    }
    conservative = n515.eccc_conservative_domain_extension(**kwargs)
    optimistic = n515.eccc_ignore_unknown_cations_diagnostic(**kwargs)
    assert conservative.supported and optimistic.supported
    assert conservative.features[n515.FEATURE_NAMES[0]] == 0.0
    assert optimistic.features[n515.FEATURE_NAMES[0]] > 0.0
    assert (
        conservative.features[n515.FEATURE_NAMES[0]]
        <= optimistic.features[n515.FEATURE_NAMES[0]]
    )


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    charge = (1.0, 1.0, -1.0, -1.0)
    symbol = ("Na", "Ac", "O", "Cl")
    edges = ((0, 2),) * 6 + ((1, 3),) * 8
    base = n515.eccc_conservative_domain_extension(
        charges=charge, symbols=symbol, endpoints=edges
    )
    reordered = n515.eccc_conservative_domain_extension(
        charges=charge, symbols=symbol, endpoints=tuple(reversed(edges))
    )
    permuted = n515.eccc_conservative_domain_extension(
        charges=(-1.0, -1.0, 1.0, 1.0),
        symbols=("Cl", "O", "Ac", "Na"),
        endpoints=((3, 1),) * 6 + ((2, 0),) * 8,
    )
    replicated = n515.eccc_conservative_domain_extension(
        charges=charge * 2,
        symbols=symbol * 2,
        endpoints=edges + tuple((i + 4, j + 4) for i, j in edges),
    )
    results = (base, reordered, permuted, replicated)
    assert all(item.supported for item in results)
    values = np.asarray([item.features[n515.FEATURE_NAMES[0]] for item in results])
    assert np.max(np.abs(values - values[0])) <= 1e-10


def test_isolated_or_empty_contact_domain_is_supported_zero() -> None:
    isolated = n515.eccc_conservative_domain_extension(
        charges=(1.0, 1.0, -2.0),
        symbols=("Na", "Na", "O"),
        endpoints=((0, 2),) * 6,
    )
    empty = n515.eccc_conservative_domain_extension(
        charges=(1.0, -1.0), symbols=("Na", "Cl"), endpoints=()
    )
    assert isolated.supported and empty.supported
    assert isolated.features[n515.FEATURE_NAMES[0]] == 0.0
    assert empty.features[n515.FEATURE_NAMES[0]] == 0.0


@pytest.mark.parametrize(
    ("charges", "symbols", "endpoints", "needle"),
    [
        ((1.0, -1.0, 0.0), ("Na", "Cl", "He"), ((0, 1),), "charged"),
        ((1.0, -2.0), ("Na", "O"), ((0, 1),), "neutral"),
        ((1.0, -1.0), ("Na", "Cl"), ((0, 2),), "index"),
    ],
)
def test_malformed_inputs_still_fail_closed(charges, symbols, endpoints, needle) -> None:
    result = n515.eccc_conservative_domain_extension(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    assert not result.supported
    assert needle in str(result.failure_reason).lower()


def test_boundary_is_unchanged_and_contains_no_dft() -> None:
    assert n515.BOUNDARY_FLAGS == n470.BOUNDARY_FLAGS
    assert all(value is False for value in n515.BOUNDARY_FLAGS.values())
    assert math.isfinite(n515.CHARACTERISTIC_CN_BY_ELEMENT["Na"][0])
    assert callable(n515.compute_eccc_ignore_unknown_diagnostic)
