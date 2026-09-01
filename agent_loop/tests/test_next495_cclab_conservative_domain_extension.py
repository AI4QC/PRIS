from __future__ import annotations

import math

import numpy as np
import pytest

import src.next490_coordination_conditioned_lewis_acidity_balance as n490
import src.next495_cclab_conservative_domain_extension as n495


def test_shared_domain_is_exactly_next490() -> None:
    kwargs = {
        "charges": (1.0, 1.0, -2.0),
        "symbols": ("Na", "Na", "O"),
        "endpoints": ((0, 2),) * 6 + ((1, 2),) * 6,
    }
    base = n490.coordination_conditioned_lewis_acidity_balance(**kwargs)
    extended = n495.cclab_conservative_domain_extension(**kwargs)
    assert base.supported and extended.supported
    assert extended.unknown_cation_count == 0
    assert extended.unknown_anion_neighborhood_count == 0
    assert extended.features[n495.FEATURE_NAMES[0]] == base.features[n490.FEATURE_NAMES[0]]


def test_unknown_donor_makes_only_its_anion_maximally_mismatched() -> None:
    result = n495.cclab_conservative_domain_extension(
        charges=(1.0, 1.0, -1.0, -1.0),
        symbols=("Na", "Ac", "O", "Cl"),
        endpoints=((0, 2),) * 6 + ((1, 3),) * 6,
    )
    received = 6.0 * 0.159
    expected = 1.0 - (abs(received - 1.0) + 1.0) / (received + 1.0 + 1.0)
    assert result.supported and result.feasible
    assert result.unknown_cation_count == 1
    assert result.unknown_anion_neighborhood_count == 1
    assert result.projected_received == pytest.approx((received, 0.0))
    assert result.features[n495.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)


def test_all_unknown_donors_produce_physical_zero() -> None:
    result = n495.cclab_conservative_domain_extension(
        charges=(1.0, -1.0), symbols=("Ac", "O"), endpoints=((0, 1),) * 8
    )
    assert result.supported
    assert result.features[n495.FEATURE_NAMES[0]] == 0.0


def test_conservative_extension_never_exceeds_ignore_unknown_diagnostic() -> None:
    kwargs = {
        "charges": (1.0, 1.0, -2.0),
        "symbols": ("Na", "Ac", "O"),
        "endpoints": ((0, 2),) * 6 + ((1, 2),) * 6,
    }
    conservative = n495.cclab_conservative_domain_extension(**kwargs)
    optimistic = n495.cclab_ignore_unknown_contacts_diagnostic(**kwargs)
    assert conservative.supported and optimistic.supported
    assert conservative.features[n495.FEATURE_NAMES[0]] == 0.0
    assert optimistic.features[n495.FEATURE_NAMES[0]] > 0.0
    assert (
        conservative.features[n495.FEATURE_NAMES[0]]
        <= optimistic.features[n495.FEATURE_NAMES[0]]
    )


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    charge = (1.0, 1.0, -1.0, -1.0)
    symbol = ("Na", "Ac", "O", "Cl")
    edges = ((0, 2),) * 6 + ((1, 3),) * 6
    base = n495.cclab_conservative_domain_extension(
        charges=charge, symbols=symbol, endpoints=edges
    )
    reordered = n495.cclab_conservative_domain_extension(
        charges=charge, symbols=symbol, endpoints=tuple(reversed(edges))
    )
    permuted = n495.cclab_conservative_domain_extension(
        charges=(-1.0, -1.0, 1.0, 1.0),
        symbols=("Cl", "O", "Ac", "Na"),
        endpoints=((3, 1),) * 6 + ((2, 0),) * 6,
    )
    replicated = n495.cclab_conservative_domain_extension(
        charges=charge * 2,
        symbols=symbol * 2,
        endpoints=edges + tuple((i + 4, j + 4) for i, j in edges),
    )
    results = (base, reordered, permuted, replicated)
    assert all(item.supported for item in results)
    values = np.asarray([item.features[n495.FEATURE_NAMES[0]] for item in results])
    assert np.max(np.abs(values - values[0])) <= 1e-10


@pytest.mark.parametrize(
    ("charges", "symbols", "endpoints", "needle"),
    [
        ((1.0, -1.0, 0.0), ("Na", "Cl", "He"), ((0, 1),), "charged"),
        ((1.0, -1.0), ("Na", "Cl"), (), "contact"),
        ((1.0, 1.0, -2.0), ("Na", "Na", "O"), ((0, 2),), "isolated"),
        ((1.0, -2.0), ("Na", "O"), ((0, 1),), "neutral"),
    ],
)
def test_malformed_inputs_still_fail_closed(charges, symbols, endpoints, needle) -> None:
    result = n495.cclab_conservative_domain_extension(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    assert not result.supported
    assert needle in str(result.failure_reason).lower()


def test_boundary_is_unchanged_and_contains_no_dft() -> None:
    assert n495.BOUNDARY_FLAGS == n490.BOUNDARY_FLAGS
    assert all(value is False for value in n495.BOUNDARY_FLAGS.values())
    assert math.isfinite(n495.CHARACTERISTIC_STATES_BY_ELEMENT["Na"][0][2])
    assert callable(n495.compute_cclab_ignore_unknown_diagnostic)
