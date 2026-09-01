from __future__ import annotations

import math

import numpy as np
import pytest

import src.next460_characteristic_lewis_anion_matching as n460
import src.next465_mixed_valence_characteristic_lewis_matching as n465


def test_exact_state_preserves_printed_acidity() -> None:
    assert n465.mixed_valence_characteristic_acidity("Fe", 2.0) == 0.352
    assert n465.mixed_valence_characteristic_acidity("Cl", 3.0) == 1.300


def test_fractional_state_uses_unique_adjacent_state_lever_rule() -> None:
    value = n465.mixed_valence_characteristic_acidity("Fe", 8.0 / 3.0)
    expected = (1.0 / 3.0) * 0.352 + (2.0 / 3.0) * 0.528
    assert value == pytest.approx(expected, abs=1e-14)
    assert n465.mixed_valence_characteristic_acidity("Ce", 3.5) == pytest.approx(
        0.5 * (0.320 + 0.450), abs=1e-14
    )


@pytest.mark.parametrize(("element", "charge"), [("Fe", 1.5), ("Fe", 4.0), ("H", 1.0), ("Xe", 2.0)])
def test_missing_or_unbracketed_state_has_no_fallback(element: str, charge: float) -> None:
    with pytest.raises(ValueError, match="lookup|bracket"):
        n465.mixed_valence_characteristic_acidity(element, charge)


def test_fe3o4_average_valence_kernel_matches_manual_formula() -> None:
    charges = (8 / 3, 8 / 3, 8 / 3, -2, -2, -2, -2)
    symbols = ("Fe", "Fe", "Fe", "O", "O", "O", "O")
    endpoints = ((0, 3), (1, 4), (2, 5), (0, 6), (1, 3), (2, 4))
    result = n465.mixed_valence_characteristic_lewis_matching(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    acidity = (0.352 + 2 * 0.528) / 3
    received = (2 * acidity, 2 * acidity, acidity, acidity)
    expected = 1 - sum(abs(value - 2) for value in received) / sum(
        value + 2 for value in received
    )
    assert result.supported and result.feasible
    assert result.cation_acidity == pytest.approx((acidity,) * 3)
    assert result.received_acidity == pytest.approx(received)
    assert result.features[n465.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    charge = (2.5, 2.5, -1.0, -2.0, -2.0)
    symbol = ("Mn", "Mn", "Cl", "O", "O")
    edges = ((0, 2), (1, 2), (0, 3), (1, 4))
    base = n465.mixed_valence_characteristic_lewis_matching(
        charges=charge, symbols=symbol, endpoints=edges
    )
    reordered = n465.mixed_valence_characteristic_lewis_matching(
        charges=charge, symbols=symbol, endpoints=tuple(reversed(edges))
    )
    permuted = n465.mixed_valence_characteristic_lewis_matching(
        charges=(-2.0, 2.5, -1.0, 2.5, -2.0),
        symbols=("O", "Mn", "Cl", "Mn", "O"),
        endpoints=((1, 2), (3, 2), (1, 0), (3, 4)),
    )
    replicated = n465.mixed_valence_characteristic_lewis_matching(
        charges=charge * 2,
        symbols=symbol * 2,
        endpoints=edges + tuple((i + 5, j + 5) for i, j in edges),
    )
    results = (base, reordered, permuted, replicated)
    assert all(item.supported for item in results)
    values = np.asarray([item.features[n465.FEATURE_NAMES[0]] for item in results])
    assert np.max(np.abs(values - values[0])) <= 1e-10


def test_unbracketed_kernel_fails_closed() -> None:
    result = n465.mixed_valence_characteristic_lewis_matching(
        charges=(1.5, 1.5, -1.0, -2.0),
        symbols=("Fe", "Fe", "Cl", "O"),
        endpoints=((0, 2), (1, 3)),
    )
    assert not result.supported
    assert "bracket" in str(result.failure_reason).lower()


def test_electronegativity_partition_is_rejected_before_kernel(monkeypatch) -> None:
    from ase import Atoms

    class Assignment:
        supported = True
        values = (0.5, -0.5)
        policy = "electronegativity_partition"

    monkeypatch.setattr(n465.n19, "infer_valence_assignment", lambda _: Assignment())
    atoms = Atoms("NaCl", scaled_positions=((0, 0, 0), (0.5, 0.5, 0.5)), cell=np.eye(3) * 4, pbc=True)
    result = n465.compute_mvclam_features(atoms)
    assert not result.supported
    assert "electronegativity" in str(result.failure_reason).lower()


def test_boundary_and_asset_identity_are_inherited_without_dft() -> None:
    assert n465.ASSET_SHA256 == n460.ASSET_SHA256
    assert n465.BOUNDARY_FLAGS == n460.BOUNDARY_FLAGS
    assert all(value is False for value in n465.BOUNDARY_FLAGS.values())
    assert math.isfinite(n465.mixed_valence_characteristic_acidity("Mn", 2.5))
