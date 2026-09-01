from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

import src.next460_characteristic_lewis_anion_matching as n460


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "data" / "characteristic_lewis_acidity_hawthorne_2026.csv"


def test_frozen_appendix3_asset_has_134_unique_non_hydrogen_keys() -> None:
    with ASSET.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row["element"], int(row["oxidation"])) for row in rows}
    assert len(rows) == len(keys) == 134
    assert not any(element == "H" for element, _ in keys)
    assert {row["source_doi"] for row in rows} == {"10.1180/mgm.2026.10215"}
    assert {row["license"] for row in rows} == {"CC-BY-4.0"}
    for row in rows:
        oxidation = int(row["oxidation"])
        cn = float(row["characteristic_cn"])
        acidity = float(row["acidity_e"])
        assert oxidation > 0 and math.isfinite(cn) and cn > 0.0
        assert math.isfinite(acidity) and acidity > 0.0
        assert abs(oxidation / cn - acidity) <= 0.06


def test_lookup_is_frozen_and_preserves_printed_values() -> None:
    assert len(n460.CHARACTERISTIC_LEWIS_ACIDITY) == 134
    assert n460.CHARACTERISTIC_LEWIS_ACIDITY[("Na", 1)] == 0.159
    assert n460.CHARACTERISTIC_LEWIS_ACIDITY[("Cl", 3)] == 1.300
    assert ("H", 1) not in n460.CHARACTERISTIC_LEWIS_ACIDITY
    assert len(n460.ASSET_SHA256) == 64


def test_single_contact_matches_manual_normalized_mismatch() -> None:
    result = n460.characteristic_lewis_anion_matching(
        charges=(1.0, -1.0), symbols=("Na", "Cl"), endpoints=((0, 1),)
    )
    expected = 1.0 - abs(0.159 - 1.0) / (0.159 + 1.0)
    assert result.supported and result.feasible
    assert result.features[n460.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)
    assert result.received_acidity == pytest.approx((0.159,))
    assert result.anion_demand == pytest.approx((1.0,))


def test_contact_multiplicity_changes_received_acidity() -> None:
    one = n460.characteristic_lewis_anion_matching(
        charges=(1.0, -1.0), symbols=("Na", "Cl"), endpoints=((0, 1),)
    )
    six = n460.characteristic_lewis_anion_matching(
        charges=(1.0, -1.0),
        symbols=("Na", "Cl"),
        endpoints=((0, 1),) * 6,
    )
    assert one.supported and six.supported
    assert six.received_acidity == pytest.approx((6 * 0.159,))
    assert six.features[n460.FEATURE_NAMES[0]] > one.features[n460.FEATURE_NAMES[0]]


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    base = n460.characteristic_lewis_anion_matching(
        charges=(2.0, 1.0, -1.0, -2.0),
        symbols=("Mg", "Na", "Cl", "O"),
        endpoints=((0, 2), (1, 2), (0, 3), (0, 3)),
    )
    reordered = n460.characteristic_lewis_anion_matching(
        charges=(2.0, 1.0, -1.0, -2.0),
        symbols=("Mg", "Na", "Cl", "O"),
        endpoints=((0, 3), (0, 2), (0, 3), (1, 2)),
    )
    permuted = n460.characteristic_lewis_anion_matching(
        charges=(-2.0, 2.0, -1.0, 1.0),
        symbols=("O", "Mg", "Cl", "Na"),
        endpoints=((1, 2), (3, 2), (1, 0), (1, 0)),
    )
    replicated = n460.characteristic_lewis_anion_matching(
        charges=(2.0, 1.0, -1.0, -2.0) * 2,
        symbols=("Mg", "Na", "Cl", "O") * 2,
        endpoints=(
            (0, 2), (1, 2), (0, 3), (0, 3),
            (4, 6), (5, 6), (4, 7), (4, 7),
        ),
    )
    values = [
        item.features[n460.FEATURE_NAMES[0]]
        for item in (base, reordered, permuted, replicated)
    ]
    assert all(item.supported for item in (base, reordered, permuted, replicated))
    assert np.max(np.abs(np.asarray(values) - values[0])) <= 1e-10


@pytest.mark.parametrize(
    ("charges", "symbols", "endpoints", "needle"),
    [
        ((1.0, -1.0), ("H", "Cl"), ((0, 1),), "lookup"),
        ((1.5, -1.5), ("Na", "Cl"), ((0, 1),), "integer"),
        ((1.0, -1.0), ("Na", "Cl"), (), "contact"),
        ((1.0, -1.0, 0.0), ("Na", "Cl", "He"), ((0, 1),), "charged"),
    ],
)
def test_unsupported_inputs_fail_closed(charges, symbols, endpoints, needle) -> None:
    result = n460.characteristic_lewis_anion_matching(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    assert not result.supported
    assert needle in str(result.failure_reason).lower()


def test_no_opposite_sign_periodic_graph_is_supported_zero(monkeypatch) -> None:
    from ase import Atoms

    class Assignment:
        supported = True
        values = (1.0, -1.0)
        policy = "test"

    class Geometry:
        supported = False
        failure_reason = "no opposite-sign periodic neighbor"

    monkeypatch.setattr(n460.n19, "infer_valence_assignment", lambda _: Assignment())
    monkeypatch.setattr(n460.n19, "build_periodic_edge_geometry", lambda *a, **k: Geometry())
    atoms = Atoms("NaCl", scaled_positions=((0, 0, 0), (0.5, 0.5, 0.5)), cell=np.eye(3) * 4, pbc=True)
    result = n460.compute_clam_features(atoms)
    assert result.supported and not result.feasible
    assert result.features[n460.FEATURE_NAMES[0]] == 0.0


def test_boundary_flags_forbid_dft_and_post_initial_information() -> None:
    assert n460.BOUNDARY_FLAGS == {
        "dft_calculation_executed": False,
        "dft_values_used": False,
        "learned_energy_force_stress_proxy_used": False,
        "model_or_proxy_potential_used": False,
        "physical_relaxation_executed": False,
        "trajectory_or_later_geometry_used": False,
        "same_composition_alternative_used": False,
    }
