from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

import src.next485_complex_anion_contact_correspondence as n485


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "data" / "complex_anion_lewis_basicity_hawthorne_2026.csv"


def test_appendix4_asset_is_exact_attributed_non_hydroxylated_subset() -> None:
    with ASSET.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 11
    assert {row["group"] for row in rows} == {
        "BO3", "BO4", "SiO4", "AlO4", "PO4", "AsO3", "AsO4", "VO4",
        "CO3", "NO3", "SO4",
    }
    assert {row["source_doi"] for row in rows} == {"10.1180/mgm.2026.10215"}
    assert {row["license"] for row in rows} == {"CC-BY-4.0"}
    assert all(float(row["lewis_basicity_e"]) > 0 for row in rows)
    assert len(n485.APPENDIX4_GROUPS) == 11


def _bo3_exact_external_nine():
    # B centre, three O ligands, nine external Na cations.
    charges = (1.0,) + (-1.0,) * 3 + (1.0,) * 9
    symbols = ("B",) + ("O",) * 3 + ("Na",) * 9
    center = ((0, 1), (0, 2), (0, 3))
    external = tuple((4 + index, 1 + index // 3) for index in range(9))
    return charges, symbols, center + external


def test_bo3_nine_external_contacts_matches_printed_basicity() -> None:
    charges, symbols, endpoints = _bo3_exact_external_nine()
    result = n485.complex_anion_contact_correspondence(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    expected_count = 3 / 0.33
    expected = 1 - abs(9 - expected_count) / (9 + expected_count)
    assert result.supported and result.feasible
    assert result.group_names == ("BO3",)
    assert result.external_contact_counts == (9,)
    assert result.expected_external_contact_counts == pytest.approx((expected_count,))
    assert result.features[n485.FEATURE_NAMES[0]] == pytest.approx(expected, abs=5e-11)


def test_hydrogen_or_no_group_is_supported_physical_zero() -> None:
    charges, symbols, endpoints = _bo3_exact_external_nine()
    hydrogen = n485.complex_anion_contact_correspondence(
        charges=charges + (1.0,), symbols=symbols + ("H",),
        endpoints=endpoints + ((13, 1),),
    )
    plain = n485.complex_anion_contact_correspondence(
        charges=(1.0, -1.0), symbols=("Na", "Cl"), endpoints=((0, 1),)
    )
    assert hydrogen.supported and plain.supported
    assert hydrogen.recognized_group_count == plain.recognized_group_count == 0
    assert hydrogen.features[n485.FEATURE_NAMES[0]] == 0.0
    assert plain.features[n485.FEATURE_NAMES[0]] == 0.0


def test_shared_oxygen_polymerized_candidates_are_excluded() -> None:
    # Two trigonal B candidates share O3, so neither is an isolated BO3 group.
    result = n485.complex_anion_contact_correspondence(
        charges=(1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        symbols=("B", "B", "O", "O", "O", "O", "O"),
        endpoints=((0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (1, 6)),
    )
    assert result.supported
    assert result.recognized_group_count == 0
    assert result.features[n485.FEATURE_NAMES[0]] == 0.0


def test_edge_order_site_permutation_and_exact_replication_are_invariant() -> None:
    charges, symbols, endpoints = _bo3_exact_external_nine()
    base = n485.complex_anion_contact_correspondence(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    reordered = n485.complex_anion_contact_correspondence(
        charges=charges, symbols=symbols, endpoints=tuple(reversed(endpoints))
    )
    permutation = tuple(reversed(range(len(charges))))
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted = n485.complex_anion_contact_correspondence(
        charges=tuple(charges[index] for index in permutation),
        symbols=tuple(symbols[index] for index in permutation),
        endpoints=tuple((inverse[left], inverse[right]) for left, right in endpoints),
    )
    offset = len(charges)
    replicated = n485.complex_anion_contact_correspondence(
        charges=charges * 2,
        symbols=symbols * 2,
        endpoints=endpoints + tuple((left + offset, right + offset) for left, right in endpoints),
    )
    results = (base, reordered, permuted, replicated)
    assert all(item.supported for item in results)
    values = np.asarray([item.features[n485.FEATURE_NAMES[0]] for item in results])
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
    result = n485.complex_anion_contact_correspondence(
        charges=charges, symbols=symbols, endpoints=endpoints
    )
    assert not result.supported
    assert needle in str(result.failure_reason).lower()


def test_boundary_forbids_dft_and_post_initial_inputs() -> None:
    assert all(value is False for value in n485.BOUNDARY_FLAGS.values())
    assert len(n485.DESIGN_SHA256) == len(n485.ASSET_SHA256) == 64
    assert math.isfinite(n485.APPENDIX4_GROUPS[("S", 4)].lewis_basicity)
