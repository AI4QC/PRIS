from __future__ import annotations

import pandas as pd
from pymatgen.core import Lattice, Structure

from experiments.pu_synthesizability_20260821.structure_overlap_control import (
    build_cross_cohort_pairs,
    canonical_reduced_formula,
    collect_selected_records,
    compare_structure_pairs,
    parse_bawl_exclusion_evidence,
    select_shared_formula_rows,
    summarize_match_results,
)


def test_canonical_reduced_formula_normalizes_formula_units() -> None:
    assert canonical_reduced_formula("Na4Cl4") == "NaCl"
    assert canonical_reduced_formula("Ca2In4As4") == "Ca(InAs)2"


def test_select_shared_formula_rows_excludes_unshared_formulas() -> None:
    positive = pd.DataFrame(
        {
            "record_index": [0, 1, 2],
            "formula": ["Na2Cl2", "LiF", "NaCl"],
        }
    )
    negative = pd.DataFrame(
        {
            "record_index": [10, 11],
            "structure_formula": ["NaCl", "KBr"],
        }
    )

    shared, positive_selected, negative_selected = select_shared_formula_rows(
        positive, negative
    )

    assert shared == ["NaCl"]
    assert positive_selected.record_index.tolist() == [0, 2]
    assert negative_selected.record_index.tolist() == [10]


def test_build_cross_cohort_pairs_is_formula_blocked_cartesian_product() -> None:
    positive = pd.DataFrame(
        {"record_index": [0, 1, 2], "reduced_formula": ["NaCl", "NaCl", "LiF"]}
    )
    negative = pd.DataFrame(
        {"record_index": [10, 11, 12], "reduced_formula": ["NaCl", "NaCl", "LiF"]}
    )

    pairs = build_cross_cohort_pairs(positive, negative)

    assert list(pairs.itertuples(index=False, name=None)) == [
        ("LiF", 2, 12),
        ("NaCl", 0, 10),
        ("NaCl", 0, 11),
        ("NaCl", 1, 10),
        ("NaCl", 1, 11),
    ]


def test_compare_structure_pairs_distinguishes_equivalent_and_distinct_cells() -> None:
    rocksalt = Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    rocksalt_perturbed = rocksalt.copy()
    rocksalt_perturbed.translate_sites([0], [0.005, 0, 0], frac_coords=True)
    layered = Structure(
        Lattice.tetragonal(3.9, 7.8),
        ["Na", "Cl"],
        [[0, 0, 0], [0, 0, 0.35]],
    )
    positive = {0: rocksalt}
    negative = {10: rocksalt_perturbed, 11: layered}
    pairs = pd.DataFrame(
        {
            "reduced_formula": ["NaCl", "NaCl"],
            "positive_index": [0, 0],
            "negative_index": [10, 11],
        }
    )

    result = compare_structure_pairs(pairs, positive, negative)

    by_index = result.set_index("negative_index")
    assert bool(by_index.loc[10, "default_match"])
    assert bool(by_index.loc[10, "strict_match"])
    assert bool(by_index.loc[10, "scale_preserving_match"])
    assert bool(by_index.loc[10, "supercell_match"])
    assert not bool(by_index.loc[11, "default_match"])
    assert not bool(by_index.loc[11, "strict_match"])
    assert not bool(by_index.loc[11, "scale_preserving_match"])
    assert not bool(by_index.loc[11, "supercell_match"])


def test_collect_selected_records_requires_every_index_exactly_once() -> None:
    records = [
        {"record_index": 3, "cif": "cif-3"},
        {"record_index": 5, "cif": "cif-5"},
        {"record_index": 8, "cif": "cif-8"},
    ]

    selected = collect_selected_records(records, {3, 8})

    assert sorted(selected) == [3, 8]
    assert selected[8]["cif"] == "cif-8"


def test_parse_bawl_exclusion_evidence_extracts_unique_logged_counts() -> None:
    strings = [
        "noise",
        "[exclusions] 99162 positive hashes, 8125912 pool hashes -> "
        "16077 excluded rows in 16s -> /cluster/pu_pool/excluded_orig_indices.txt\\nmore-json",
        "same [exclusions] 99162 positive hashes, 8125912 pool hashes -> "
        "16077 excluded rows in 16s -> /cluster/pu_pool/excluded_orig_indices.txt",
    ]

    evidence = parse_bawl_exclusion_evidence(strings)

    assert evidence == {
        "positive_hashes": 99_162,
        "pool_hashes": 8_125_912,
        "excluded_rows": 16_077,
        "elapsed_seconds": 16,
        "remote_exclusion_path": "/cluster/pu_pool/excluded_orig_indices.txt",
    }


def test_summarize_match_results_reports_each_tolerance_and_union() -> None:
    frame = pd.DataFrame(
        {
            "default_match": [True, False, False],
            "default_error": [None, None, None],
            "strict_match": [True, False, False],
            "strict_error": [None, None, None],
            "scale_preserving_match": [True, False, False],
            "scale_preserving_error": [None, None, None],
            "supercell_match": [True, True, False],
            "supercell_error": [None, None, None],
        }
    )

    summary = summarize_match_results(frame)

    assert summary["comparison_pairs"] == 3
    assert summary["default"]["matches"] == 1
    assert summary["strict"]["matches"] == 1
    assert summary["scale_preserving"]["matches"] == 1
    assert summary["supercell"]["matches"] == 2
    assert summary["matched_by_any_regime"] == 2
