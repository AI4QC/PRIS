from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

from experiments.pu_synthesizability_20260821.combined_screening import (
    apply_rank_rule,
    bootstrap_matched_comparison,
    build_matched_frontier,
    build_natural_serial_results,
    build_observation_support,
    build_overlap_table,
    calibrate_rank_rule,
    deduplicate_binary_cifs,
    formula_decisions_at_experimental_preservation,
    plot_combined_screening,
    score_synthesis_variants,
)
from experiments.pu_synthesizability_20260821.formula_scores import FrozenFormula


def _frame(
    scores: list[float],
    l4: list[str],
    *,
    cohort: str,
    hashes: list[str] | None = None,
) -> pd.DataFrame:
    if hashes is None:
        hashes = [f"{index:064x}" for index in range(1, len(scores) + 1)]
    return pd.DataFrame(
        {
            "cohort": cohort,
            "cif_sha256": hashes,
            "L4_verdict": l4,
            "S_syn": scores,
        }
    )


def test_formula_rank_rule_hits_exact_experimental_retention() -> None:
    experimental = _frame(
        [0.0, 1.0, 2.0, 3.0],
        ["pass"] * 4,
        cohort="experimental",
    )
    pu = _frame(
        [0.5, 2.5],
        ["pass", "pass"],
        cohort="pu_negative",
        hashes=[f"{20:064x}", f"{21:064x}"],
    )

    rule = calibrate_rank_rule(
        experimental,
        score_column="S_syn",
        target_retention=0.5,
        include_l4=False,
    )

    assert apply_rank_rule(experimental, rule).tolist() == [True, True, False, False]
    assert apply_rank_rule(pu, rule).tolist() == [True, False]
    assert rule.experimental_screened_n == 2
    assert rule.experimental_retention == 0.5


def test_rank_rule_retains_boundary_ties_and_never_uses_pu_hash() -> None:
    experimental = _frame(
        [0.0, 1.0, 1.0, 2.0],
        ["pass"] * 4,
        cohort="experimental",
        hashes=[f"{index:064x}" for index in range(10, 14)],
    )
    pu = _frame(
        [1.0, 0.5],
        ["pass", "pass"],
        cohort="pu_negative",
        hashes=[f"{0:064x}", f"{99:064x}"],
    )

    rule = calibrate_rank_rule(
        experimental,
        score_column="S_syn",
        target_retention=0.5,
        include_l4=False,
    )

    assert rule.boundary_tie_n == 2
    assert rule.experimental_target_screened_n == 2
    assert rule.experimental_screened_n == 1
    assert apply_rank_rule(pu, rule).tolist() == [False, True]


def test_combined_rule_reports_budget_infeasible_when_l4_alone_exceeds_it() -> None:
    experimental = _frame(
        [0.0, 1.0, 2.0, 3.0],
        ["explicit_violation"] * 3 + ["pass"],
        cohort="experimental",
    )
    rule = calibrate_rank_rule(
        experimental,
        score_column="S_syn",
        target_retention=0.5,
        include_l4=True,
    )

    assert not rule.feasible
    assert rule.experimental_target_screened_n == 2
    assert rule.experimental_screened_n == 3
    assert rule.experimental_retention == 0.25


def test_combined_rule_prioritizes_all_l4_violations_then_low_formula() -> None:
    experimental = _frame(
        [0.0, 3.0, 1.0, 2.0],
        ["pass", "explicit_violation", "pass", "explicit_violation"],
        cohort="experimental",
    )
    pu = _frame(
        [100.0, -0.5, 2.5],
        ["explicit_violation", "pass", "pass"],
        cohort="pu_negative",
        hashes=[f"{20:064x}", f"{21:064x}", f"{22:064x}"],
    )

    rule = calibrate_rank_rule(
        experimental,
        score_column="S_syn",
        target_retention=0.25,
        include_l4=True,
    )

    assert apply_rank_rule(experimental, rule).tolist() == [True, True, False, True]
    assert apply_rank_rule(pu, rule).tolist() == [True, True, False]
    assert rule.experimental_screened_n == 3


def test_overlap_table_counts_four_disjoint_categories() -> None:
    table = build_overlap_table(
        pris_screened=np.array([True, True, False, False]),
        formula_screened=np.array([True, False, True, False]),
        cohort="experimental",
        score_variant="S_syn",
    )

    assert table.set_index("category")["n"].to_dict() == {
        "both": 1,
        "pris_only": 1,
        "formula_only": 1,
        "neither": 1,
    }
    assert np.isclose(table["rate"].sum(), 1.0)


def test_exact_cif_deduplication_is_equal_weight_and_fail_closed() -> None:
    experimental = _frame([1.0], ["pass"], cohort="experimental")
    pu = _frame(
        [0.0, 0.0],
        ["explicit_violation", "explicit_violation"],
        cohort="pu_negative",
        hashes=[f"{10:064x}", f"{10:064x}"],
    )
    combined = pd.concat((experimental, pu), ignore_index=True)

    unique, summary = deduplicate_binary_cifs(
        combined, score_columns=("S_syn",)
    )

    assert len(unique) == 2
    assert summary == {
        "experimental_input_rows": 1,
        "experimental_unique_cif": 1,
        "experimental_duplicate_extra_rows": 0,
        "pu_negative_input_rows": 2,
        "pu_negative_unique_cif": 1,
        "pu_negative_duplicate_extra_rows": 1,
    }


def test_bootstrap_matched_gain_is_paired_and_deterministic() -> None:
    experimental = _frame(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ["pass", "explicit_violation", "pass", "pass", "pass", "pass"],
        cohort="experimental",
    )
    pu = _frame(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [
            "pass",
            "explicit_violation",
            "explicit_violation",
            "pass",
            "pass",
            "pass",
        ],
        cohort="pu_negative",
        hashes=[f"{index:064x}" for index in range(20, 26)],
    )

    first = bootstrap_matched_comparison(
        experimental,
        pu,
        score_column="S_syn",
        target_retention=0.5,
        n_bootstrap=50,
        seed=17,
    )
    second = bootstrap_matched_comparison(
        experimental,
        pu,
        score_column="S_syn",
        target_retention=0.5,
        n_bootstrap=50,
        seed=17,
    )

    assert first == second
    assert first["n_bootstrap"] == 50
    assert first["combined_detection"] >= first["formula_detection"]
    assert np.isclose(
        first["paired_gain"],
        first["combined_detection"] - first["formula_detection"],
    )
    assert len(first["paired_gain_ci95"]) == 2


def test_natural_formula_threshold_uses_strict_low_score_decision() -> None:
    experimental = _frame(
        [0.0, 1.0, 2.0, 3.0], ["pass"] * 4, cohort="experimental"
    )
    pu = _frame(
        [0.5, 1.0, 1.5],
        ["pass"] * 3,
        cohort="pu_negative",
        hashes=[f"{index:064x}" for index in range(20, 23)],
    )

    result = formula_decisions_at_experimental_preservation(
        experimental,
        pu,
        score_column="S_syn",
        preservation=0.5,
    )

    assert result["threshold"] == 2.0
    assert result["experimental_screened"].tolist() == [True, True, False, False]
    assert result["pu_screened"].tolist() == [True, True, True]


def test_natural_serial_reports_incremental_pris_screening() -> None:
    experimental = _frame(
        [0.0, 1.0, 2.0, 3.0],
        ["pass", "explicit_violation", "pass", "pass"],
        cohort="experimental",
    )
    pu = _frame(
        [0.0, 1.0, 2.0, 3.0],
        ["pass", "explicit_violation", "explicit_violation", "pass"],
        cohort="pu_negative",
        hashes=[f"{index:064x}" for index in range(20, 24)],
    )

    summary, overlap = build_natural_serial_results(
        experimental,
        pu,
        score_column="S_syn",
        preservation=0.75,
    )

    row = summary.iloc[0]
    assert row["pu_formula_screened_n"] == 1
    assert row["pu_pris_screened_n"] == 2
    assert row["pu_combined_screened_n"] == 3
    assert row["pu_incremental_pris_beyond_formula_n"] == 2
    assert overlap.groupby("cohort")["n"].sum().to_dict() == {
        "experimental": 4,
        "pu_negative": 4,
    }


def test_matched_frontier_has_exact_budget_and_gain_columns() -> None:
    experimental = _frame(
        list(np.arange(10, dtype=float)),
        ["explicit_violation"] * 2 + ["pass"] * 8,
        cohort="experimental",
    )
    pu = _frame(
        list(np.arange(10, dtype=float)),
        ["explicit_violation"] * 4 + ["pass"] * 6,
        cohort="pu_negative",
        hashes=[f"{index:064x}" for index in range(20, 30)],
    )

    got = build_matched_frontier(
        experimental,
        pu,
        score_column="S_syn",
        retentions=(0.7,),
        n_bootstrap=20,
        seed=4,
    )

    assert len(got) == 1
    row = got.iloc[0]
    assert row["experimental_screened_n"] == 3
    assert row["pu_combined_screened_n"] >= row["pu_formula_screened_n"]
    assert np.isclose(
        row["paired_gain"],
        row["combined_detection"] - row["formula_detection"],
    )


def test_score_synthesis_variants_keeps_frozen_coefficients_in_sensitivity() -> None:
    formula = FrozenFormula(
        name="S_syn",
        features=("wyckoff_econ_001", "bv_rel_mean", "vol_per_atom"),
        beta=np.array([1.0, 2.0, 3.0]),
        impute_median={
            "wyckoff_econ_001": 0.0,
            "bv_rel_mean": 0.0,
            "vol_per_atom": 0.0,
        },
        mu={
            "wyckoff_econ_001": 0.0,
            "bv_rel_mean": 0.0,
            "vol_per_atom": 0.0,
        },
        sd={
            "wyckoff_econ_001": 1.0,
            "bv_rel_mean": 1.0,
            "vol_per_atom": 1.0,
        },
        source_path=Path("synthetic.json"),
    )
    frame = pd.DataFrame(
        {
            "formula_syn_wyckoff_econ_001": [1.0],
            "formula_syn_bv_rel_mean": [2.0],
            "formula_syn_vol_per_atom": [3.0],
        }
    )

    got = score_synthesis_variants(frame, formula)

    assert got["S_syn"].tolist() == [14.0]
    assert got["S_syn_no_D7_D8"].tolist() == [9.0]
    assert got["S_syn_n_observed"].tolist() == [3]
    assert got["S_syn_all_observed"].tolist() == [True]
    assert got["S_syn_no_D7_D8_n_observed"].tolist() == [1]
    assert got["S_syn_no_D7_D8_all_observed"].tolist() == [True]


def test_observation_support_counts_every_structure_once_per_variant() -> None:
    frame = pd.DataFrame(
        {
            "cohort": ["experimental", "experimental", "pu_negative"],
            "S_syn_n_observed": [6, 5, 0],
            "S_syn_all_observed": [True, False, False],
            "S_syn_no_D7_D8_n_observed": [4, 4, 1],
            "S_syn_no_D7_D8_all_observed": [True, True, False],
        }
    )

    got = build_observation_support(frame)

    assert got.groupby(["score_variant", "cohort"])["n"].sum().to_dict() == {
        ("S_syn", "experimental"): 2,
        ("S_syn", "pu_negative"): 1,
        ("S_syn_no_D7_D8", "experimental"): 2,
        ("S_syn_no_D7_D8", "pu_negative"): 1,
    }
    complete = got.loc[got.all_observed].set_index(["score_variant", "cohort"])
    assert complete.loc[("S_syn", "experimental"), "n"] == 1
    assert complete.loc[("S_syn_no_D7_D8", "experimental"), "n"] == 2


def test_combined_figure_renders_cjk_without_missing_glyph_warning(tmp_path) -> None:
    experimental = _frame(
        list(np.arange(10, dtype=float)),
        ["explicit_violation"] * 2 + ["pass"] * 8,
        cohort="experimental",
    )
    pu = _frame(
        list(np.arange(10, dtype=float)),
        ["explicit_violation"] * 4 + ["pass"] * 6,
        cohort="pu_negative",
        hashes=[f"{index:064x}" for index in range(20, 30)],
    )
    for frame in (experimental, pu):
        frame["S_syn_no_D7_D8"] = frame["S_syn"]
    frontier = pd.concat(
        [
            build_matched_frontier(
                experimental,
                pu,
                score_column=variant,
                retentions=(0.7,),
                n_bootstrap=2,
                seed=4,
            )
            for variant in ("S_syn", "S_syn_no_D7_D8")
        ],
        ignore_index=True,
    )
    for column in (
        "formula_detection_ci95",
        "combined_detection_ci95",
        "paired_gain_ci95",
    ):
        frontier[f"{column}_low"] = frontier[column].map(lambda value: value[0])
        frontier[f"{column}_high"] = frontier[column].map(lambda value: value[1])
    frontier = frontier.drop(
        columns=[
            "formula_detection_ci95",
            "combined_detection_ci95",
            "paired_gain_ci95",
        ]
    )
    natural_parts = []
    overlap_parts = []
    for variant in ("S_syn", "S_syn_no_D7_D8"):
        natural, overlap = build_natural_serial_results(
            experimental,
            pu,
            score_column=variant,
            preservation=0.75,
        )
        natural_parts.append(natural)
        overlap_parts.append(overlap)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_combined_screening(
            frontier,
            pd.concat(natural_parts, ignore_index=True),
            pd.concat(overlap_parts, ignore_index=True),
            output_png=tmp_path / "figure.png",
            output_pdf=tmp_path / "figure.pdf",
        )

    assert (tmp_path / "figure.png").stat().st_size > 0
    assert not [warning for warning in caught if "Glyph" in str(warning.message)]
