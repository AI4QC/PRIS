from __future__ import annotations

import pandas as pd
import pytest

from experiments.pu_synthesizability_20260821.analysis import (
    add_common_support_weights,
    add_unique_structure_weights,
    assign_score_quantile_bands,
    count_explicit_violations,
    conjoin_reader_verdicts,
    summarize_score_bands,
    summarize_binary_verdict,
    summarize_mechanism_status_counts,
    summarize_weighted_verdict,
    translate_verdict,
)
from experiments.pu_synthesizability_20260821.analyze_results import (
    _within_strata_row,
    build_parser,
)


@pytest.mark.parametrize(
    ("internal", "reader"),
    [("pass", "pass"), ("reject", "explicit_violation"), ("no verdict", "no_verdict")],
)
def test_translate_verdict_preserves_three_states(internal, reader):
    assert translate_verdict(internal) == reader


def test_binary_summary_never_counts_no_verdict_as_pass_or_violation():
    frame = pd.DataFrame(
        {
            "cohort": ["experimental"] * 5 + ["pu_negative"] * 4,
            "L2": ["pass", "pass", "explicit_violation", "no_verdict", "no_verdict"]
            + ["explicit_violation", "explicit_violation", "pass", "no_verdict"],
        }
    )
    got = summarize_binary_verdict(frame, verdict_column="L2").set_index("cohort")
    exp = got.loc["experimental"]
    neg = got.loc["pu_negative"]
    assert exp["satisfaction"] == pytest.approx(2 / 3)
    assert exp["queue_retained"] == pytest.approx(4 / 5)
    assert neg["damage_detection"] == pytest.approx(2 / 4)
    assert exp[["pass_n", "explicit_violation_n", "no_verdict_n"]].sum() == 5


def test_score_quantiles_collapse_tied_edges_instead_of_splitting_equal_scores():
    scores = pd.Series([0.0] * 8 + [0.2, 0.4])
    bands = assign_score_quantile_bands(scores, n_quantiles=5)
    assert bands[scores.eq(0)].nunique() == 1
    assert bands.notna().all()
    assert 2 <= bands.nunique() < 5


def test_common_support_weights_drop_unshared_strata_and_balance_shared_mix():
    frame = pd.DataFrame(
        {
            "cohort": ["experimental"] * 5 + ["pu_negative"] * 4,
            "stratum": ["A", "A", "A", "B", "only_exp", "A", "B", "B", "only_neg"],
        }
    )
    got = add_common_support_weights(frame, cohort_column="cohort", strata_columns=["stratum"])
    assert got.loc[got.stratum.str.startswith("only"), "common_support_weight"].isna().all()
    shared = got.dropna(subset=["common_support_weight"])
    assert set(shared.stratum) == {"A", "B"}
    sums = shared.groupby("cohort").common_support_weight.sum()
    assert sums["experimental"] == pytest.approx(1.0)
    assert sums["pu_negative"] == pytest.approx(1.0)


def test_common_support_can_require_minimum_rows_in_each_cohort():
    frame = pd.DataFrame(
        {
            "cohort": ["experimental", "pu_negative", "experimental", "experimental", "pu_negative", "pu_negative"],
            "stratum": ["sparse", "sparse", "dense", "dense", "dense", "dense"],
        }
    )
    got = add_common_support_weights(
        frame,
        cohort_column="cohort",
        strata_columns=["stratum"],
        min_per_cohort=2,
    )
    assert got.loc[got.stratum.eq("sparse"), "common_support_weight"].isna().all()
    assert got.loc[got.stratum.eq("dense"), "common_support_weight"].notna().all()


def test_unique_structure_weights_make_each_cif_count_once():
    frame = pd.DataFrame({"cif_sha256": ["a", "a", "b"], "value": [1, 1, 0]})
    got = add_unique_structure_weights(frame)
    assert got.unique_structure_weight.tolist() == [0.5, 0.5, 1.0]
    assert got.unique_structure_weight.sum() == pytest.approx(2.0)


def test_count_explicit_violations_ignores_no_verdict():
    frame = pd.DataFrame(
        {
            "D1_verdict": ["explicit_violation", "no_verdict", "pass"],
            "D2_verdict": ["pass", "no_verdict", "explicit_violation"],
        }
    )
    assert count_explicit_violations(frame, ["D1_verdict", "D2_verdict"]).tolist() == [1, 0, 1]


def test_mechanism_evidence_fraction_excludes_unknown_and_not_applicable():
    frame = pd.DataFrame(
        {
            "D1_status": ["violated", "unknown"],
            "D2_status": ["not applicable", "satisfied"],
            "D3_status": ["satisfied", "violated"],
        }
    )
    got = summarize_mechanism_status_counts(
        frame, ["D1_status", "D2_status", "D3_status"]
    )
    assert got.explicit_violation_count.tolist() == [1, 1]
    assert got.evaluable_mechanism_count.tolist() == [2, 2]
    assert got.violation_fraction_evaluable.tolist() == pytest.approx([0.5, 0.5])


def test_score_band_summary_reports_all_three_states_and_counts():
    frame = pd.DataFrame(
        {
            "score": [0.0, 0.0, 0.2, 0.4],
            "L4_verdict": ["explicit_violation", "no_verdict", "pass", "pass"],
        }
    )
    got = summarize_score_bands(
        frame,
        score_column="score",
        verdict_column="L4_verdict",
        n_quantiles=3,
    )
    assert got.n.sum() == 4
    assert set(got.columns).issuperset(
        {"score_band", "n", "explicit_violation_rate", "no_verdict_rate", "pass_rate"}
    )


def test_weighted_verdict_summary_uses_effective_structure_weights():
    frame = pd.DataFrame(
        {
            "verdict": ["pass", "explicit_violation", "no_verdict"],
            "weight": [0.5, 0.5, 1.0],
        }
    )
    got = summarize_weighted_verdict(
        frame, verdict_column="verdict", weight_column="weight"
    )
    assert got["total_weight"] == pytest.approx(2.0)
    assert got["satisfaction"] == pytest.approx(0.5)
    assert got["explicit_violation_rate"] == pytest.approx(0.25)
    assert got["queue_retained"] == pytest.approx(0.75)


def test_conjoin_reader_verdicts_lets_any_explicit_violation_override_unknown():
    frame = pd.DataFrame(
        {
            "a": ["pass", "pass", "no_verdict", "no_verdict"],
            "b": ["pass", "no_verdict", "explicit_violation", "no_verdict"],
        }
    )
    assert conjoin_reader_verdicts(frame, ["a", "b"]).tolist() == [
        "pass",
        "no_verdict",
        "explicit_violation",
        "no_verdict",
    ]


def test_within_strata_correlation_residualizes_new_rank_column():
    frame = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "outcome": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            "source": ["a", "a", "a", "b", "b", "b"],
        }
    )
    got = _within_strata_row(
        frame,
        score="score",
        outcome="outcome",
        strata=["source"],
        min_group_rows=3,
    )
    assert got["n"] == 6
    assert got["n_groups"] == 2
    assert got["spearman_rho"] > 0


def test_within_strata_correlation_excludes_constant_outcome_groups():
    informative = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.3],
            "outcome": [0.0, 1.0, 2.0],
            "source": ["informative"] * 3,
        }
    )
    constant = pd.DataFrame(
        {
            "score": [index / 100 for index in range(100)],
            "outcome": [0.0] * 100,
            "source": ["constant"] * 100,
        }
    )
    got = _within_strata_row(
        pd.concat([informative, constant], ignore_index=True),
        score="score",
        outcome="outcome",
        strata=["source"],
        min_group_rows=3,
    )
    assert got["n"] == 3
    assert got["n_groups"] == 1
    assert got["excluded_constant_rows"] == 100
    assert got["spearman_rho"] == pytest.approx(1.0)


def test_within_strata_correlation_returns_nan_if_all_groups_are_constant():
    frame = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.3],
            "outcome": [1.0, 1.0, 1.0],
            "source": ["constant"] * 3,
        }
    )
    got = _within_strata_row(
        frame,
        score="score",
        outcome="outcome",
        strata=["source"],
        min_group_rows=3,
    )
    assert got["n"] == 0
    assert got["n_groups"] == 0
    assert got["excluded_constant_rows"] == 3
    assert pd.isna(got["spearman_rho"])


def test_analysis_cli_accepts_multiple_cohort_segments():
    args = build_parser().parse_args(
        [
            "--experimental-dir",
            "exp-a",
            "--experimental-dir",
            "exp-b",
            "--negative-dir",
            "neg-a",
        ]
    )
    assert args.experimental_dirs == ["exp-a", "exp-b"]
    assert args.negative_dirs == ["neg-a"]
