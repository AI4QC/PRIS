import math

import numpy as np

from src.next6_wbm_protocol import (
    clopper_pearson_upper,
    evaluate_triage,
    formula_split,
    reduced_formula_key,
    select_rejection_threshold,
    stable_from_wbm_hull,
    wbm_stage,
)


def test_reduced_formula_split_keeps_equivalent_compositions_together():
    # Break caught: hashing the raw formula would leak scaled versions across splits.
    assert reduced_formula_key("Li2O2") == "LiO"
    assert formula_split("Li2O2") == formula_split("LiO")


def test_wbm_stage_nests_two_calibration_roles_without_splitting_compositions():
    # Break caught: selecting a formula and calibrating its threshold on the same
    # labels invalidates the risk guarantee.
    assert wbm_stage("CH4") == "formula_selection"
    assert wbm_stage("H2O") == "threshold_calibration"
    assert wbm_stage("Li2O2") == wbm_stage("LiO") == "test"


def test_wbm_stability_uses_the_frozen_corrected_hull_boundary():
    # Break caught: using a strict inequality would discard structures exactly on the hull.
    values = np.array([-0.1, 0.0, 0.000001, np.nan])
    assert stable_from_wbm_hull(values).tolist() == [True, True, False, False]


def test_abstentions_are_sent_to_dft_and_never_count_as_savings():
    # Break caught: treating unknown/abstain as a successful rejection inflates savings.
    stable = np.array([True, True, False, False])
    decisions = np.array(["KEEP", "REJECT", "REJECT", "ABSTAIN"])
    got = evaluate_triage(stable, decisions)

    assert got["n"] == 4
    assert got["n_stable"] == 2
    assert got["n_reject"] == 2
    assert got["n_abstain"] == 1
    assert got["stable_recall"] == 0.5
    assert got["false_negative_rate"] == 0.5
    assert got["dft_savings"] == 0.5
    assert got["abstention_rate"] == 0.25


def test_clopper_pearson_upper_is_conservative_with_zero_errors():
    # Break caught: returning empirical zero would falsely certify a tiny calibration set.
    got = clopper_pearson_upper(0, 10, confidence=0.95)
    assert math.isclose(got, 0.2588655509, rel_tol=0, abs_tol=1e-10)


def test_threshold_calibration_maximizes_rejection_under_risk_bound():
    # 300 stable examples make zero false rejects certifiable below 1%; lowering the
    # threshold by one step rejects a stable example and violates that frozen risk.
    stable_scores = np.arange(300, dtype=float) / 1000.0
    unstable_scores = 1.0 + np.arange(100, dtype=float) / 1000.0
    scores = np.concatenate([stable_scores, unstable_scores, [np.nan]])
    stable = np.concatenate(
        [np.ones(300, dtype=bool), np.zeros(100, dtype=bool), [False]]
    )

    chosen = select_rejection_threshold(
        scores,
        stable,
        max_false_negative_ucb=0.01,
        confidence=0.95,
    )

    assert chosen["threshold"] == 1.0
    assert chosen["n_reject"] == 100
    assert chosen["n_abstain"] == 1
    assert chosen["stable_false_rejects"] == 0
    assert chosen["false_negative_ucb"] < 0.01
    assert math.isclose(chosen["dft_savings"], 100 / 401)
