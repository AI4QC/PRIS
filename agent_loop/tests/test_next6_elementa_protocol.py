import math

import numpy as np
import pandas as pd
import pytest

from src.next6_elementa_protocol import (
    apply_group_threshold,
    attach_energy_labels,
    elementa_stage,
    evaluate_group_triage,
    group_conformal_threshold,
)


pytestmark = pytest.mark.filterwarnings("error")


def test_composition_stage_is_deterministic_and_never_splits_a_group():
    keys = ["Be1|Li1|N1", "Be1|Li1|N1", "Fe1|Mn1|Se2", "Cl1|Na1"]
    stages = [elementa_stage(key) for key in keys]
    assert stages[0] == stages[1]
    assert set(stages) <= {
        "search_calibration",
        "formula_selection",
        "threshold_calibration",
        "test",
    }


def test_energy_labels_are_computed_only_within_composition():
    data = pd.DataFrame(
        {"rk": ["a", "a", "b", "b"], "e_per_atom": [-2.0, -1.9, -5.0, -4.7]}
    )
    got = attach_energy_labels(data)
    assert got.delta_e.tolist() == pytest.approx([0.0, 0.1, 0.0, 0.3])
    assert got.near_min.tolist() == [True, False, True, False]
    assert got.valuable.tolist() == [True, False, True, False]
    assert got.high_energy.tolist() == [False, False, False, True]


def test_group_conformal_quantile_and_strict_threshold_avoid_tie_rejection():
    data = pd.DataFrame(
        {
            "rk": ["a", "b", "c", "d"],
            "valuable": [True, True, True, True],
            "score": [1.0, 2.0, 3.0, 4.0],
            "supported": [True, True, True, True],
        }
    )
    calibrated = group_conformal_threshold(data, alpha=0.4)
    assert calibrated["n_groups"] == 4
    assert calibrated["order_index"] == 3
    assert calibrated["threshold"] == 3.0

    decisions = apply_group_threshold(
        data.score.to_numpy(), data.supported.to_numpy(), calibrated["threshold"]
    )
    assert decisions.tolist() == ["KEEP", "KEEP", "KEEP", "REJECT"]


def test_group_conformal_can_protect_at_least_one_member_instead_of_every_member():
    data = pd.DataFrame(
        {
            "rk": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "valuable": [True] * 8,
            "score": [1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
            "supported": [True] * 8,
        }
    )
    protect_every = group_conformal_threshold(data, alpha=0.4, within_group="max")
    protect_one = group_conformal_threshold(data, alpha=0.4, within_group="min")
    assert protect_every["threshold"] == 6.0
    assert protect_one["threshold"] == 3.0
    assert protect_one["within_group"] == "min"

    with pytest.raises(ValueError, match="within_group"):
        group_conformal_threshold(data, alpha=0.4, within_group="median")


def test_small_calibration_sample_returns_no_rejection_threshold():
    data = pd.DataFrame(
        {
            "rk": ["a", "b", "c", "d"],
            "valuable": [True, True, True, True],
            "score": [1.0, 2.0, 3.0, 4.0],
            "supported": [True, True, True, True],
        }
    )
    calibrated = group_conformal_threshold(data, alpha=0.05)
    assert math.isinf(calibrated["threshold"])
    assert calibrated["order_index"] == 5


def test_missing_or_non_x0_input_abstains_and_never_counts_as_savings():
    decisions = apply_group_threshold(
        np.array([4.0, np.nan, 9.0]),
        np.array([True, True, False]),
        threshold=3.0,
    )
    assert decisions.tolist() == ["REJECT", "ABSTAIN", "ABSTAIN"]


def test_group_metrics_preserve_near_tied_minimum_and_count_abstentions_as_dft():
    data = pd.DataFrame(
        {
            "rk": ["g1", "g1", "g1", "g2", "g2"],
            "delta_e": [0.0, 0.0005, 0.2, 0.0, 0.1],
            "decision": ["REJECT", "KEEP", "REJECT", "ABSTAIN", "REJECT"],
        }
    )
    got = evaluate_group_triage(data)

    assert got["n"] == 5 and got["n_groups"] == 2
    assert got["dft_savings"] == pytest.approx(3 / 5)
    assert got["macro_dft_savings"] == pytest.approx(((2 / 3) + (1 / 2)) / 2)
    assert got["abstention_rate"] == pytest.approx(1 / 5)
    assert got["exact_min_retention"] == pytest.approx(1 / 2)
    assert got["near_min_retention"] == 1.0
    assert got["valuable_item_recall"] == pytest.approx(2 / 3)
    assert got["valuable_all_retained_group_rate"] == pytest.approx(1 / 2)
    assert got["high_energy_removal_recall"] == 1.0
    assert got["reject_high_energy_precision"] == pytest.approx(1 / 3)
    assert got["regret_max"] == pytest.approx(0.0005)


def test_rejecting_an_entire_group_yields_infinite_regret():
    data = pd.DataFrame(
        {"rk": ["g", "g"], "delta_e": [0.0, 0.2], "decision": ["REJECT", "REJECT"]}
    )
    got = evaluate_group_triage(data)
    assert got["all_rejected_groups"] == 1
    assert math.isinf(got["regret_p95"])
    assert math.isinf(got["regret_max"])
