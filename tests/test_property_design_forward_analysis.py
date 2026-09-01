from __future__ import annotations

import pandas as pd
import pytest

from experiments.property_design_20260821.forward_analysis import (
    summarize_forward_operating_point,
    summarize_theoretical_quality,
    wilson_interval,
)


def test_wilson_interval_contains_observed_fraction():
    low, high = wilson_interval(3, 10)

    assert low < 0.3 < high


def test_forward_operating_point_separates_theoretical_queue_from_experimental_loss():
    frame = pd.DataFrame(
        {
            "made": [True, True, False, False, False],
            "rung_L2_verdict": ["pass", "reject", "reject", "pass", "no verdict"],
        }
    )

    summary = summarize_forward_operating_point(
        frame,
        verdict_column="rung_L2_verdict",
    )

    assert summary["selected_count"] == 5
    assert summary["experimental_high_property_count"] == 2
    assert summary["experimental_high_property_removed_count"] == 1
    assert summary["experimental_high_property_retention"] == 0.5
    assert summary["theoretical_dft_queue_count"] == 3
    assert summary["theoretical_dft_queue_removed_count"] == 1
    assert summary["theoretical_dft_queue_reduction"] == 1 / 3


def test_theoretical_quality_compares_removed_and_retained_candidates_only():
    frame = pd.DataFrame(
        {
            "made": [False, False, False, False, True],
            "rung_L2_verdict": ["reject", "reject", "pass", "no verdict", "reject"],
            "energy_above_hull": [0.40, 0.30, 0.01, 0.02, 1.00],
            "dyn_stable": [False, False, True, True, False],
        }
    )

    summary = summarize_theoretical_quality(
        frame,
        verdict_column="rung_L2_verdict",
    )

    assert summary["removed"]["count"] == 2
    assert summary["removed"]["median_energy_above_hull_ev_atom"] == pytest.approx(0.35)
    assert summary["removed"]["dynamically_unstable_fraction"] == 1.0
    assert summary["removed"]["within_50mev_hull_fraction"] == 0.0
    assert summary["retained"]["count"] == 2
    assert summary["retained"]["median_energy_above_hull_ev_atom"] == pytest.approx(0.015)
    assert summary["retained"]["dynamically_unstable_fraction"] == 0.0
    assert summary["retained"]["within_50mev_hull_fraction"] == 1.0
