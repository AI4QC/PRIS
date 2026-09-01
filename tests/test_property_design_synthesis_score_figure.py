import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.property_design_20260821 import plot_synthesis_score_design as design_plot

from experiments.property_design_20260821.plot_synthesis_score_design import (
    _rule_marker_sizes,
    build_design_frontier,
    build_transferred_frontier,
    render_design_figure,
    rule_operating_point,
)


def test_rule_marker_sizes_keep_coincident_methods_visible():
    points = pd.DataFrame(
        {
            "method": ["L1", "L2", "L3", "L4"],
            "inverse_priority_retention": [1.0, 1.0, 1.0, 0.4],
            "inverse_queue_reduction": [0.0, 0.0, 0.0, 0.6],
        }
    )

    sizes = _rule_marker_sizes(points)

    assert sizes["L1"] > sizes["L2"] > sizes["L3"]
    assert sizes["L1"] >= 6 * sizes["L3"]
    assert sizes["L4"] > 0


def test_draw_design_panel_reuses_inverse_design_visual_contract(tmp_path):
    assert hasattr(design_plot, "draw_design_panel")

    frontier = pd.DataFrame(
        {
            "inverse_priority_retention": [0.40, 0.70, 1.00],
            "inverse_queue_reduction": [0.80, 0.45, 0.00],
        }
    )
    rule_points = pd.DataFrame(
        {
            "method": ["L1", "L2", "L3", "L4"],
            "inverse_priority_retention": [1.00, 1.00, 1.00, 0.40],
            "inverse_queue_reduction": [0.00, 0.01, 0.01, 0.67],
        }
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    metadata = design_plot.draw_design_panel(
        ax,
        frontier,
        rule_points,
        inverse_priority_label="UMA-proxy candidates retained (%)",
        panel_letter="f",
    )
    path = tmp_path / "panel.svg"
    fig.savefig(path)
    plt.close(fig)

    svg = path.read_text()
    assert "UMA-proxy candidates retained (%)" in svg
    assert "Generated queue" in svg
    assert "generated queue" not in svg
    assert "PSS" in svg
    assert "Synthesis score" not in svg
    assert "operating point" not in svg
    assert "same threshold" not in svg
    assert metadata["plot_limits"]["x"][1] > 100


def test_build_design_frontier_uses_one_forward_calibration_for_both_tasks():
    forward = pd.DataFrame(
        {
            "made": [True, True, True, True, False, False, False, False],
            "synthesis_score": [-3.0, -1.0, 1.0, 3.0, -4.0, -2.0, 0.0, 2.0],
        }
    )
    inverse = pd.DataFrame({"synthesis_score": [-5.0, -1.5, 0.5, 4.0]})

    frontier = build_design_frontier(
        forward,
        inverse,
        score_column="synthesis_score",
        target_retentions=(0.50, 0.75, 1.00),
    )

    assert frontier["target_experimental_retention"].tolist() == [0.50, 0.75, 1.00]
    assert np.allclose(frontier["experimental_retention"], [0.50, 0.75, 1.00])
    assert np.allclose(frontier["forward_theory_reduction"], [0.75, 0.50, 0.00])
    assert np.allclose(frontier["inverse_queue_reduction"], [0.75, 0.50, 0.00])
    assert frontier["cutoff_score"].iloc[:2].tolist() == [1.0, -1.0]
    assert np.isneginf(frontier["cutoff_score"].iloc[2])


def test_build_design_frontier_accepts_support_matched_calibration_subset():
    forward = pd.DataFrame(
        {
            "made": [True, True, True, True, False, False],
            "synthesis_score": [-4.0, -1.0, 2.0, 5.0, -3.0, 3.0],
        }
    )
    inverse = pd.DataFrame({"synthesis_score": [-2.0, 0.0, 4.0]})
    matched = np.array([True, True, False, False, False, False])

    frontier = build_design_frontier(
        forward,
        inverse,
        score_column="synthesis_score",
        target_retentions=(0.50,),
        experimental_calibration_mask=matched,
    )

    row = frontier.iloc[0]
    assert row.experimental_count == 2
    assert row.cutoff_score == -1.0
    assert row.experimental_retention == 0.5
    assert row.inverse_queue_reduction == 1 / 3


def test_rule_operating_point_counts_only_explicit_violations():
    forward = pd.DataFrame(
        {
            "made": [True, True, False, False],
            "rung_L2_verdict": ["pass", "no verdict", "reject", "pass"],
        }
    )
    inverse = pd.DataFrame(
        {"rung_L2_verdict": ["reject", "no verdict", "pass", "reject"]}
    )

    point = rule_operating_point(
        forward,
        inverse,
        verdict_column="rung_L2_verdict",
        method="L2",
    )

    assert point["method"] == "L2"
    assert point["experimental_retention"] == 1.0
    assert point["forward_theory_reduction"] == 0.5
    assert point["inverse_queue_reduction"] == 0.5


def test_build_transferred_frontier_keeps_external_cutoffs_fixed():
    forward = pd.DataFrame(
        {
            "made": [True, True, False, False],
            "synthesis_score": [-1.0, 2.0, -2.0, 3.0],
        }
    )
    inverse = pd.DataFrame({"synthesis_score": [-3.0, 0.0, 4.0]})
    cutoffs = pd.DataFrame(
        {
            "reference_experimental_retention": [0.90, 0.80],
            "cutoff_score": [-1.5, 1.0],
        }
    )

    result = build_transferred_frontier(
        forward,
        inverse,
        cutoffs=cutoffs,
        score_column="synthesis_score",
    )

    assert result["cutoff_score"].tolist() == [-1.5, 1.0]
    assert np.allclose(result["forward_experimental_retention"], [1.0, 0.5])
    assert np.allclose(result["forward_theory_reduction"], [0.5, 0.5])
    assert np.allclose(result["inverse_queue_reduction"], [1 / 3, 2 / 3])


def test_render_design_figure_writes_preview_and_machine_readable_data(tmp_path):
    forward = pd.DataFrame(
        {
            "made": [True] * 4 + [False] * 4,
            "synthesis_score": [-3.0, -1.0, 1.0, 3.0, -4.0, -2.0, 0.0, 2.0],
            "rung_L1_verdict": ["pass"] * 8,
            "rung_L2_verdict": ["pass"] * 6 + ["reject", "pass"],
            "rung_L3_verdict": ["pass"] * 5 + ["reject", "reject", "pass"],
            "rung_L4_verdict": ["pass", "reject", "pass", "pass", "reject", "reject", "pass", "pass"],
        }
    )
    inverse = pd.DataFrame(
        {
            "synthesis_score": [-5.0, -1.5, 0.5, 4.0],
            "rung_L1_verdict": ["pass"] * 4,
            "rung_L2_verdict": ["reject", "pass", "pass", "pass"],
            "rung_L3_verdict": ["reject", "reject", "pass", "pass"],
            "rung_L4_verdict": ["reject", "reject", "reject", "pass"],
        }
    )

    status = render_design_figure(
        forward,
        inverse,
        output_dir=tmp_path,
        highlight_retention=0.50,
        target_retentions=(0.50, 0.75, 1.00),
    )

    assert (tmp_path / "property_design_synthesis_score_candidate.pdf").is_file()
    assert (tmp_path / "property_design_synthesis_score_candidate.png").is_file()
    assert (tmp_path / "property_design_synthesis_score_candidate.svg").is_file()
    assert (tmp_path / "plot_data.csv").is_file()
    assert (tmp_path / "STATUS.json").is_file()
    svg = (tmp_path / "property_design_synthesis_score_candidate.svg").read_text()
    assert "Forward screening" not in svg
    assert "Inverse design" not in svg
    assert 'id="axes_2"' not in svg
    assert "operating point" not in svg
    assert "same threshold transferred" not in svg
    assert "inverse score:" not in svg
    assert "screened;" not in svg
    assert status["plot_limits"]["x"][1] > 100
    assert status["plot_limits"]["y"][0] < 0
    assert status["highlight"]["drawn_in_figure"] is False
    assert status["highlight"]["experimental_retained_n"] == 2
    assert status["highlight"]["forward_theory_screened_n"] == 3
    assert status["highlight"]["inverse_screened_n"] == 3
