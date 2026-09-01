import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.text import Text
from pathlib import Path
import pandas as pd
import re
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.pu_synthesizability_20260821 import plot_merged_fig45_nature as merged


def _require_private_outputs(*paths: Path) -> None:
    """Skip checks whose frozen analysis inputs are not distributed on GitHub."""

    missing = [Path(path) for path in paths if not Path(path).exists()]
    if missing:
        rendered = ", ".join(
            str(path.relative_to(merged.ROOT))
            if path.is_relative_to(merged.ROOT)
            else str(path)
            for path in missing
        )
        pytest.skip(f"requires non-distributed analysis output(s): {rendered}")


def _assert_axis_labels_start_with_capital(fig):
    labels = []
    for ax in fig.findobj(Axes):
        labels.extend([ax.get_xlabel(), ax.get_ylabel()])
    labels = [label for label in labels if label.strip()]
    assert labels
    for label in labels:
        first_cased = next(
            (char for char in label if char.isalpha() and char.upper() != char.lower()),
            None,
        )
        if first_cased is not None:
            assert first_cased.isupper(), label


def _toy_frames():
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
    return forward, inverse


def test_panel_f_is_inverse_design_queue_tradeoff(tmp_path):
    forward, inverse = _toy_frames()
    fig, ax = plt.subplots(figsize=(4, 3))
    record = merged.panel_f(
        ax,
        forward,
        inverse,
        inverse_priority_mask=[False, True, True, True],
        target_retentions=(0.50, 0.75, 1.00),
        inverse_priority_label="UMA-proxy candidates retained (%)",
    )
    path = tmp_path / "merged_f.svg"
    fig.savefig(path)
    plt.close(fig)

    svg = path.read_text()
    assert record["task"] == "inverse_design_queue_tradeoff"
    assert record["inverse_total_n"] == 4
    assert record["inverse_priority_total_n"] == 3
    assert len(record["rule_operating_points"]) == 4
    assert "UMA-proxy candidates retained (%)" in svg
    assert "DFT-validation queue reduction (%)" in svg
    assert "energy and phonon state" not in svg
    assert "short-range repulsion" not in svg
    assert "crystallographic site complexity" not in svg
    assert "operating point" not in svg


def test_panel_f_matches_pss_descriptor_support_before_calibration():
    forward, inverse = _toy_frames()
    forward["n_observed"] = [2, 2, 6, 6, 2, 6, 2, 6]
    inverse["synthesis_n_observed"] = 2
    fig, ax = plt.subplots(figsize=(4, 3))

    record = merged.panel_f(
        ax,
        forward,
        inverse,
        inverse_priority_mask=[False, True, True, True],
        target_retentions=(0.50, 1.00),
    )
    plt.close(fig)

    assert record["calibration_support"]["n_observed"] == 2
    assert record["calibration_support"]["experimental_n"] == 2
    assert {row["experimental_count"] for row in record["frontier"]} == {2}


def test_panels_c_and_d_match_panel_f_legend_font_size(monkeypatch):
    binary_dir = merged.ROOT / "outputs/20260821_pu_synthesizability/analysis_v1"
    independent_dir = (
        merged.ROOT / "outputs/20260822_pu_formula_scores/independent_choices_v1"
    )
    _require_private_outputs(
        binary_dir / "rung_summary.csv",
        binary_dir / "distance_cutoff_summary.csv",
        independent_dir / "independent_frontier.csv",
        merged.FULL_POOL_DEFAULT / "score_deciles.csv",
        merged.FORMULA_NPZ_DEFAULT,
    )
    monkeypatch.syspath_prepend(
        str(merged.ROOT / "experiments/pu_synthesizability_20260821")
    )
    merged.style()
    cdata, _, ehull, _ = merged._load_c(
        binary_dir,
        independent_dir,
        merged.EHULL_DEFAULT,
    )
    ddata = merged.load_d_series(merged.FULL_POOL_DEFAULT, merged.FORMULA_NPZ_DEFAULT)
    forward, inverse = _toy_frames()

    fig = plt.figure(figsize=(12, 3))
    grid = fig.add_gridspec(1, 3)
    c_ax = fig.add_subplot(grid[0, 0])
    d_outer = fig.add_subplot(grid[0, 1])
    f_ax = fig.add_subplot(grid[0, 2])
    merged.panel_c(c_ax, cdata, ehull)
    axes_before_d = set(fig.axes)
    merged.panel_d(d_outer, ddata)
    d_axes = [axis for axis in fig.axes if axis not in axes_before_d]
    merged.panel_f(
        f_ax,
        forward,
        inverse,
        inverse_priority_mask=[False, True, True, True],
        target_retentions=(0.50, 0.75, 1.00),
    )

    c_legend = c_ax.get_legend()
    d_legend = next(axis.get_legend() for axis in d_axes if axis.get_legend())
    f_legend = f_ax.get_legend()
    sizes = [
        legend.get_texts()[0].get_fontsize()
        for legend in (c_legend, d_legend, f_legend)
    ]
    plt.close(fig)

    assert sizes[0] == pytest.approx(sizes[2])
    assert sizes[1] == pytest.approx(sizes[2])


def test_panel_d_legend_sits_above_the_curve_area(monkeypatch):
    _require_private_outputs(
        merged.FULL_POOL_DEFAULT / "score_deciles.csv",
        merged.FORMULA_NPZ_DEFAULT,
    )
    monkeypatch.syspath_prepend(
        str(merged.ROOT / "experiments/pu_synthesizability_20260821")
    )
    merged.style()
    ddata = merged.load_d_series(merged.FULL_POOL_DEFAULT, merged.FORMULA_NPZ_DEFAULT)

    fig, outer = plt.subplots(figsize=(4, 4))
    axes_before = set(fig.axes)
    merged.panel_d(outer, ddata)
    panel_axes = [axis for axis in fig.axes if axis not in axes_before]
    top = next(axis for axis in panel_axes if axis.get_legend())
    legend = top.get_legend()
    anchor = legend.get_bbox_to_anchor().transformed(top.transAxes.inverted())
    plt.close(fig)

    assert anchor.y1 >= 1.25


def test_panel_b_uses_compact_class_labels_at_the_y_tick_font_size():
    merged.style()
    fig, ax = plt.subplots(figsize=(4, 3))
    merged.panel_b(ax)

    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()
    assert [label.get_text() for label in xlabels] == [
        "D1", "D2", "D3", "D4", "D5"
    ]
    assert ylabels
    assert all(
        label.get_fontsize() == pytest.approx(ylabels[0].get_fontsize())
        for label in xlabels
    )
    plt.close(fig)


def test_inverse_design_diagnostics_explain_the_fixed_pss_screen() -> None:
    _require_private_outputs(merged.INVERSE_DESIGN_DEFAULT)
    inverse = pd.read_parquet(merged.INVERSE_DESIGN_DEFAULT)
    priority = inverse["clamped_bulk_modulus_proxy_gpa"].ge(400).to_numpy(bool)

    diagnostics = merged.build_inverse_design_diagnostics(
        inverse,
        priority_mask=priority,
        cutoff=-0.6368790173149083,
        screened_example_id="candidate_0009",
        retained_example_id="candidate_0801",
    )

    assert diagnostics["screened_n"] == 61
    assert diagnostics["priority_retained_n"] == 140
    assert diagnostics["priority_total_n"] == 140
    assert diagnostics["screened_distance_0p7_pass_n"] == 61
    assert diagnostics["screened_d7_violation_n"] == 61
    assert diagnostics["screened_site_fraction_median"] == 1.0
    assert diagnostics["screened_volume_per_atom_median"] == pytest.approx(
        15.273025575250852
    )
    assert diagnostics["screened_volume_per_atom_mean"] == pytest.approx(
        15.329090265523147
    )
    assert diagnostics["retained_volume_per_atom_mean"] == pytest.approx(
        14.330610730498508
    )
    assert [row["candidate_id"] for row in diagnostics["examples"]] == [
        "candidate_0009",
        "candidate_0801",
    ]


def test_panel_f_draws_provenance_backed_structure_diagnostics() -> None:
    _require_private_outputs(
        merged.FORWARD_DESIGN_DEFAULT,
        merged.INVERSE_DESIGN_DEFAULT,
        merged.INVERSE_ARCHIVE_DEFAULT,
    )
    forward, inverse, priority, _ = merged.load_design_queues()
    archive = merged.INVERSE_ARCHIVE_DEFAULT
    fig, ax = plt.subplots(figsize=(4, 3))
    record = merged.panel_f(
        ax,
        forward,
        inverse,
        inverse_priority_mask=priority,
        diagnostic_target_retention=0.975,
        diagnostic_archive=archive,
    )
    visible_text = "\n".join(
        item.get_text() for item in fig.findobj(Text) if item.get_visible()
    )
    rasterized_scatter = [
        item for item in fig.findobj(PathCollection) if item.get_rasterized()
    ]
    plt.close(fig)

    assert record["diagnostics"]["screened_n"] == 61
    assert record["diagnostics"]["thumbnail_count"] == 2
    assert record["diagnostics"]["graphic_inset"]["queue_points"] == 1081
    assert record["diagnostics"]["graphic_inset"]["screened_points"] == 61
    assert record["diagnostics"]["graphic_inset"]["decision_boundary"] is True
    assert r"\eta_{\mathrm{site}}" in visible_text
    assert r"V/N" in visible_text
    assert "Site fraction" not in visible_text
    assert "Atomic volume" not in visible_text
    assert "P1" not in visible_text
    assert "C2/m" not in visible_text
    assert "61/61" not in visible_text
    assert rasterized_scatter == []


def test_structure_thumbnail_uses_a_2_by_2_supercell_with_smaller_atoms() -> None:
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(4.0),
        ["Ir", "Os"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, projection="3d")
    record = merged._draw_structure_thumbnail(ax, structure)
    plt.close(fig)

    assert record["supercell"] == [2, 2, 1]
    assert record["rendered_atoms"] == 8
    assert record["maximum_marker_area_pt2"] <= 6.0
    assert record["view"] == {"elevation_deg": 40.0, "azimuth_deg": -55.0}
    assert record["element_colours"] == {"Ir": merged.BLU, "Os": merged.ORA}
    assert record["display_dimensions"] == [2.0, 2.0, 1.0]
    assert record["zoom"] == pytest.approx(1.28)
    assert record["lattice_line"] == {
        "colour": "#989A9D",
        "width_pt": 0.26,
        "alpha": 0.58,
    }
    assert [position[0] for position in merged.STRUCTURE_INSET_POSITIONS] == [
        0.450,
        0.450,
    ]


def test_load_design_queues_records_inverse_score_coverage(tmp_path):
    verdicts = {
        "rung_L1_verdict": ["pass", "pass"],
        "rung_L2_verdict": ["pass", "pass"],
        "rung_L3_verdict": ["pass", "pass"],
        "rung_L4_verdict": ["pass", "reject"],
    }
    forward = pd.DataFrame(
        {
            "fit_valid": [True, True],
            "bulk_modulus_gpa": [250.0, 300.0],
            "made": [True, False],
            "synthesis_score": [-1.0, 1.0],
            **verdicts,
        }
    )
    inverse = pd.DataFrame(
        {
            "fit_valid": [True, True],
            "clamped_bulk_modulus_proxy_gpa": [350.0, 450.0],
            "synthesis_score": [-2.0, 2.0],
            "synthesis_n_observed": [2, 6],
            "synthesis_all_observed": [False, True],
            **verdicts,
        }
    )
    forward_path = tmp_path / "forward.parquet"
    inverse_path = tmp_path / "inverse.parquet"
    forward.to_parquet(forward_path)
    inverse.to_parquet(inverse_path)

    loaded_forward, loaded_inverse, priority, metadata = merged.load_design_queues(
        forward_path,
        inverse_path,
    )

    assert len(loaded_forward) == 2
    assert len(loaded_inverse) == 2
    assert priority.tolist() == [False, True]
    assert metadata["inverse_score_complete_n"] == 1
    assert metadata["inverse_score_observed_terms"] == {"2": 1, "6": 1}


def test_physical_state_panel_can_be_rendered_without_main_text_letter():
    fig, ax = plt.subplots(figsize=(4, 3))
    record = merged.panel_physical_states_si(
        ax,
        merged.DATA / "fig6_threeaxis.csv",
        panel_letter=None,
    )
    _assert_axis_labels_start_with_capital(fig)
    plt.close(fig)

    assert record["n_structures"] == 26600


def test_reader_facing_panels_use_pris_synthesis_score_abbreviation(monkeypatch):
    binary_dir = merged.ROOT / "outputs/20260821_pu_synthesizability/analysis_v1"
    independent_dir = (
        merged.ROOT / "outputs/20260822_pu_formula_scores/independent_choices_v1"
    )
    _require_private_outputs(
        binary_dir / "rung_summary.csv",
        binary_dir / "distance_cutoff_summary.csv",
        independent_dir / "independent_frontier.csv",
        merged.EHULL_DEFAULT,
        merged.FULL_POOL_DEFAULT / "score_deciles.csv",
        merged.FORMULA_NPZ_DEFAULT,
        merged.F3_DEFAULT,
    )
    monkeypatch.syspath_prepend(
        str(merged.ROOT / "experiments/pu_synthesizability_20260821")
    )
    cdata, _, ehull, _ = merged._load_c(
        binary_dir,
        independent_dir,
        merged.EHULL_DEFAULT,
    )
    ddata = merged.load_d_series(merged.FULL_POOL_DEFAULT, merged.FORMULA_NPZ_DEFAULT)
    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(2, 3)
    merged.panel_a(fig.add_subplot(grid[0, 0]))
    merged.panel_b(fig.add_subplot(grid[0, 1]))
    merged.panel_c(fig.add_subplot(grid[0, 2]), cdata, ehull)
    merged.panel_d(fig.add_subplot(grid[1, 0]), ddata)
    merged.panel_e(fig.add_subplot(grid[1, 1]), merged.F3_DEFAULT)
    visible_text = "\n".join(
        text.get_text() for text in fig.findobj(Text) if text.get_visible()
    )
    _assert_axis_labels_start_with_capital(fig)
    plt.close(fig)

    assert "PSS" in visible_text
    assert "MatterSim hull energy" in visible_text
    assert "DFT hull energy" in visible_text
    assert "MatterSim basin–hull" not in visible_text
    assert "MatterSim proxy" not in visible_text
    assert "Hull energy" not in visible_text
    assert "Synthesis score" not in visible_text
    assert "S_syn" not in visible_text


def test_main_caption_draft_only_describes_content_and_reading():
    caption = merged.CAPTION_DRAFT
    flat_caption = " ".join(caption.split())

    for panel in "abcdef":
        assert f"**{panel}**," in caption
    assert "threshold on the MatterSim-computed hull energy" in flat_caption
    assert "DFT hull energy" in flat_caption
    assert not re.search(r"\d+\.\d+%", caption)
    for result_phrase in (
        "compared with",
        "a gain of",
        "while retaining",
        "rather than",
        "the latter is a proxy",
    ):
        assert result_phrase not in caption.lower()


def test_mathtext_labels_render_without_cursive_font_fallback(
    tmp_path, caplog, monkeypatch, recwarn
):
    _require_private_outputs(merged.F3_DEFAULT)
    monkeypatch.syspath_prepend(
        str(merged.ROOT / "experiments/pu_synthesizability_20260821")
    )
    merged.style()
    fig, ax = plt.subplots(figsize=(4, 3))
    merged.panel_e(ax, merged.F3_DEFAULT)
    fig.savefig(tmp_path / "panel_e.pdf")
    plt.close(fig)

    assert "Font family ['cursive'] not found" not in caplog.text
    assert not [warning for warning in recwarn if "py23 module" in str(warning.message)]
