"""Regression tests for the requested August 29 main-figure polish."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
for search_path in (
    ROOT,
    ROOT / "src",
    ROOT / "experiments" / "pu_synthesizability_20260821",
):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_translucent_hued_bars(
    bars,
    *,
    min_face_alpha: float = 0.0,
    max_face_alpha: float = 0.40,
) -> None:
    assert bars
    for bar in bars:
        face = bar.get_facecolor()
        edge = bar.get_edgecolor()
        assert bar.get_alpha() is None
        assert min_face_alpha <= face[3] <= max_face_alpha
        assert edge[3] >= 0.99
        assert np.allclose(face[:3], edge[:3])
        assert bar.get_linewidth() >= 0.80


def _assert_opaque_light_hued_bars(bars) -> None:
    assert bars
    for bar in bars:
        face = bar.get_facecolor()
        edge = bar.get_edgecolor()
        assert bar.get_alpha() == pytest.approx(1.0)
        assert face[3] == pytest.approx(1.0)
        assert edge[3] == pytest.approx(1.0)
        assert np.all(np.asarray(face[:3]) >= np.asarray(edge[:3]))
        assert not np.allclose(face[:3], edge[:3])
        assert bar.get_linewidth() >= 0.80


def test_fig1_points_are_vector_and_requested_labels_clear_lines() -> None:
    module = _load("paper_figs_aug29", ROOT / "src" / "paper_figs.py")
    captured = {}
    module.save = lambda fig, name: captured.update(fig=fig, name=name)
    module.fig1()
    fig = captured["fig"]
    fig.canvas.draw()

    rulespace = fig.axes[1]
    dense_points, surviving_laws = rulespace.collections[:2]
    assert dense_points.get_rasterized() is False
    assert dense_points.get_alpha() is None
    assert dense_points.get_linewidths()[0] >= 0.15
    assert dense_points.get_facecolors()[:, 3].max() <= 0.30
    assert dense_points.get_edgecolors()[:, 3].min() >= 0.55
    assert surviving_laws.get_alpha() is None
    assert surviving_laws.get_linewidths()[0] >= 1.25
    assert surviving_laws.get_facecolors()[0, 3] <= 0.40
    assert surviving_laws.get_edgecolors()[0, 3] == pytest.approx(1.0)

    catalogue = fig.axes[3]
    set_one_prime = next(
        text for text in catalogue.texts if text.get_text() == "Set 1$'$"
    )
    conditional = next(
        text for text in catalogue.texts if text.get_text() == "two-sided\nwindow"
    )
    header_gap_px = (
        set_one_prime.get_window_extent().y0
        - conditional.get_window_extent().y1
    )
    assert header_gap_px >= fig.dpi / 72.0

    trajectory, inset = fig.axes[4], fig.axes[5]
    reference_y = trajectory.transData.transform((0.0, 0.0))[1]
    tick_boxes = [label.get_window_extent() for label in inset.get_xticklabels()]
    assert all(not (box.y0 <= reference_y <= box.y1) for box in tick_boxes)
    clearance_px = min(
        min(abs(reference_y - box.y0), abs(reference_y - box.y1))
        for box in tick_boxes
    )
    assert clearance_px >= fig.dpi / 72.0
    assert trajectory.get_position().height >= 3.30 / 20.35
    plt.close(fig)


def test_fig2b_bars_overlap_left_over_right_with_opaque_light_fills() -> None:
    module = _load("paper_figs_aug29_fig2", ROOT / "src" / "paper_figs.py")
    captured = {}
    module.save = lambda fig, name: captured.update(fig=fig, name=name)
    module.fig2()
    fig = captured["fig"]
    bars = fig.axes[1].patches

    assert len(bars) == 30
    _assert_opaque_light_hued_bars(bars)

    first_class = [bars[index * 6] for index in range(5)]
    first_class.sort(key=lambda bar: bar.get_x() + bar.get_width() / 2)
    centres = np.array(
        [bar.get_x() + bar.get_width() / 2 for bar in first_class]
    )
    widths = np.array([bar.get_width() for bar in first_class])
    assert np.all(np.diff(centres) < widths[:-1])
    assert centres[-1] - centres[0] + widths[0] <= 0.74

    containers = fig.axes[1].containers
    assert len(containers) == 5
    for class_index in range(6):
        class_bars = [container.patches[class_index] for container in containers]
        class_bars.sort(key=lambda bar: bar.get_x() + bar.get_width() / 2)
        zorders = np.array([bar.get_zorder() for bar in class_bars])
        assert np.all(np.diff(zorders) < 0.0)
    plt.close(fig)


def test_all_main_figure_bars_have_opaque_hued_edges_and_translucent_fills() -> None:
    paper = _load("paper_figs_aug29_all_bars", ROOT / "src" / "paper_figs.py")

    captured = {}
    paper.save = lambda fig, name: captured.update(fig=fig, name=name)
    paper.fig1()
    fig1 = captured["fig"]
    _assert_translucent_hued_bars(fig1.axes[2].patches)
    plt.close(fig1)

    captured.clear()
    paper.fig2()
    fig2 = captured["fig"]
    _assert_opaque_light_hued_bars(fig2.axes[1].patches)
    _assert_translucent_hued_bars(fig2.axes[2].patches)
    plt.close(fig2)

    from experiments.pu_synthesizability_20260821 import (
        plot_merged_fig45_nature as merged,
    )

    fig4, ax4 = plt.subplots(figsize=(4, 3))
    merged.panel_a(ax4)
    _assert_translucent_hued_bars(ax4.patches)
    plt.close(fig4)

    fig5 = _load("fig5_aug29_all_bars", ROOT / "src" / "fig6_deployment.py")
    frozen = json.loads(
        (ROOT / "paper" / "data" / "fig7_wrong_site.json").read_text(
            encoding="utf-8"
        )
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    fig5.panel_e(fig, ax, frozen)
    bar_axes = [child for child in ax.child_axes if child.patches]
    assert len(bar_axes) == 2
    for bar_axis in bar_axes:
        _assert_translucent_hued_bars(bar_axis.patches)
    plt.close(fig)


def test_fig2d_uses_law_colours_and_adds_transfer_impact_inset() -> None:
    module = _load("paper_figs_aug29_fig2d", ROOT / "src" / "paper_figs.py")
    captured = {}
    module.save = lambda fig, name: captured.update(fig=fig, name=name)
    module.fig2()
    fig = captured["fig"]
    panel = fig.axes[3]

    row_colours = {
        tuple(np.round(collection.get_edgecolors()[0, :3], 4))
        for collection in panel.collections
        if (
            len(collection.get_edgecolors())
            and len(collection.get_sizes())
            and collection.get_sizes()[0] >= 20
        )
    }
    assert len(row_colours) >= 5
    colours_used_elsewhere = {
        tuple(np.round(to_rgba(colour)[:3], 4))
        for colour in (
            module.BLU,
            module.ORA,
            module.RED,
            "#0A5A3C",
            "#535557",
        )
    }
    assert row_colours <= colours_used_elsewhere

    inset = next(
        child
        for child in panel.child_axes
        if child.get_label() == "Fig2d-transfer-impact"
    )
    assert "Experimental" in inset.get_xlabel()
    assert "Damaged" in inset.get_ylabel()
    point_collections = [
        collection
        for collection in inset.collections
        if len(collection.get_offsets()) == 1
    ]
    assert len(point_collections) == 8
    observed_points = np.array(
        [collection.get_offsets()[0] for collection in point_collections],
        dtype=float,
    )
    expected_points = np.array(
        [
            [0.189, 1.633],
            [0.170, 1.190],
            [0.117, 0.114],
            [0.635, 0.535],
            [0.249, 5.011],
            [0.326, 7.807],
            [2.878, 3.196],
            [0.000, 0.000],
        ]
    )
    assert np.allclose(observed_points, expected_points, atol=0.001)
    assert all(collection.get_clip_on() is False for collection in point_collections)
    # Marker shape carries the law identity here as it does in the main panel,
    # so the inset points are deliberately unlabelled.
    assert [text for text in inset.texts if text.get_text().strip()] == []
    assert inset.xaxis.label.get_fontsize() > 6.2
    assert inset.yaxis.label.get_fontsize() > 5.6
    assert any(line.get_linestyle() != "None" for line in inset.lines)
    assert inset.get_xlim()[0] <= 0.0
    assert inset.get_ylim()[0] <= 0.0
    assert 19.5 <= inset.get_xlim()[1] <= 21.0
    assert 19.5 <= inset.get_ylim()[1] <= 21.0
    assert np.allclose(inset.get_xticks(), [0.0, 10.0, 20.0])
    assert np.allclose(inset.get_yticks(), [0.0, 10.0, 20.0])
    diagonal = next(line for line in inset.lines if line.get_linestyle() != "None")
    assert np.allclose(diagonal.get_xdata(), [0.0, 20.0])
    assert np.allclose(diagonal.get_ydata(), [0.0, 20.0])
    inset_colours = {
        tuple(np.round(colour[:3], 4))
        for collection in inset.collections
        for colour in collection.get_edgecolors()
    }
    assert inset_colours <= colours_used_elsewhere
    plt.close(fig)


def test_fig4_primary_artist_contract_is_shared_across_panels() -> None:
    from experiments.pu_synthesizability_20260821 import (
        plot_merged_fig45_nature as merged,
    )

    assert merged.F4_DATA_LINE_WIDTH == pytest.approx(1.20)
    assert merged.F4_DATA_LINE_ALPHA == pytest.approx(0.82)
    assert merged.F4_MARKER_SIZE_PT == pytest.approx(5.30)
    assert merged.F4_MARKER_AREA_PT2 == pytest.approx(5.30**2)
    assert merged.F4_MARKER_FACE_ALPHA == pytest.approx(0.62)
    assert merged.F4_MARKER_EDGE_WIDTH == pytest.approx(0.65)
    assert merged.F4_BAND_ALPHA == pytest.approx(0.10)

    merged.style()
    ddata = merged.load_d_series(merged.FULL_POOL_DEFAULT, merged.FORMULA_NPZ_DEFAULT)
    fig = plt.figure(figsize=(8, 4))
    outer = fig.add_subplot(121)
    axes_before = set(fig.axes)
    merged.panel_d(outer, ddata)
    d_axes = [axis for axis in fig.axes if axis not in axes_before]
    e_axis = fig.add_subplot(122)
    merged.panel_e(e_axis, merged.F3_DEFAULT)

    series = [
        line
        for axis in [*d_axes, e_axis]
        for line in axis.lines
        if line.get_gid() == "f4-primary-series"
    ]
    assert len(series) >= 8
    for line in series:
        assert line.get_linewidth() == pytest.approx(merged.F4_DATA_LINE_WIDTH)
        assert to_rgba(line.get_color())[3] == pytest.approx(
            merged.F4_DATA_LINE_ALPHA
        )
        assert line.get_markersize() == pytest.approx(merged.F4_MARKER_SIZE_PT)
        assert line.get_markeredgewidth() == pytest.approx(
            merged.F4_MARKER_EDGE_WIDTH
        )
        assert to_rgba(line.get_markerfacecolor())[3] == pytest.approx(
            merged.F4_MARKER_FACE_ALPHA
        )

    bands = [
        collection
        for axis in [*d_axes, e_axis]
        for collection in axis.collections
        if collection.get_gid() == "f4-confidence-band"
    ]
    assert len(bands) >= 4
    assert all(
        collection.get_alpha() == pytest.approx(merged.F4_BAND_ALPHA)
        for collection in bands
    )
    plt.close(fig)


def test_fig4f_uses_the_shared_contract_and_light_structure_frames(monkeypatch) -> None:
    from experiments.pu_synthesizability_20260821 import (
        plot_merged_fig45_nature as merged,
    )

    forward = pd.DataFrame(
        {
            "made": [True] * 4 + [False] * 4,
            "synthesis_score": [-3.0, -1.0, 1.0, 3.0, -4.0, -2.0, 0.0, 2.0],
            "rung_L1_verdict": ["pass"] * 8,
            "rung_L2_verdict": ["pass"] * 6 + ["reject", "pass"],
            "rung_L3_verdict": ["pass"] * 5 + ["reject", "reject", "pass"],
            "rung_L4_verdict": [
                "pass", "reject", "pass", "pass",
                "reject", "reject", "pass", "pass",
            ],
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
    dft_frontier = pd.DataFrame(
        {
            "inverse_priority_retention": [0.50, 0.75, 1.00],
            "inverse_queue_reduction": [0.45, 0.25, 0.00],
        }
    )
    monkeypatch.setattr(
        merged,
        "_dft_design_frontier",
        lambda forward, inverse, target_retentions, calibration_mask: (
            dft_frontier,
            {"legend": "PSS, DFT ≥376 GPa"},
        ),
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    merged.panel_f(
        ax,
        forward,
        inverse,
        inverse_priority_mask=[False, True, True, True],
        target_retentions=(0.50, 0.75, 1.00),
    )
    primary_lines = [line for line in ax.lines if line.get_gid() == "f4-primary-series"]
    primary_markers = [
        collection
        for collection in ax.collections
        if collection.get_gid() == "f4-primary-marker"
    ]
    assert primary_lines
    assert primary_markers
    assert all(
        line.get_linewidth() == pytest.approx(merged.F4_DATA_LINE_WIDTH)
        for line in primary_lines
    )
    assert all(
        collection.get_linewidths()[0]
        == pytest.approx(merged.F4_MARKER_EDGE_WIDTH)
        for collection in primary_markers
    )
    # Every primary marker shares one area except the Set 4 star, which is
    # enlarged so its thin concave outline carries comparable visual weight.
    assert all(
        collection.get_sizes()[0]
        in (
            pytest.approx(merged.F4_MARKER_AREA_PT2),
            pytest.approx(merged.F4_STAR_AREA_PT2),
        )
        for collection in primary_markers
    )
    star_markers = [
        collection
        for collection in primary_markers
        if collection.get_sizes()[0] == pytest.approx(merged.F4_STAR_AREA_PT2)
    ]
    assert len(star_markers) == 1
    dft_line = next(
        line for line in ax.lines if line.get_gid() == "f4-dft-reference-series"
    )
    assert np.allclose(
        to_rgba(dft_line.get_color()),
        to_rgba(merged.EHULL, merged.F4_DATA_LINE_ALPHA),
    )
    assert dft_line.get_linewidth() == pytest.approx(merged.F4_DATA_LINE_WIDTH)
    assert dft_line.get_linestyle() == "--"
    assert "DFT" in ax.get_ylabel()
    assert "\n" not in ax.get_ylabel()
    plt.close(fig)

    examples = [
        {"site_fraction": 0.35, "volume_per_atom": 12.0},
        {"site_fraction": 0.85, "volume_per_atom": 15.0},
    ]
    monkeypatch.setattr(
        merged,
        "_load_diagnostic_structures",
        lambda archive_path, requested: [(object(), row) for row in examples],
    )
    monkeypatch.setattr(
        merged,
        "_draw_structure_thumbnail",
        lambda axis, structure: {},
    )
    diagnostic_inverse = pd.DataFrame(
        {
            "formula_syn_wyckoff_econ_001": [0.2, 0.5, 0.8, 1.0],
            "formula_syn_vol_per_atom": [10.0, 12.0, 14.0, 16.0],
        }
    )
    diagnostic_inverse["synthesis_score"] = (
        -2.0
        + 0.7 * diagnostic_inverse["formula_syn_wyckoff_econ_001"]
        + 0.2 * diagnostic_inverse["formula_syn_vol_per_atom"]
    )
    diagnostics = {
        "examples": examples,
        "cutoff": 0.8,
        "priority_retained_n": 3,
        "priority_total_n": 4,
        "screened_n": 2,
        "retained_n": 2,
    }
    fig, ax = plt.subplots(figsize=(4, 3))
    merged.draw_inverse_design_diagnostics(
        ax,
        diagnostics,
        archive_path=Path("unused.zip"),
        inverse=diagnostic_inverse,
    )
    assert merged.F4_DESCRIPTOR_POINT_AREA_PT2 >= 9.0
    assert merged.F4_DESCRIPTOR_FACE_ALPHA == pytest.approx(0.14)
    assert merged.F4_DESCRIPTOR_EDGE_ALPHA == pytest.approx(0.45)
    assert merged.F4_DESCRIPTOR_EDGE_WIDTH == pytest.approx(0.55)
    assert merged.F4_DESCRIPTOR_TICK_PAD == pytest.approx(0.0)
    descriptor = next(
        child for child in ax.child_axes if child.get_label() == "f4f-descriptor-map"
    )
    cloud = descriptor.collections[:2]
    assert len(cloud) == 2
    for collection in cloud:
        assert collection.get_sizes()[0] >= 9.0
        assert collection.get_facecolors()[0, 3] == pytest.approx(0.14)
        assert collection.get_edgecolors()[0, 3] == pytest.approx(0.45)
        assert collection.get_linewidths()[0] == pytest.approx(0.55)
    assert descriptor.yaxis.majorTicks[0].get_pad() == pytest.approx(0.0)
    ax.set_ylim(-2, 70)
    ax.set_yticks(np.arange(0, 71, 10))
    ax.set_ylabel("DFT validation queue reduction (%)", labelpad=3)
    fig.canvas.draw()
    outer_tick_boxes = [
        label.get_window_extent()
        for label in ax.get_yticklabels()
        if label.get_visible()
    ]
    inset_tick_boxes = [
        label.get_window_extent()
        for label in descriptor.get_yticklabels()
        if label.get_visible()
    ]
    assert all(
        not outer.overlaps(inner)
        for outer in outer_tick_boxes
        for inner in inset_tick_boxes
    )
    inset_left = descriptor.get_window_extent().x0
    assert all(inner.x1 <= inset_left for inner in inset_tick_boxes)
    outer_right = max(outer.x1 for outer in outer_tick_boxes)
    assert min(inner.x0 for inner in inset_tick_boxes) - outer_right >= 8.0
    frames = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    assert len(frames) == 2
    assert all(
        frame.get_linewidth() == pytest.approx(merged.F4_STRUCTURE_FRAME_WIDTH)
        for frame in frames
    )
    for frame, colour in zip(frames, (merged.RED, merged.BLU), strict=True):
        frame_rgb = np.asarray(frame.get_edgecolor()[:3])
        dark_rgb = np.asarray(merged.deep_color(colour)[:3])
        assert frame_rgb.mean() >= dark_rgb.mean() + 0.20
    plt.close(fig)


def test_fig2d_and_fig4f_explanations_match_the_revised_visuals() -> None:
    for path in (ROOT / "tex-submission" / "body.tex", ROOT / "tex" / "body.tex"):
        text = path.read_text(encoding="utf-8")
        flattened = " ".join(text.split())
        assert "common 0--20\\% scale" in text
        assert "blue and red points" in text
        assert "correspondingly labelled structure thumbnails" in flattened
        assert "376\\,GPa" in text


def test_fig5d_points_have_lighter_edges_and_translucent_interiors() -> None:
    module = _load("fig5_aug29_d", ROOT / "src" / "fig6_deployment.py")
    ladder = pd.read_csv(ROOT / "paper" / "data" / "fig7_mattergen_ladder_energy.csv")
    fig, ax = plt.subplots(figsize=(4, 3))
    module.panel_c(fig, ax, ladder)

    assert ax.collections
    for collection in ax.collections:
        assert collection.get_alpha() is None
        assert collection.get_linewidths()[0] >= 0.80
        assert collection.get_facecolors()[0, 3] <= 0.40
        assert collection.get_edgecolors()[0, 3] == pytest.approx(0.40)
    plt.close(fig)


def test_fig5b_and_fig5f_use_the_fig5e_dark_blue() -> None:
    module = _load("fig5_aug29_bf", ROOT / "src" / "fig6_deployment.py")
    frozen = json.loads(
        (ROOT / "paper" / "data" / "fig7_wrong_site.json").read_text(
            encoding="utf-8"
        )
    )
    module.wrong_site = frozen

    fig, ax = plt.subplots(figsize=(4, 3))
    module.panel_b_ordering(fig, ax)
    gnome_tick = next(label for label in ax.get_yticklabels() if label.get_text() == "GNoME")
    assert np.allclose(to_rgba(gnome_tick.get_color())[:3], to_rgba(module.BLU)[:3])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    module.panel_f(fig, ax)
    all_laws = next(
        text for text in ax.texts if text.get_text() == "all eight laws, reference code"
    )
    assert np.allclose(to_rgba(all_laws.get_color())[:3], to_rgba(module.BLU)[:3])
    plt.close(fig)


def test_fig5e_structure_block_is_materially_larger() -> None:
    module = _load("fig5_aug29_e", ROOT / "src" / "fig6_deployment.py")
    frozen = json.loads(
        (ROOT / "paper" / "data" / "fig7_wrong_site.json").read_text(
            encoding="utf-8"
        )
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    module.panel_e(fig, ax, frozen)
    fig.canvas.draw()

    structure_axes = [ax.child_axes[index] for index in (0, 1, 3, 4)]
    parent_box = ax.get_position()
    relative_widths = [
        child.get_position().width / parent_box.width for child in structure_axes
    ]
    assert min(relative_widths) >= 0.33

    points, _, corners = module._project_structure(
        frozen["exemplar"], "parent_species"
    )
    xmin = min(points[:, 0].min(), corners[:, 0].min())
    xmax = max(points[:, 0].max(), corners[:, 0].max())
    dx = xmax - xmin
    xlim = structure_axes[0].get_xlim()
    assert (xmin - xlim[0]) / dx <= 0.18
    assert (xlim[1] - xmax) / dx <= 0.18
    plt.close(fig)
