"""Regression tests for the requested August 30 main-figure revisions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import patheffects
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


def _path_vertices(collection) -> np.ndarray:
    assert len(collection.get_paths()) == 1
    return np.asarray(collection.get_paths()[0].vertices)


def test_fig2d_distinguishes_law5_and_law7_with_consistent_markers() -> None:
    module = _load("paper_figs_aug30_fig2d", ROOT / "src" / "paper_figs.py")
    captured = {}
    module.save = lambda fig, name: captured.update(fig=fig, name=name)
    module.fig2()
    fig = captured["fig"]
    panel = fig.axes[3]
    inset = next(
        child for child in panel.child_axes
        if child.get_label() == "Fig2d-transfer-impact"
    )

    main_law5 = next(
        collection for collection in panel.collections
        if collection.get_gid() == "fig2d-fixed-law-5"
    )
    main_law7 = next(
        collection for collection in panel.collections
        if collection.get_gid() == "fig2d-fixed-law-7"
    )
    inset_law5 = next(
        collection for collection in inset.collections
        if collection.get_gid() == "fig2d-impact-law-5"
    )
    inset_law7 = next(
        collection for collection in inset.collections
        if collection.get_gid() == "fig2d-impact-law-7"
    )

    assert np.allclose(_path_vertices(main_law5), _path_vertices(inset_law5))
    assert np.allclose(_path_vertices(main_law7), _path_vertices(inset_law7))
    law5_vertices = _path_vertices(main_law5)
    law7_vertices = _path_vertices(main_law7)
    assert (
        law5_vertices.shape != law7_vertices.shape
        or not np.allclose(law5_vertices, law7_vertices)
    )
    legend_note = next(text for text in panel.texts if "fixed before testing" in text.get_text())
    assert "●" not in legend_note.get_text()
    assert "○" not in legend_note.get_text()
    plt.close(fig)


def test_fig2b_bar_faces_are_a_lighter_opaque_mix_of_the_outline() -> None:
    module = _load("paper_figs_aug30_fig2b_lighter", ROOT / "src" / "paper_figs.py")
    captured = {}
    module.save = lambda fig, name: captured.update(fig=fig, name=name)
    module.fig2()
    bars = captured["fig"].axes[1].patches

    assert len(bars) == 30
    for bar in bars:
        face = np.asarray(bar.get_facecolor()[:3])
        edge = np.asarray(bar.get_edgecolor()[:3])
        assert np.allclose(face, 0.55 * edge + 0.45, atol=1e-6)
        assert bar.get_alpha() == pytest.approx(1.0)
    plt.close(captured["fig"])


def test_fig2d_inset_carries_no_per_point_labels() -> None:
    """The inset identifies each law by marker shape, not by a printed number.

    Five of the eight points sit within 1.7 percentage points of the origin,
    where the numbers and their leader lines cost more than they returned.
    """
    module = _load("paper_figs_aug30_fig2d_labels", ROOT / "src" / "paper_figs.py")
    captured = {}
    module.save = lambda fig, name: captured.update(fig=fig, name=name)
    module.fig2()
    panel = captured["fig"].axes[3]
    inset = next(
        child for child in panel.child_axes
        if child.get_label() == "Fig2d-transfer-impact"
    )

    assert [text for text in inset.texts if text.get_text().strip()] == []
    assert len(inset.collections) == 8
    plt.close(captured["fig"])


def test_fig4f_uses_fig4b_blue_and_labels_structures_without_overlap(monkeypatch) -> None:
    from experiments.pu_synthesizability_20260821 import (
        plot_merged_fig45_nature as merged,
    )

    examples = [
        {
            "role": "screened",
            "site_fraction": 0.35,
            "volume_per_atom": 12.0,
            "space_group": "P1",
        },
        {
            "role": "retained high property",
            "site_fraction": 0.85,
            "volume_per_atom": 15.0,
            "space_group": "C2/m",
        },
    ]
    monkeypatch.setattr(
        merged,
        "_load_diagnostic_structures",
        lambda archive_path, requested: [(object(), row) for row in examples],
    )
    monkeypatch.setattr(merged, "_draw_structure_thumbnail", lambda axis, structure: {})
    inverse = pd.DataFrame(
        {
            "formula_syn_wyckoff_econ_001": [0.2, 0.5, 0.8, 1.0],
            "formula_syn_vol_per_atom": [10.0, 12.0, 14.0, 16.0],
        }
    )
    inverse["synthesis_score"] = (
        -2.0
        + 0.7 * inverse["formula_syn_wyckoff_econ_001"]
        + 0.2 * inverse["formula_syn_vol_per_atom"]
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
        ax, diagnostics, archive_path=Path("unused.zip"), inverse=inverse
    )
    descriptor = next(
        child for child in ax.child_axes if child.get_label() == "f4f-descriptor-map"
    )
    retained_cloud = descriptor.collections[0]
    fig4b_blue = plt.get_cmap("palmatrix")(1.0)
    assert np.allclose(to_rgba(merged.F4_DIAGNOSTIC_BLUE), fig4b_blue)
    assert np.allclose(retained_cloud.get_facecolors()[0, :3], fig4b_blue[:3])
    assert np.allclose(retained_cloud.get_edgecolors()[0, :3], fig4b_blue[:3])

    labels = [text for text in ax.texts if text.get_text() in {"Screened", "Retained"}]
    assert {text.get_text() for text in labels} == {"Screened", "Retained"}
    assert all(text.get_rotation() == pytest.approx(90.0) for text in labels)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frames = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    for text, frame in zip(labels, frames, strict=True):
        label_box = text.get_window_extent(renderer)
        frame_box = frame.get_window_extent(renderer)
        left_blank_edge = frame_box.x0 + (frame_box.width - frame_box.height) / 2
        assert frame_box.contains(label_box.x0, label_box.y0)
        assert frame_box.contains(label_box.x1, label_box.y1)
        assert label_box.x1 <= left_blank_edge
    plt.close(fig)


def test_fig5b_points_have_translucent_faces_thick_edges_and_left_clearance() -> None:
    module = _load("fig5_aug30_b", ROOT / "src" / "fig6_deployment.py")
    d7 = json.loads(
        (ROOT / "paper" / "data" / "fig7_gnome_d7.json").read_text(
            encoding="utf-8"
        )
    )

    fig, (upper, lower) = plt.subplots(2, 1, figsize=(4, 4))
    module.panel_d(fig, upper, d7)
    module.panel_b_ordering(fig, lower)

    upper_points = [line for line in upper.lines if line.get_marker() == "o"]
    assert len(upper_points) == len(d7["by_sg"])
    for point in upper_points:
        assert point.get_alpha() is None
        assert to_rgba(point.get_markerfacecolor())[3] <= 0.30
        assert to_rgba(point.get_markeredgecolor())[3] >= 0.85
        assert point.get_markeredgewidth() >= 0.80

    assert len(lower.collections) == 2
    for collection in lower.collections:
        assert collection.get_alpha() is None
        assert collection.get_facecolors()[0, 3] <= 0.30
        assert collection.get_edgecolors()[0, 3] >= 0.85
        assert collection.get_linewidths()[0] >= 0.80
    assert lower.get_xlim()[0] < 1.0

    gnome = next(
        collection
        for collection in lower.collections
        if np.mean(collection.get_offsets()[:, 1]) > 0.5
    )
    assert np.count_nonzero(np.isclose(gnome.get_offsets()[:, 0], 1.0)) == 6
    fig.canvas.draw()
    leftmost_x_px = lower.transData.transform((1.0, 1.0))[0]
    marker_radius_px = (
        np.sqrt(gnome.get_sizes()[0]) / 2 + gnome.get_linewidths()[0] / 2
    ) * fig.dpi / 72.0
    assert leftmost_x_px - marker_radius_px > lower.bbox.x0
    plt.close(fig)


def test_fig5d_removes_005_reference_and_uses_white_failure_medians() -> None:
    module = _load("fig5_aug30_d_contrast", ROOT / "src" / "fig6_deployment.py")
    ladder = pd.read_csv(ROOT / "paper" / "data" / "fig7_mattergen_ladder_energy.csv")
    fig, ax = plt.subplots(figsize=(4, 3))
    module.panel_c(fig, ax, ladder)

    def vertical_line_at(value: float):
        return next(
            line for line in ax.lines
            if len(line.get_xdata()) == 2
            and np.allclose(np.asarray(line.get_xdata(), dtype=float), value)
        )

    displacement = vertical_line_at(6.05)
    assert displacement.get_linewidth() >= 1.1
    assert displacement.get_alpha() is None or displacement.get_alpha() >= 0.95
    assert displacement.get_zorder() >= 3
    assert not any(
        len(line.get_xdata()) == 2
        and np.allclose(np.asarray(line.get_xdata(), dtype=float), 0.05)
        for line in ax.lines
    )
    assert all(text.get_text() != "0.05" for text in ax.texts)

    median_lines = [
        line for line in ax.lines if line.get_gid() == "fig5d-failure-median"
    ]
    assert len(median_lines) == 3
    assert all(
        np.allclose(to_rgba(line.get_color()), to_rgba("white"))
        for line in median_lines
    )
    assert all(line.get_linewidth() >= 1.8 for line in median_lines)
    assert all(line.get_zorder() >= 6 for line in median_lines)
    for line in median_lines:
        strokes = [
            effect
            for effect in line.get_path_effects()
            if isinstance(effect, patheffects.Stroke)
        ]
        assert len(strokes) == 1
        stroke = strokes[0]._gc
        edge = to_rgba(stroke["foreground"])
        assert stroke["linewidth"] > line.get_linewidth()
        assert min(edge[:3]) >= 0.78
        assert max(edge[:3]) - min(edge[:3]) <= 0.05
        assert edge[3] <= 0.85
    plt.close(fig)


def test_fig5d_captions_match_white_failure_medians_and_removed_reference() -> None:
    for body_path in (ROOT / "tex-submission" / "body.tex", ROOT / "tex" / "body.tex"):
        body = " ".join(body_path.read_text(encoding="utf-8").split())
        assert (
            "White vertical segments with pale-grey outlines mark failure-group medians"
            in body
        )
        assert "the dashed vertical line gives the same reference as in panel c" not in body
