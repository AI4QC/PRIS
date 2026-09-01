"""Regression tests for the August 24 Fig. 2d/Fig. 5 layout revision."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.text import Annotation


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fig2b_uses_compact_class_labels_at_the_y_tick_font_size() -> None:
    module = _load("paper_figs_aug24_b", ROOT / "src" / "paper_figs.py")
    captured = {}
    module.save = lambda fig, name: captured.update(fig=fig, name=name)
    module.fig2()

    ax = captured["fig"].axes[1]
    xlabels = ax.get_xticklabels()
    ylabels = ax.get_yticklabels()
    assert [label.get_text() for label in xlabels] == [
        "D1", "D2", "D3", "D4", "D5", "All"
    ]
    assert ylabels
    assert all(
        np.isclose(label.get_fontsize(), ylabels[0].get_fontsize())
        for label in xlabels
    )
    plt.close(captured["fig"])


def test_fig2d_uses_the_true_linear_constant_ratio_scale() -> None:
    module = _load("paper_figs_aug24", ROOT / "src" / "paper_figs.py")
    captured = {}
    module.save = lambda fig, name: captured.update(fig=fig, name=name)
    module.fig2()

    fig = captured["fig"]
    ax = fig.axes[3]
    transfer = json.loads(
        (ROOT / "outputs" / "20260815_threshold_transfer" / "transfer.json")
        .read_text(encoding="utf-8")
    )
    d5 = transfer["D5 15.17 (max VM)"]
    d5_ratio = d5["calib_value"] / d5["threshold"]
    plotted_x = np.concatenate(
        [collection.get_offsets()[:, 0] for collection in ax.collections]
    )

    assert ax.get_xlim()[0] <= 0.20
    assert ax.get_xlim()[1] >= 1.20
    assert np.isclose(plotted_x, d5_ratio, atol=1e-5).any()
    assert ax.get_xlabel() == "Re-derived / fixed constant"
    assert not any(
        text.get_text() == "0.25" and text.get_position()[0] < 0.50
        for text in ax.texts
    )

    numeric_labels = {
        "0.745",
        "0.809",
        "1.04",
        "1.07",
        "26.4",
        "3.86",
        "0.75",
    }
    plotted_numbers = [text for text in ax.texts if text.get_text() in numeric_labels]
    assert {text.get_text() for text in plotted_numbers} == numeric_labels
    assert all(float(text.get_position()[1]).is_integer() for text in plotted_numbers)
    assert all(text.get_va() == "center" for text in plotted_numbers)
    ratio_by_label = {
        f"{row['calib_value']:.3g}": row["calib_value"] / row["threshold"]
        for row in transfer.values()
        if f"{row['calib_value']:.3g}" in numeric_labels
    }
    assert all(
        abs(text.get_position()[0] - ratio_by_label[text.get_text()]) >= 0.019
        for text in plotted_numbers
    )

    fig.canvas.draw()
    d5_label = next(text for text in plotted_numbers if text.get_text() == "3.86")
    assert d5_label.get_window_extent().x0 > ax.bbox.x0 + 2.0
    plt.close(fig)


def test_fig5a_directly_labels_every_dataset_with_a_leader_line() -> None:
    module = _load("fig5_aug24_a", ROOT / "src" / "fig6_deployment.py")
    gen = pd.read_csv(ROOT / "paper" / "data" / "fig7_generators.csv")
    fig, ax = plt.subplots()
    module.panel_a(fig, ax, gen)
    fig.canvas.draw()

    expected = {
        "CrystalFormer",
        "DiffCSP",
        "DiffCSP++",
        "MP-20",
        "MatterGen",
        "MiAD",
        "SymmCD",
        "WyFormer-DiffCSP++",
    }
    annotations = {
        text.get_text(): text
        for text in ax.texts
        if isinstance(text, Annotation)
    }
    assert set(annotations) == expected
    assert all(annotation.arrow_patch is not None for annotation in annotations.values())

    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "no imposed symmetry",
        "symmetry imposed",
    ]
    assert legend._ncols == 1
    frame = legend.get_frame()
    assert frame.get_visible()
    assert np.allclose(frame.get_edgecolor(), to_rgba("#B6B8BB"))
    assert frame.get_linewidth() >= 0.5
    legend_anchor = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    assert legend_anchor.x0 >= 0.95
    assert ax.get_xlabel() == "Set 4 satisfaction"
    assert ax.get_ylabel() == "0.7-Å distance-filter\nsatisfaction"
    plt.close(fig)


def test_fig5e_has_a_taller_row_spaced_structures_and_atom_padding() -> None:
    module = _load("fig5_aug24_e", ROOT / "src" / "fig6_deployment.py")
    assert module.ROW_CM[-1] >= 3.40
    assert module.TOP_CM + sum(module.ROW_CM) + module.BOT_CM <= 19.00

    frozen = json.loads(
        (ROOT / "paper" / "data" / "fig7_wrong_site.json").read_text(
            encoding="utf-8"
        )
    )
    fig, ax = plt.subplots()
    module.panel_e(fig, ax, frozen)
    fig.canvas.draw()
    assert len(ax.child_axes) == 6

    (
        top_parent_ax,
        top_damaged_ax,
        _top_rate_ax,
        _bottom_parent_ax,
        _bottom_damaged_ax,
        bottom_rate_ax,
    ) = ax.child_axes
    parent_box = ax.get_position()

    def _relative_box(child):
        box = child.get_position()
        return (
            (box.x0 - parent_box.x0) / parent_box.width,
            (box.y0 - parent_box.y0) / parent_box.height,
            box.width / parent_box.width,
            box.height / parent_box.height,
        )

    parent_rel = _relative_box(top_parent_ax)
    damaged_rel = _relative_box(top_damaged_ax)
    assert parent_rel[0] < 0.0
    assert damaged_rel[0] - (parent_rel[0] + parent_rel[2]) >= 0.025

    points, _, corners = module._project_structure(
        frozen["exemplar"], "parent_species"
    )
    xmin = min(points[:, 0].min(), corners[:, 0].min())
    xmax = max(points[:, 0].max(), corners[:, 0].max())
    ymin = min(points[:, 1].min(), corners[:, 1].min())
    ymax = max(points[:, 1].max(), corners[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin
    xlim, ylim = top_parent_ax.get_xlim(), top_parent_ax.get_ylim()
    assert (xmin - xlim[0]) / dx >= 0.12
    assert (xlim[1] - xmax) / dx >= 0.12
    assert (ymin - ylim[0]) / dy >= 0.09
    assert (ylim[1] - ymax) / dy >= 0.09

    p988 = next(text for text in bottom_rate_ax.texts if text.get_text() == "98.8%")
    assert p988.get_position()[0] > 98.8
    assert p988.get_ha() == "left"
    assert p988.get_window_extent().x1 < bottom_rate_ax.bbox.x1
    plt.close(fig)


def test_fig5_float_precedes_the_d7_charge_assignment_paragraph() -> None:
    body = (ROOT / "tex" / "body.tex").read_text(encoding="utf-8")
    label = body.index(r"\label{fig:deploy}")
    figure_start = body.rfind(r"\begin{figure}", 0, label)
    figure_end = body.index(r"\end{figure}", label) + len(r"\end{figure}")
    paragraph = body.index("Because Law~7 requires no charge assignment")
    assert figure_start >= 0
    assert figure_end < paragraph


def test_fig5_source_declares_its_standalone_export_contract() -> None:
    source = (ROOT / "src" / "fig6_deployment.py").read_text(encoding="utf-8")
    assert '"font.family": "Arial"' in source
    assert '"pdf.fonttype": 42' in source
    assert '"svg.fonttype": "none"' in source
    assert 'fig.savefig(OUT / f"{name}.pdf")' in source
    assert 'fig.savefig(OUT / f"{name}.svg")' in source
    assert 'fig.savefig(OUT / f"{name}.png", dpi=400)' in source


def test_fig2d_caption_no_longer_describes_an_axis_break() -> None:
    body = (ROOT / "tex" / "body.tex").read_text(encoding="utf-8")
    start = body.index(r"\caption{\textbf{Experimental-structure satisfaction")
    end = body.index(r"\label{fig:rules}", start)
    assert "axis break" not in body[start:end]
