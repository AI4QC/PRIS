#!/usr/bin/env python3
"""Create SI-ready figures for the PRIS composition-held-out evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CM = 1 / 2.54
DOUBLE_COLUMN = 18.3 * CM
SET_ORDER = ("Set 1", "Set 1-prime", "Set 2", "Set 3", "Set 4")
SET_LABELS = {
    "Set 1": "Set 1",
    "Set 1-prime": "Set 1′",
    "Set 2": "Set 2",
    "Set 3": "Set 3",
    "Set 4": "Set 4",
}
SET_COLORS = {
    "Set 1": "#005B93",
    "Set 1-prime": "#35A7D8",
    "Set 2": "#9861B0",
    "Set 3": "#D6564C",
    "Set 4": "#0A5A3C",
}
COHORT_STYLE = {
    "heldout_all": {"label": "All held-out", "marker": "D", "color": "#777777", "fill": True},
    "composition_shared": {"label": "Composition shared", "marker": "o", "color": None, "fill": False},
    "composition_unseen": {"label": "Composition unseen", "marker": "o", "color": None, "fill": True},
    "chemical_system_unseen": {
        "label": "Chemical system unseen",
        "marker": "s",
        "color": "#0A5A3C",
        "fill": True,
    },
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 400,
            "savefig.dpi": 400,
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_dir = Path(results_dir)
    counts = pd.read_csv(results_dir / "cohort_counts.csv")
    metrics = pd.read_csv(results_dir / "metrics.csv")
    per_class = pd.read_csv(results_dir / "set4_per_damage_class.csv")
    expected_metrics = {
        (endpoint, cohort, lawset)
        for endpoint in ("experimental_satisfaction", "damage_detection")
        for cohort in COHORT_STYLE
        for lawset in SET_ORDER
    }
    observed_metrics = set(metrics[["endpoint", "cohort", "lawset"]].itertuples(index=False, name=None))
    if not expected_metrics <= observed_metrics:
        raise ValueError("metrics.csv does not contain the complete frozen-set grid")
    return counts, metrics, per_class


def _panel_letter(ax: plt.Axes, letter: str, x: float = -0.17) -> None:
    ax.text(
        x,
        1.055,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _point_interval(
    ax: plt.Axes,
    *,
    x: float,
    low: float,
    high: float,
    y: float,
    color: str,
    marker: str,
    filled: bool,
    zorder: int,
) -> None:
    face = color if filled else "white"
    ax.errorbar(
        x,
        y,
        xerr=np.array([[x - low], [high - x]]),
        fmt=marker,
        ms=4.2,
        mfc=face,
        mec=color,
        mew=0.9,
        ecolor=color,
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.7,
        linestyle="none",
        zorder=zorder,
    )


def _metric_panel(ax: plt.Axes, metrics: pd.DataFrame, endpoint: str) -> None:
    data = metrics[metrics.endpoint.eq(endpoint)].set_index(["lawset", "cohort"])
    y = np.arange(len(SET_ORDER), dtype=float)
    offsets = {"heldout_all": -0.16, "composition_shared": 0.0, "composition_unseen": 0.16}
    for index, lawset in enumerate(SET_ORDER):
        set_color = SET_COLORS[lawset]
        for cohort in offsets:
            row = data.loc[(lawset, cohort)]
            style = COHORT_STYLE[cohort]
            color = style["color"] or set_color
            _point_interval(
                ax,
                x=float(row.estimate_micro),
                low=float(row.micro_ci_low),
                high=float(row.micro_ci_high),
                y=y[index] + offsets[cohort],
                color=color,
                marker=style["marker"],
                filled=style["fill"],
                zorder=4 if cohort == "composition_unseen" else 3,
            )
    ax.set_yticks(y)
    ax.set_yticklabels([SET_LABELS[name] for name in SET_ORDER])
    for tick, lawset in zip(ax.get_yticklabels(), SET_ORDER):
        tick.set_color(SET_COLORS[lawset])
    ax.set_ylim(len(SET_ORDER) - 0.45, -0.55)
    if endpoint == "experimental_satisfaction":
        ax.set_xlim(0.77, 1.005)
        ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_xlabel("Experimental satisfaction")
    else:
        ax.set_xlim(0.20, 1.005)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Damage detection")
    ax.set_ylabel("")
    ax.grid(False)


def make_overview_figure(counts: pd.DataFrame, metrics: pd.DataFrame) -> plt.Figure:
    """Three-panel overlap and frozen-law performance overview."""

    configure_style()
    figure = plt.figure(figsize=(DOUBLE_COLUMN, 8.9 * CM))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.03, 1.36, 1.42],
        left=0.095,
        right=0.985,
        top=0.82,
        bottom=0.19,
        wspace=0.48,
    )
    ax_overlap = figure.add_subplot(grid[0, 0])
    ax_sat = figure.add_subplot(grid[0, 1])
    ax_det = figure.add_subplot(grid[0, 2])

    table = counts.set_index(["population", "cohort"])
    populations = ("experimental", "chemically_damaged")
    labels = ("Experimental\nstructures", "Chemically\ndamaged\nstructures")
    y = np.array([1.0, 0.0])
    segment_colors = ("#B9B9B9", "#7CC5A5", "#0A5A3C")
    segment_labels = (
        "Composition shared",
        "Composition unseen, system shared",
        "Chemical system unseen",
    )
    for row_index, population in enumerate(populations):
        shared = float(table.loc[(population, "composition_shared"), "fraction_of_heldout_rows"])
        comp_unseen = float(table.loc[(population, "composition_unseen"), "fraction_of_heldout_rows"])
        sys_unseen = float(table.loc[(population, "chemical_system_unseen"), "fraction_of_heldout_rows"])
        fractions = (shared, comp_unseen - sys_unseen, sys_unseen)
        left = 0.0
        for segment_index, (fraction, color) in enumerate(zip(fractions, segment_colors)):
            ax_overlap.barh(y[row_index], fraction, left=left, height=0.55, color=color, lw=0)
            if fraction >= 0.09 and segment_index != 1:
                ax_overlap.text(
                    left + fraction / 2,
                    y[row_index],
                    f"{100 * fraction:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if color == "#0A5A3C" else "#222222",
                    fontsize=7.4,
                )
            left += fraction
        ax_overlap.text(
            shared + comp_unseen / 2,
            y[row_index] - 0.43,
            f"{100 * comp_unseen:.1f}% unseen",
            ha="center",
            va="center",
            color="#397E62",
            fontsize=7.2,
        )
        n = int(table.loc[(population, "heldout_all"), "n_rows"])
        ax_overlap.text(1.10, y[row_index], f"n={n:,}", ha="left", va="center", fontsize=7.5)
    ax_overlap.set_yticks(y)
    ax_overlap.set_yticklabels(labels)
    ax_overlap.set_xlim(0, 1.32)
    ax_overlap.set_ylim(-0.58, 1.58)
    ax_overlap.set_xticks([0, 0.5, 1.0])
    ax_overlap.set_xticklabels(["0", "50", "100"])
    ax_overlap.set_xlabel("Held-out population (%)")
    ax_overlap.spines["left"].set_visible(False)
    ax_overlap.tick_params(axis="y", length=0)
    ax_overlap.grid(False)

    _metric_panel(ax_sat, metrics, "experimental_satisfaction")
    _metric_panel(ax_det, metrics, "damage_detection")
    _panel_letter(ax_overlap, "a", x=-0.32)
    _panel_letter(ax_sat, "b", x=-0.22)
    _panel_letter(ax_det, "c", x=-0.21)

    overlap_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, ec="none", label=label)
        for color, label in zip(segment_colors, segment_labels)
    ]
    ax_overlap.legend(
        handles=overlap_handles,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.05, 1.03),
        fontsize=7.2,
        handlelength=1.15,
        labelspacing=0.25,
        borderpad=0,
    )
    metric_handles = [
        mlines.Line2D(
            [],
            [],
            color="#777777" if cohort == "heldout_all" else "#0A5A3C",
            marker=COHORT_STYLE[cohort]["marker"],
            markerfacecolor=(
                "white" if not COHORT_STYLE[cohort]["fill"] else
                ("#777777" if cohort == "heldout_all" else "#0A5A3C")
            ),
            markeredgewidth=0.9,
            linestyle="none",
            markersize=4.5,
            label=COHORT_STYLE[cohort]["label"],
        )
        for cohort in ("heldout_all", "composition_shared", "composition_unseen")
    ]
    figure.legend(
        handles=metric_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.70, 0.965),
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.35,
        borderpad=0,
    )
    return figure


def make_set4_sensitivity_figure(
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
) -> plt.Figure:
    """Set 4 transfer to composition- and exact-system-unseen chemistry."""

    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 8.3 * CM))
    figure.subplots_adjust(left=0.12, right=0.985, top=0.80, bottom=0.18, wspace=0.42)
    cohorts = ("heldout_all", "composition_unseen", "chemical_system_unseen")
    cohort_colors = {
        "heldout_all": "#777777",
        "composition_unseen": "#62B58D",
        "chemical_system_unseen": "#0A5A3C",
    }
    offsets = {"heldout_all": -0.16, "composition_unseen": 0.0, "chemical_system_unseen": 0.16}

    ax = axes[0]
    set4 = metrics[metrics.lawset.eq("Set 4")].set_index(["endpoint", "cohort"])
    endpoint_rows = (
        ("experimental_satisfaction", "Experimental satisfaction"),
        ("damage_detection", "Damage detection"),
    )
    for row_index, (endpoint, _) in enumerate(endpoint_rows):
        for cohort in cohorts:
            row = set4.loc[(endpoint, cohort)]
            style = COHORT_STYLE[cohort]
            _point_interval(
                ax,
                x=float(row.estimate_micro),
                low=float(row.micro_ci_low),
                high=float(row.micro_ci_high),
                y=row_index + offsets[cohort],
                color=cohort_colors[cohort],
                marker=style["marker"],
                filled=style["fill"],
                zorder=3,
            )
    ax.set_yticks(range(len(endpoint_rows)))
    ax.set_yticklabels([label for _, label in endpoint_rows])
    ax.set_ylim(len(endpoint_rows) - 0.45, -0.55)
    ax.set_xlim(0.77, 1.005)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Set 4 rate")
    ax.grid(False)

    ax = axes[1]
    per = per_class.set_index(["damage_class", "cohort"])
    classes = ("S1", "S2", "S3", "S4", "S5")
    y = np.arange(len(classes), dtype=float)
    for row_index, damage_class in enumerate(classes):
        for cohort in cohorts:
            row = per.loc[(damage_class, cohort)]
            style = COHORT_STYLE[cohort]
            _point_interval(
                ax,
                x=float(row.estimate_micro),
                low=float(row.micro_ci_low),
                high=float(row.micro_ci_high),
                y=y[row_index] + offsets[cohort],
                color=cohort_colors[cohort],
                marker=style["marker"],
                filled=style["fill"],
                zorder=3,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_ylim(len(classes) - 0.45, -0.55)
    ax.set_xlim(0.66, 1.005)
    ax.set_xticks([0.70, 0.80, 0.90, 1.00])
    ax.set_xlabel("Set 4 damage detection")
    ax.grid(False)

    _panel_letter(axes[0], "a", x=-0.24)
    _panel_letter(axes[1], "b", x=-0.20)
    handles = [
        mlines.Line2D(
            [],
            [],
            color=cohort_colors[cohort],
            marker=COHORT_STYLE[cohort]["marker"],
            markerfacecolor=(
                "white" if not COHORT_STYLE[cohort]["fill"] else cohort_colors[cohort]
            ),
            markeredgewidth=0.9,
            linestyle="none",
            markersize=4.5,
            label=COHORT_STYLE[cohort]["label"],
        )
        for cohort in cohorts
    ]
    figure.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.955),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.35,
        borderpad=0,
    )
    return figure


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.pdf")
    figure.savefig(output_dir / f"{name}.svg")
    figure.savefig(output_dir / f"{name}.png", dpi=400)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()
    counts, metrics, per_class = load_results(args.results_dir)
    save_figure(
        make_overview_figure(counts, metrics),
        args.output_dir,
        "pris_composition_holdout_overview",
    )
    save_figure(
        make_set4_sensitivity_figure(metrics, per_class),
        args.output_dir,
        "pris_set4_unseen_chemistry_sensitivity",
    )
    print(f"wrote SI-ready figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
