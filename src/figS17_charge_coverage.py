#!/usr/bin/env python3
"""Supplementary Fig. S17: coverage of charge-dependent PRIS evaluation."""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figs import CM, GRY, PUR, RED, W2, plt, save

import matplotlib.patches as mp
from matplotlib.colors import to_rgb


DATA = pathlib.Path(__file__).resolve().parent.parent / "paper" / "data"
INK = "#323335"
L4C = "#0A5A3C"
PUR2 = "#AE82C3"
FS = 8.0


def panel_charge_coverage(ax, frozen: dict) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n_total = frozen["n_sampled"]
    modes = frozen["charge_assignment"]["modes"]
    n_integer = modes["integer"]
    n_fallback = modes["fractional"]
    n_evaluated = n_integer + n_fallback
    n_not_evaluated = n_total - n_evaluated

    x0, x1 = 0.025, 0.063
    x2, x3 = 0.245, 0.292
    label_x = 0.355
    y0, y1 = 0.160, 0.800
    height = y1 - y0
    evaluated_fraction = n_evaluated / n_total

    ax.add_patch(mp.Rectangle(
        (x0, y0), x1 - x0, height * (1 - evaluated_fraction),
        facecolor="#E2E2E2", edgecolor="#B5B5B5", linewidth=0.45,
    ))
    ax.add_patch(mp.Rectangle(
        (x0, y1 - height * evaluated_fraction), x1 - x0,
        height * evaluated_fraction, facecolor=INK, edgecolor="#1F1F1F",
        linewidth=0.45,
    ))
    ax.text(x0, y1 + 0.095, f"n = {n_total:,}", fontsize=FS,
            color=INK, ha="left", va="bottom")
    ax.text(x0, y0 - 0.030,
            f"{n_not_evaluated:,}\ncharge-dependent laws not evaluated",
            fontsize=FS, color="#8E8E8E", ha="left", va="top",
            linespacing=1.30)
    ax.text(x2, y1 + 0.095, f"charge-dependent laws evaluated: {n_evaluated}",
            fontsize=FS, color=INK, ha="left", va="bottom")
    ax.text(x2, y1 + 0.040,
            f"{n_integer} integer assignment   ·   {n_fallback} mean-valence fallback",
            fontsize=FS, color=INK, ha="left", va="bottom")

    slice_bottom = y1 - height * evaluated_fraction
    ax.add_patch(mp.Polygon(
        [[x1, y1], [x1, slice_bottom], [x2, y0], [x2, y1]],
        closed=True, facecolor="#000000", alpha=0.045, edgecolor="none",
        zorder=0,
    ))
    for source_y, target_y in ((y1, y1), (slice_bottom, y0)):
        ax.plot([x1, x2], [source_y, target_y], color="#B5B5B5", lw=0.6,
                ls=(0, (3, 2)), zorder=1)

    n_l3_failure = frozen["L3"]["n_reject"]
    n_d78 = frozen["L4_overlap"]["n_pass_L3_fail_D7_or_D8"]
    n_pass = frozen["L4"]["n_pass"]
    n_unavailable = frozen["L4"]["n_undetermined"]
    n_d78_unavailable = frozen["L4"]["n_reject"] - n_l3_failure - n_d78
    groups = [
        (n_pass, L4C, "satisfies Set 4"),
        (n_unavailable, GRY, "some law inputs unavailable"),
        (n_d78, PUR, "Law 7 or Law 8 only"),
        (n_d78_unavailable, PUR2, "Law 7 or Law 8; other inputs unavailable"),
        (n_l3_failure, RED, "Law 1 or Law 3–Law 6"),
    ]
    assert sum(number for number, _, _ in groups) == n_evaluated

    tops = [0.795, 0.665, 0.535, 0.405, 0.275]
    cursor = y1
    for (number, colour, label), text_y in zip(groups, tops):
        fraction = number / n_evaluated
        bottom = cursor - height * fraction
        ax.add_patch(mp.Rectangle(
            (x2, bottom), x3 - x2, height * fraction,
            facecolor=colour,
            edgecolor=tuple(0.68 * channel for channel in to_rgb(colour)),
            linewidth=0.45, alpha=0.82, zorder=3,
        ))
        middle = 0.5 * (cursor + bottom)
        ax.plot([x3 + 0.006, label_x - 0.015], [middle, text_y - 0.018],
                color=colour, lw=0.65, alpha=0.85, zorder=2)
        ax.text(label_x, text_y, f"{number}  ({100 * fraction:.0f}%)",
                fontsize=FS, color=colour, fontweight="bold",
                ha="left", va="top")
        ax.text(label_x, text_y - 0.048, label, fontsize=FS, color=colour,
                ha="left", va="top")
        cursor = bottom


def main() -> None:
    frozen = json.loads((DATA / "fig7_gnome_ladder.json").read_text())
    fig, ax = plt.subplots(figsize=(W2, 7.2 * CM))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.98)
    panel_charge_coverage(ax, frozen)
    save(fig, "figS17_charge_coverage")


if __name__ == "__main__":
    main()
