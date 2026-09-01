#!/usr/bin/env python3
"""Additive, full a--f PRIS Fig. 4--5 redraw in the manuscript figure style.

This module deliberately leaves ``tex/`` and ``paper/`` untouched.  It keeps
the original controlled-damage panels a/b, then places the application and
mechanism panels c--f below them.  The visual contract is the same as the
existing manuscript figures: Arial, white background, no grid, left/bottom
spines only, and the original PRIS palette.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.colors import to_rgba, to_rgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import figure_palette  # noqa: F401  (registers the ramps, caps bar opacity)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.property_design_20260821.plot_synthesis_score_design import (
    build_design_frontier,
    draw_design_panel,
    rule_operating_point,
    support_matched_calibration_mask,
)

DATA = ROOT / "paper" / "data"
DEFAULT_OUTPUT = ROOT / "outputs/20260823_fig45_merged_nature_inverse_f_v1"
EHULL_DEFAULT = ROOT / "outputs/20260822_pu_mattersim_basin_hull_gpu_stratified_pilot_v3/gpu_results_corrected/e_hull_threshold_curve.csv"
F3_DEFAULT = ROOT / "outputs/20260814_f3_synth/resolve_f3.json"
FULL_POOL_DEFAULT = ROOT / "outputs/20260821_pu_synthesizability/full_pool_analysis_v1"
FORMULA_NPZ_DEFAULT = ROOT / "outputs/20260822_pu_formula_scores/full_pool_dual_v2/direct_formula_plots_v3/clscore_formula_density_data.npz"
FORWARD_DESIGN_DEFAULT = ROOT / "outputs/20260822_property_design_synthesis_score/forward_scores.parquet"
INVERSE_DESIGN_DEFAULT = ROOT / "outputs/20260822_property_design_synthesis_score/inverse_scores.parquet"
INVERSE_ARCHIVE_DEFAULT = ROOT / (
    "outputs/20260821_property_design/"
    "mattergen_bulk400_progress_13shards_adaptive/generated_crystals_cif.zip"
)
INVERSE_DIAGNOSTIC_TARGET = 0.975
INVERSE_SCREENED_EXAMPLE_ID = "candidate_0009"
INVERSE_RETAINED_EXAMPLE_ID = "candidate_0801"

CM = 1 / 2.54
BLU = "#005B93"
RED = "#D6564C"
ORA = "#9861B0"
PSSC = "#E88A8E"
GRN = "#156646"
L4C = "#0A5A3C"
GRY = "#8A8C8E"
PUR = "#35A7D8"
INK = "#1F2022"
SLATE = "#29445C"
EHULL = "#5C6B7A"
# Match Fig. 4b's darkest matrix blue for the retained diagnostic population.
F4_DIAGNOSTIC_BLUE = figure_palette.ramp("matrix")[-1]

# Structure thumbnails use the same series colours and viewing convention as
# the manuscript figures.  Keeping this explicit prevents a generic colormap
# from assigning unrelated hues to the same chemical species.
STRUCTURE_ELEMENT_COLOURS = {"Ir": BLU, "Os": ORA}
STRUCTURE_VIEW = {"elevation_deg": 40.0, "azimuth_deg": -55.0}
STRUCTURE_ZOOM = 1.28
STRUCTURE_INSET_POSITIONS = (
    [0.450, 0.248, 0.278, 0.248],
    [0.450, 0.002, 0.278, 0.248],
)
RULE_COLORS = {"L1": BLU, "L2": ORA, "L3": RED, "L4": L4C}
RULE_MARKERS = {"L1": "o", "L2": "s", "L3": "D", "L4": "*"}

# One visual contract for every primary quantitative series in Fig. 4.  Guide
# lines, uncertainty bands, density clouds and crystal atoms remain quieter
# because they carry different semantic roles.
F4_DATA_LINE_WIDTH = 1.20
F4_DATA_LINE_ALPHA = 0.82
F4_MARKER_SIZE_PT = 5.30
F4_MARKER_AREA_PT2 = F4_MARKER_SIZE_PT**2
# The Set 4 star is the only concave marker in Fig. 4.  At the shared area it
# puts roughly half the ink of a disk inside the same bounding box, so it reads
# as the smallest symbol on panels c and f even though it marks the operating
# point both panels are built around.  Enlarge that marker alone.
F4_STAR_AREA_SCALE = 2.0
F4_STAR_AREA_PT2 = F4_MARKER_AREA_PT2 * F4_STAR_AREA_SCALE
F4_STAR_SIZE_PT = F4_MARKER_SIZE_PT * F4_STAR_AREA_SCALE**0.5
F4_MARKER_FACE_ALPHA = 0.62
F4_MARKER_EDGE_WIDTH = 0.65
F4_BAR_FACE_ALPHA = 0.38
F4_BAR_EDGE_WIDTH = 0.85
F4_GUIDE_LINE_WIDTH = 0.65
F4_GUIDE_ALPHA = 0.58
F4_BAND_ALPHA = 0.10
F4_HIGHLIGHT_FRAME_WIDTH = 0.85
F4_STRUCTURE_FRAME_WIDTH = 0.55
F4_STRUCTURE_FRAME_WHITE_MIX = 0.45
F4_DESCRIPTOR_POINT_AREA_PT2 = 10.0
F4_DESCRIPTOR_FACE_ALPHA = 0.14
F4_DESCRIPTOR_EDGE_ALPHA = 0.45
F4_DESCRIPTOR_EDGE_WIDTH = 0.55
F4_DESCRIPTOR_TICK_PAD = 0.0

F4_DESIGN_VISUAL_STYLE = {
    "line_width": F4_DATA_LINE_WIDTH,
    "line_alpha": F4_DATA_LINE_ALPHA,
    "marker_area": F4_MARKER_AREA_PT2,
    "marker_size": F4_MARKER_SIZE_PT,
    "marker_face_alpha": F4_MARKER_FACE_ALPHA,
    "rule_face_alpha": F4_MARKER_FACE_ALPHA,
    "marker_edge_width": F4_MARKER_EDGE_WIDTH,
    "uniform_rule_markers": True,
    "rule_marker_shapes": RULE_MARKERS,
    "rule_marker_area_overrides": {"L4": F4_STAR_AREA_PT2},
    "gid_prefix": "f4",
}

CAPTION_DRAFT = r"""**Figure 4 | Physicochemical damage detection, synthesizability screening and property-conditioned inverse design.**

**a**, Overall damage detection for fixed-distance, composition-only and
successive PRIS criteria. Colours identify families. Bar and upper-margin labels
give damage detection and experimental-structure satisfaction, and dotted
dividers separate the families. **b**, Damage detection by
composition-preserving perturbation class. Rows are baselines, contact laws and
PRIS law sets, and columns are the five controlled perturbations. Cell colour
and text encode detection. Lines separate method families, and outlines mark the
isotropic-expansion blind region of lower-contact screening. **c**,
Experimental-structure satisfaction versus the fraction of hard-to-synthesize
structures screened.
Symbols locate Set 1–Set 4, the orange curve sweeps PSS thresholds, grey squares denote
distance cutoffs, and blue triangles sweep a threshold on the MatterSim-computed
hull energy. Connectors trace the law-set ladder, the true
distance-cutoff ordinates and the matched-satisfaction comparison of Set 4 with PSS.
**d**, Set 4 violation and mean PSS across within-model CLscore
percentiles. Blue and orange circles denote CGCNN-PU and
MatterSim-1M-MLP-PU, black diamonds give their aligned pointwise mean, and
lower-panel shading spans each model's interquartile PSS range, and each panel gives
the coefficient of determination of a straight-line fit in the decile index. **e**, Held-out
same-composition pair accuracy as progressively larger confidence fractions are
retained. Blue circles denote PSS, green triangles denote the DFT
hull energy, and shading gives the corresponding confidence
intervals. **f**, Validation-queue reduction versus retention of the
high-property subset defined by the UMA bulk-modulus proxy in a
property-conditioned MatterGen run. The orange curve sweeps PSS thresholds, and
symbols mark Set 1–Set 4. The inset maps crystallographic site fraction against atomic
volume, with retained and screened points linked to same-composition thumbnails
in a common supercell and view.
"""


def style() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The py23 module has been deprecated.*",
        category=DeprecationWarning,
    )
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.cursive": ["Arial", "Liberation Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
        # Keep panel letters, the heat-map colour bar and the right-edge D1
        # labels intact.  The source canvas remains the manuscript's 18.3-cm
        # width; TeX scales the complete bounding box to \textwidth.
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def clean(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def require(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def deep_color(color: str, factor: float = 0.72) -> tuple[float, float, float, float]:
    """Return a darker, opaque edge colour from the series colour.

    White marker outlines made the previous draft look sticker-like.  A darker
    version of the same hue keeps the alpha hierarchy while preserving the
    visual identity of each curve.
    """
    rgb = np.asarray(to_rgb(color), dtype=float) * factor
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def soft_structure_frame_color(color: str) -> tuple[float, float, float, float]:
    """Return a light same-hue frame that does not dominate crystal insets."""

    rgb = np.asarray(to_rgb(color), dtype=float)
    mixed = (
        (1.0 - F4_STRUCTURE_FRAME_WHITE_MIX) * rgb
        + F4_STRUCTURE_FRAME_WHITE_MIX * np.ones(3)
    )
    return (float(mixed[0]), float(mixed[1]), float(mixed[2]), 1.0)


def letter(ax: plt.Axes, value: str) -> None:
    ax.text(-0.14, 1.04, value, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left", color=INK)


def panel_a(ax: plt.Axes) -> dict:
    """Overall controlled-damage benchmark, ported from the original Fig. 4a.

    The numbers above the bars are part of the result, rather than explanatory
    text, so they are deliberately retained.  Law 1 is kept in panel b (where it
    resolves the D1--D5 mechanism) but not as a separate bar here, matching the
    original benchmark panel.
    """
    v = pd.read_csv(DATA / "fig6_validity.csv").set_index("criterion")
    order = ["min pair distance > 0.5 A", "min pair distance > 0.7 A",
             "min pair distance > 1.0 A", "SMACT charge neutrality",
             "L1 (D1, tau=0.735)", "L1' (D1+D2)", "L2 (D1,D3-D5)",
             "L3 (D1,D3-D6)", "L4 (D1,D3-D8)"]
    labels = ["d 0.5 Å", "d 0.7 Å", "d 1.0 Å", "SMACT",
              "Set 1", "Set 1′", "Set 2", "Set 3", "Set 4"]
    colors = [GRY, GRY, GRY, "#B3B8BD", BLU, PUR, ORA, RED, L4C]
    detection = v.loc[order, "exclusion_total"].to_numpy(float)
    satisfaction = v.loc[order, "real_satisfaction"].to_numpy(float)
    x = np.arange(len(order))
    bars = ax.bar(
        x,
        detection,
        width=0.60,
        color=colors,
        edgecolor=colors,
        linewidth=F4_BAR_EDGE_WIDTH,
        zorder=2,
    )
    for bar, color in zip(bars, colors, strict=True):
        bar.set_alpha(None)
        bar.set_facecolor(to_rgba(color, F4_BAR_FACE_ALPHA))
        bar.set_edgecolor(to_rgba(color, 1.0))
        bar.set_linewidth(F4_BAR_EDGE_WIDTH)
    ax.set_ylim(0, 1.20)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Damage detection")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    # The grey row is the original panel's compact satisfaction readout.  It
    # is intentionally data-rich but prose-light.
    ax.text(-0.55, 1.135, "satisfaction of experimental structures", fontsize=7.4,
            color="#97999C", ha="left", va="center")
    for i, (e, s, c) in enumerate(zip(detection, satisfaction, colors)):
        ax.text(i, e + 0.017, f"{e:.3f}", ha="center", fontsize=7.0,
                color=c)
        ax.text(i, 1.055, f"{s:.2f}", ha="center", fontsize=6.9, color="#97999C")
    ax.axvline(2.5, color=to_rgba("#B9BBBE", F4_GUIDE_ALPHA),
               lw=F4_GUIDE_LINE_WIDTH, ls=":")
    ax.axvline(3.5, color=to_rgba("#86888A", F4_GUIDE_ALPHA),
               lw=F4_GUIDE_LINE_WIDTH, ls=":")
    ax.set_xlim(-0.62, 8.62)
    clean(ax); letter(ax, "a")
    return {"source": str(DATA / "fig6_validity.csv"), "l4_detection": float(detection[-1]),
            "l4_satisfaction": float(satisfaction[-1]), "n_rows": len(order)}


def panel_b(ax: plt.Axes) -> dict:
    """Per-damage-class heatmap, including every value and group boundary."""
    v = pd.read_csv(DATA / "fig6_validity.csv").set_index("criterion")
    order = ["min pair distance > 0.5 A", "min pair distance > 0.7 A",
             "min pair distance > 1.0 A", "SMACT charge neutrality",
             "L1 (D1, tau=0.735)", "D1 alone, tau=0.804", "L1' (D1+D2)",
             "L2 (D1,D3-D5)", "L3 (D1,D3-D6)", "L4 (D1,D3-D8)"]
    labels = ["d 0.5 Å", "d 0.7 Å", "d 1.0 Å", "SMACT",
              "Set 1", "Law 1", "Set 1′", "Set 2", "Set 3", "Set 4"]
    full_cols = [c for c in v.columns if c.startswith("S") and c[1:2].isdigit()]
    full_cols = sorted(full_cols, key=lambda c: int(c[1:c.find("_")]))
    if len(full_cols) != 5:
        raise ValueError(f"expected five damage columns, found {full_cols}")
    matrix = v.loc[order, full_cols].to_numpy(float)
    im = ax.imshow(matrix, cmap="palmatrix", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    # 归档列名仍是 S1--S5;图内显示 D1--D5,避免与正文的 Set 1--Set 5 混读
    class_labels = ["D1", "D2", "D3", "D4", "D5"]
    # 集合成分见 Fig. 1d;这里只留集合名,避免行标签伸进左边的面板 a。
    row_labels = ["min d > 0.5 Å", "min d > 0.7 Å", "min d > 1.0 Å",
                  "SMACT (composition)", "Set 1",
                  "Law 1 only, τ=0.804", "Set 1′", "Set 2",
                  "Set 3", "Set 4"]
    ax.set_xticks(np.arange(5), class_labels)
    ax.tick_params(axis="x", labelsize=plt.rcParams["ytick.labelsize"], pad=2)
    ax.set_yticks(np.arange(len(order)), row_labels)
    ax.tick_params(axis="y", pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # Numeric cell labels are the measurements the user asked to preserve;
    # separator lines and the rectangle reproduce the original blind-region
    # cue without adding explanatory prose.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.7, color="white" if matrix[i, j] > 0.55 else "#212224")
    ax.axhline(2.5, color=to_rgba(INK, F4_GUIDE_ALPHA),
               lw=F4_GUIDE_LINE_WIDTH)
    ax.axhline(3.5, color=to_rgba(INK, F4_GUIDE_ALPHA),
               lw=F4_GUIDE_LINE_WIDTH)
    ax.add_patch(Rectangle((2.5, 3.5), 1, 2, fill=False,
                           ec=to_rgba(INK, 0.78),
                           lw=F4_HIGHLIGHT_FRAME_WIDTH, zorder=5))
    for tick in ax.get_yticklabels():
        if "Set 4" in tick.get_text():
            tick.set_color(L4C)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, .2, .4, .6, .8, 1])
    cb.set_label("Damage detection", fontsize=8)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    clean(ax); letter(ax, "b")
    return {"source": str(DATA / "fig6_validity.csv"), "l4_row": matrix[-1].tolist()}


def _load_c(binary_dir: Path, independent_dir: Path, ehull_path: Path | None):
    # Reuse the audited loaders so the denominator and ABSTAIN policy stay
    # identical to the c--f draft.
    from plot_fig45_cdef_draft import load_c_data, load_ehull
    cdata, cmeta = load_c_data(binary_dir, independent_dir)
    ehull, emeta = load_ehull(ehull_path)
    return cdata, cmeta, ehull, emeta


def panel_c(ax: plt.Axes, cdata: dict, ehull: pd.DataFrame | None) -> dict:
    """Screening choices in one uncluttered satisfaction/screening plane.

    Set 1--Set 4 are displayed as measured points, while the descriptor-derived
    synthesis score and the two reference methods retain their curves.  Only
    the matched Set 4 comparison is annotated; every other number remains in the
    machine-readable status file and caption rather than competing for space
    with the data.
    """
    ladder = cdata["rule_ladder"].copy()
    rule_x = ladder["experimental_retention"].to_numpy(float) * 100
    rule_y = ladder["pu_screened_rate"].to_numpy(float) * 100
    # A very light guide shows the rule ladder without suggesting that the
    # gates are a sequential cascade.
    ax.plot(
        rule_x,
        rule_y,
        color=to_rgba("#989A9D", F4_GUIDE_ALPHA),
        lw=F4_GUIDE_LINE_WIDTH,
        ls=(0, (2, 2)),
        zorder=1,
    )
    label_positions = {"L1": (95.3, 7.2), "L2": (89.0, 7.2), "L3": (83.8, 12.5)}
    for row in ladder.itertuples(index=False):
        rule = str(row.rule)
        # L4 is drawn once below as the emphasized matched-comparison star.
        # Drawing it here as well made the marker look blurred and oversized.
        if rule == "L4":
            continue
        color = RULE_COLORS[rule]
        x0 = float(row.experimental_retention) * 100
        y0 = float(row.pu_screened_rate) * 100
        marker = RULE_MARKERS[rule]
        # L2 and L3 are separated by only 1.2 percentage points in x and
        # 0.1 percentage points in y.  Their true coordinates are retained;
        # a compact marker avoids turning two measurements into one blob.
        mark = ax.scatter(
            [x0], [y0], s=F4_MARKER_AREA_PT2, marker=marker,
            color=to_rgba(color, F4_MARKER_FACE_ALPHA),
            edgecolor=deep_color(color), linewidth=F4_MARKER_EDGE_WIDTH,
            zorder=5,
        )
        mark.set_gid("f4-primary-marker")
        tx, ty = label_positions[rule]
        ax.annotate("Set " + rule[1:], (x0, y0), xytext=(tx, ty), textcoords="data",
                    color=color, fontsize=7.2,
                    ha="center", va="bottom",
                    arrowprops=dict(
                        arrowstyle="-", color=to_rgba(color, F4_GUIDE_ALPHA),
                        lw=F4_GUIDE_LINE_WIDTH, shrinkA=2, shrinkB=2,
                    ))

    syn = cdata["s_syn"].sort_values("experimental_retention")
    l4 = cdata["l4"].iloc[0]
    xr = syn["experimental_retention"].to_numpy(float) * 100
    yr = syn["pu_screened_rate"].to_numpy(float) * 100
    (pss_line,) = ax.plot(
        xr, yr, color=to_rgba(PSSC, F4_DATA_LINE_ALPHA),
        lw=F4_DATA_LINE_WIDTH, zorder=2,
    )
    pss_line.set_gid("f4-primary-series")
    # A few semi-transparent markers preserve the threshold samples without
    # turning the curve into a row of opaque dots.
    idx_match = int(np.nanargmin(np.abs(xr - float(l4.experimental_retention) * 100)))
    # The matched operating point is emphasized once below; exclude it from
    # the ordinary threshold markers to avoid a double-stroked orange disk.
    keep_idx = np.delete(np.arange(len(xr)), idx_match)
    threshold_marks = ax.scatter(
        xr[keep_idx], yr[keep_idx], s=F4_MARKER_AREA_PT2,
        color=to_rgba(PSSC, F4_MARKER_FACE_ALPHA),
        edgecolor=deep_color(PSSC), linewidth=F4_MARKER_EDGE_WIDTH, zorder=3,
    )
    threshold_marks.set_gid("f4-primary-marker")
    lx = float(l4.experimental_retention) * 100
    ly = float(l4.pu_screened_rate) * 100
    sy = float(yr[idx_match])
    matched_mark = ax.scatter(
        [xr[idx_match]], [yr[idx_match]], s=F4_MARKER_AREA_PT2,
        color=to_rgba(PSSC, F4_MARKER_FACE_ALPHA), edgecolor=deep_color(PSSC),
        linewidth=F4_MARKER_EDGE_WIDTH, zorder=5,
    )
    matched_mark.set_gid("f4-primary-marker")
    l4_mark = ax.scatter(
        [lx], [ly], marker=RULE_MARKERS["L4"], s=F4_STAR_AREA_PT2,
        color=to_rgba(L4C, F4_MARKER_FACE_ALPHA), edgecolor=deep_color(L4C),
        linewidth=F4_MARKER_EDGE_WIDTH, zorder=6,
    )
    l4_mark.set_gid("f4-primary-marker")
    ax.vlines(lx, ly, sy, color=to_rgba("#535557", F4_GUIDE_ALPHA),
              lw=F4_GUIDE_LINE_WIDTH, zorder=1)
    advantage = sy - ly
    ax.annotate(f"+{advantage:.1f} pp", xy=(lx, (ly + sy) / 2),
                xytext=(-7, 0), textcoords="offset points", va="center",
                ha="right", color=PSSC, fontsize=8.0)
    ax.annotate("Set 4", (lx, ly), xytext=(7, -9), textcoords="offset points",
                color=L4C, fontsize=7.8)
    points = cdata["distance"].sort_values("retention")
    if not points.empty:
        # Both distance screens have a measured screening rate of exactly 0%
        # and lie beside the 99%-satisfaction synthesis-score point. Display their
        # markers at separate y offsets and connect them to the true y=0
        # values so three distinct measurements remain visible.
        y_display = np.linspace(4.0, 10.0, len(points))
        x_distance = points["retention"].to_numpy(float) * 100
        y_true = points["screening"].to_numpy(float) * 100
        ax.vlines(x_distance, y_true, y_display,
                  color=to_rgba(GRY, F4_GUIDE_ALPHA),
                  lw=F4_GUIDE_LINE_WIDTH, zorder=2)
        distance_marks = ax.scatter(
            x_distance, y_display, marker="s", s=F4_MARKER_AREA_PT2,
            facecolor=to_rgba(GRY, F4_MARKER_FACE_ALPHA),
            edgecolor=deep_color(GRY), linewidth=F4_MARKER_EDGE_WIDTH, zorder=4,
        )
        distance_marks.set_gid("f4-primary-marker")
    if ehull is not None and not ehull.empty:
        (hull_line,) = ax.plot(
            ehull["retention"] * 100,
            ehull["screening"] * 100,
            color=to_rgba(EHULL, F4_DATA_LINE_ALPHA),
            lw=F4_DATA_LINE_WIDTH, ls="--", marker="^",
            ms=F4_MARKER_SIZE_PT,
            mfc=to_rgba(EHULL, F4_MARKER_FACE_ALPHA),
            mec=deep_color(EHULL), mew=F4_MARKER_EDGE_WIDTH,
            label="MatterSim hull energy",
        )
        hull_line.set_gid("f4-primary-series")
    ax.set_xlim(55, 100.5); ax.set_ylim(-2, 101)
    ax.set_xticks([60, 70, 80, 90, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Experimental-structure satisfaction (%)")
    # two lines: on one line the label is taller than the axes and prints over the panel
    # letter, the same way the wider panel f label is already broken
    ax.set_ylabel("Hard-to-synthesize\nstructures screened (%)")
    handles = [Line2D([], [], color=to_rgba(PSSC, F4_DATA_LINE_ALPHA),
                      lw=F4_DATA_LINE_WIDTH, marker="o",
                      mfc=to_rgba(PSSC, F4_MARKER_FACE_ALPHA),
                      mec=deep_color(PSSC), ms=F4_MARKER_SIZE_PT,
                      mew=F4_MARKER_EDGE_WIDTH,
                      label="PSS"),
               Line2D([], [], color=to_rgba("#989A9D", F4_GUIDE_ALPHA),
                      ls=(0, (2, 2)), marker="o",
                      mfc=to_rgba(SLATE, F4_MARKER_FACE_ALPHA),
                      mec=deep_color(SLATE), lw=F4_GUIDE_LINE_WIDTH,
                      ms=F4_MARKER_SIZE_PT, mew=F4_MARKER_EDGE_WIDTH,
                      label="PRIS (Set 1–Set 4)"),
               Line2D([], [], color=to_rgba(GRY, F4_DATA_LINE_ALPHA),
                      marker="s", mfc=to_rgba(GRY, F4_MARKER_FACE_ALPHA),
                      mec=deep_color(GRY), lw=0, ms=F4_MARKER_SIZE_PT,
                      mew=F4_MARKER_EDGE_WIDTH, label="distance cutoffs"),
               Line2D([], [], color=to_rgba(EHULL, F4_DATA_LINE_ALPHA),
                      ls="--", marker="^",
                      mfc=to_rgba(EHULL, F4_MARKER_FACE_ALPHA),
                      mec=deep_color(EHULL), lw=F4_DATA_LINE_WIDTH,
                      ms=F4_MARKER_SIZE_PT, mew=F4_MARKER_EDGE_WIDTH,
                      label="MatterSim hull energy")]
    # Four short rows in the empty lower-left quadrant keep the data region
    # clear and remove the large inter-row gap caused by an outside legend.
    ax.legend(handles=handles, frameon=False, loc="lower left",
              bbox_to_anchor=(0.02, 0.025), ncol=1,
              handlelength=1.35, columnspacing=0.8, labelspacing=0.24,
              borderpad=0.0, fontsize=plt.rcParams["legend.fontsize"])
    clean(ax); letter(ax, "c")
    return {"l4_satisfaction": lx, "l4_screening": ly,
            "s_syn_matched_satisfaction": float(xr[idx_match]),
            "s_syn_matched_screening": sy, "advantage_pp": advantage,
            "rule_ladder": [
                {"rule": str(row.rule),
                 "experimental_retention": float(row.experimental_retention),
                 "pu_screened_rate": float(row.pu_screened_rate)}
                for row in ladder.itertuples(index=False)
            ]}


def load_d_series(full_pool: Path, formula_npz: Path) -> dict:
    """Load the two CLscore curves and their explicit pointwise mean.

    The source files contain ten within-model deciles.  We retain that
    protocol, use the mean (rather than the median) for the synthesis formula,
    and never call the two model names simply ``A`` and ``B`` in the figure.
    """
    decile_path = Path(full_pool) / "score_deciles.csv"
    dec = pd.read_csv(decile_path)
    require(dec, {"analysis_unit", "provenance", "score_model", "decile",
                  "L4_explicit_violation_rate"}, decile_path)
    dec = dec.loc[(dec["analysis_unit"] == "orig_index_records")
                  & (dec["provenance"] == "all")].copy()
    models = [
        ("CLscore_A", "CGCNN-PU", BLU),
        ("CLscore_B", "MatterSim-1M-MLP-PU", ORA),
    ]
    data = np.load(formula_npz, allow_pickle=False)
    series = {}
    for key, name, color in models:
        sub = dec.loc[dec["score_model"].eq(key)].sort_values("decile")
        if len(sub) != 10 or sub["decile"].tolist() != list(range(1, 11)):
            raise ValueError(f"{decile_path} does not contain ten ordered deciles for {key}")
        prefix = f"S_syn__{key}__"
        required = [prefix + s for s in ["y_mean", "y_q25", "y_q75"]]
        if any(k not in data.files for k in required):
            raise ValueError(f"{formula_npz} lacks mean/quantile arrays for {key}")
        series[key] = {
            "name": name,
            "color": color,
            "x": (np.arange(1, 11, dtype=float) - 0.5) * 10,
            "l4": sub["L4_explicit_violation_rate"].to_numpy(float) * 100,
            "syn": np.asarray(data[prefix + "y_mean"], dtype=float),
            "q25": np.asarray(data[prefix + "y_q25"], dtype=float),
            "q75": np.asarray(data[prefix + "y_q75"], dtype=float),
        }
    mean = {
        "name": "mean of the two models",
        "color": INK,
        "x": series["CLscore_A"]["x"],
        "l4": (series["CLscore_A"]["l4"] + series["CLscore_B"]["l4"]) / 2,
        "syn": (series["CLscore_A"]["syn"] + series["CLscore_B"]["syn"]) / 2,
    }
    return {"series": series, "mean": mean,
            "source_score_deciles": str(decile_path),
            "source_formula_npz": str(formula_npz),
            "protocol": "aligned within-model deciles; formulas use decile means",
            "status": "aligned_decile_curves"}


def panel_d(outer: plt.Axes, ddata: dict) -> dict:
    """CLscore--PRIS/formula relation, with three lines in each row."""
    outer.set_axis_off()
    grid = outer.get_subplotspec().subgridspec(2, 1, hspace=0.06, height_ratios=(1, 1))
    top = outer.figure.add_subplot(grid[0])
    bot = outer.figure.add_subplot(grid[1], sharex=top)
    x = ddata["mean"]["x"]
    for item in ddata["series"].values():
        col = to_rgba(item["color"], F4_DATA_LINE_ALPHA)
        (top_line,) = top.plot(
            x, item["l4"], color=col, lw=F4_DATA_LINE_WIDTH,
            marker="o", ms=F4_MARKER_SIZE_PT,
            mfc=to_rgba(item["color"], F4_MARKER_FACE_ALPHA),
            mec=deep_color(item["color"]), mew=F4_MARKER_EDGE_WIDTH,
            zorder=3, label=item["name"],
        )
        top_line.set_gid("f4-primary-series")
        band = bot.fill_between(
            x, item["q25"], item["q75"], color=item["color"],
            alpha=F4_BAND_ALPHA, lw=0, zorder=1,
        )
        band.set_gid("f4-confidence-band")
        (bot_line,) = bot.plot(
            x, item["syn"], color=col, lw=F4_DATA_LINE_WIDTH,
            marker="o", ms=F4_MARKER_SIZE_PT,
            mfc=to_rgba(item["color"], F4_MARKER_FACE_ALPHA),
            mec=deep_color(item["color"]), mew=F4_MARKER_EDGE_WIDTH,
            zorder=3,
        )
        bot_line.set_gid("f4-primary-series")
    mean = ddata["mean"]
    (mean_top,) = top.plot(
        x, mean["l4"], color=to_rgba(INK, F4_DATA_LINE_ALPHA),
        lw=F4_DATA_LINE_WIDTH, ls="--", marker="D",
        ms=F4_MARKER_SIZE_PT, mfc=to_rgba(INK, F4_MARKER_FACE_ALPHA),
        mec=deep_color(INK), mew=F4_MARKER_EDGE_WIDTH, zorder=4,
        label=mean["name"],
    )
    mean_top.set_gid("f4-primary-series")
    (mean_bot,) = bot.plot(
        x, mean["syn"], color=to_rgba(INK, F4_DATA_LINE_ALPHA),
        lw=F4_DATA_LINE_WIDTH, ls="--", marker="D",
        ms=F4_MARKER_SIZE_PT, mfc=to_rgba(INK, F4_MARKER_FACE_ALPHA),
        mec=deep_color(INK), mew=F4_MARKER_EDGE_WIDTH, zorder=4,
    )
    mean_bot.set_gid("f4-primary-series")
    # Both trends are linear in the decile index, and the abstract calls them so; the fit
    # quality belongs on the panel that makes the claim rather than only in the text.
    def _r2(xv, yv):
        xv, yv = np.asarray(xv, float), np.asarray(yv, float)
        pred = np.polyval(np.polyfit(xv, yv, 1), xv)
        return 1.0 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()

    # each label goes to the corner its own curve leaves empty: violation falls with the
    # percentile, PSS rises with it.  Two numbers on one axis have to say which model each
    # belongs to, so the model names are written out at the size of the panel key.
    for axis, key, ax_x, ax_ha in ((top, "l4", 0.015, "left"), (bot, "syn", 0.985, "right")):
        parts = [f"{_r2(x, item[key]):.2f} ({item['name']})"
                 for item in ddata["series"].values()]
        axis.text(ax_x, 0.045, r"$R^2$ = " + parts[0] + "\nand " + parts[1],
                  transform=axis.transAxes,
                  fontsize=plt.rcParams["legend.fontsize"], color=INK,
                  ha=ax_ha, va="bottom", linespacing=1.30)

    top.set_ylabel("Set 4 violation (%)", labelpad=1)
    bot.set_ylabel("Mean PSS", labelpad=1)
    bot.set_xlabel("CLscore percentile within each model", labelpad=2)
    top.set_ylim(25, 60); bot.set_ylim(-15, 2)
    bot.set_xlim(0, 100); bot.set_xticks([5, 25, 45, 65, 85, 95])
    top.set_yticks([30, 40, 50, 60]); bot.set_yticks([-15, -10, -5, 0])
    top.tick_params(labelbottom=False)
    # Lift the model key into the inter-row whitespace so it remains close to
    # the upper panel without obscuring any of the three score curves.
    top.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.99, 1.30),
               ncol=1, handlelength=1.35, labelspacing=.16, borderpad=0.0,
               fontsize=plt.rcParams["legend.fontsize"])
    clean(top); clean(bot)
    outer.text(-0.14, 1.04, "d", transform=outer.transAxes, fontsize=10,
               fontweight="bold", va="bottom", color=INK)
    return {"status": ddata["status"], "source_score_deciles": ddata["source_score_deciles"],
            "source_formula_npz": ddata["source_formula_npz"],
            "lines_per_row": 3,
            "models": [v["name"] for v in ddata["series"].values()] + [ddata["mean"]["name"]]}


def panel_e(ax: plt.Axes, f3_path: Path) -> dict:
    """Held-out validation of PSS against DFT energy above the hull."""
    from plot_fig45_cdef_draft import _curve_bounds
    f3 = json.loads(Path(f3_path).read_text(encoding="utf-8"))
    keys = sorted(float(k) for k in f3["F3_curve"])
    x = np.asarray(keys) * 100
    rec = [f3["F3_curve"][f"{k:.2f}"] for k in keys]
    y, lo, hi = zip(*(_curve_bounds(r) for r in rec))
    (pss_line,) = ax.plot(
        x, np.asarray(y) * 100, "o-",
        color=to_rgba(BLU, F4_DATA_LINE_ALPHA), lw=F4_DATA_LINE_WIDTH,
        ms=F4_MARKER_SIZE_PT, mfc=to_rgba(BLU, F4_MARKER_FACE_ALPHA),
        mec=deep_color(BLU), mew=F4_MARKER_EDGE_WIDTH, label="PSS",
    )
    pss_line.set_gid("f4-primary-series")
    pss_band = ax.fill_between(
        x, np.asarray(lo) * 100, np.asarray(hi) * 100,
        color=BLU, alpha=F4_BAND_ALPHA, lw=0,
    )
    pss_band.set_gid("f4-confidence-band")
    if isinstance(f3.get("e_hull_curve"), dict):
        hr = [f3["e_hull_curve"].get(f"{k:.2f}") for k in keys]
        if all(r is not None for r in hr):
            yh, lh, hh = zip(*(_curve_bounds(r) for r in hr))
            (hull_line,) = ax.plot(
                x, np.asarray(yh) * 100, "--^",
                color=to_rgba(GRN, F4_DATA_LINE_ALPHA),
                lw=F4_DATA_LINE_WIDTH, ms=F4_MARKER_SIZE_PT,
                mfc=to_rgba(GRN, F4_MARKER_FACE_ALPHA),
                mec=deep_color(GRN), mew=F4_MARKER_EDGE_WIDTH,
                label="DFT hull energy",
            )
            hull_line.set_gid("f4-primary-series")
            hull_band = ax.fill_between(
                x, np.asarray(lh) * 100, np.asarray(hh) * 100,
                color=GRN, alpha=F4_BAND_ALPHA, lw=0,
            )
            hull_band.set_gid("f4-confidence-band")
    ax.set_xlim(0, 103); ax.set_ylim(68, 101)
    ax.set_xticks([5, 10, 20, 30, 50, 100])
    ax.set_xlabel("Held-out pairs retained (%)")
    ax.set_ylabel("Pair accuracy (%)")
    ax.legend(frameon=False, loc="lower right", ncol=2, handlelength=1.3,
              borderpad=0.0, labelspacing=0.2)
    ax.annotate("high-confidence tail", xy=(20, 94.4), xytext=(29, 97.2),
                arrowprops=dict(arrowstyle="-", color=to_rgba(BLU, F4_GUIDE_ALPHA),
                                lw=F4_GUIDE_LINE_WIDTH),
                color=BLU, fontsize=7.5)
    clean(ax); letter(ax, "e")
    return {"f3_pairs": int(f3.get("n_pairs", 0)),
            "formula": "PRIS-derived synthesis score (PSS) only"}


def panel_physical_states_si(
    outer: plt.Axes,
    threeaxis: Path,
    *,
    panel_letter: str | None = "f",
) -> dict:
    """Show the two complementary bounds across the four physical states.

    The former 2x2 grouped-bar layout made the comparison difficult to follow:
    the state labels, the rule labels and the recorded/computed split all had
    different visual directions.  A pair of shared-x profiles keeps the exact
    measurements but makes the question explicit.  D1 is nearly invariant
    across energy/phonon classes, whereas D7 separates recorded from
    computed-only structures within every class.
    """
    table = pd.read_csv(threeaxis)
    outer.set_axis_off()
    grid = outer.get_subplotspec().subgridspec(2, 1, hspace=0.18,
                                               height_ratios=(1.0, 1.0))
    top = outer.figure.add_subplot(grid[0])
    bottom = outer.figure.add_subplot(grid[1], sharex=top)

    # Put the energy state first and the phonon state second.  The previous
    # order interleaved the two axes and made the four physical cells look
    # arbitrary.
    states = [(True, True, "hull\nno imaginary"),
              (True, False, "hull\nimaginary"),
              (False, True, "metastable\nno imaginary"),
              (False, False, "metastable\nimaginary")]
    x = np.arange(len(states), dtype=float)
    n_by_state: list[int] = []
    values: dict[str, dict[bool, np.ndarray]] = {
        "D1": {True: [], False: []}, "D7": {True: [], False: []}}
    for on_hull, dyn, _ in states:
        sub = table[(table.on_hull == on_hull) & (table.dyn_stable == dyn)]
        n_by_state.append(int(sub["n"].sum()))
        for made in (True, False):
            row = sub[sub.experimental == made]
            if len(row) != 1:
                raise ValueError(
                    f"{threeaxis}: expected one row for on_hull={on_hull}, "
                    f"dyn_stable={dyn}, experimental={made}"
                )
            values["D1"][made].append(float(row.iloc[0].pass_d1_735) * 100)
            values["D7"][made].append(float(row.iloc[0].pass_d7) * 100)
    for law in values:
        for made in values[law]:
            values[law][made] = np.asarray(values[law][made], dtype=float)

    # Colour encodes the record status; line style and marker shape make the
    # distinction robust in print and do not require an external legend.
    rec_col, comp_col = BLU, SLATE
    statuses = [(True, rec_col, "recorded", "o", "-"),
                (False, comp_col, "computed-only", "s", "--")]

    def profile(ax: plt.Axes, law: str, title: str, *, annotate_all: bool,
                title_y: float, title_va: str = "top") -> None:
        for made, col, label, marker, ls in statuses:
            vals = values[law][made]
            # Small horizontal offsets keep paired markers legible without
            # changing the state ordering or the measured ordinate.
            dx = -0.075 if made else 0.075
            xx = x + dx
            ax.plot(xx, vals, color=to_rgba(col, .86), lw=1.05, ls=ls,
                    marker=marker, ms=6.2, mfc=to_rgba(col, .78),
                    mec=deep_color(col), mew=.65, zorder=3)
            # D7 carries the central physical comparison, so label every
            # measured point.  D1 labels are kept at the right edge to avoid
            # a stack of nearly coincident 100% labels at the top of the row.
            if annotate_all:
                for xi, yi in zip(xx, vals):
                    ax.text(xi, yi + (6.0 if made else -6.0), f"{yi:.0f}",
                            ha="center", va="bottom" if made else "top",
                            fontsize=6.0, color=deep_color(col))
            else:
                yi = float(vals[-1])
                # Use one common label column rather than letting the two
                # near-100% endpoint labels collide with each other and with
                # the computed-only square.
                label_y = yi - 1.3 if made else yi + 1.3
                ax.annotate(f"{yi:.0f}%  {label}", xy=(xx[-1], yi),
                            xytext=(3.22, label_y), textcoords="data",
                            ha="left", va="top" if made else "bottom",
                            fontsize=6.2, color=deep_color(col),
                            arrowprops=dict(arrowstyle="-", lw=0.45,
                                            color=to_rgba(col, .65),
                                            shrinkA=2, shrinkB=2))
        ax.set_ylabel("Satisfaction (%)", labelpad=2)
        ax.text(0.01, title_y, title, transform=ax.transAxes, ha="left",
                va=title_va, fontsize=7.2, color=INK, fontweight="bold")
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 50, 100])
        clean(ax)

    # Keep the mechanism labels readable in the panel; the exact bounds
    # (rho >= 0.735 and N_distinct/N_sites <= 2/3) are retained in the
    # caption/status record.  The D1 profile is almost exactly at 100%, so its
    # label sits in the open lower-left area rather than over the data.
    profile(top, "D1", "Law 1  short-range repulsion", annotate_all=False,
            title_y=0.68)
    profile(bottom, "D7", "Law 7  crystallographic site complexity", annotate_all=True,
            title_y=1.01, title_va="bottom")
    top.tick_params(labelbottom=False)
    bottom.set_xticks(x)
    bottom.set_xticklabels([label for _, _, label in states],
                           fontsize=6.4, linespacing=1.05)
    bottom.set_xlabel("Energy and phonon state", labelpad=4)
    bottom.set_xlim(-0.42, 3.62)
    if panel_letter:
        outer.text(-0.14, 1.04, str(panel_letter), transform=outer.transAxes,
                   fontsize=10, fontweight="bold", va="bottom", color=INK)
    return {
        "source": str(threeaxis),
        "n_structures": int(table["n"].sum()),
        "state_order": [label.replace("\n", " / ") for _, _, label in states],
        "n_by_state": n_by_state,
        "d1_satisfaction": {
            "recorded": values["D1"][True].tolist(),
            "computed_only": values["D1"][False].tolist(),
        },
        "d7_satisfaction": {
            "recorded": values["D7"][True].tolist(),
            "computed_only": values["D7"][False].tolist(),
        },
        "bounds": {
            "D1": "rho >= 0.735 (reduced-contact bound)",
            "D7": "N_distinct/N_sites <= 2/3 (distinct-site bound)",
        },
        "charge_evaluable": True,
    }


def load_design_queues(
    forward_path: Path = FORWARD_DESIGN_DEFAULT,
    inverse_path: Path = INVERSE_DESIGN_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """Load the fixed forward-calibration and MatterGen inverse queues."""

    forward_all = pd.read_parquet(forward_path)
    inverse_all = pd.read_parquet(inverse_path)
    require(
        forward_all,
        {
            "fit_valid",
            "bulk_modulus_gpa",
            "made",
            "synthesis_score",
            "rung_L1_verdict",
            "rung_L2_verdict",
            "rung_L3_verdict",
            "rung_L4_verdict",
        },
        forward_path,
    )
    require(
        inverse_all,
        {
            "fit_valid",
            "clamped_bulk_modulus_proxy_gpa",
            "synthesis_score",
            "rung_L1_verdict",
            "rung_L2_verdict",
            "rung_L3_verdict",
            "rung_L4_verdict",
        },
        inverse_path,
    )
    forward_mask = (
        forward_all["fit_valid"].astype(bool)
        & pd.to_numeric(forward_all["bulk_modulus_gpa"], errors="coerce").ge(200)
    )
    inverse_mask = (
        inverse_all["fit_valid"].astype(bool)
        & pd.to_numeric(
            inverse_all["clamped_bulk_modulus_proxy_gpa"], errors="coerce"
        ).ge(200)
    )
    forward = forward_all.loc[forward_mask].reset_index(drop=True)
    inverse = inverse_all.loc[inverse_mask].reset_index(drop=True)
    priority = pd.to_numeric(
        inverse["clamped_bulk_modulus_proxy_gpa"], errors="coerce"
    ).ge(400).to_numpy(bool)
    if not priority.any():
        raise ValueError("inverse queue contains no UMA-proxy candidates at or above 400 GPa")
    complete_column = next(
        (name for name in ("synthesis_all_observed", "all_observed") if name in inverse),
        None,
    )
    observed_column = next(
        (name for name in ("synthesis_n_observed", "n_observed") if name in inverse),
        None,
    )
    observed_terms = (
        {
            str(int(key)): int(value)
            for key, value in pd.to_numeric(
                inverse[observed_column], errors="coerce"
            ).value_counts(dropna=False).sort_index().items()
            if pd.notna(key)
        }
        if observed_column
        else {}
    )
    return forward, inverse, priority, {
        "forward_source": str(forward_path),
        "inverse_source": str(inverse_path),
        "forward_all_n": int(len(forward_all)),
        "forward_calibration_n": int(len(forward)),
        "forward_experimental_n": int(forward["made"].astype(bool).sum()),
        "forward_theoretical_n": int((~forward["made"].astype(bool)).sum()),
        "inverse_all_n": int(len(inverse_all)),
        "inverse_queue_n": int(len(inverse)),
        "inverse_priority_n": int(priority.sum()),
        "inverse_score_complete_n": (
            int(inverse[complete_column].fillna(False).astype(bool).sum())
            if complete_column
            else None
        ),
        "inverse_score_observed_terms": observed_terms,
        "forward_property_gate": "valid UMA fit and bulk_modulus_gpa >= 200",
        "inverse_property_gate": "valid UMA fit and clamped_bulk_modulus_proxy_gpa >= 200",
        "inverse_priority_gate": "clamped_bulk_modulus_proxy_gpa >= 400",
    }


def build_inverse_design_diagnostics(
    inverse: pd.DataFrame,
    *,
    priority_mask,
    cutoff: float,
    screened_example_id: str = INVERSE_SCREENED_EXAMPLE_ID,
    retained_example_id: str = INVERSE_RETAINED_EXAMPLE_ID,
) -> dict:
    """Explain a fixed PSS operating point using measured queue descriptors.

    The MatterGen queue supplies only the site-economy and atomic-volume terms
    of PSS.  This function therefore reports those two observed quantities and
    the independent D7/distance verdicts without attributing information to the
    four frozen-median terms.
    """

    required = {
        "candidate_id",
        "archive_member",
        "cif_sha256",
        "formula",
        "num_sites",
        "synthesis_score",
        "formula_syn_wyckoff_econ_001",
        "formula_syn_vol_per_atom",
        "clamped_bulk_modulus_proxy_gpa",
        "pred_D7_verdict",
        "distance_0.7_verdict",
        "rung_L4_verdict",
    }
    require(inverse, required, INVERSE_DESIGN_DEFAULT)
    priority = np.asarray(tuple(priority_mask), dtype=bool)
    if len(priority) != len(inverse) or not priority.any():
        raise ValueError("priority mask must match the inverse queue and be non-empty")
    score = pd.to_numeric(inverse["synthesis_score"], errors="raise").to_numpy(float)
    if not np.isfinite(float(cutoff)) or not np.isfinite(score).all():
        raise ValueError("diagnostic cutoff and PSS values must be finite")
    screened = score < float(cutoff)
    retained = ~screened
    if not screened.any() or not retained.any():
        raise ValueError("diagnostic cutoff must split the inverse queue")

    site_fraction = pd.to_numeric(
        inverse["formula_syn_wyckoff_econ_001"], errors="raise"
    ).to_numpy(float)
    atomic_volume = pd.to_numeric(
        inverse["formula_syn_vol_per_atom"], errors="raise"
    ).to_numpy(float)
    d7_violation = inverse["pred_D7_verdict"].astype(str).eq("reject").to_numpy()
    distance_pass = inverse["distance_0.7_verdict"].astype(str).eq("pass").to_numpy()
    l4_violation = inverse["rung_L4_verdict"].astype(str).eq("reject").to_numpy()

    by_id = inverse.set_index("candidate_id", drop=False)
    examples = []
    for role, candidate_id, expected_mask in (
        ("screened", screened_example_id, screened),
        ("retained high property", retained_example_id, retained & priority),
    ):
        if candidate_id not in by_id.index:
            raise ValueError(f"diagnostic example is absent: {candidate_id}")
        positions = np.flatnonzero(inverse["candidate_id"].astype(str).eq(candidate_id))
        if len(positions) != 1 or not bool(expected_mask[positions[0]]):
            raise ValueError(f"diagnostic example is outside its declared cohort: {candidate_id}")
        row = by_id.loc[candidate_id]
        examples.append(
            {
                "role": role,
                "candidate_id": str(candidate_id),
                "archive_member": str(row["archive_member"]),
                "cif_sha256": str(row["cif_sha256"]),
                "formula": str(row["formula"]),
                "num_sites": int(row["num_sites"]),
                "synthesis_score": float(row["synthesis_score"]),
                "site_fraction": float(row["formula_syn_wyckoff_econ_001"]),
                "volume_per_atom": float(row["formula_syn_vol_per_atom"]),
                "bulk_modulus_proxy_gpa": float(
                    row["clamped_bulk_modulus_proxy_gpa"]
                ),
            }
        )

    screened_n = int(screened.sum())
    return {
        "cutoff": float(cutoff),
        "screened_n": screened_n,
        "retained_n": int(retained.sum()),
        "priority_total_n": int(priority.sum()),
        "priority_retained_n": int((priority & retained).sum()),
        "screened_distance_0p7_pass_n": int((screened & distance_pass).sum()),
        "screened_d7_violation_n": int((screened & d7_violation).sum()),
        "screened_l4_violation_n": int((screened & l4_violation).sum()),
        "l4_violation_n": int(l4_violation.sum()),
        "l4_priority_screened_n": int((priority & l4_violation).sum()),
        "screened_site_fraction_mean": float(site_fraction[screened].mean()),
        "screened_site_fraction_median": float(np.median(site_fraction[screened])),
        "screened_volume_per_atom_mean": float(atomic_volume[screened].mean()),
        "screened_volume_per_atom_median": float(np.median(atomic_volume[screened])),
        "screened_volume_per_atom_range": [
            float(atomic_volume[screened].min()),
            float(atomic_volume[screened].max()),
        ],
        "retained_volume_per_atom_mean": float(atomic_volume[retained].mean()),
        "retained_volume_per_atom_median": float(np.median(atomic_volume[retained])),
        "examples": examples,
    }


def _load_diagnostic_structures(archive_path: Path, examples: list[dict]):
    """Load and hash-check the two exact MatterGen structures used in Fig. 4f."""

    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structures = []
    with ZipFile(archive_path, "r") as archive:
        members = set(archive.namelist())
        for example in examples:
            member = example["archive_member"]
            if member not in members:
                raise ValueError(f"diagnostic CIF is absent from archive: {member}")
            raw = archive.read(member)
            if hashlib.sha256(raw).hexdigest() != example["cif_sha256"]:
                raise ValueError(f"diagnostic CIF hash mismatch: {member}")
            structure = Structure.from_str(raw.decode("utf-8"), fmt="cif")
            symbol = SpacegroupAnalyzer(structure, symprec=0.01).get_space_group_symbol()
            enriched = dict(example)
            enriched["space_group"] = str(symbol)
            structures.append((structure, enriched))
    return structures


def _draw_structure_thumbnail(ax, structure) -> dict:
    """Render a comparable 2 x 2 x 1 supercell in a normalized crystal frame."""

    rendered = structure.copy()
    rendered.make_supercell([2, 2, 1])
    display_dimensions = np.asarray([2.0, 2.0, 1.0])
    centre = display_dimensions / 2
    display_coordinates = np.asarray(rendered.frac_coords, dtype=float)
    display_coordinates = display_coordinates * display_dimensions - centre
    corners = np.asarray(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
        dtype=float,
    )
    cell_segments: dict[tuple, np.ndarray] = {}
    for shift_a in (0, 1):
        for shift_b in (0, 1):
            local_corners = (
                corners + np.asarray([shift_a, shift_b, 0], dtype=float)
                - centre
            )
            for first in range(8):
                for second in range(first + 1, 8):
                    if np.abs(corners[first] - corners[second]).sum() != 1:
                        continue
                    points = local_corners[[first, second]]
                    key = tuple(sorted(tuple(np.round(point, 8)) for point in points))
                    cell_segments[key] = points
    atomic_numbers = [int(site.specie.Z) for site in rendered]
    symbols = [site.specie.symbol for site in rendered]
    colours = [STRUCTURE_ELEMENT_COLOURS.get(symbol, SLATE) for symbol in symbols]
    edge_colours = [deep_color(colour, 0.64) for colour in colours]
    sizes = [1.5 + 0.04 * number for number in atomic_numbers]
    ax.scatter(
        display_coordinates[:, 0], display_coordinates[:, 1], display_coordinates[:, 2],
        s=sizes, c=colours, edgecolors=edge_colours, linewidths=0.16,
        alpha=0.90, depthshade=False, zorder=2,
    )
    # Draw the four constituent cells after the atoms so that the 2 x 2
    # periodicity remains legible even for the denser P1 example.
    for points in cell_segments.values():
        ax.plot(
            points[:, 0], points[:, 1], points[:, 2],
            color="#989A9D", lw=0.26, alpha=0.58, zorder=4,
        )
    extent = 1.10
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_proj_type("ortho")
    ax.view_init(
        elev=STRUCTURE_VIEW["elevation_deg"],
        azim=STRUCTURE_VIEW["azimuth_deg"],
    )
    ax.set_box_aspect([1, 1, 1], zoom=STRUCTURE_ZOOM)
    ax.set_axis_off()
    return {
        "supercell": [2, 2, 1],
        "rendered_atoms": int(len(rendered)),
        "maximum_marker_area_pt2": float(max(sizes)),
        "view": dict(STRUCTURE_VIEW),
        "orientation": "a, b and c aligned in a common normalized crystallographic frame",
        "display_dimensions": [2.0, 2.0, 1.0],
        "zoom": STRUCTURE_ZOOM,
        "lattice_line": {
            "colour": "#989A9D",
            "width_pt": 0.26,
            "alpha": 0.58,
        },
        "element_colours": {
            symbol: STRUCTURE_ELEMENT_COLOURS.get(symbol, SLATE)
            for symbol in sorted(set(symbols))
        },
    }


def draw_inverse_design_diagnostics(
    ax: plt.Axes,
    diagnostics: dict,
    *,
    archive_path: Path,
    inverse: pd.DataFrame,
) -> dict:
    """Draw a descriptor map and matched structures in Fig. 4f's empty region."""

    loaded = _load_diagnostic_structures(Path(archive_path), diagnostics["examples"])
    site_fraction = pd.to_numeric(
        inverse["formula_syn_wyckoff_econ_001"], errors="raise"
    ).to_numpy(float)
    atomic_volume = pd.to_numeric(
        inverse["formula_syn_vol_per_atom"], errors="raise"
    ).to_numpy(float)
    score = pd.to_numeric(inverse["synthesis_score"], errors="raise").to_numpy(float)
    screened = score < float(diagnostics["cutoff"])

    descriptor_ax = ax.inset_axes([0.070, 0.070, 0.330, 0.402])
    descriptor_ax.set_label("f4f-descriptor-map")
    descriptor_ax.scatter(
        site_fraction[~screened], atomic_volume[~screened],
        s=F4_DESCRIPTOR_POINT_AREA_PT2,
        facecolors=to_rgba(F4_DIAGNOSTIC_BLUE, F4_DESCRIPTOR_FACE_ALPHA),
        edgecolors=to_rgba(F4_DIAGNOSTIC_BLUE, F4_DESCRIPTOR_EDGE_ALPHA),
        linewidths=F4_DESCRIPTOR_EDGE_WIDTH, zorder=1,
    )
    descriptor_ax.scatter(
        site_fraction[screened], atomic_volume[screened],
        s=F4_DESCRIPTOR_POINT_AREA_PT2,
        facecolors=to_rgba(RED, F4_DESCRIPTOR_FACE_ALPHA),
        edgecolors=to_rgba(RED, F4_DESCRIPTOR_EDGE_ALPHA),
        linewidths=F4_DESCRIPTOR_EDGE_WIDTH, zorder=2,
    )

    design = np.column_stack((np.ones(len(inverse)), site_fraction, atomic_volume))
    intercept, site_weight, volume_weight = np.linalg.lstsq(design, score, rcond=None)[0]
    reconstructed = design @ np.array([intercept, site_weight, volume_weight])
    if float(np.max(np.abs(reconstructed - score))) > 1e-8:
        raise ValueError("inverse-queue PSS is not an exact two-descriptor score")
    boundary_x = np.linspace(0.18, 1.04, 300)
    boundary_y = (
        float(diagnostics["cutoff"]) - intercept - site_weight * boundary_x
    ) / volume_weight
    descriptor_ax.plot(
        boundary_x, boundary_y, color=to_rgba(PSSC, F4_GUIDE_ALPHA),
        lw=F4_GUIDE_LINE_WIDTH, ls=(0, (3, 2)), zorder=3,
    )
    descriptor_ax.set_xlim(0.18, 1.04)
    descriptor_ax.set_ylim(8.3, 16.9)
    descriptor_ax.set_xticks([0.2, 0.6, 1.0])
    descriptor_ax.set_yticks([10, 13, 16])
    descriptor_ax.text(
        0.97, 0.04, r"$\eta_{\mathrm{site}}$", transform=descriptor_ax.transAxes,
        ha="right", va="bottom", fontsize=7.4, color=INK,
    )
    descriptor_ax.text(
        0.09, 0.96, r"$V/N$", transform=descriptor_ax.transAxes,
        ha="left", va="top", fontsize=7.4, color=INK,
    )
    descriptor_ax.tick_params(
        axis="x", labelsize=6.8, pad=1.0, width=0.45, length=2.0,
    )
    descriptor_ax.tick_params(
        axis="y", labelsize=6.8, pad=F4_DESCRIPTOR_TICK_PAD,
        width=0.45, length=2.0,
    )
    descriptor_ax.spines["left"].set_linewidth(0.45)
    descriptor_ax.spines["bottom"].set_linewidth(0.45)
    descriptor_ax.spines["top"].set_visible(False)
    descriptor_ax.spines["right"].set_visible(False)
    descriptor_ax.set_facecolor("white")
    descriptor_ax.grid(False)

    positions = STRUCTURE_INSET_POSITIONS
    colours = (RED, F4_DIAGNOSTIC_BLUE)
    for role_index, ((structure, example), position, colour) in enumerate(
        zip(loaded, positions, colours, strict=True)
    ):
        point = (example["site_fraction"], example["volume_per_atom"])
        selected_marker = descriptor_ax.scatter(
            [point[0]], [point[1]], s=F4_MARKER_AREA_PT2, marker="D",
            facecolors=to_rgba(colour, F4_MARKER_FACE_ALPHA),
            edgecolors=deep_color(colour),
            linewidths=F4_MARKER_EDGE_WIDTH, zorder=5, clip_on=False,
        )
        selected_marker.set_gid("f4-primary-marker")
        inset = ax.inset_axes(position, projection="3d")
        _draw_structure_thumbnail(inset, structure)
        ax.add_patch(
            Rectangle(
                (position[0], position[1]), position[2], position[3],
                transform=ax.transAxes, fill=False,
                edgecolor=soft_structure_frame_color(colour),
                linewidth=F4_STRUCTURE_FRAME_WIDTH, zorder=7, clip_on=False,
            )
        )
        fallback_role = ("screened", "retained")[role_index]
        role_label = str(example.get("role", fallback_role)).split()[0].capitalize()
        ax.text(
            position[0] + 0.0215,
            position[1] + position[3] / 2,
            role_label,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.0,
            color=deep_color(colour),
            zorder=8,
            clip_on=False,
        )
        ax.add_artist(
            ConnectionPatch(
                xyA=point, coordsA=descriptor_ax.transData,
                xyB=(position[0], position[1] + position[3] / 2),
                coordsB=ax.transAxes, color=soft_structure_frame_color(colour),
                linewidth=F4_GUIDE_LINE_WIDTH, alpha=F4_GUIDE_ALPHA,
                zorder=6, clip_on=False,
            )
        )

    work_x = 100 * diagnostics["priority_retained_n"] / diagnostics["priority_total_n"]
    work_y = 100 * diagnostics["screened_n"] / (
        diagnostics["screened_n"] + diagnostics["retained_n"]
    )
    workpoint = ax.scatter(
        [work_x], [work_y], s=F4_MARKER_AREA_PT2,
        facecolors=to_rgba(PSSC, F4_MARKER_FACE_ALPHA),
        edgecolors=deep_color(PSSC), linewidths=F4_MARKER_EDGE_WIDTH,
        zorder=8, clip_on=False,
    )
    workpoint.set_gid("f4-primary-marker")
    diagnostics["workpoint"] = {
        "priority_retention": float(work_x / 100),
        "queue_reduction": float(work_y / 100),
    }
    diagnostics["thumbnail_count"] = len(loaded)
    diagnostics["example_archive"] = str(archive_path)
    diagnostics["examples"] = [example for _, example in loaded]
    diagnostics["graphic_inset"] = {
        "queue_points": int(len(inverse)),
        "screened_points": int(screened.sum()),
        "decision_boundary": True,
        "variables": ["site fraction", "atomic volume"],
    }
    return diagnostics



# the property target the inverse-design run was conditioned on
DESIGN_TARGET_GPA = 400.0


def _dft_design_frontier(forward, inverse, target_retentions, calibration_mask):
    """The same sweep read against a first-principles high-property subset.

    Returns None when the campaign's results are absent, so the figure still builds from
    the archived data alone.  The threshold is the design target carried onto the DFT
    scale by the measured proxy-to-DFT ratio; see dft/PREREG-DFT.md, amendment 9.
    """
    import json
    from pathlib import Path as _P

    # the curve is an addition to an archived panel, so anything missing -- the campaign's
    # results, or the identifier that joins them to the queue, as in the synthetic frames
    # the tests build -- drops it silently and leaves the panel as it was
    if "candidate_id" not in inverse:
        return None
    path = _P(__file__).resolve().parents[2] / "dft" / "E4_design" / "bulk_moduli.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    measured = {r["candidate_id"]: (r["dft_bulk_modulus_gpa"], r["uma_bulk_modulus_gpa"])
                for r in rows if r.get("candidate_id")
                and r.get("dft_bulk_modulus_gpa") and r.get("uma_bulk_modulus_gpa")}
    if not measured:
        return None
    ratio = float(np.median([d / u for d, u in measured.values()]))
    threshold = DESIGN_TARGET_GPA * ratio
    ids = inverse["candidate_id"].astype(str).to_numpy()
    mask = np.array([measured.get(i, (0.0, 0.0))[0] >= threshold for i in ids], dtype=bool)
    if not mask.any():
        return None
    frontier = build_design_frontier(
        forward, inverse, score_column="synthesis_score",
        target_retentions=target_retentions, inverse_priority_mask=mask,
        experimental_calibration_mask=calibration_mask,
    )
    return frontier, {
        "n_measured": int(len(measured)),
        "n_priority": int(mask.sum()),
        "proxy_to_dft_ratio": ratio,
        "threshold_gpa": float(threshold),
        "legend": f"PSS, DFT \u2265{threshold:.0f} GPa",
    }


def panel_f(
    ax: plt.Axes,
    forward: pd.DataFrame,
    inverse: pd.DataFrame,
    *,
    inverse_priority_mask,
    target_retentions=None,
    inverse_priority_label: str = "High-property candidates retained (%)",
    diagnostic_target_retention: float | None = None,
    diagnostic_archive: Path | None = None,
) -> dict:
    """Prospective queue reduction for a property-conditioned inverse task."""

    if target_retentions is None:
        target_retentions = np.linspace(0.84, 1.0, 161)
    target_retentions = tuple(float(value) for value in target_retentions)
    priority = np.asarray(tuple(inverse_priority_mask), dtype=bool)
    if len(priority) != len(inverse) or not priority.any():
        raise ValueError("inverse priority mask must match the queue and be non-empty")
    calibration_mask, calibration_support = support_matched_calibration_mask(
        forward,
        inverse,
    )
    frontier = build_design_frontier(
        forward,
        inverse,
        score_column="synthesis_score",
        target_retentions=target_retentions,
        inverse_priority_mask=priority,
        experimental_calibration_mask=calibration_mask,
    )
    rule_columns = {
        "L1": "rung_L1_verdict",
        "L2": "rung_L2_verdict",
        "L3": "rung_L3_verdict",
        "L4": "rung_L4_verdict",
    }
    rule_points = pd.DataFrame(
        [
            rule_operating_point(
                forward,
                inverse,
                verdict_column=column,
                method=method,
                inverse_priority_mask=priority,
            )
            for method, column in rule_columns.items()
        ]
    )
    # Set 1--Set 3 coincide at the far right of this panel, where all three retain the whole
    # high-property subset and remove almost none of the queue.  Three nested markers on one
    # point crowded the legend without adding a measurement, so only Set 4, the operating
    # point the text compares PSS against, is drawn; the other three stay in STATUS.json.
    drawn_rules = rule_points.loc[rule_points["method"].eq("L4")].reset_index(drop=True)
    visual = draw_design_panel(
        ax,
        frontier,
        drawn_rules,
        inverse_priority_label=inverse_priority_label,
        panel_letter="f",
        visual_style=F4_DESIGN_VISUAL_STYLE,
    )
    ax.set_ylabel("DFT-validation queue reduction (%)", labelpad=3)
    # With those markers gone the curve no longer needs headroom above its maximum, and
    # lowering the ceiling lifts the whole trade-off, which is what opens the lower-left
    # corner for the descriptor inset and the two structure thumbnails.
    y_top = 1.05 * max(
        100 * float(frontier["inverse_queue_reduction"].max()),
        100 * float(drawn_rules["inverse_queue_reduction"].max()),
    ) + 1.5
    ax.set_ylim(-2.0, y_top)
    visual["plot_limits"]["y"] = [-2.0, float(y_top)]

    # The high-property subset on this axis is defined by a machine-learned proxy.  Bulk
    # moduli computed with DFT for 260 of these candidates let the same sweep be read
    # against a subset defined from first principles, and the two curves share their
    # thresholds and their y axis exactly: only the definition of "high property" differs,
    # so the horizontal gap between them is the whole of the disagreement.  The proxy runs
    # high, so the target is carried onto the DFT scale by the measured ratio rather than
    # transferred as an absolute number, which would test the proxy's calibration instead
    # of the screen's selection.
    dft_curve = _dft_design_frontier(forward, inverse, target_retentions,
                                     calibration_mask)
    if dft_curve is not None:
        dft_frontier, dft_note = dft_curve
        (dft_line,) = ax.plot(
            100 * dft_frontier["inverse_priority_retention"].to_numpy(float),
            100 * dft_frontier["inverse_queue_reduction"].to_numpy(float),
            color=to_rgba(EHULL, F4_DATA_LINE_ALPHA),
            lw=F4_DATA_LINE_WIDTH, ls="--", zorder=4,
        )
        dft_line.set_gid("f4-dft-reference-series")
        from matplotlib.lines import Line2D as _L2D
        leg = ax.get_legend()
        handles = list(leg.legend_handles)
        labels = [t.get_text() for t in leg.get_texts()]
        handles.insert(
            1,
            _L2D(
                [0], [0], color=to_rgba(EHULL, F4_DATA_LINE_ALPHA),
                lw=F4_DATA_LINE_WIDTH, ls="--",
            ),
        )
        labels.insert(1, dft_note["legend"])
        # the first entry named the score alone; with two subsets on one axis it has to
        # name which subset its curve is drawn against
        labels[0] = f"PSS, predicted \u2265{DESIGN_TARGET_GPA:.0f} GPa"
        ax.legend(handles=handles, labels=labels, loc="upper right", frameon=False,
                  ncol=1, columnspacing=0.8, handlelength=1.5, handletextpad=0.45,
                  borderaxespad=0.2)
        visual["dft_frontier"] = dft_note
    result = {
        "task": "inverse_design_queue_tradeoff",
        "score": "PRIS-derived synthesis score (PSS)",
        "calibration": "thresholds fixed by descriptor-support-matched forward experimental structures",
        "calibration_support": calibration_support,
        "inverse_total_n": int(len(inverse)),
        "inverse_priority_total_n": int(priority.sum()),
        "inverse_priority_definition": "UMA bulk-modulus proxy >= 400 GPa",
        "rule_operating_points": rule_points.to_dict(orient="records"),
        "frontier": frontier.to_dict(orient="records"),
        **visual,
    }
    if diagnostic_target_retention is not None:
        offsets = np.abs(
            frontier["target_experimental_retention"].to_numpy(float)
            - float(diagnostic_target_retention)
        )
        chosen = frontier.iloc[int(np.argmin(offsets))]
        if float(offsets.min()) > 1e-9:
            raise ValueError("diagnostic target retention is absent from the frontier")
        diagnostics = build_inverse_design_diagnostics(
            inverse,
            priority_mask=priority,
            cutoff=float(chosen.cutoff_score),
        )
        if diagnostic_archive is None:
            raise ValueError("diagnostic archive is required for structure thumbnails")
        result["diagnostics"] = draw_inverse_design_diagnostics(
            ax,
            diagnostics,
            archive_path=Path(diagnostic_archive),
            inverse=inverse,
        )
    return result


def render(output_dir: Path, *, ehull_path: Path | None = EHULL_DEFAULT,
           binary_dir: Path = ROOT / "outputs/20260821_pu_synthesizability/analysis_v1",
           independent_dir: Path = ROOT / "outputs/20260822_pu_formula_scores/independent_choices_v1",
           full_pool: Path = FULL_POOL_DEFAULT, formula_npz: Path = FORMULA_NPZ_DEFAULT,
           forward_design: Path = FORWARD_DESIGN_DEFAULT,
           inverse_design: Path = INVERSE_DESIGN_DEFAULT,
           consensus_path: Path | None = None) -> dict:
    style(); output_dir.mkdir(parents=True, exist_ok=True)
    cdata, cmeta, ehull, emeta = _load_c(binary_dir, independent_dir, ehull_path)
    ddata = load_d_series(full_pool, formula_npz)
    design_forward, design_inverse, design_priority, design_meta = load_design_queues(
        forward_design,
        inverse_design,
    )
    # Match the 18.3-cm source width used by the original TeX figures while
    # avoiding the artificially narrow appearance of the former 24-cm-tall
    # canvas.  Row-specific subgrids let a/b retain room for their categorical
    # labels and give the denser c/f panels a materially wider plotting area.
    fig = plt.figure(figsize=(18.3 * CM, 20.0 * CM), facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=(1.06, 0.95, 1.18),
                             left=0.087, right=0.982, bottom=0.073, top=0.982,
                             hspace=0.30)
    top_row = outer[0].subgridspec(1, 2, width_ratios=(1.0, 1.03), wspace=0.42)
    mid_row = outer[1].subgridspec(1, 2, width_ratios=(1.0, 1.02), wspace=0.28)
    bottom_row = outer[2].subgridspec(1, 2, width_ratios=(1.0, 1.02), wspace=0.28)
    a = fig.add_subplot(top_row[0, 0]); b = fig.add_subplot(top_row[0, 1])
    c = fig.add_subplot(mid_row[0, 0]); d = fig.add_subplot(mid_row[0, 1])
    e = fig.add_subplot(bottom_row[0, 0]); f = fig.add_subplot(bottom_row[0, 1])
    records = {
        "a": panel_a(a), "b": panel_b(b),
        "c": panel_c(c, cdata, ehull), "d": panel_d(d, ddata),
        "e": panel_e(e, F3_DEFAULT),
        "f": panel_f(
            f,
            design_forward,
            design_inverse,
            inverse_priority_mask=design_priority,
            diagnostic_target_retention=INVERSE_DIAGNOSTIC_TARGET,
            diagnostic_archive=INVERSE_ARCHIVE_DEFAULT,
        ),
    }
    stem = output_dir / "pris_fig45_merged_nature"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", metadata={"Title": "PRIS Figure 4 a-f"})
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    status = {
        "figure_scope": "production main-text Figure 4 a-f",
        "style": {"font": "Arial", "grid": False, "background": "white", "palette": "manuscript"},
        "layout": {"source_size_cm": [18.3, 20.0],
                   "intended_tex_width": "\\textwidth (152 mm)",
                   "row_specific_column_spacing": True},
        "logic": "controlled mechanism benchmark a-b; complete Set 1-Set 4 ladder plus the PRIS-derived synthesis score (PSS) and MatterSim E_hull threshold c; cross-PU-model relation d; held-out PSS and DFT E_hull energy e; support-matched property-conditioned inverse-design screening with descriptor-space and matched-structure diagnosis f",
        "panel_records": records,
        "panel_c_ehull": emeta,
        "panel_d": records["d"],
        "panel_f_design_data": design_meta,
        "formula_vs_L4": {"matched_advantage_pp": records["c"]["advantage_pp"],
                           "interpretation": "PSS exceeds Set 4 at the matched screening point, not at every satisfaction value"},
    }
    (output_dir / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "README.md").write_text(
        "# Full Figure 4 a–f additive redraw\n\n"
        "Panels a/b restore the original controlled-damage values, cell labels, group separators, and blind-region rectangle. "
        "Panel c shows the complete Set 1-Set 4 law ladder (94.8/0.2, 88.3/1.8, 87.1/1.9, and 80.7/51.9% satisfaction/screening) together with the task-specific PRIS-derived synthesis score (PSS) frontier and two reference choices. "
        "Panel d shows three explicitly named PU-model curves in both rows. "
        "Panel e plots PSS and DFT E_hull energy on held-out composition pairs. "
        "Panel f transfers descriptor-support-matched PSS thresholds to 1,081 MatterGen candidates, shows generated-queue reduction against retention of the 140 candidates with UMA bulk-modulus proxy at or above 400 GPa, and visualises the two observed PSS descriptors with a matched-composition structural pair. "
        "The former energy/phonon/record panel remains available as an SI candidate through panel_physical_states_si. "
        "The figure uses the manuscript Arial/colour/no-grid style.\n",
        encoding="utf-8")
    (output_dir / "CAPTION_DRAFT.md").write_text(CAPTION_DRAFT, encoding="utf-8")
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--ehull", type=Path, default=EHULL_DEFAULT)
    args = ap.parse_args()
    status = render(args.output_dir, ehull_path=args.ehull)
    print(json.dumps({"status": str(args.output_dir / "STATUS.json"), "advantage_pp": status["formula_vs_L4"]["matched_advantage_pp"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
