#!/usr/bin/env python3
"""Fig. 5 — physical plausibility beyond distance and relaxation energy.

Three rows of two panels, read left to right:

a  distance-filter satisfaction versus Set 4 satisfaction across generators
b  MLIP relaxation energy for GNoME parents and controlled damage
c  MLIP relaxation energy for MatterGen structures, resolved by Set 1--Set 4 verdict
d  the distinct-site law Law 7, resolved by the GNoME space-group label
e  a fixed-coordinate wrong-site example and paired detection benchmark
f  cost per structure, from the contact law to DFT

Panels b and c share one x scale and one decade width; c stops just past the
displacement median, because nothing it plots reaches the tail that b has to show.

Data are frozen under ``paper/data`` from the protocol-frozen external runs.
"""
from __future__ import annotations
import json, pathlib, sys
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figs import (
    plt, np, W2, CM, BLU, RED, ORA, GRN, GRY, PUR, OUT, stamp,
    style_bar_container,
)
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import offset_copy

DATA = pathlib.Path(__file__).resolve().parent.parent / "paper" / "data"
# results of the pre-registered first-principles campaign (dft/PREREG-DFT.md)
DFT = pathlib.Path(__file__).resolve().parent.parent / "dft"
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 400,
})
L4C = "#0A5A3C"          # satisfies the seven-law rule set
PUR2 = "#AE82C3"         # chemical order, contact laws unavailable
INK = "#323335"
SOFT = "#757779"
FAMC = {"free": INK, "sym": INK, "real": L4C}
FS = 8.0                 # in-panel annotation size, at the page scale of Fig. 5
F5D_MARKER_EDGE_ALPHA = 0.40
F5D_MARKER_EDGE_WIDTH = 0.85
F5B_MARKER_FACE_ALPHA = 0.24
F5B_MARKER_EDGE_ALPHA = 0.88
F5B_MARKER_EDGE_WIDTH = 0.85
ROW_CM = (5.30, 2.95, 3.05, 2.30, 3.44)
TOP_CM, BOT_CM = 1.00, 0.95
PANEL_ORDER = (
    "generator_screening",
    "controlled_relaxation",
    "mattergen_ladder",
    "gnome_site_complexity",
    "wrong_site_identity",
    "cost",
)

# c and d plot the same quantity on the same logarithmic decade.  Panel c has to reach the
# displacement tail; panel d ends just past the displacement median, because everything it
# plots sits below 1 eV per atom and the decades beyond that were empty.
E_LO, E_HI = 3e-4, 5e2
E_TICKS = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
E_HI_D = 2e1
E_TICKS_D = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
E_LABEL = "Energy released on relaxation (eV per atom)"


def save(fig, name: str) -> None:
    """Export editable vectors and a high-resolution review raster."""
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=400)
    plt.close(fig)
    print("wrote", name)


def _darken(colour: str, factor: float = 0.68) -> tuple[float, float, float]:
    """Return a darker edge colour while preserving the marker hue."""
    from matplotlib.colors import to_rgb

    return tuple(factor * channel for channel in to_rgb(colour))


def _verdict(value) -> str:
    if pd.isna(value):
        return "not evaluated"
    if value is True or str(value).lower() == "true":
        return "satisfies"
    if value is False or str(value).lower() == "false":
        return "fails"
    raise ValueError(f"unrecognised verdict: {value!r}")


def ladder_verdict_counts(data: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Count satisfaction, failure and coverage separately for Set 1--Set 4."""
    out: dict[str, dict[str, int]] = {}
    for level in ("L1", "L2", "L3", "L4"):
        counts = data[level].map(_verdict).value_counts()
        out[level] = {
            state: int(counts.get(state, 0))
            for state in ("satisfies", "fails", "not evaluated")
        }
    return out


def wrong_site_detection_rates(frozen: dict) -> dict[str, dict[str, float]]:
    """Recover paired detection rates from mutually exclusive count cells."""
    out: dict[str, dict[str, float]] = {}
    for klass in ("S2", "S5"):
        row = frozen["classes"][klass]
        n = row["n"]
        out[klass] = {
            "coordinate checks": (row["both"] + row["coordinate_only"]) / n,
            "PRIS": (row["both"] + row["pris_only"]) / n,
        }
    return out


# ─────────────────────────────────────────────────────────────────────── (a)
def panel_a(fig, ax, gen):
    g = gen.set_index("model")
    ax.set_xlim(-0.045, 1.045)
    ax.set_ylim(0.882, 1.030)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.90, 0.95, 1.00])
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    ax.spines["left"].set_bounds(0.90, 1.00)
    ax.set_xlabel("Set 4 satisfaction")
    ax.set_ylabel("0.7-Å distance-filter\nsatisfaction")

    # family by marker shape alone; colour is reserved for the experimental-structure
    # control, so that no hue carries two meanings inside the figure
    MK = {"free": ("o", 6.0), "sym": ("s", 6.0), "real": ("D", 7.2)}
    # The three upper-left points are almost coincident.  Their display markers
    # are offset by only a few points and connected to small true-position anchors;
    # data coordinates and every quantitative comparison remain unchanged.
    DISPLAY_OFFSET = {
        "DiffCSP": (-2.8, 3.2),
        "MatterGen": (3.2, 2.0),
        "MiAD": (-2.8, -3.2),
    }
    marker_transforms = {}
    for m, r in g.iterrows():
        c = FAMC[r.family]
        mk, ms = MK[r.family]
        dx, dy = DISPLAY_OFFSET.get(m, (0.0, 0.0))
        transform = offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points")
        marker_transforms[m] = transform
        if dx or dy:
            ax.plot([r.pass_L4], [r.md07], ls="", marker="o", ms=2.2,
                    mfc=c, mec=c, alpha=0.28, zorder=3)
            ax.add_artist(ConnectionPatch(
                xyA=(r.pass_L4, r.md07), coordsA=ax.transData,
                xyB=(r.pass_L4, r.md07), coordsB=transform,
                color="#757779", lw=0.45, alpha=0.75, zorder=3,
            ))
        ax.plot([r.pass_L4], [r.md07], ls="", marker=mk, ms=ms,
                mfc="white" if r.family == "sym" else c, mec=c,
                mew=1.4 if r.family == "real" else 1.1, alpha=0.86,
                transform=transform, zorder=5)

    label_specs = {
        "DiffCSP": (0, 25, "center", "bottom"),
        "MatterGen": (12, 17, "left", "bottom"),
        "MiAD": (10, -23, "left", "top"),
        "SymmCD": (-8, -10, "right", "top"),
        "CrystalFormer": (-10, 10, "right", "bottom"),
        "DiffCSP++": (8, 18, "left", "bottom"),
        "WyFormer-DiffCSP++": (10, -18, "left", "top"),
        "MP20-test": (12, 10, "left", "bottom"),
    }
    display_name = {"MP20-test": "MP-20"}
    for model, (dx, dy, ha, va) in label_specs.items():
        row = g.loc[model]
        colour = L4C if model == "MP20-test" else INK
        ax.annotate(
            display_name.get(model, model),
            xy=(row.pass_L4, row.md07), xycoords=marker_transforms[model],
            xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=FS, color=colour,
            arrowprops=dict(
                arrowstyle="-", color=_darken(colour) if model == "MP20-test"
                else "#6E7072", lw=0.55, shrinkA=2.0, shrinkB=3.0,
            ),
            annotation_clip=False,
        )

    hs = [Line2D([], [], ls="", marker="o", ms=6.0, mfc=INK, mec=INK,
                 alpha=0.86),
          Line2D([], [], ls="", marker="s", ms=6.0, mfc="white", mec=INK,
                 mew=1.1, alpha=0.86)]
    leg = ax.legend(
        hs, ["no imposed symmetry", "symmetry imposed"],
        frameon=True, framealpha=1.0, facecolor="white", edgecolor="#B6B8BB",
        fancybox=False, fontsize=FS, loc="lower right", ncol=1,
        bbox_to_anchor=(0.995, 0.025), handlelength=0.9,
        handletextpad=0.35, labelspacing=0.50, borderpad=0.30,
    )
    leg.get_frame().set_linewidth(0.6)
    for text in leg.get_texts():
        text.set_color(INK)


# ─────────────────────────────────────────────────────────────────────── (b)
def panel_b(fig, ax, ml):
    e1 = ml[ml.experiment == "exp1"]
    SER = [("parent", INK, "GNoME parents", 1.5),
           ("S1", ORA, "D1 compression", 1.2),
           ("S4", BLU, "D4 expansion", 1.2),
           ("S3", RED, "D3 displacement", 1.2)]
    for k, c, lab, lw in SER:
        v = np.sort(np.clip(e1[e1.kind == k].e_released_per_atom.values,
                            E_LO, None))
        ax.step(np.concatenate([[E_LO], v]),
                np.concatenate([[0.0], np.arange(1, len(v) + 1) / len(v)]),
                where="post", color=c, lw=lw, zorder=4, label=lab)
    for k, c, dy in (("S2", GRN, 0.170), ("S5", PUR, 0.060)):
        v = np.clip(e1[e1.kind == k].e_released_per_atom.values, E_LO, None)
        ax.plot(v, np.full(len(v), dy), ls="", marker="|", ms=6.0, mew=1.2,
                color=c, zorder=5)
        ax.text(4.2e2, dy, f"D{k[1:]}  (n = {len(v)})", fontsize=FS, color=c,
                ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(E_LO, E_HI)
    ax.set_xticks(E_TICKS)
    ax.tick_params(axis="x", labelsize=8.8)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(E_LABEL)
    ax.set_ylabel("Cumulative fraction")
    # Every energy in this panel comes from a machine-learned potential.  The claim that
    # carries the most weight is the flat one: unmodified GNoME parents release almost
    # nothing, which is what makes a large release diagnostic of damage.  Twenty of those
    # parents were recomputed here with plane-wave DFT, and the dashed curve is their
    # release.  Only the parents are redrawn: the damaged cells in this campaign come from
    # experimental parents, a different population, and overlaying those would invite a
    # sampling difference to be read as a method difference.  The matched cell-by-cell
    # comparison against MatterSim is the Supplementary scatter.
    pair = json.loads((DFT / "E3_crosscheck" / "paired_energies.json").read_text())
    gp = np.sort(np.clip([r["dft_release_ions_ev_per_atom"] for r in pair
                          if r["kind"] == "gnome" and r["variant"] == "P0"], E_LO, None))
    ax.step(np.concatenate([[E_LO], gp]),
            np.concatenate([[0.0], np.arange(1, len(gp) + 1) / len(gp)]),
            where="post", color=INK, lw=1.2, ls=(0, (2.6, 1.6)), zorder=6)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, lw=1.2, ls=(0, (2.6, 1.6))))
    labels.append(f"GNoME parents, DFT (n = {len(gp)})")
    leg = ax.legend(handles, labels, frameon=False, fontsize=FS, loc="lower left",
                    ncol=2, bbox_to_anchor=(-0.008, 1.005), handlelength=1.4,
                    handletextpad=0.45, labelspacing=0.30, columnspacing=1.1,
                    borderpad=0.0)
    for t, c in zip(leg.get_texts(), [x[1] for x in SER] + [INK]):
        t.set_color(c)
    ax.text(4.2e2, 0.60, "n = 150 each", fontsize=FS, color=SOFT, ha="right",
            va="center")
    ax.axvline(0.05, color="#C2C4C7", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(0.042, 0.50, "0.05", fontsize=FS, color="#989A9D", rotation=90,
            ha="right", va="center")




# ─────────────────────────────────────────────────────────────────────── (c)
def panel_c(fig, ax, ladder):
    """Resolve relaxation energy by nested PRIS level without conflating coverage."""
    rng = np.random.default_rng(20260823)
    levels = ("L1", "L2", "L3", "L4")
    styles = {
        "not evaluated": (GRY, 0.14, 12),
        "satisfies": (L4C, 0.24, 18),
        "fails": (RED, 0.38, 22),
    }
    for row, level in enumerate(levels[::-1]):
        states = ladder[level].map(_verdict)
        for state in ("not evaluated", "satisfies", "fails"):
            values = np.clip(
                ladder.loc[states.eq(state), "e_released_per_atom"].to_numpy(float),
                E_LO,
                None,
            )
            if not len(values):
                continue
            colour, alpha, size = styles[state]
            jitter = rng.uniform(-0.20, 0.20, len(values))
            ax.scatter(
                values,
                row + jitter,
                s=size,
                facecolors=to_rgba(colour, alpha),
                edgecolors=to_rgba(_darken(colour), F5D_MARKER_EDGE_ALPHA),
                linewidths=F5D_MARKER_EDGE_WIDTH,
                zorder=2 if state == "not evaluated" else 3,
            )
            if state == "fails":
                median = float(np.median(values))
                (median_line,) = ax.plot(
                    [median, median],
                    [row - 0.30, row + 0.30],
                    color="white",
                    lw=2.0,
                    zorder=6,
                    solid_capstyle="butt",
                    path_effects=[
                        patheffects.Stroke(
                            linewidth=3.2,
                            foreground=to_rgba("#D2D5D8", 0.70),
                        ),
                        patheffects.Normal(),
                    ],
                )
                median_line.set_gid("fig5d-failure-median")

    handles = [
        Line2D(
            [], [], ls="", marker="o", ms=4.8,
            mfc=to_rgba(styles[state][0], styles[state][1]),
            mec=to_rgba(_darken(styles[state][0]), F5D_MARKER_EDGE_ALPHA),
            mew=F5D_MARKER_EDGE_WIDTH,
        )
        for state in ("satisfies", "fails", "not evaluated")
    ]
    legend = ax.legend(
        handles,
        ["satisfies", "fails", "not evaluated"],
        frameon=False,
        fontsize=FS,
        loc="lower left",
        ncol=3,
        bbox_to_anchor=(-0.008, 1.005),
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.9,
        borderpad=0.0,
    )
    for text, state in zip(legend.get_texts(), ("satisfies", "fails", "not evaluated")):
        text.set_color(styles[state][0])

    # The comparison the text makes is drawn into the panel rather than left to the reader:
    # every generated output sits decades below where compression and displacement put a
    # cell.  Both names are set over two lines to the left of their own line, which is what
    # lets the axis stop just past the displacement median instead of at 500 eV per atom.
    for value, name, colr in ((0.22, "compression", ORA), (6.05, "displacement", RED)):
        ax.axvline(
            value, color=colr, lw=1.2, ls=(0, (2.4, 1.8)), alpha=1.0,
            zorder=4,
        )
        ax.text(value * 0.86, 4.30, f"{name}\nmedian", linespacing=1.15,
                fontsize=FS - 0.8, color=colr, ha="right", va="top")
    ax.set_yticks(range(len(levels)))
    # 归档表的列名仍是 L1--L4;图内一律用正文的 Set 1--Set 4。
    ax.set_yticklabels(["Set " + lv[1:] for lv in levels[::-1]])
    ax.set_xscale("log")
    ax.set_xlim(E_LO, E_HI_D)
    ax.set_xticks(E_TICKS_D)
    ax.tick_params(axis="x", labelsize=8.8)
    ax.set_ylim(-0.52, 4.34)
    ax.spines["left"].set_bounds(0, 3)
    ax.set_xlabel(E_LABEL)


# ─────────────────────────────────────────────────────────────────────── (d)
def panel_d(fig, ax, d7):
    rows = sorted(d7["by_sg"].items(), key=lambda kv: kv[1]["rate"])
    ov = d7["overall"]["rate"]
    XN = 1.34
    ax.plot([ov, ov], [-0.45, len(rows) + 0.10], color=INK, lw=0.9,
            ls=(0, (4, 2)), zorder=2, clip_on=False)
    # the note sits in the block the low-rate rows leave empty, so that the
    # panel needs no headroom above the top row
    ax.text(ov + 0.055, 1.94,
            f"whole sample\n{ov:.3f}   (n = {d7['overall']['n']:,})",
            fontsize=FS, color=INK, ha="left", va="top", linespacing=1.30)
    ax.text(XN, len(rows) - 0.62, "n", fontsize=FS, color=SOFT, ha="right",
            va="bottom")
    for i, (sg, v) in enumerate(rows):
        c = GRY if sg == "others" else BLU
        ax.plot([0, v["rate"]], [i, i], color="#D4D6D9", lw=0.7, zorder=1)
        ax.plot(
            [v["rate"]], [i], ls="", marker="o", ms=6.0,
            mfc=to_rgba(c, F5B_MARKER_FACE_ALPHA),
            mec=to_rgba(_darken(c), F5B_MARKER_EDGE_ALPHA),
            mew=F5B_MARKER_EDGE_WIDTH, zorder=4,
        )
        ax.text(v["rate"] + 0.042, i, f"{v['rate']:.3f}", fontsize=FS, color=c,
                ha="left", va="center")
        ax.text(XN, i, f"{v['n']:,}", fontsize=FS, color=SOFT, ha="right",
                va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["all others" if sg == "others" else sg for sg, _ in rows])
    for t in ax.get_yticklabels():
        t.set_ha("right")
    ax.tick_params(axis="y", pad=2.0)
    ax.set_ylim(-0.72, len(rows) - 0.24)
    ax.set_xlim(-0.03, 1.37)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    ax.spines["left"].set_bounds(-0.4, len(rows) - 1)
    ax.set_xlabel("Fraction not satisfying Law 7 (distinct sites)", x=0.379,
                  ha="center")


# ───────────────────────────────────────────────── (b, lower)
def panel_b_ordering(fig, ax):
    """Whether the ordering that splits those sites is thermodynamically real.

    The panel above localises the low-symmetry excess to split sites but cannot say
    whether the ordering that splits them costs anything.  For each merge group every
    symmetry-distinct ordering of its mergeable species was relaxed with DFT; the
    temperature plotted is where the cost of leaving the best ordering for a random one is
    matched by the configurational entropy.  Below room temperature the orderings are
    degenerate and the compound is a solid solution, not an ordered crystal.
    """
    ordering = json.loads((DFT / "E2_ordering" / "ordering_energies.json").read_text())
    T_FLOOR = 1.0
    rng_i = np.random.default_rng(20260828)
    # the two groups are named on the axis rather than inside the panel: with only two
    # rows an in-panel label has nowhere to sit that is not on top of a point
    labels, colours = [], []
    for row, (kind, colr, lab) in enumerate((("experimental", L4C, "controls"),
                                             ("gnome", BLU, "GNoME"))):
        t = np.clip([r["T_disorder_K"] for r in ordering if r["kind"] == kind],
                    T_FLOOR, None)
        ax.scatter(
            t, row + rng_i.uniform(-0.16, 0.16, len(t)), s=13,
            facecolors=to_rgba(colr, F5B_MARKER_FACE_ALPHA),
            edgecolors=to_rgba(_darken(colr), F5B_MARKER_EDGE_ALPHA),
            linewidths=F5B_MARKER_EDGE_WIDTH, zorder=3,
        )
        ax.plot([np.median(t)] * 2, [row - 0.32, row + 0.32], color=_darken(colr),
                lw=1.3, zorder=4)
        ax.text(2.7e4, row, f"{len(t)}", fontsize=FS, color=SOFT, ha="right",
                va="center")
        labels.append(lab); colours.append(colr)
    ax.axvline(300, color="#989A9D", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.text(262, 1.86, "300 K", fontsize=FS, color="#75777A", ha="right", va="top")
    ax.set_xscale("log")
    # Six clipped GNoME values sit exactly at the 1 K reporting floor.  Keep a
    # small logarithmic margin so their marker outlines are not cut by the axis.
    ax.set_xlim(0.8, 3e4)
    ax.set_xticks([1e0, 1e1, 1e2, 1e3, 1e4])
    ax.set_ylim(-0.70, 1.95)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    for tick, colr in zip(ax.get_yticklabels(), colours):
        tick.set_color(colr)
    ax.tick_params(axis="y", length=0, pad=2.0)
    ax.tick_params(axis="x", labelsize=8.8)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("DFT order–disorder temperature (K)")


# ─────────────────────────────────────────────────────────────────────── (e)
def _project_structure(exemplar: dict, species_key: str):
    """Project the fixed unit cell using one identical orthographic view."""
    lattice = np.asarray(exemplar["lattice_matrix_angstrom"], float)
    fractional = np.asarray(exemplar["fractional_coordinates"], float)
    labels = exemplar[species_key]
    points = fractional @ lattice

    azimuth, elevation = np.deg2rad(-58.0), np.deg2rad(45.0)
    view = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0.0],
        [np.sin(azimuth) * np.sin(elevation),
         np.cos(azimuth) * np.sin(elevation), -np.cos(elevation)],
    ])
    projected = points @ view.T

    corners_f = np.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float
    )
    corners = (corners_f @ lattice) @ view.T
    return projected, labels, corners


def _draw_structure(ax, exemplar: dict, species_key: str, label: str) -> None:
    points, species, corners = _project_structure(exemplar, species_key)
    bits = [[(idx >> 2) & 1, (idx >> 1) & 1, idx & 1] for idx in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            if sum(abs(a - b) for a, b in zip(bits[i], bits[j])) != 1:
                continue
            ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                color="#A6A8AB", lw=0.48, alpha=0.58, zorder=0,
            )

    colours = {"K": ORA, "Er": BLU, "Br": RED}
    sizes = {"K": 50, "Er": 54, "Br": 34}
    for idx in np.argsort(points[:, 1]):
        symbol = species[idx]
        ax.scatter(
            [points[idx, 0]], [points[idx, 1]], s=sizes[symbol],
            facecolors=colours[symbol], edgecolors=_darken(colours[symbol]),
            linewidths=0.38, alpha=0.88, zorder=2 + idx / 1000, clip_on=False,
        )
        if idx in exemplar["swapped_sites"]:
            ax.scatter(
                [points[idx, 0]], [points[idx, 1]], s=sizes[symbol] * 1.65,
                facecolors="none", edgecolors=_darken(colours[symbol]),
                linewidths=0.65, alpha=0.88, zorder=4, clip_on=False,
            )
    xmin = min(points[:, 0].min(), corners[:, 0].min())
    xmax = max(points[:, 0].max(), corners[:, 0].max())
    ymin = min(points[:, 1].min(), corners[:, 1].min())
    ymax = max(points[:, 1].max(), corners[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.15 * dx, xmax + 0.15 * dx)
    ax.set_ylim(ymin - 0.12 * dy, ymax + 0.12 * dy)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(0.5, -0.03, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=FS, color=INK)


def panel_e(fig, ax, wrong_site):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rates = wrong_site_detection_rates(wrong_site)

    # One parent, two fixed-coordinate exchanges; each row pairs its own
    # structures with the detection rates for that exchange class.
    rows = [
        ("S2", wrong_site["exemplar"], "K ↔ Er", "cation–cation  (n = 69)",
         0.585, 0.545, 0.945, False),
        ("S5", wrong_site["exemplar_s5"], "K ↔ Br", "cation–anion  (n = 83)",
         0.085, 0.000, 0.400, True),
    ]
    for klass, exemplar, swap_label, header, y0, cell_y, header_y, is_bottom in rows:
        parent_ax = ax.inset_axes([-0.190, y0 - 0.020, 0.360, 0.420])
        swapped_ax = ax.inset_axes([0.180, y0 - 0.020, 0.360, 0.420])
        _draw_structure(parent_ax, exemplar, "parent_species", "parent")
        _draw_structure(swapped_ax, exemplar, "damaged_species", swap_label)

        ax.text(0.545, header_y, header, transform=ax.transAxes, fontsize=FS,
                color=INK, ha="left", va="bottom")
        cell = ax.inset_axes([0.545, cell_y, 0.440, 0.375])
        for method, y, colour in (("coordinate checks", 1.30, BLU),
                                  ("PRIS", 0.36, L4C)):
            value = 100.0 * rates[klass][method]
            bars = cell.barh(
                y, value, height=0.30, color=colour, edgecolor=colour,
                linewidth=0.85, zorder=2,
            )
            style_bar_container(bars, colour)
            cell.text(
                value + 2.0, y, f"{value:.1f}%", fontsize=FS,
                color=_darken(colour), ha="left", va="center",
            )
            cell.text(0, y + 0.24, method, fontsize=FS, color=colour,
                      ha="left", va="bottom")
        cell.set_yticks([])
        cell.set_xlim(0, 120)
        cell.set_ylim(0.0, 2.0)
        cell.set_xticks([0, 50, 100])
        cell.spines["bottom"].set_bounds(0, 100)
        cell.spines["left"].set_visible(False)
        if is_bottom:
            cell.set_xlabel("Damage detected (%)")
        else:
            cell.set_xticklabels([])

    hs = [
        Line2D([], [], ls="", marker="o", ms=4.8, mfc=colour,
               mec=_darken(colour), mew=0.45, alpha=0.85)
        for colour in (ORA, BLU, RED)
    ]
    legend = ax.legend(
        hs, ["K", "Er", "Br"], frameon=False, fontsize=FS,
        loc="lower left", bbox_to_anchor=(0.010, 0.945), ncol=3,
        handlelength=0.7, handletextpad=0.25, columnspacing=0.75,
        borderpad=0.0,
    )
    for text, colour in zip(legend.get_texts(), (ORA, BLU, RED)):
        text.set_color(colour)


# ─────────────────────────────────────────────────────────────────────── (f)
def panel_f(fig, ax):
    H = 3600.0
    ROWS = [
        (0.05e-3, 10.7e-3, RED, "contact law alone, 8 → 216 atoms",
         "0.05 ms", "10.7 ms", False),
        (58e-3, 2.4, BLU, "all eight laws, reference code",
         "58 ms", "2.4 s", False),
        (10 * H, 1000 * H, GRY, "DFT relaxation, literature estimate",
         None, "order 10² CPU-hours", True),
    ]
    ax.set_xscale("log")
    ax.set_xlim(2e-5, 4e7)
    ax.set_ylim(-0.55, 2.52)
    for i, (lo, hi, c, lab, ltxt, rtxt, fuzzy) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        if fuzzy:
            ax.plot([lo, hi], [y, y], color=c, lw=1.1, ls=(0, (2.2, 1.8)),
                    zorder=3)
            ax.plot([100 * H], [y], ls="", marker="o", ms=5.6, mfc="white",
                    mec=c, mew=1.4, zorder=4)
        else:
            ax.plot([lo, hi], [y, y], color=c, lw=1.6, zorder=3,
                    solid_capstyle="round")
            for xv in {lo, hi}:
                ax.plot([xv], [y], ls="", marker="o", ms=5.3, mfc=c,
                        mec=_darken(c), mew=0.50, alpha=0.86, zorder=4)
        ax.text(2.6e-5, y + 0.15, lab, fontsize=FS, color=c, ha="left",
                va="bottom")
        if ltxt:
            ax.text(lo, y - 0.13, ltxt, fontsize=FS, color=c, ha="left",
                    va="top")
        if fuzzy:
            ax.text(hi / 1.6, y - 0.13, rtxt, fontsize=FS, color=c, ha="right",
                    va="top")
        else:
            ax.text(hi * 3.6, y, rtxt, fontsize=FS, color=c, ha="left",
                    va="center")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("CPU time per structure (s)")
    ax.set_xticks([1e-4, 1e-2, 1, 1e2, 1e4, 1e6])
    ax.tick_params(axis="x", labelsize=8.8)

    sec = ax.secondary_xaxis("top", functions=(lambda x: x * 1e4,
                                               lambda x: x / 1e4))
    sec.set_xticks([1.0, 60.0, H, 365 * 24 * H, 100 * 365 * 24 * H])
    sec.set_xticklabels(["1 s", "1 min", "1 h", "1 yr", "100 yr"])
    sec.set_xlabel("Total for a queue of 10,000 structures", labelpad=3)
    sec.tick_params(labelsize=8, width=0.6, length=2.5)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    gen = pd.read_csv(DATA / "fig7_generators.csv")
    d7 = json.load(open(DATA / "fig7_gnome_d7.json"))
    ml = pd.read_csv(DATA / "fig7_mlip.csv")
    ladder = pd.read_csv(DATA / "fig7_mattergen_ladder_energy.csv")
    wrong_site = json.load(open(DATA / "fig7_wrong_site.json"))

    # Three thematic rows: generator-scale symptom and Law 7 localization,
    # paired relaxation-energy tests, then chemical identity and cost.
    # Explicit spacer rows keep the two gaps fixed as panel contents change.
    H_CM = TOP_CM + sum(ROW_CM) + BOT_CM
    fig = plt.figure(figsize=(W2, H_CM * CM))
    gs = fig.add_gridspec(5, 2, width_ratios=[1.0, 1.0],
                          height_ratios=ROW_CM, hspace=0.0, wspace=0.30,
                          left=0.094, right=0.976,
                          top=1.0 - TOP_CM / H_CM, bottom=BOT_CM / H_CM)
    axa = fig.add_subplot(gs[0, 0])
    # Panel b carries two steps of one argument -- where the excess sits, and whether the
    # ordering behind it costs anything -- so the cell is split rather than overlaid; an
    # inset on a six-row dot plot buries both.
    gsb = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42)
    axb, axb2 = fig.add_subplot(gsb[0]), fig.add_subplot(gsb[1])
    axc, axd = (fig.add_subplot(gs[2, j]) for j in range(2))
    axe, axf = (fig.add_subplot(gs[4, j]) for j in range(2))

    panel_a(fig, axa, gen)
    panel_d(fig, axb, d7)
    panel_b_ordering(fig, axb2)
    panel_b(fig, axc, ml)
    panel_c(fig, axd, ladder)
    panel_e(fig, axe, wrong_site)
    panel_f(fig, axf)

    stamp(fig, [(axa, "a", (0.0, 0.030)), (axb, "b", (0.0, 0.030)),
                (axc, "c", (0.0, 0.032)), (axd, "d", (0.0, 0.032)),
                (axe, "e", (0.0, 0.005)), (axf, "f", (0.0, 0.005))])
    save(fig, "fig6_deployment")


if __name__ == "__main__":
    main()
