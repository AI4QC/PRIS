#!/usr/bin/env python3
"""Figure: exclusion power vs corruption amplitude, one panel per corruption
type. Style follows src/paper_figs.py rcParams."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

plt.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial", "Liberation Sans"],
    "font.size": 8.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})

CM = 1 / 2.54
SETC = {"L1": "#005B93", "L1'": "#35A7D8", "L2": "#9861B0",
        "L3": "#D6564C", "L4": "#0A5A3C"}
GRY = "#8A8C8E"

ROOT = pathlib.Path(__file__).resolve().parent.parent
SI_DATA, OUT = ROOT / "paper" / "si_data", ROOT / "paper" / "figs"
df = pd.read_csv(SI_DATA / "s7_amplitude_response.csv")

panels = [("iso_expansion", "Linear strain (%)", 100),
          ("uni_compression", "Linear strain (%)", 100),
          ("gauss_disp", r"$\sigma$ ($\mathrm{\AA}$)", 1)]

fig, axes = plt.subplots(1, 3, figsize=(18.3 * CM, 8.0 * CM), sharey=True)
for ax, (ctype, xlab, scale) in zip(axes, panels):
    g = df[df.corruption == ctype].sort_values("amplitude")
    x = g.amplitude.values * scale
    for rule in ("L1", "L1'", "L2", "L3", "L4"):
        ax.plot(x, g[f"excl_{rule}"].values, "-o", color=SETC[rule],
                lw=1.0, ms=2.6, label="Set " + rule[1:].replace("'", "\u2032"),
                clip_on=False, zorder=3)
    ax.plot(x, g["excl_md_0.5"].values, "--s", color=GRY, lw=0.9, ms=2.4,
            label=r"min $d$ > 0.5 $\mathrm{\AA}$", clip_on=False)
    ax.plot(x, g["excl_md_0.7"].values, ":^", color="#4d4d4d", lw=0.9, ms=2.4,
            label=r"min $d$ > 0.7 $\mathrm{\AA}$", clip_on=False)
    ax.set_xlabel(xlab)
    ax.set_ylim(-0.02, 1.34)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axes[0].set_ylabel("Damage detection")
axes[0].legend(frameon=False, loc="upper left", handlelength=1.6, ncol=2,
               columnspacing=1.0, borderaxespad=0.2)
for ax, letter in zip(axes, "abc"):
    ax.text(-0.15 if letter == "a" else -0.07, 1.05, letter,
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")
fig.tight_layout()
for _ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"figS7_amplitude_response.{_ext}")
print("wrote figS7_amplitude_response")
