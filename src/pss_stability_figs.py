#!/usr/bin/env python3
"""Supplementary Figs. S12-S13. Arial, no titles (captions carry them),
data from paper/si_data (written by src/pss_stability_analysis.py)."""
from __future__ import annotations
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figs import (plt, stamp, save, CM, W2,          # noqa: E402
                        BLU, RED, ORA, GRN, GRY, PUR, SI_DATA)

TERMS = ["v_atom", "M_z", "D_BV", "eta_site", "k_max", "f_iso"]
MATH = {"v_atom": r"$\widetilde{v}_{\mathrm{atom}}$",
        "M_z": r"$\widetilde{M}_{z}$",
        "D_BV": r"$\widetilde{\Delta}_{\mathrm{BV}}$",
        "eta_site": r"$\widetilde{\eta}_{\mathrm{site}}$",
        "k_max": r"$\widetilde{k}_{\mathrm{max}}$",
        "f_iso": r"$\widetilde{f}_{\mathrm{iso}}$"}
TERMC = {"v_atom": BLU, "M_z": RED, "D_BV": GRN,
         "eta_site": ORA, "k_max": PUR, "f_iso": GRY}


def _box(ax, x, values, colour, width):
    """Median bar, interquartile box and 2.5-97.5% whisker, in the SI bar style."""
    lo, q1, med, q3, hi = np.percentile(values, [2.5, 25, 50, 75, 97.5])
    ax.plot([x, x], [lo, hi], color=colour, lw=0.6, solid_capstyle="butt", zorder=2)
    ax.add_patch(plt.Rectangle((x - width / 2, q1), width, q3 - q1, lw=0,
                               facecolor=colour, alpha=0.85, zorder=3))
    ax.plot([x - width / 2, x + width / 2], [med, med], color="white", lw=0.9,
            solid_capstyle="butt", zorder=4)


def coefficient_stability():
    sub = pd.read_csv(SI_DATA / "s8_pss_coefficient_stability.csv")
    bt = pd.read_csv(SI_DATA / "s8_pss_bootstrap.csv").set_index("term")
    fracs = sorted(sub.fraction.unique())

    fig = plt.figure(figsize=(W2, 6.6 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.94, 1.20], wspace=0.34,
                          left=0.056, right=0.965, top=0.90, bottom=0.155)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # a — refit the same six terms on random subsets of the development structures
    w, step = 0.115, 0.135
    for k, t in enumerate(TERMS):
        for i, f in enumerate(fracs):
            v = sub.loc[sub.fraction == f, t].values / bt.published[t]
            _box(axa, i + (k - 2.5) * step, v, TERMC[t], w)
    axa.axhline(1.0, color="#444444", lw=0.7, ls="--", zorder=1)
    axa.axhline(0.0, color=GRY, lw=0.6, zorder=1)
    axa.set_xticks(range(len(fracs)))
    axa.set_xticklabels([f"{int(f * 100)}" for f in fracs])
    axa.set_xlim(-0.55, len(fracs) - 0.45)
    axa.set_ylim(-0.45, 2.72)
    axa.set_xlabel("Development structures (%)")
    axa.set_ylabel("Refitted coefficient / published value")
    axa.legend(handles=[plt.Line2D([], [], color=TERMC[t], lw=3.2, label=MATH[t])
                        for t in TERMS],
               ncol=3, frameon=False, fontsize=7.4, loc="upper center",
               handlelength=0.9, handletextpad=0.4, columnspacing=1.0,
               borderpad=0.0, bbox_to_anchor=(0.5, 1.035))

    # b — cluster bootstrap over composition groups, all development data
    ypos = np.arange(len(TERMS))[::-1]
    for yv, t in zip(ypos, TERMS):
        axb.plot([bt.ci_lo[t], bt.ci_hi[t]], [yv, yv], color=TERMC[t], lw=1.5,
                 solid_capstyle="round", zorder=2)
        axb.plot([bt.published[t]], [yv], "o", ms=4.0, color=TERMC[t],
                 mec="white", mew=0.7, zorder=3)
    axb.axvline(0.0, color="#444444", lw=0.7, ls="--", zorder=1)
    axb.set_yticks(ypos)
    axb.set_yticklabels([MATH[t] for t in TERMS])
    axb.set_ylim(-0.7, len(TERMS) - 0.3)
    axb.set_xlim(-8.3, 5.4)
    axb.set_xticks([-6, -4, -2, 0])
    axb.set_xlabel("Coefficient (standardised units)")
    for yv, t in zip(ypos, TERMS):
        axb.text(5.3, yv, f"|$\\beta$|/SE {bt.abs_z[t]:.1f}", ha="right",
                 va="center", fontsize=7.0, color="#555555")

    # c — what the refitted score does, relative to the published one
    for lab, col, colour, mark in (
            ("Rank correlation with published PSS", "spearman_vs_published", BLU, "o"),
            (r"Cosine similarity of $\beta$", "cosine", RED, "^")):
        m = [sub.loc[sub.fraction == f, col].mean() for f in fracs]
        lo = [sub.loc[sub.fraction == f, col].quantile(0.025) for f in fracs]
        hi = [sub.loc[sub.fraction == f, col].quantile(0.975) for f in fracs]
        axc.fill_between(fracs, lo, hi, color=colour, alpha=0.16, lw=0)
        axc.plot(list(fracs) + [1.0], m + [1.0], "-", marker=mark, ms=3.4, lw=1.0,
                 color=colour, mec="white", mew=0.5, label=lab)
    axc.set_ylim(0.90, 1.008)
    axc.set_xlim(0.46, 1.04)
    axc.set_xticks(list(fracs) + [1.0])
    axc.set_xticklabels([f"{int(f * 100)}" for f in list(fracs) + [1.0]])
    axc.set_xlabel("Development structures (%)")
    axc.set_ylabel("Agreement with the published score")
    axc.legend(frameon=False, fontsize=7.0, loc="lower right", handlelength=1.4,
               handletextpad=0.5, borderpad=0.0)

    stamp(fig, [(axa, "a"), (axb, "b"), (axc, "c")])
    save(fig, "pss_coefficient_stability")


def term_correlation():
    C = json.loads((SI_DATA / "s9_pss_term_correlation.json").read_text())
    lab = [MATH[t] for t in C["terms"]]
    n = len(C["terms"])
    struct = np.array(C["spearman_structures"])
    pair = np.array(C["pearson_pair_design"])
    vif = np.array(C["vif_pair_design"])

    fig = plt.figure(figsize=(W2, 7.0 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.90], wspace=0.46,
                          left=0.085, right=0.965, top=0.90, bottom=0.175)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # a — Spearman across structures above the diagonal, Pearson on the
    #     within-composition differences the fit sees below it
    M = np.tril(pair) + np.triu(struct, 1)
    np.fill_diagonal(M, np.nan)
    im = axa.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            axa.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.6,
                     color="white" if abs(M[i, j]) > 0.40 else "#222222")
    axa.set_xticks(range(n)); axa.set_xticklabels(lab)
    axa.set_yticks(range(n)); axa.set_yticklabels(lab)
    axa.tick_params(length=0)
    for sp in axa.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=axa, fraction=0.045, pad=0.035)
    cb.set_label("Correlation coefficient", fontsize=8, labelpad=2)
    cb.outline.set_linewidth(0.6)
    axa.text(-0.5, -0.80, "upper: Spearman, across structures", fontsize=7.2,
             color="#555555", ha="left", va="center")
    axa.set_xlabel("lower: Pearson, within composition", fontsize=7.2,
                   color="#555555", labelpad=4)

    # b — variance inflation on the pairwise design matrix
    x = np.arange(n)
    axb.bar(x, vif, 0.62, lw=0, color=[TERMC[t] for t in C["terms"]], alpha=0.9)
    axb.axhline(1.0, color="#444444", lw=0.7, ls="--", zorder=3)
    for xi, v in zip(x, vif):
        axb.text(xi, v + 0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=7.0)
    axb.set_xticks(x); axb.set_xticklabels(lab)
    axb.set_ylim(0, 1.58)
    axb.set_ylabel("Variance inflation factor")
    axb.set_xlabel("Term of the fitted synthesis score")
    axb.text(0.03, 0.955,
             f"condition number {C['condition_number']:.2f}\n"
             f"{C['n_structures']:,} structures, {C['n_groups']} groups, "
             f"{C['n_pairs']:,} pairs",
             transform=axb.transAxes, fontsize=7.0, color="#555555",
             ha="left", va="top")

    stamp(fig, [(axa, "a"), (axb, "b")])
    save(fig, "pss_term_correlation")


if __name__ == "__main__":
    coefficient_stability()
    term_correlation()
