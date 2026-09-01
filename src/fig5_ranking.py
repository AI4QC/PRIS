#!/usr/bin/env python3
"""Fig. 5 — the same-composition ranking task, in one figure.

1,508 compositions, 18,920 same-composition pairs. The upper half asks what the
binary plausibility rules can do on that task; the lower half asks what two
label-fitted continuous scores say about it.

a  commitment plane: fraction of pairs a criterion distinguishes vs its accuracy
   there — Pauling's rules, the law sets read as pass/fail tests, geometric
   baselines, DFT energy above the hull, and rho as a continuous quantity
b  fraction of composition groups in which no single structure can be picked
c  accuracy restricted to the pairs DFT energy above the hull ranks wrongly
d  sealed commitment profile of the synthesis score F3 against hull energy,
   each on its own confidence ranking
e  standardised coefficients of F2R (stability) and F3 (synthesis): two nearly
   disjoint vocabularies, and the density sign reversal
f  ladder profile on 26,600 phonon-computed materials: both conditional
   directions between the five rungs Set 1–Set 4 and three binary axes (on hull,
   no imaginary modes, made), each against its own base rate

Panels a–c are the former Fig. 4a/4b/4d (src/paper_figs.py), panels d–e the
former Fig. 6a/6b (src/fig6_scores.py); numbers, colours and text are
verbatim, only the page geometry is new. The two demoted panels (top-1 lift,
F2R accuracy vs energy-gap floor) live in figS12_ranking_extras.

Data: paper/data/{fig3_coverage_accuracy,fig7_top1,rank_rulesets,
fig5_twoway}.csv and the frozen json under outputs/20260814_f3_synth/.
fig5_twoway.csv is written by experiments/twoway_threeaxis/twoway.py.
"""
from __future__ import annotations
import json, pathlib, sys
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from matplotlib.legend_handler import HandlerTuple
from paper_figs import plt, np, W2, CM, BLU, RED, ORA, GRN, GRY, PUR, save, stamp

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
F3D = ROOT / "outputs" / "20260814_f3_synth"

# ── criterion naming and colouring (verbatim from paper_figs, Fig. 4 block)
RHO = r"$\rho$"
SHORT = {"Pauling 2 (bond-strength dev.)": "Pauling 2",
         "Pauling 3 (edge/face sharing)": "Pauling 3",
         "Pauling 4 (high-valence contact)": "Pauling 4",
         "Pauling 5 (parsimony)": "Pauling 5",
         "This work: bl_min": RHO,
         "Volume per atom": "Volume/atom",
         "Shannon packing": "Shannon packing",
         "DFT E_hull (baseline)": "DFT energy"}
SLATE = "#41556B"
RSET = ["L1", "L1'", "L2", "L3", "L4"]
# 归档表仍以 L1--L4 为键;图内显示名与正文一致(Set 1--Set 4)。
RLAB = {"L1": "Set 1", "L1'": "Set 1′", "L2": "Set 2", "L3": "Set 3",
        "L4": "Set 4"}
SETC = {"L1": BLU, "L1'": PUR, "L2": ORA, "L3": RED, "L4": "#1B7837"}
CMAP = {"Pauling 2": SLATE, "Pauling 3": SLATE, "Pauling 4": SLATE, "Pauling 5": SLATE,
        RHO: BLU, "Volume/atom": GRY, "Shannon packing": GRY,
        "DFT energy": GRN}
# b and c are the two horizontal-category panels: category labels end flush
# against their own axis (right-aligned, modest pad), no category ticks.
PAD_BC = 3.0

# ── the two fitted scores (verbatim from fig6_scores)
F3_TERMS = [("volume per atom", -4.90), ("mean site Madelung / |z|", -1.24),
            ("mean rel. BV dev.", -1.18), ("Wyckoff economy", -0.84),
            ("max polyhedral degree", -0.22), ("frac. isolated polyhedra", 0.59)]
F2R_TERMS = [("global instability index", -0.623), ("max ECoN", -0.517),
             ("max Pauling rule-2 dev.", -0.312), ("max cation CN", -0.239),
             ("Wyckoff economy", -0.168), ("min ECoN", -0.156)]

# ── page geometry (cm).  Three rows of two panels on an 18.3 cm page.
#    Left column: a, c, e share one left and one right edge; the 3.35 cm gutter
#    holds e's longest term label.  Right column: b, d, f share XR.
#    Rows share their top and bottom edges, hence their baselines.
_W, _H = 18.3, 18.64
XL, WL = 3.35, 5.60
XE, WE = XL, WL
XR, WR = 11.35, 6.50
Y1, H1 = 0.58, 4.72
Y2, H2 = 6.60, 3.62
Y3, H3 = 11.84, 5.55
# f is a two-row strip inside the same rect: one row per conditional direction,
# sharing the x groups.  The gap carries f's key.
FH1, FGAP, FH2 = 2.60, 0.75, 2.20
# b's right-aligned row labels now define the right column's leftmost
# decoration, and they clear panel e's box on their own; no nudge needed.
DXR = 0.0


def _rect(x0, w, ytop, h):
    return [x0 / _W, (_H - ytop - h) / _H, w / _W, h / _H]


# ─────────────────────────────────────────────────────────────── (a) plane
def panel_a(ax, ca, rr):
    # the abstention band the caption names: coverage < 0.30, i.e. criteria that
    # score both structures alike in more than 70% of pairs.  A background tint
    # only, so nothing it lies under loses contrast.
    ax.axvspan(-0.03, 0.30, color="#C0392B", alpha=0.055, lw=0, zorder=0)
    ax.axhline(0.5, color="#BBBBBB", lw=0.6, ls=":", zorder=1)
    ax.text(0.015, 0.505, "chance", fontsize=7.7, color="#666666",
            va="bottom", ha="left")
    for _, r in ca.iterrows():
        ax.scatter(r.coverage, r.accuracy, s=48 if r.c == RED else 30,
                   c=r.c, ec="w", lw=0.7, zorder=3,
                   marker="D" if r.s == "DFT energy" else "o")
    # label offsets are the Fig. 4a ones, except for three that had been resting
    # on graphics: "Pauling 4" moves 4 pt left off the Pauling 3 marker,
    # "Shannon packing" 3 pt down off the Pauling 2 marker, and rho now hangs
    # straight under its marker, between it and the chance line.
    off = {"Pauling 2": (0, 11), "Pauling 3": (0, -14), "Pauling 4": (8, 16),
           "Pauling 5": (10, -12), RHO: (-2, -13),
           "Volume/atom": (-34, 8), "Shannon packing": (-42, -18), "DFT energy": (-24, 4)}
    for _, r in ca.iterrows():
        ax.annotate(r.s, (r.coverage, r.accuracy), textcoords="offset points",
                    xytext=off.get(r.s, (5, 5)), fontsize=8.0, color=r.c,
                    ha="center", fontweight="bold" if r.c == RED else "normal")
    for k in RSET:
        ax.scatter(rr.loc[k, "coverage"], rr.loc[k, "accuracy"], s=40, c=SETC[k],
                   ec="w", lw=0.7, zorder=4, marker="s")
    # L1 and L1' sit hard against the left edge at the top of the plane; their
    # labels hang to the right of the markers, which is also the only part of
    # the box no marker can reach, so the upper left stops being a void.
    roff = {"L1": (14, -1), "L1'": (14, -5), "L2": (16, 3), "L3": (14, -9),
            "L4": (16, 6)}
    for k in RSET:
        ax.annotate(RLAB[k], (rr.loc[k, "coverage"], rr.loc[k, "accuracy"]),
                    textcoords="offset points", xytext=roff[k], fontsize=8.0,
                    color=SETC[k], fontweight="bold", ha="center")
    # No connector between rho and L1: every route between the two markers
    # grazes a label ("Pauling 2", "Volume/atom") or the L4 square, so the two
    # readings of D1 are tied together in the caption instead.
    # a hair of negative room so the two law sets at coverage ~0.014 are not
    # cut in half by the left spine; the ticks still start at 0
    ax.set_xlim(-0.03, 1.04); ax.set_ylim(0.34, 0.99)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("fraction of pairs distinguished")
    ax.set_ylabel("accuracy on distinguished pairs")


# ─────────────────────────────────────────────────────────── (b) abstention
def _cat_labels(ax, labs):
    """Right-aligned category labels, flush against the axis, no category ticks."""
    ax.set_yticklabels(labs, fontsize=7.5, ha="right")
    ax.tick_params(axis="y", length=0, pad=PAD_BC)


def panel_b(ax, t1, rr):
    extra = pd.DataFrame({"s": [RLAB[k] for k in RSET],
                          "c": [SETC[k] for k in RSET],
                          "tie_rate": [float(rr.loc[k, "tie_rate"]) for k in RSET]})
    o = pd.concat([t1[["s", "c", "tie_rate"]], extra]).sort_values(
        "tie_rate", ascending=False)
    y = np.arange(len(o))[::-1]
    ax.barh(y, o.tie_rate * 100, color=o.c, height=0.70, lw=0, alpha=0.9)
    ax.set_yticks(y)
    _cat_labels(ax, o.s)
    ax.set_xlabel("groups where no single structure\ncan be picked (%)")
    # room for the value label of the longest bar inside the box: at 99 % the
    # text would otherwise run across the right spine
    ax.set_xlim(0, 113)
    for i, (val, c) in enumerate(zip(o.tie_rate * 100, o.c)):
        ax.text(val + 2.2, len(o) - 1 - i, f"{val:.0f}%", va="center", fontsize=7.7,
                color=c)


# ──────────────────────────────────────────── (c) where DFT energy is wrong
def panel_c(ax, ca):
    o = ca[ca.s != "DFT energy"].sort_values("acc_energy_wrong")
    ax.axvline(0.5, color="#666666", lw=0.7, ls="--")
    ax.barh(np.arange(len(o)), o.acc_energy_wrong, color=o.c, height=0.66, lw=0,
            alpha=0.9)
    ax.set_yticks(np.arange(len(o)))
    _cat_labels(ax, o.s)
    ax.set_xlim(0.38, 0.665); ax.set_xticks([0.4, 0.5, 0.6])
    ax.set_xlabel("accuracy on pairs\nDFT energy ranks wrongly")
    ax.text(0.5, 1.015, "chance", transform=ax.get_xaxis_transform(), fontsize=7.7,
            color="#666666", va="bottom", ha="center")
    for i, (val, c) in enumerate(zip(o.acc_energy_wrong, o.c)):
        xp = val + 0.007
        if xp < 0.5 < xp + 0.072:      # 别让 chance 虚线穿过数值标签
            xp = 0.506
        ax.text(xp, i, f"{val:.3f}", va="center", fontsize=7.5, color=c)


# ─────────────────────────────────────────── (d) sealed commitment profile
def panel_d(ax, r3, ci3):
    keeps = [1.00, 0.50, 0.30, 0.20, 0.10]
    f3 = [r3["F3_curve"][f"{k:.2f}"] for k in keeps]
    ehc = [r3["e_hull_curve"][f"{k:.2f}"] for k in keeps]
    x = np.arange(len(keeps))
    ax.plot(x, [v["acc"] for v in ehc], "-o", color=GRN, lw=1.2, ms=4,
            label="DFT hull energy")
    lo3 = [v["acc"] - v["ci"][0] for v in f3]
    hi3 = [v["ci"][1] - v["acc"] for v in f3]
    ax.errorbar(x, [v["acc"] for v in f3], yerr=[lo3, hi3], fmt="-o", color=BLU,
                lw=1.2, ms=4, capsize=2, elinewidth=0.7, label="synthesis score")
    ax.axhline(0.8, color="#BBBBBB", lw=0.7, ls=":")
    d = ci3["0.20"]
    ax.annotate(f"$\\Delta$ = {d['delta']:+.3f}\n[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]",
                xy=(3, f3[3]["acc"]), xytext=(0.06, 0.998), fontsize=7.7, color=BLU,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=BLU,
                                mutation_scale=6))
    ax.set_xticks(x)
    ax.set_xticklabels(["all", "1/2", "3/10", "1/5", "1/10"], fontsize=8.0)
    # 单行刻度 + 小 labelpad:两行刻度加两行标签会把 x 轴标签推到面板 f 的
    # 字母同一高度,读起来像 f 的标题。"most confident first" 的口径在图注里。
    ax.set_xlabel("fraction of held-out pairs retained", labelpad=2)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.70, 1.01)
    # the widest empty band is the strip under both curves along the bottom;
    # a one-row key lying in it leaves no corner void.
    ax.legend(frameon=False, fontsize=7.7, loc="lower center", ncol=2,
              handlelength=1.6, columnspacing=1.4, borderaxespad=0.15,
              borderpad=0.0)


# ───────────────────────────────────────── (e) the two fitted vocabularies
def panel_e(ax):
    # two empty slots between the blocks: the block heading needs two lines of
    # clear space above the longest bar of its own block
    terms = F2R_TERMS + [("", 0)] * 2 + F3_TERMS
    ypos = np.arange(len(terms))[::-1]
    cols = [ORA] * len(F2R_TERMS) + ["#FFFFFF"] * 2 + [BLU] * len(F3_TERMS)
    ax.barh(ypos, [t[1] for t in terms], color=cols, height=0.62, lw=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([t[0] for t in terms], fontsize=7.2)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color="#444444", lw=0.7)
    ax.text(-4.9, ypos[2] + 0.5, "stability score\n(hull-energy ordering)",
            fontsize=8.0, color=ORA, fontweight="bold", va="top")
    ax.text(-4.9, ypos[len(F2R_TERMS)] + 0.70, "synthesis score\n(experimental record)",
            fontsize=8.0, color=BLU, fontweight="bold", va="top")
    ax.set_xlabel("standardised coefficient\n(higher score = stable / recorded)")
    ax.set_xlim(-5.2, 1.2)
    # keep Fig. 6b's unit ticks; the narrower axis would otherwise auto-thin to
    # every second one and the −1.24 / −1.18 terms would lose their reference
    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1])


# ────────────────────────────────────────── (f) the ladder profile
#   Two stacked rows over one shared x axis, the five rungs of the ladder in
#   order from gentle to strict.  Upper row, P(rung accepts | class): a solid
#   curve through the structures in the class, a dashed curve through those
#   not in it, one colour per axis, plus the unconditional acceptance rate in
#   grey.  Lower row, the other direction: solid, P(class | rung accepts);
#   dashed, P(class | rung rejects); faint, that axis's own base rate on the
#   rung's determinate subset.  Wilson 95% intervals as light bands.
RUNGS = ["L1", "L1'", "L2", "L3", "L4"]
#   one colour per axis, each carried over from its own panel: hull energy is
#   green as in a and d, the experimental-record label blue as in d and e
TW_AXES = [("on_hull", "on hull", GRN),
           ("dyn_stable", "no imaginary modes", ORA),
           ("made", "experimentally recorded", BLU)]
REFC, KEYC = "#9A9A9A", "#555555"


def panel_f(ax1, ax2, tl):
    fw = tl[tl.direction == "forward"].set_index(["axis", "rung", "class"])
    rv = tl[tl.direction == "reverse"].set_index(["axis", "rung", "class"])
    mar = tl[tl.direction == "marginal"]
    acc = mar[(mar.axis == "-")
              & (mar["class"] == "rung accepts | determinate")]
    base = mar[mar["class"] == "class true | determinate"].set_index(
        ["axis", "rung"])
    ndet = mar[(mar.axis == "-")
               & (mar["class"] == "determinate | all")].set_index("rung")
    x = np.arange(len(RUNGS))

    def curve(ax, idx, axis, cls, col, ls):
        r = [idx.loc[(axis, k, cls)] for k in RUNGS]
        p = np.array([v.p for v in r])
        ax.fill_between(x, [v.lo for v in r], [v.hi for v in r], color=col,
                        alpha=0.13, lw=0, zorder=2)
        ax.plot(x, p, ls=ls, color=col, lw=1.1, marker="o", ms=2.4, mew=0.8,
                mfc=col if ls == "-" else "w", mec=col, zorder=3)

    for a, _lab, col in TW_AXES:
        curve(ax1, fw, a, "class true", col, "-")
        curve(ax1, fw, a, "class false", col, "--")
        curve(ax2, rv, a, "rung accepts", col, "-")
        curve(ax2, rv, a, "rung rejects", col, "--")
        # the axis base rate drifts a little across rungs because each rung
        # has its own determinate subset; draw it as it is, not as a flat line
        ax2.plot(x, [float(base.loc[(a, k), "p"]) for k in RUNGS], color=col,
                 lw=0.8, alpha=0.38, zorder=1)
    # dotted, so the unconditional rate cannot be mistaken for one of the six
    # conditional curves it runs between
    ax1.plot(x, acc.set_index("rung").loc[RUNGS, "p"].to_numpy(), color="#777777",
             lw=1.0, ls=":", zorder=1)

    lo, hi = int(ndet.k.min()), int(ndet.k.max())
    ax2.text(4.25, 0.665, f"n = {lo:,}–{hi:,} per law set", ha="right",
             va="center", fontsize=7.2, color="#888888")

    # key A, the three axes and the unconditional rate, in the upper row's
    # empty lower-left corner
    ha = [plt.Line2D([], [], color=c, lw=1.1) for _, _, c in TW_AXES]
    ha.append(plt.Line2D([], [], color="#777777", lw=1.0, ls=":"))
    ax1.legend(ha, [b for _, b, _ in TW_AXES] + ["marginal"], frameon=False,
               fontsize=7.2, loc="lower left", bbox_to_anchor=(0.005, 0.005),
               handlelength=1.5, handletextpad=0.5, labelspacing=0.24,
               borderpad=0.0, borderaxespad=0.0)

    # key B, the two line styles and the base rate, in the gap between rows
    hb = [plt.Line2D([], [], color=KEYC, lw=1.1, ls="-", marker="o", ms=2.4,
                     mew=0.8, mfc=KEYC, mec=KEYC),
          plt.Line2D([], [], color=KEYC, lw=1.1, ls="--", marker="o", ms=2.4,
                     mew=0.8, mfc="w", mec=KEYC),
          plt.Line2D([], [], color=REFC, lw=0.8, alpha=0.6)]
    lb = ["in group and satisfies law set",
          "outside group and does not satisfy law set",
          "base rate"]
    ax2.legend(hb, lb, frameon=False, fontsize=7.2, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), handlelength=1.9,
               handletextpad=0.5, labelspacing=0.24, columnspacing=1.0,
               borderpad=0.0, borderaxespad=0.0)

    for ax in (ax1, ax2):
        ax.set_xlim(-0.3, 4.3)
        ax.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.set_ylim(0.66, 1.00)
    ax1.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax1.set_ylabel("satisfaction within class", fontsize=7.5)
    ax2.set_xticklabels([RLAB[k] for k in RUNGS], fontsize=7.5)
    ax2.set_ylim(0.25, 0.70)
    ax2.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax2.set_ylabel("class enrichment", fontsize=7.5)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ca = pd.read_csv(DATA / "fig3_coverage_accuracy.csv")
    ca["s"] = ca.rule.map(SHORT); ca["c"] = ca.s.map(CMAP)
    t1 = pd.read_csv(DATA / "fig7_top1.csv")
    t1["s"] = t1.rule.map(SHORT); t1["c"] = t1.s.map(CMAP)
    rr = pd.read_csv(DATA / "rank_rulesets.csv").set_index("rule")
    r3 = json.load(open(F3D / "resolve_f3.json"))
    ci3 = json.load(open(F3D / "paired_curve_ci.json"))
    tw = pd.read_csv(DATA / "fig5_twoway_ladder.csv")

    fig = plt.figure(figsize=(W2, _H * CM))
    axa = fig.add_axes(_rect(XL, WL, Y1, H1))
    axb = fig.add_axes(_rect(XR, WR, Y1, H1))
    axc = fig.add_axes(_rect(XL, WL, Y2, H2))
    axd = fig.add_axes(_rect(XR, WR, Y2, H2))
    axe = fig.add_axes(_rect(XE, WE, Y3, H3))
    axf1 = fig.add_axes(_rect(XR, WR, Y3, FH1))
    axf2 = fig.add_axes(_rect(XR, WR, Y3 + FH1 + FGAP, FH2))

    panel_a(axa, ca, rr)
    panel_b(axb, t1, rr)
    panel_c(axc, ca)
    panel_d(axd, r3, ci3)
    panel_e(axe)
    panel_f(axf1, axf2, tw)

    stamp(fig, [(axa, "a"), (axb, "b", (DXR, 0.0)), (axc, "c"),
                (axd, "d", (DXR, 0.0)), (axe, "e"), (axf1, "f", (DXR, 0.0))])
    save(fig, "fig5_ranking")


if __name__ == "__main__":
    main()
