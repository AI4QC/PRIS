#!/usr/bin/env python3
"""Supplementary figures. Arial, no titles (captions carry them), data from paper/si_data."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np, pandas as pd, pathlib, sys, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figs import (plt as _p, stamp, save, CM, W1, W2,
                        BLU, RED, ORA, GRN, GRY, PUR, DATA, OUT, ROOT,
                        SHORT, CMAP, setlab, dmglab)

SI = DATA.parent / "si_data"


def _split(split="calibration"):
    """与 paper_figs 同名助手一致:按划分取法则集逐行指标。"""
    d = pd.read_csv(SI / "s5_split_consistency.csv")
    return d[d.split == split].set_index("ruleset")


def si1_chemistry():
    """Set 1 与 Set 1' 按阴离子族分层 —— 守卫恰好在离子域内起作用。"""
    d = pd.read_csv(SI / "s1_chemistry.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 8.6 * CM))
    fig.subplots_adjust(left=0.092, right=0.985, top=0.905, bottom=0.155, wspace=0.24)
    x = np.arange(len(d)); w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, d.L1_sat, w, color=BLU, lw=0, alpha=0.9, label="Set 1")
    ax.bar(x + w / 2, d.L1p_sat, w, color=PUR, lw=0, alpha=0.9, label="Set 1′")
    ax.set_ylim(0.93, 1.005); ax.set_xticks(x); ax.set_xticklabels(d.anion, fontsize=8.0)
    ax.set_xlabel("Anion"); ax.set_ylabel("Satisfaction of experimental structures")
    ax.axhline(0.99, color="#86888A", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=8.0, ncol=2, loc="lower left",
              bbox_to_anchor=(-0.01, 1.00), borderpad=0.1, handlelength=1.2)

    ax = axes[1]
    ax.bar(x - w / 2, d.L1_excl, w, color=BLU, lw=0, alpha=0.9)
    ax.bar(x + w / 2, d.L1p_excl, w, color=PUR, lw=0, alpha=0.9)
    ax.set_ylim(0, 0.72); ax.set_xticks(x); ax.set_xticklabels(d.anion, fontsize=8.0)
    ax.set_xlabel("Anion"); ax.set_ylabel("Damage detection")
    for i, (a, b) in enumerate(zip(d.L1_excl, d.L1p_excl)):
        if b - a > 0.05:
            ax.annotate("", xy=(i + w / 2, b - 0.006), xytext=(i - w / 2, a + 0.006),
                        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#424446",
                                        mutation_scale=5))
    stamp(fig, [(axes[0], "a"), (axes[1], "b")])
    save(fig, "figS1_chemistry")


def si2_threshold():
    """阈值敏感性:为什么是 0.735,以及膨胀类恒为零。"""
    d = pd.read_csv(SI / "s2_threshold.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 8.8 * CM))
    # 顶部留出一条带:b 的五行图例移到数据区之外,两个面板因此共用 0-1 的纵轴
    fig.subplots_adjust(left=0.088, right=0.985, top=0.815, bottom=0.135, wspace=0.26)

    ax = axes[0]
    ax.plot(d.tau, d.sat, "-", color=BLU, lw=1.3, label="satisfaction (experimental)")
    ax.plot(d.tau, d.excl, "-", color=RED, lw=1.3, label="damage detection")
    for t, lb, c in [(0.7353, "Set 1\n0.735", BLU), (0.8044, "Set 2\n0.804", ORA)]:
        ax.axvline(t, color=c, lw=0.8, ls="--")
        ax.text(t, 1.02, lb, fontsize=7.7, color=c, ha="center", va="bottom",
                linespacing=1.3)
    ax.set_xlim(0.55, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"Threshold $\tau$ in $\rho \geq \tau$")
    ax.set_ylabel("Fraction")
    # 放大后图例挡住上升的红曲线,移到两条曲线之间的空白带
    ax.legend(frameon=False, fontsize=8.0, loc="center left",
              bbox_to_anchor=(0.02, 0.70))

    ax = axes[1]
    for k, c, lb in [("S3", GRN, "D3 random displacement"),
                     ("S2", ORA, "D2 cation–cation swap"),
                     ("S1", BLU, "D1 uniaxial compression"),
                     ("S5", PUR, "D5 cation–anion swap"),
                     ("S4", RED, "D4 isotropic expansion")]:
        ax.plot(d.tau, d[k], "-", color=c, lw=1.2, label=lb)
    ax.set_xlim(0.55, 1.0); ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"Threshold $\tau$ in $\rho \geq \tau$")
    ax.set_ylabel("Damage detection by type")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2,
              handlelength=1.3, labelspacing=0.30, columnspacing=1.4,
              borderpad=0.1, bbox_to_anchor=(-0.02, 1.005))
    stamp(fig, [(axes[0], "a"), (axes[1], "b")])
    save(fig, "figS2_threshold")


def si3_band_grid():
    """(tau_lo, tau_hi) 二维扫描:上界的价值只在低满足率处出现。"""
    z = np.load(SI / "s3_band_grid.npz")
    lo, hi, sat, exc = z["lo"], z["hi"], z["sat"], z["excl"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 9.2 * CM))
    fig.subplots_adjust(left=0.090, right=0.918, top=0.945, bottom=0.130, wspace=0.52)
    ext = [hi[0], hi[-1], lo[0], lo[-1]]

    ax = axes[0]
    im = ax.imshow(exc, origin="lower", aspect="auto", extent=ext, cmap="magma")
    cs = ax.contour(hi, lo, sat, levels=[0.90, 0.95, 0.98, 0.99],
                    colors="w", linewidths=0.7)
    ax.clabel(cs, fmt="%.2f", fontsize=7.5, inline=True)
    ax.set_xlabel(r"Upper bound $\tau_{\mathrm{hi}}$")
    ax.set_ylabel(r"Lower bound $\tau_{\mathrm{lo}}$")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Damage detection", fontsize=8.0); cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.5)
    ax.scatter([1.05], [0.735], s=34, c="w", ec="k", lw=0.8, zorder=5, marker="s")
    ax.text(1.10, 0.748, "Set 1′", fontsize=8.0, color="w", fontweight="bold")
    ax.text(0.98, 0.03, "white contours: satisfaction", transform=ax.transAxes,
            fontsize=7.5, color="w", ha="right")

    ax = axes[1]
    bf = pd.read_csv(DATA / "fig4_band.csv")
    ax.plot(bf.floor, bf.one_excl, "o-", color=RED, lw=1.2, ms=3.4, mec="w", mew=0.6,
            label="best one-sided")
    ax.plot(bf.floor, bf.two_excl, "s-", color=PUR, lw=1.2, ms=3.4, mec="w", mew=0.6,
            label="best two-sided")
    ax.set_xlabel("Minimum satisfaction"); ax.set_ylabel("Damage detection")
    ax.set_xlim(0.995, 0.895)
    ax.legend(frameon=False, fontsize=8.0, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(bf.floor, bf.two_hi, "^--", color="#646668", lw=0.9, ms=3.0)
    ax2.set_ylabel(r"Selected $\tau_{\mathrm{hi}}$", fontsize=8.5, color="#646668")
    ax2.tick_params(labelsize=7.5, colors="#646668")
    ax2.spines["right"].set_visible(True); ax2.spines["right"].set_linewidth(0.6)
    ax2.axvspan(0.945, 0.955, color="#000000", alpha=0.06, lw=0)
    stamp(fig, [(axes[0], "a"), (axes[1], "b")])
    save(fig, "figS3_band_grid")


def si4_split():
    """discovery 与留出集逐条对照 —— 无过拟合的直接证据。"""
    d = pd.read_csv(SI / "s5_split_consistency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 8.6 * CM))
    fig.subplots_adjust(left=0.095, right=0.940, top=0.885, bottom=0.135, wspace=0.34)
    sets = ["L1", "L1'", "L2", "L3", "L4"]
    cols = [RED, PUR, ORA, GRN, "#0A5A3C"]
    for ax, met, lab, lt in [(axes[0], "sat", "Satisfaction", "a"),
                             (axes[1], "excl", "Damage detection", "b")]:
        x = np.arange(len(sets)); w = 0.36
        dv = [float(d[(d.split == "discovery") & (d.ruleset == s)][met].iloc[0]) for s in sets]
        cv = [float(d[(d.split == "calibration") & (d.ruleset == s)][met].iloc[0]) for s in sets]
        ax.bar(x - w / 2, dv, w, color=cols, lw=0, alpha=0.42)
        ax.bar(x + w / 2, cv, w, color=cols, lw=0, alpha=0.95)
        ax.set_xticks(x)
        # 归档表以 L1--L4 为键;图内用正文的 Set 1--Set 4
        ax.set_xticklabels([setlab(v) for v in sets], fontsize=8.5)
        ax.set_ylabel(lab)
        for i in range(len(sets)):
            ax.text(x[i], max(dv[i], cv[i]) + 0.014, f"{cv[i]-dv[i]:+.4f}", ha="center",
                    fontsize=7.5, color="#424446")
        ax.set_ylim(0, 1.06)          # 两个面板同为 0-1 的比率:同刻度同上限
    # 一个图例,不是每个面板一份:两行都放在整幅图的顶端居中
    fig.text(0.5175, 0.972, "pale: discovery (thresholds fitted here)      "
                            "solid: held-out split      "
                            "bar labels: held-out minus discovery",
             ha="center", va="bottom", fontsize=7.4, color="#535557")
    stamp(fig, [(axes[0], "a"), (axes[1], "b")])
    save(fig, "figS4_split_consistency")


def si5_perclass():
    """四个法则集的分类排除力矩阵,两个划分并排 —— 主文图 1c 的完整版。"""
    d = pd.read_csv(SI / "s5_split_consistency.csv")
    KS = ["S1", "S2", "S3", "S4", "S5"]
    sets = ["L1", "L1'", "L2", "L3", "L4"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 8.4 * CM))
    fig.subplots_adjust(left=0.070, right=0.90, top=0.895, bottom=0.255, wspace=0.20)
    for ax, sp_name in zip(axes, ["discovery", "calibration"]):
        M = d[d.split == sp_name].set_index("ruleset").loc[sets, KS].values
        im = ax.imshow(M, cmap="palmatrix", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(["D1\nuniaxial compression", "D2\n\ncation–cation swap",
                            "D3\n\n\nrandom displacement", "D4\nisotropic expansion",
                            "D5\n\ncation–anion swap"], fontsize=7.5)
        ax.set_yticks(range(5))
        ax.set_yticklabels(["Set 1", r"Set 1$'$", "Set 2", "Set 3", "Set 4"],
                           fontsize=8.5)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.0,
                        color="w" if M[i, j] > 0.55 else "#212224")
        display_name = "held-out" if sp_name == "calibration" else sp_name
        ax.text(0.0, 1.018, f"{display_name} split", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=8.0, color="#646668")
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    cb.set_label("Damage detection", fontsize=8.0); cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.5)
    stamp(fig, [(axes[0], "a"), (axes[1], "b")])
    save(fig, "figS5_perclass")


def si6_loko():
    """留一扰动类:认证最优树 vs 单阈值 vs 守卫带。"""
    lk = pd.read_csv(DATA / "fig3_loko.csv")
    bl = pd.read_csv(DATA / "fig4_band_loko.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 8.8 * CM))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.945, bottom=0.130, wspace=0.26)

    ax = axes[0]
    ax.plot([0, 0.9], [0, 0.9], color="#CACCCF", lw=0.8, ls=":")
    ax.scatter(lk.tree_seen, lk.tree_held, s=34, c=PUR, ec="w", lw=0.6,
               label="best tree, depth $\\leq$ 3")
    ax.scatter(lk.thr_seen, lk.thr_held, s=34, c=RED, ec="w", lw=0.6, marker="s",
               label=r"single $\rho$ threshold")
    for _, r in lk.iterrows():
        ax.annotate(dmglab(r.held), (r.tree_seen, r.tree_held), textcoords="offset points",
                    xytext=(7, -1), fontsize=7.5, color=PUR)
    ax.set_xlim(0, 0.90); ax.set_ylim(0, 0.95)
    ax.set_xlabel("Damage detection, types used during selection")
    ax.set_ylabel("Damage detection, omitted type")
    ax.legend(frameon=False, fontsize=7.7, loc="upper left",
              bbox_to_anchor=(0.185, 1.00))

    ax = axes[1]
    x = np.arange(len(bl)); w = 0.38
    ax.bar(x - w / 2, bl.base, w, color=GRY, lw=0, alpha=0.75,
           label=r"Set 1 (one-sided) on omitted type")
    ax.bar(x + w / 2, bl.excl_held, w, color=PUR, lw=0, alpha=0.95,
           label="Set 1′ (conditional) on omitted type")
    ax.set_xticks(x); ax.set_xticklabels([dmglab(v) for v in bl.held], fontsize=8.0)
    ax.set_xlabel("Damage type omitted during threshold selection")
    ax.set_ylabel("Damage detection")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7.7, loc="upper left")
    i = int(np.argmax(bl.gain.values))
    ax.annotate(f"+{bl.gain.iloc[i]:.3f}",
                (x[i] + w / 2, bl.excl_held.iloc[i]), textcoords="offset points",
                xytext=(0, 20), fontsize=7.7, color=PUR, ha="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=PUR, mutation_scale=6))
    stamp(fig, [(axes[0], "a"), (axes[1], "b")])
    save(fig, "figS6_loko")


def si9_ranking_data():
    """Database scale, energy-span matching and composition concentration."""
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(W2, 8.8 * CM), gridspec_kw={"width_ratios": [0.82, 1.35]}
    )
    fig.subplots_adjust(left=0.083, right=0.908, top=0.94, bottom=0.22, wspace=0.43)

    # ---- (a) evidence base behind the cross-database comparison
    db = pd.read_csv(DATA / "fig4_db_concentration.csv")
    db = db[db.database != "MP synthesis target"]
    axa.bar(range(len(db)), db.n_structures, color=[BLU, ORA, GRN, PUR], width=0.62,
            lw=0, alpha=0.82)
    axa.set_yscale("log")
    axa.set_ylim(1e3, 6e5)
    axa.set_xticks(range(len(db)))
    axa.set_xticklabels(["MP exp.", "ELEMENTA", "Alexandria", "LeMat"],
                        fontsize=6.5, rotation=0, ha="center")
    axa.tick_params(axis="x", pad=3)
    axa.set_ylabel("Structures in comparison")
    for i, n in enumerate(db.n_structures):
        axa.text(i, n * 1.28, f"{n:,}", ha="center", fontsize=7.4, color="#424446")

    # ---- (b) matched energy window and composition concentration
    names = ["MP\nexperimental", "ELEMENTA", "Alexandria", "LeMat"]
    acc = [0.6912, 0.6323, 0.6287, 0.5987]
    span = [25, 99, 186, 449]
    cc = [BLU, GRN, ORA, PUR]
    axb.bar(range(4), acc, color=cc, width=0.38, lw=0, alpha=0.78)
    axb.axhline(0.5, color="#646668", lw=0.7, ls="--")
    axb.set_xticks(range(4))
    axb.set_xticklabels([f"{n}\n{sp} meV span" for n, sp in zip(names, span)],
                        fontsize=7.6)
    axb.set_xlim(-0.55, 3.55)
    axb.set_ylim(0.45, 0.78)
    axb.set_ylabel("Group-equal accuracy\n([50, 200) meV per atom)", fontsize=8.0)
    for i, a in enumerate(acc):
        axb.text(i, a + 0.008, f"{a:.4f}", ha="center", fontsize=7.7, color=cc[i])
    ax2 = axb.twinx()
    frac = [59.3, 0.04, 2.8, 0.006]
    ax2.plot(range(4), frac, "D--", color="#424446", ms=5.0, lw=0.8,
             mec="#212224", mew=0.6, alpha=0.78)
    ax2.set_yscale("log")
    ax2.set_ylim(3e-3, 3e3)
    ax2.set_ylabel("Largest-composition share (%)", fontsize=8.0)
    ax2.tick_params(labelsize=8.2)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)

    stamp(fig, [(axa, "a"), (axb, "b")])
    save(fig, "figS10_ranking_data")


def si10_unguarded_band():
    """整条法则阶梯对上"无守卫的最优双侧带",以及膨胀类的分解。

    整体搬自主图原 Fig. 3c;无守卫双侧带的三个数字取自 fig4_band.csv 的
    0.99 满足率行(two_excl / two_S4 / two_sat),与原面板逐位一致。
    """
    # every other \textwidth SI figure is ~17.8 cm native and is therefore reduced by
    # LaTeX; at 12.6 cm this one was enlarged 1.23x and printed larger than its neighbours
    fig, ax = plt.subplots(figsize=(18.8 * CM, 8.8 * CM))
    fig.subplots_adjust(left=0.108, right=0.988, top=0.972, bottom=0.148)

    dis = _split("discovery")
    bd = pd.read_csv(DATA / "fig4_band.csv")
    bd = bd[np.isclose(bd.floor, 0.99)].iloc[0]

    names = ["Set 1\nLaw 1 alone", "\n\nunconditional\ntwo-sided range",
             "Set 1′\nLaw 1 + Law 2", "Set 2\n\nLaw 1, Law 3–Law 5",
             "Set 3\nLaw 1, Law 3–Law 6", "Set 4\n\nLaw 1, Law 3–Law 8"]
    tot = [float(dis.loc["L1", "excl"]), float(bd.two_excl),
           float(dis.loc["L1'", "excl"]), float(dis.loc["L2", "excl"]),
           float(dis.loc["L3", "excl"]), float(dis.loc["L4", "excl"])]
    s4v = [float(dis.loc["L1", "S4"]), float(bd.two_S4),
           float(dis.loc["L1'", "S4"]), float(dis.loc["L2", "S4"]),
           float(dis.loc["L3", "S4"]), float(dis.loc["L4", "S4"])]
    sat = [float(dis.loc["L1", "sat"]), float(bd.two_sat),
           float(dis.loc["L1'", "sat"]), float(dis.loc["L2", "sat"]),
           float(dis.loc["L3", "sat"]), float(dis.loc["L4", "sat"])]
    cc = [BLU, GRY, PUR, ORA, RED, "#0A5A3C"]
    x = np.arange(6); w = 0.32
    ax.bar(x - w / 2, tot, w, color=cc, lw=0, alpha=0.92)
    ax.bar(x + w / 2, s4v, w, color=cc, lw=0, alpha=0.42, hatch="////", edgecolor="w")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.8)
    ax.set_ylabel("Damage detection (discovery split)")
    ax.set_ylim(0, 1.30)
    for i in range(6):
        # 两个数值标签横向必然相碰(组内间距 < 标签宽),因此靠得近时强制错层
        ytot = tot[i] + 0.014
        ys4 = s4v[i] + 0.014
        if abs(ys4 - ytot) < 0.100:
            ys4 = ytot + 0.100
        ax.text(x[i] - w / 2, ytot, f"{tot[i]:.3f}", ha="center",
                fontsize=7.5, color=cc[i], fontweight="bold")
        ax.text(x[i] + w / 2, ys4, f"{s4v[i]:.3f}", ha="center",
                fontsize=7.5, color=cc[i])
        ax.text(x[i], 1.120, f"satisfaction\n{sat[i]:.4f}", ha="center", fontsize=7.5,
                color="#757779", linespacing=1.30)
    # 实心/斜纹此前在图内没有任何说明
    hs = [mp.Patch(fc="#757779", alpha=0.92, lw=0),
          mp.Patch(fc="#757779", alpha=0.42, hatch="////", ec="w", lw=0)]
    ax.legend(hs, ["all five classes pooled", "D4 isotropic expansion"],
              frameon=False, fontsize=7.8, loc="upper left", handlelength=1.3,
              labelspacing=0.30, borderpad=0.1, bbox_to_anchor=(-0.012, 0.855))
    save(fig, "figS11_unguarded_band")


def si12_ranking_extras():
    """主图排序图(Fig. 5)合并时被降级的两块面板。

    a  原 Fig. 4c:top-1 命中率减去随机基线(1,508 个同成分组)。规则以"能不能
       挑出唯一结构"计分,弃权按随机记账,因此本图与 Fig. 5b 的弃权率同源。
    b  原 Fig. 6c:F2R 的组等权稳定性准确率随 |ΔE_hull| 下限抬升,以及在
       ≥25 meV/atom 对上重新拟合的 F2RG 单点(菱形)。

    两块面板的数值、配色、刻度与文字均逐字搬自原面板,只是换了版面。
    数据:paper/data/fig7_top1.csv,outputs/20260814_f2r_stability/resolve_f2r.json,
    outputs/20260814_f2rg_gap25/calib_result.json。
    """
    t1 = pd.read_csv(DATA / "fig7_top1.csv")
    t1["s"] = t1.rule.map(SHORT); t1["c"] = t1.s.map(CMAP)
    r2 = json.load(open(ROOT / "outputs" / "20260814_f2r_stability" /
                        "resolve_f2r.json"))
    fg = json.load(open(ROOT / "outputs" / "20260814_f2rg_gap25" /
                        "calib_result.json"))

    # 版面(cm):左栏留 3.05 cm 给长行标签,右栏留 2.6 cm 给两行 y 轴标题,
    # 两个面板同上沿同高,面板字母因此落在两条竖线上。
    _W, _H = 18.3, 8.60
    fig = plt.figure(figsize=(W2, _H * CM))

    def _rect(x0, w, ytop, h):
        return [x0 / _W, (_H - ytop - h) / _H, w / _W, h / _H]

    # ---- (a) top-1 lift over random  (原 Fig. 4c)
    axa = fig.add_axes(_rect(3.05, 5.60, 0.50, 6.45))
    o = t1.sort_values("lift")
    axa.axvline(0, color="#646668", lw=0.7)
    axa.barh(np.arange(len(o)), o.lift, color=o.c, height=0.66, lw=0, alpha=0.9)
    axa.set_yticks(np.arange(len(o))); axa.set_yticklabels(o.s, fontsize=7.5)
    axa.set_xlabel("Top-1 hit rate − random baseline")
    axa.set_xlim(-0.265, 0.60); axa.set_xticks([0, 0.2, 0.4])
    for i, (val, c) in enumerate(zip(o.lift, o.c)):
        axa.text(val + (0.012 if val >= 0 else -0.012), i, f"{val:+.3f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=7.5, color=c)

    # ---- (b) stability accuracy vs energy-gap floor  (原 Fig. 6c)
    axb = fig.add_axes(_rect(11.55, 6.30, 0.50, 6.45))
    thr = [0, 10, 25, 50, 100]
    gb = [r2["gap_bins"][f"{t/1000:.3f}"] for t in thr]
    y = [v["group_equal_acc"] for v in gb]
    ylo = [v["group_equal_acc"] - v["ci_group"][0] for v in gb]
    yhi = [v["ci_group"][1] - v["group_equal_acc"] for v in gb]
    axb.errorbar(range(len(thr)), y, yerr=[ylo, yhi], fmt="-o", color=ORA, lw=1.2,
                 ms=4, capsize=2, elinewidth=0.7, label="stability score, held-out data")
    a, ci = fg["result"]["accuracy"], fg["ci"]
    axb.errorbar([2.18], [a], yerr=[[a - ci[0]], [ci[1] - a]], fmt="D", color=RED,
                 ms=4.5, capsize=2, elinewidth=0.7,
                 label="refit using pairs $\\geq$25 meV")
    axb.axhline(0.8, color="#B9BBBE", lw=0.7, ls=":")
    axb.text(3.55, 0.808, "0.8", fontsize=7.5, color="#97999C", va="bottom")
    axb.set_xticks(range(len(thr)))
    axb.set_xticklabels(["all", "$\\geq$10", "$\\geq$25", "$\\geq$50", "$\\geq$100"],
                        fontsize=8.0)
    axb.set_xlabel("Pairs restricted to $|\\Delta E_{\\mathrm{hull}}|$ (meV/atom)")
    axb.set_ylabel("Accuracy (equal weight\nper composition)")
    axb.set_ylim(0.55, 0.92)
    axb.legend(frameon=False, fontsize=7.7, loc="upper left", handlelength=1.6)

    stamp(fig, [(axa, "a"), (axb, "b")])
    save(fig, "figS12_ranking_extras")


def si13_failure_modes():
    """图 S13:反驳账本按失败模式分组(原主文图 1d,按作者决定移入 SI)。"""
    _W, _H = 12.0, 5.20
    fig = plt.figure(figsize=(_W * CM, _H * CM))
    GUT = 3.05
    ax = fig.add_axes([(0.30 + GUT) / _W, 0.95 / _H,
                       (_W - 0.55 - GUT - 0.30) / _W, (_H - 1.35) / _H])
    r = pd.read_csv(DATA / "fig5_retractions.csv")
    order = ["Metric artefact", "Aggregation hides structure",
             "Leakage of the negative generator", "Degenerate or confounded control",
             "Search or implementation defect"]
    cnt = r.failure_mode.value_counts().reindex(order)
    ids = r.groupby("failure_mode", group_keys=False).apply(
        lambda g: ", ".join(f"R{i+1}" for i in g.index)).reindex(order)
    yy = np.arange(len(cnt))[::-1]
    ax.barh(yy, cnt.values, color=RED, alpha=0.80, height=0.60, lw=0)
    ax.set_yticks(yy)
    ax.set_yticklabels([c.replace("Leakage of the", "Leakage of the\n").replace(
                        "Aggregation hides", "Aggregation hides\n").replace(
                        "Degenerate or", "Degenerate or\n").replace(
                        "Search or", "Search or\n") for c in cnt.index],
                       fontsize=7.4, linespacing=1.15, ha="left")
    ax.tick_params(axis="y", pad=GUT / 2.54 * 72, length=0)
    ax.set_xlabel("Claims refuted")
    ax.set_xlim(0, 4.6); ax.set_xticks([0, 1, 2, 3])
    for i, lb in enumerate(ids.values):
        ax.text(cnt.values[i] + 0.13, yy[i], lb, va="center", fontsize=7.4,
                color="#757779")
    save(fig, "figS9_failure_modes")


if __name__ == "__main__":
    si1_chemistry(); si2_threshold(); si3_band_grid()
    si4_split(); si5_perclass(); si6_loko()
    si9_ranking_data(); si10_unguarded_band()
    si12_ranking_extras(); si13_failure_modes()
