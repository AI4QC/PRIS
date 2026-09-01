#!/usr/bin/env python3
"""论文主图 —— 本脚本产出 Fig 1/2/4/5,Arial,多子图拼版。数据一律从 paper/data/*.csv 读。

图序(发现导向的叙事主线):
  Fig 1  规则是怎么被发现的:agentic law learning 回路、规模、反驳账本、证据库
  Fig 2  发现的规则本身:全部规则+公式一次呈现,联合表现,与 Pauling 正面对比
  Fig 3  规则的验证、盲区、守卫 —— 已移出本脚本,由 src/fig3_detection.py 生成
         (跨库比较面板位于 legacy asset figS10 的面板 d，正文显示为 Supplementary Fig. S2d；见 src/si_figs.py)
  Fig 4  规则替代了什么:Pauling 规则以"弃权"而非"排错"为主要失效模式
  Fig 5  反驳账本全表

命名体系(与正文一致):
  ALL         agentic law learning,智能体自主定律归纳
  rho         约化接触比 min_bonds d/(r_cat + r_an)
  Born floor  rho 的下界律(D1)
  commitment profile   (承诺率, 承诺时准确率)

面板字母对齐:一律用 stamp(),按 gridspec 的行/列聚类,同列共用一个 x、
同行共用一个 y,x 取该列所有面板 y 轴装饰(刻度标签+轴标题)的最左端。
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.colors import to_rgba
import numpy as np, pandas as pd, pathlib, json

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = pathlib.Path(__file__).resolve().parent.parent / "paper"
DATA, OUT = D / "data", D / "figs"
SI_DATA = D / "si_data"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial", "Liberation Sans"],
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
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

import figure_palette  # noqa: F401  (registers the ramps, caps bar opacity)

CM = 1 / 2.54
W1, W2 = 8.9 * CM, 18.3 * CM                      # Nature 单栏 / 双栏
BLU, RED, ORA, GRN, GRY, PUR = ("#005B93", "#D6564C", "#9861B0",
                                "#156646", "#8A8C8E", "#35A7D8")
SETC = {"L1": BLU, "L1'": PUR, "L2": ORA, "L3": RED, "L4": "#0A5A3C"}
BAR_FACE_ALPHA = 0.38
BAR_EDGE_WIDTH = 0.85


def style_bar_container(
    bars,
    colours,
    *,
    face_alpha: float = BAR_FACE_ALPHA,
    edge_width: float = BAR_EDGE_WIDTH,
    opaque_fill_weight: float | None = None,
):
    """Give bars a same-hue outline and either translucent or pale opaque fill."""
    if isinstance(colours, str):
        colours = [colours] * len(bars)
    if len(colours) != len(bars):
        raise ValueError("one bar colour is required for every patch")
    for bar, colour in zip(bars, colours, strict=True):
        # figure_palette applies an artist-level opacity cap to every bar call.
        # Clear it so the face and outline can have independent alpha values.
        if opaque_fill_weight is None:
            bar.set_alpha(None)
            bar.set_facecolor(to_rgba(colour, face_alpha))
        else:
            if not 0.0 <= opaque_fill_weight <= 1.0:
                raise ValueError("opaque_fill_weight must lie in [0, 1]")
            base = np.asarray(to_rgba(colour)[:3])
            face = opaque_fill_weight * base + (1.0 - opaque_fill_weight)
            bar.set_alpha(1.0)
            bar.set_facecolor((*face, 1.0))
        bar.set_edgecolor(to_rgba(colour, 1.0))
        bar.set_linewidth(edge_width)
    return bars


def setlab(key, prime="′"):
    """规则集的归档键 -> 图内显示名（与正文一致的 Set 1--Set 4）。

    归档的结果表仍以 L1/L1'/L2/L3/L4 为键，所以键与显示名分开：
    只在绘制时转换，数据索引一律用原键。
    """
    return "Set " + key[1:].replace("'", prime)


def dmglab(key):
    """损伤类型的归档键 -> 图内显示名（S1--S5 -> D1--D5）。

    归档表与 kind 列仍以 S1--S5 为键，改名只发生在绘制时，
    否则会与正文的 Set 1--Set 5 混读。
    """
    return "D" + key[1:]


# ─────────────────────────────────────────────────────────────────────────────
# 面板字母:网格对齐
# ─────────────────────────────────────────────────────────────────────────────
def _ydeco_left(ax, fig, rend):
    """该 axes 左侧所有 y 轴装饰的最左端,图坐标。"""
    inv = fig.transFigure.inverted()
    xs = [ax.get_position().x0]
    for t in ax.get_yticklabels():
        if t.get_visible() and t.get_text():
            xs.append(inv.transform_bbox(t.get_window_extent(rend)).x0)
    lab = ax.yaxis.get_label()
    if lab.get_text():
        xs.append(inv.transform_bbox(lab.get_window_extent(rend)).x0)
    return min(xs)


def stamp(fig, items, pad_x=0.013, pad_y=0.011, size=10):
    """网格对齐的面板字母。

    items: [(ax, "a"), ...] 或 [(ax, "a", (dx, dy)), ...] 作局部微调。
    同一 gridspec 列的面板共用一个 x(取该列最左的 y 轴装饰边界再左移 pad_x),
    同一行的面板共用一个 y(取该行 axes 顶边再上移 pad_y)。
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    rows: dict[float, list[float]] = {}
    cols: dict[float, list[float]] = {}
    rec = []
    for it in items:
        ax, letter = it[0], it[1]
        dx, dy = it[2] if len(it) > 2 else (0.0, 0.0)
        bb = ax.get_position()
        ck, rk = round(bb.x0, 3), round(bb.y1, 3)
        cols.setdefault(ck, []).append(_ydeco_left(ax, fig, rend))
        rows.setdefault(rk, []).append(bb.y1)
        rec.append((letter, ck, rk, dx, dy))
    for letter, ck, rk, dx, dy in rec:
        fig.text(min(cols[ck]) - pad_x + dx, max(rows[rk]) + pad_y + dy, letter,
                 fontsize=size, fontweight="bold", va="bottom", ha="left")


def panel(ax, letter, dx=-0.16, dy=1.06):
    """旧式单面板字母;仅 SI 中不便网格对齐的散图仍在用。"""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=400)
    plt.close(fig)
    print("wrote", name)


def _split(split="calibration"):
    d = pd.read_csv(SI_DATA / "s5_split_consistency.csv")
    return d[d.split == split].set_index("ruleset")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 2 — 发现的规则:一次全部列出,再看它们联合起来的表现
# ─────────────────────────────────────────────────────────────────────────────
RULES = [
    ("Law 1", r"$\rho \geq \tau$",
     "shortest cation–anion contact, relative to the radius sum",
     {"L1": "0.735", "L1'": "0.735", "L2": "0.804", "L3": "0.804", "L4": "0.804"}),
    ("Law 2", r"if  $f_{\mathrm{i}} > 0.50$   then   $\rho \leq 1.05$",
     "ionicity condition: an upper bound applied only where the ionic model holds",
     {"L1'": "•"}),
    ("Law 3", r"if  mean anion CN $\leq$ 3.333   then   mean $d/(r_{\mathrm{cat}}{+}r_{\mathrm{an}}) \leq 1.081$",
     "low-coordination structures may not also be loosely packed on average",
     {"L2": "•", "L3": "•", "L4": "•"}),
    ("Law 4", r"range of  $E_{\mathrm{M}}/|z|$  across sites  $\leq$ 31.45",
     "no site may be electrostatically far out of line with the others",
     {"L2": "•", "L3": "•", "L4": "•"}),
    ("Law 5", r"$\mathrm{max}_i\ E_{\mathrm{M}}(i) \leq$ 15.17",
     "no single site may have an implausible Madelung energy",
     {"L2": "•", "L3": "•", "L4": "•"}),
    ("Law 6", r"if  $f_{\mathrm{i}} > 0.55$   then   no like-charge bonds",
     "charge topology: ionic compounds do not bond like to like",
     {"L3": "•", "L4": "•"}),
    ("Law 7", r"inequivalent sites / sites $\leq 2/3$",
     "distinct-site bound consistent with a preference for simpler structures",
     {"L4": "•"}),
    ("Law 8", r"mean relative bond-valence deviation $\leq$ 0.7143",
     "a permissive tail bound on the valence sum, the quantity of rule 2",
     {"L4": "•"}),
]
SETS = ["L1", "L1'", "L2", "L3", "L4"]
# 五类扰动的统一两/三行标签(Fig 2c 与 Fig 4b 共用,行数一致以便对齐)
CLASSLAB = ["D1\nuniaxial\ncompression", "D2\ncation–cation\nswap",
            "D3\nrandom\ndisplacement", "D4\nisotropic\nexpansion",
            "D5\ncation–anion\nswap", "all five\nclasses\npooled"]
# 规则集的名字按"整套 set 所要求的模型"起,不按新增的那一条:
# hard-sphere -> rigid-ion lattice -> ionic network -> crystal chemistry,
# 每一个都包含前一个。列心间距只有 1.43 cm,所以每行不超过 11 个字符。
# Set 1' 不重复写 "hard-sphere":两列间距只有 1.43 cm,同一个词并排时净间距
# 只剩 0.9 mm,会读成一串。"two-sided" 与正文措辞逐字对应。
SETNAME = {"L1": "hard-sphere\nfloor", "L1'": "two-sided\nwindow",
           "L2": "rigid-ion\nlattice", "L3": "ionic\nnetwork",
           "L4": "crystal\nchemistry"}


def fig2():
    cal = _split("calibration")
    _W, _H = 18.3, 12.45
    FH = _H * CM
    fig = plt.figure(figsize=(W2, FH))
    # 显式面板矩形(cm -> figure fraction);行高按字号放大后的行距需求分配,
    # 两个数据行的列宽不同(d/e 的 y 标签比 b/c 长得多),但左右列的 x0 保持一致,
    # 使 stamp 的分列逻辑仍把 a/b/d 与 c/e 各自对齐。
    # 总高受版面约束:这一页的浮动体本来就装不下(见 body.tex 的 [p] 图注),
    # 所以图高不得超过改版前的 20.3 cm 一线,否则图注会压到页码上。
    def _rect(x0, w, ytop, h):
        return [x0 / _W, (_H - ytop - h) / _H, w / _W, h / _H]

    # 两列一格:a/c 共用左列的 [X1, X1R],b/d 共用右列的 [X2, XR],
    # 两列之间的 2.00 cm 留给 d 的常数标签(它们右对齐,贴在右列的纵轴上)。
    X1, X1R, X2, XR = 1.95, 7.60, 9.85, 17.95
    YB, HB_ = 0.50, 4.25
    YC, HC_ = 6.35, 4.95

    # ---- (a) satisfaction–damage-detection plane, discovery split
    axb = fig.add_axes(_rect(X1, X1R - X1, YB, HB_))
    cf = pd.read_csv(DATA / "fig3_certified_frontier.csv").sort_values("sat")
    hcert, = axb.plot(cf.sat, cf.excl, "-", color="#9A9C9F", lw=1.1, zorder=1,
                      label="provably optimal decision trees (depth $\\leq$ 3)")
    # 前沿上的散点只是"线上的结点":标记做小,否则与它旁边的规则集标记
    # (L1'/L2 与前沿点相距只有 0.02--0.06 个数据单位)糊成一团双色斑。
    axb.scatter(cf.sat, cf.excl, s=9, c="#9A9C9F", ec="w", lw=0.5, zorder=2)

    fr = pd.read_csv(DATA / "fig2_frontier.csv").dropna(subset=["exclusion"])
    fr = fr.sort_values("satisfaction")
    # 短周期虚线:前沿在 L1'/L1 之间只有 0.1 cm,长划线在那里会剩下两截
    # 孤立的残墨,细密的划线则明显读成"一条虚线的一部分"。
    hbeam, = axb.plot(fr.satisfaction, fr.exclusion, color="#D0D2D5", lw=0.9,
                      ls=(0, (3.0, 1.6)), zorder=1,
                      label="systematic search over law combinations")
    axb.scatter(fr.satisfaction, fr.exclusion, s=8, c="#D0D2D5", ec="w", lw=0.4,
                zorder=2)
    # 图例收进数据区内的右上角:面板变窄后,原来悬在轴外的图例右端会撞上
    # 右列的面板字母;顶部再抬高 0.09 个单位,腾出图例与前沿曲线之间的间距。
    axb.legend(handles=[hcert, hbeam], frameon=False, fontsize=7.15,
               loc="upper right", handlelength=1.5, handletextpad=0.6,
               borderpad=0.1, labelspacing=0.35, bbox_to_anchor=(1.0, 1.005))

    dis = _split("discovery")
    mks = {"L1": ("o", (2, -13)), "L1'": ("s", (-17, -3)),
           "L2": ("o", (-1, -13)), "L3": ("o", (2, -13)),
           "L4": ("D", (4, -13))}
    for s in SETS:
        m, off = mks[s]
        axb.scatter([dis.loc[s, "sat"]], [dis.loc[s, "excl"]], s=52, c=SETC[s],
                    ec="w", lw=0.8, zorder=5, marker=m)
        # At the base 8.5 pt these five in-plot labels were the largest type in
        # the figure and read as headings rather than as point labels.  Match
        # the row labels of panel d instead.
        axb.annotate(setlab(s), (dis.loc[s, "sat"], dis.loc[s, "excl"]),
                     textcoords="offset points", xytext=off, fontsize=7.6,
                     color=SETC[s], ha="center")
    p = pd.read_csv(DATA / "fig2_pauling_structure.csv")
    # Pauling 2--5 has measured satisfaction but no damage-detection value on this
    # damage benchmark.  Do not place it on the two-axis plane using a proxy
    # coordinate; the measured satisfaction comparison is shown separately in (c).
    axb.set_xlim(0.795, 1.004); axb.set_ylim(0.0, 1.20)
    axb.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axb.spines["left"].set_bounds(0, 1.0)
    axb.set_xlabel("Satisfaction of experimental structures\n(discovery split)",
                   labelpad=2.0, linespacing=1.3)
    axb.set_ylabel("Damage detection", labelpad=4.0,
                   linespacing=1.3)

    # ---- (b) profile by damage type for the four sets, held-out split
    axc = fig.add_axes(_rect(X2, XR - X2, YB, HB_))
    KS = ["S1", "S2", "S3", "S4", "S5"]
    x = np.arange(6)
    bar_width, bar_step = 0.190, 0.135
    for k, s in enumerate(SETS):
        v = [float(cal.loc[s, c]) for c in KS] + [float(cal.loc[s, "excl"])]
        bars = axc.bar(
            x + (k - 2.0) * bar_step,
            v,
            bar_width,
            color=SETC[s],
            edgecolor=SETC[s],
            linewidth=0.85,
            label=setlab(s),
            zorder=3.0 - 0.10 * k,
        )
        # Keep the fill only slightly lighter than the fully opaque, same-hue
        # outline.  Set the patch properties separately because the shared
        # palette wrapper caps artist-level alpha and would otherwise fade the
        # outline together with the interior.
        style_bar_container(bars, SETC[s], opaque_fill_weight=0.55)
    axc.axvline(4.5, color="#B9BBBE", lw=0.7, ls=":")
    axc.set_xlim(-0.60, 5.50)
    axc.set_xticks(x)
    axc.set_xticklabels(
        [dmglab(k) for k in KS] + ["All"],
        fontsize=plt.rcParams["ytick.labelsize"],
    )
    axc.set_ylabel("Damage detection\n(held-out data)", labelpad=4.0,
                   linespacing=1.3)
    axc.set_ylim(0, 1.05)
    axc.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axc.legend(frameon=False, fontsize=7.8, ncol=5, loc="upper left",
               handlelength=1.0, columnspacing=1.0, borderpad=0.1,
               bbox_to_anchor=(-0.015, 1.13))

    # ---- (c) structure-level comparison on the same held-out experimental structures
    axd = fig.add_axes(_rect(X1, X1R - X1, YC, HC_))
    pa = p.set_index("rule")
    PAU, PAUJ = "#989A9D", "#585A5C"     # 中性灰:颜色在本图内只用于标识规则集
    # The separate historical audit uses site-orbit, orbit-pair and structure
    # denominators and therefore cannot support a common-axis comparison here.
    # These rows instead apply each Pauling criterion at structure level to the
    # same 5,297 held-out structures used for the PRIS bars.
    rows = [("Pauling 2", float(pa.loc["Pauling 2", "satisfaction"]), np.nan, PAU),
            ("Pauling 3", float(pa.loc["Pauling 3", "satisfaction"]), np.nan, PAU),
            ("Pauling 4", float(pa.loc["Pauling 4", "satisfaction"]), np.nan, PAU),
            ("Pauling 5", float(pa.loc["Pauling 5", "satisfaction"]), np.nan, PAU),
            ("Pauling 2–5\njointly",
             float(pa.loc["Pauling 2-5 jointly", "satisfaction"]), np.nan, PAUJ)]
    rows += [(setlab(s), float(cal.loc[s, "sat"]), np.nan, SETC[s])
             for s in SETS]
    # 行位置不等距:"Pauling 2--5 / jointly" 是本面板唯一的两行刻度标签,
    # 等距时它的两行会各自吃掉上下邻行 0.05 cm 的留白。给这一行前后各加
    # 0.30 个单位,它与上下邻居的净间距就与其余各行一致(约 0.15 cm)。
    yy = np.array([9.6, 8.6, 7.6, 6.6,             # Pauling 2--5
                   5.35,                           # Pauling 2--5 jointly
                   4.0, 3.0, 2.0, 1.0, 0.0])       # L1 ... L4
    satisfaction_bars = axd.barh(
        yy,
        [r[1] for r in rows],
        color=[r[3] for r in rows],
        height=0.66,
        lw=0,
    )
    style_bar_container(satisfaction_bars, [r[3] for r in rows])
    for i, (lab, v, vhi, c) in enumerate(rows):
        if not np.isnan(vhi) and vhi > v + 1e-9:
            axd.plot([v, vhi], [yy[i]] * 2, color="#535557", lw=0.7, zorder=4)
            axd.plot([vhi] * 2, [yy[i] - 0.17, yy[i] + 0.17], color="#535557",
                     lw=0.7, zorder=4)
        axd.text(max(v, 0 if np.isnan(vhi) else vhi) + 0.016, yy[i], f"{v:.3f}",
                 va="center", fontsize=7.5, color="#535557" if c == PAU else c)
    axd.set_yticks(yy)
    axd.set_yticklabels([r[0] for r in rows], fontsize=7.2, linespacing=1.05,
                        ha="right", multialignment="right")
    axd.tick_params(axis="y", pad=2.5)
    axd.set_xlim(0, 1.12); axd.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axd.set_xlabel("Structure-level satisfaction\n(5,297 held-out crystals)")
    # 分隔线放在 "jointly" 的降部与 L1 的字冠正中间,两侧各留 0.08 cm
    axd.axhline(4.50, color="#B9BBBE", lw=0.7)

    # ---- (d) do the constants remain stable? re-derive each bound at the same percentile
    #          on the held-out split and compare with the selected value
    axe = fig.add_axes(_rect(X2, XR - X2, YC, HC_))
    tr = json.loads((ROOT / "outputs" / "20260815_threshold_transfer"
                     / "transfer.json").read_text())
    # The archive keys retain their historical labels; RLABE below controls the
    # scientifically corrected text shown in the figure.
    KEY = ["D1 tau=0.735 (rho)", "D1 tau=0.804 (rho)", "D2 1.05 (rho, ionic)",
           "D3 1.081 (mean d/rsum)", "D4 31.45 (range VM/z)", "D5 15.17 (max VM)",
           "D7 2/3 (economy)", "D8 0.714 (BV dev)"]
    RLABE = [r"Law 1   $\tau$ = 0.735", r"Law 1   $\tau$ = 0.804", "Law 2   1.05",
             "Law 3   1.081", "Law 4   31.45", "Law 5   15.17", "Law 7   2/3",
             "Law 8   0.7143"]
    XTXT = 1.315                       # verdict-column anchor
    NEU = "#535557"
    # Reuse only colours already established in panels a--c.  Laws 1--4 retain
    # the corresponding set identities; laws without a set identity are neutral.
    ROWC = [BLU, BLU, ORA, RED, SETC["L4"], NEU, NEU, NEU]
    LAW_NUMBERS = (1, 1, 2, 3, 4, 5, 7, 8)
    LAW_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
    yy = np.arange(len(KEY))[::-1]
    # 参考线只画在数据行的高度上:满高的 axvline 会从上方的表头一路穿到
    # 下方的图例文字里去。上下界与左脊线的 set_bounds 保持一致。
    axe.plot([1.0, 1.0], [-0.30, len(KEY) - 0.70], color="#B9BBBE", lw=0.7,
             zorder=1, solid_capstyle="butt")
    for k, y, row_colour, law_number, law_marker in zip(
        KEY, yy, ROWC, LAW_NUMBERS, LAW_MARKERS, strict=True
    ):
        d = tr[k]
        r = d["calib_value"] / d["threshold"]
        impact_colour = RED if abs(d["verdict_flip_real_pct"]) >= 1.0 else row_colour
        if abs(r - 1.0) > 1e-9:
            axe.plot([1.0, r], [y, y], color=to_rgba(row_colour, 0.58),
                     lw=1.0, zorder=2, solid_capstyle="butt")
            axe.scatter(
                [r], [y], s=26, marker=law_marker,
                facecolors=to_rgba(row_colour, 0.12),
                edgecolors=row_colour, lw=1.0, zorder=4, clip_on=False,
            )
            # Full linear scaling makes the small D1--D3 shifts nearly overlap the
            # fixed point.  Put the number on the side away from unity so that the
            # label remains readable without magnifying the displacement itself.
            sgn = 1.0 if r >= 1.0 else -1.0
            axe.text(r + sgn * 0.022, y, f"{d['calib_value']:.3g}",
                     fontsize=7.0, color=row_colour, va="center",
                     ha="left" if sgn > 0 else "right")
        else:
            axe.scatter(
                [1.0], [y], s=34, marker=law_marker, facecolors="none",
                edgecolors=row_colour, lw=0.9, zorder=4,
            )
            axe.text(1.022, y, "identical", fontsize=7.0, color=row_colour, ha="left",
                     va="center")
        # 冻结常数的实心点画在空心圈的白底之上,并加白描边;随后把空心圈的
        # 轮廓重描一遍压在最上层。两个标记最近处只差 0.6%(约 0.1 cm),
        # 这样圆环始终是完整的一圈,实心点像是压在它上面,而不是把它啃掉一块。
        fixed_marker = axe.scatter(
            [1.0], [y], s=13, marker=law_marker,
            c=row_colour, ec="w", lw=0.5, zorder=5,
        )
        fixed_marker.set_gid(f"fig2d-fixed-law-{law_number}")
        if abs(r - 1.0) > 1e-9:
            axe.scatter(
                [r], [y], s=26, marker=law_marker, facecolors="none",
                edgecolors=row_colour, lw=1.0, zorder=6, clip_on=False,
            )
        axe.text(XTXT, y, f"{d['verdict_flip_real_pct']:.2f}", fontsize=7.6,
                 color=impact_colour,
                 ha="right", va="center")

    # The main row plot shows how far each frozen constant moves.  This inset
    # uses the otherwise empty upper-left region to answer a different question:
    # whether re-derivation changes experimental and chemically damaged verdicts
    # in the same way.  A common 0--20% scale resolves every measured change
    # while retaining ample headroom above the observed sub-8% range.
    impact = axe.inset_axes([0.170, 0.615, 0.340, 0.295])
    impact.set_label("Fig2d-transfer-impact")
    real_flip = np.array([tr[key]["verdict_flip_real_pct"] for key in KEY])
    damaged_flip = np.array(
        [tr[key]["verdict_flip_corrupted_pct"] for key in KEY]
    )
    impact.plot(
        [0.0, 20.0], [0.0, 20.0], color="#B9BBBE", lw=0.6,
        ls=(0, (2.5, 1.8)), zorder=1,
    )
    for x, y, colour, marker, law_number in zip(
        real_flip, damaged_flip, ROWC, LAW_MARKERS, LAW_NUMBERS, strict=True
    ):
        impact_marker = impact.scatter(
            [x], [y], s=18, marker=marker,
            facecolors=to_rgba(colour, 0.14),
            edgecolors=to_rgba(colour, 0.96), linewidths=0.60,
            zorder=4, clip_on=False,
        )
        impact_marker.set_gid(f"fig2d-impact-law-{law_number}")

    # Marker shape already identifies the law in both the main panel and this
    # inset, so the points carry no per-point label: five of them sit within
    # 1.7 percentage points of the origin, where the numbers and their leader
    # lines cost more legibility than they returned.
    impact.set_xlim(-0.4, 20.4)
    impact.set_ylim(-0.4, 20.4)
    impact.set_xticks([0, 10, 20])
    impact.set_yticks([0, 10, 20])
    impact.set_xlabel("Experimental flips (%)", fontsize=7.2, labelpad=1.2)
    impact.set_ylabel("Damaged flips (%)", fontsize=7.0, labelpad=1.0)
    impact.tick_params(labelsize=6.8, width=0.45, length=2.0, pad=1.2)
    for spine in impact.spines.values():
        spine.set_linewidth(0.45)
        spine.set_zorder(0.5)
    impact.spines["top"].set_visible(False)
    impact.spines["right"].set_visible(False)
    axe.set_yticks(yy); axe.set_yticklabels(RLABE, fontsize=7.6, ha="right")
    axe.tick_params(axis="y", pad=2.5)
    axe.set_ylim(-0.95, len(KEY) + 0.20)
    axe.set_xlim(0.14, 1.34)
    axe.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    axe.set_xlabel("Re-derived / fixed constant", labelpad=1.5)
    axe.spines["bottom"].set_bounds(0.20, 1.20)
    axe.spines["left"].set_bounds(-0.30, len(KEY) - 0.70)
    axe.text(XTXT, len(KEY) - 0.35, "experimental-structure\nverdicts flipped (%)", fontsize=7.4,
             color=NEU, ha="right", va="bottom", linespacing=1.30)
    axe.text(0.205, -0.60, "filled = fixed before testing      open = re-derived", fontsize=7.4,
             color="#535557", ha="left", va="center")

    stamp(fig, [(axb, "a"), (axc, "b"),
                (axd, "c"), (axe, "d")])
    save(fig, "fig2_rules")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 1 — Agentic law learning: the loop, its scale, and its refutation ledger
# ─────────────────────────────────────────────────────────────────────────────
def _rule_table(ax, cal, W, H):
    """D1--D8 与五个嵌套规则集的完整目录(原图 2a,现为图 1d)。

    坐标系直接用厘米:x 向右 0..W,y 向下 0..H。行高因此按字号算得出,
    面板加宽加高时字不会跟着被拉伸,多出来的高度全部变成行间留白。
    """
    ax.set_xticks([]); ax.set_yticks([])     # 让 stamp 取到面板框而非隐藏刻度
    ax.axis("off")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    xc = [W * f for f in (0.628, 0.708, 0.788, 0.868, 0.948)]
    x0, x1 = 0.0, W          # 表格横线与斑马纹与面板 a 的流程框左右缘严格对齐
    # 编号列改用 Law 1--Law 8 后比旧的 D1--D8 宽约 0.35 cm，表达式列右移到
    # 0.078W，否则"Law 1"的末字与紧随其后的公式首字连成一体。
    XR, XG = 0.014 * W, 0.078 * W            # 规则编号列;表达式与释义列
    # 表头高度必须容得下"L1 / contact / floor"三行:0.95 时第三行的基线正好
    # 压在表格顶线上,印刷时会连成一体。1.12 给出约 0.15 cm 的净间距。
    HHEAD, HSUM, HGAP = 1.12, 0.38, 0.20     # 表头 / 每行汇总 / 分隔线上方留白
    DY = 0.135                               # 表达式行与释义行相对行中线的偏移
    HRULE = (H - HHEAD - HGAP - 2 * HSUM) / len(RULES)

    # 表头标签下移到 0.32:名字与表格顶线之间留 0.11 cm,与上方 c 的轴题
    # 之间则拉开到 0.22 cm —— 标签因此明确属于下面这张表,而不是浮在两者之间。
    yh = 0.32
    for x, s in zip(xc, SETS):
        c = SETC[s]
        ax.text(x, yh, setlab(s, "$'$"), fontsize=9.2,
                color=c, ha="center", va="center")
        ax.text(x, yh + 0.53, SETNAME[s], fontsize=7.2, color=c, ha="center",
                va="center", linespacing=1.30)
    ax.plot([x0, x1], [HHEAD] * 2, color="#323335", lw=0.8)

    y = HHEAD
    for i, (rid, expr, gloss, member) in enumerate(RULES):
        if i % 2 == 0:
            ax.add_patch(mp.Rectangle((x0, y), x1 - x0, HRULE, fc="#EEF2F5",
                                      lw=0, zorder=0, clip_on=False))
        yc = y + HRULE / 2                  # 行中线:成员标记落在这里
        ye = yc - DY                        # 表达式行,与释义行对称居中
        ax.text(XR, ye, rid, fontsize=8.6,
                color="#212224", va="center")
        ax.text(XG, ye, expr, fontsize=8.8, color="#101113", va="center")
        ax.text(XG, yc + DY, gloss, fontsize=7.1, color="#535557",
                va="center", style="italic")
        for x, s in zip(xc, SETS):
            mk = member.get(s)
            if mk is None:
                ax.text(x, yc, "–", fontsize=7.4, color="#CACCCF", ha="center",
                        va="center")
            elif mk == "•":
                ax.scatter([x], [yc], s=30, c=SETC[s], lw=0, zorder=3,
                           transform=ax.transData)
            else:
                ax.text(x, yc, mk, fontsize=7.4, color=SETC[s], ha="center",
                        va="center")
        y += HRULE

    ax.plot([x0, x1], [y + HGAP] * 2, color="#323335", lw=0.8)
    y += HGAP
    for lab, key, fmt in [("satisfaction (experimental structures)", "sat", "{:.4f}"),
                          ("damage detection", "excl", "{:.4f}")]:
        y += HSUM
        ax.text(XG, y - 0.16, lab, fontsize=8.3, color="#323335", va="center")
        for x, s in zip(xc, SETS):
            ax.text(x, y - 0.16, fmt.format(float(cal.loc[s, key])), fontsize=8.3,
                    color=SETC[s], ha="center", va="center")



def _fig1_rulespace(ax):
    """面板 b:2,047,123 条候选律的 t-SNE 投影,只有散点与八条存活法则的标记。

    数据是 experiments/rule_space_tsne/aggregate_for_fig1.py 聚合出的 220x220
    计数网格(原始 2M 行投影不入库)。不按课题着色,不加任何图内说明文字:
    读法交给图注,规模交给面板 c。
    """
    from matplotlib.colors import LogNorm
    g = pd.read_csv(DATA / "fig1_rulespace_grid.csv")
    st = pd.read_csv(DATA / "fig1_rulespace_stars.csv")

    # 密度用散点画,不用 imshow:面板是横向长方形,imshow 的方格会被拉成短横线,
    # 而散点标记在显示空间里绘制,坐标系怎么拉都保持圆形。半透明叠加自然显出密度。
    order = np.argsort(g.n.values)                 # 稀疏的先画,密的压在上层
    norm = LogNorm(vmin=1, vmax=g.n.max())
    point_colours = plt.get_cmap("palseq")(norm(g.n.values[order]))
    point_faces = point_colours.copy()
    point_edges = point_colours.copy()
    point_faces[:, 3] = 0.24
    point_edges[:, 3] = 0.64
    ax.scatter(
        g.x.values[order], g.y.values[order], s=1.8,
        facecolors=point_faces, edgecolors=point_edges,
        linewidths=0.18, zorder=1,
    )

    # 八条存活法则:填充单独变淡，白色边框保持完全不透明；点云和标记都
    # 保留为矢量路径，避免放大 PDF 时出现 400-dpi 栅格层的模糊。
    ax.scatter(
        st.x,
        st.y,
        s=30,
        facecolors=[to_rgba(RED, 0.34)],
        edgecolors="white",
        linewidths=1.30,
        zorder=4,
    )

    # 八条法则在 t-SNE 上落成四个可分辨的斑点,每个斑点各得一条引线和一个标签。
    # 名字一律写在点云之外:x 轴右端留出一段空白列,标签排在那里,只有引线进入点云。
    # 面板 c 的类别标签紧贴在这条列的右侧,所以标签不能越出面板自己的右边界。
    P = {r.rule: (r.x, r.y) for r in st.itertuples()}
    # runs of consecutive laws are written as ranges: eight names crowded the embedding they
    # point into, and the leader lines are what identify the clusters
    # D6/D7 sit about one marker width to the left of D1/D3/D8, so their leader has to
    # end on the marker itself: a 3-pt shrink would retract its tip onto the neighbouring
    # cluster and label the wrong blob.
    groups = [(["D2"], "Law 2", (138, 104)),
              (["D1", "D3", "D8"], "Law 1, 3, 8", (138, 36)),
              (["D6", "D7"], "Law 6–7", (138, -32)),
              (["D4", "D5"], "Law 4–5", (138, -100))]
    for names, lab, (tx, ty) in groups:
        xs = float(np.mean([P[n][0] for n in names]))
        ys = float(np.mean([P[n][1] for n in names]))
        ax.annotate(lab, xy=(xs, ys), xytext=(tx, ty), textcoords="data",
                    fontsize=7.2, color=RED,
                    ha="left", va="center", zorder=6, annotation_clip=False,
                    linespacing=1.25,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=RED,
                                    shrinkA=1.0, shrinkB=1.0))

    # x 轴只延长到刚好放下那条标签列为止:再宽点云就白白缩小,再窄标签就要撞上
    # 面板 c 的类别标签。最长的一条 "Law 1, 3, 8" 约 1.3 cm,列宽按它定。
    ax.set_xlim(-136.0, 178.0)
    ax.set_ylim(-142.0, 142.0)
    ax.set_xticks([]); ax.set_yticks([])
    # 轴题带上量级:这张图里的点云正是聚合前的 2,047,123 条候选律
    ax.set_xlabel(
        r"Explored $2\times10^{6}$ law space", labelpad=2.0,
        fontsize=plt.rcParams["axes.labelsize"],
    )
    # 轴题跟着点云走,不跟着现在偏右的面板中线走
    ax.xaxis.set_label_coords((0.0 + 136.0) / 314.0, -0.052)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _fig1_trajectory(ax, fig, inset_rect):
    at = pd.read_csv(DATA / "fig1_attempts.csv")

    def steps(ms, x_end):
        xs, ys = [ms[0][0]], [ms[0][1]]
        for r, v in ms[1:]:
            xs += [r, r]; ys += [ys[-1], v]
        xs.append(x_end); ys.append(ys[-1])
        return xs, ys

    MS = [(1, 0.0), (4, 0.2890), (13, 0.6121), (14, 0.7004)]
    xs, ys = steps(MS, 594)
    ax.plot(xs, ys, color=BLU, lw=1.6, zorder=4,
            label="best damage detection at satisfaction $\\geq$ 0.90")
    # 虚线周期按这条折线的总长挑:(4,2) 时最后一段只剩 1 pt 的残墨,
    # 悬在 L4 标记与 L4 标签之间像一粒脏点。(5,2.2) 让末端收在一整段上。
    ax.plot([560, 585, 585, 594], [0.7004, 0.7004, 0.8180, 0.8180],
            color="#0A5A3C", lw=1.6, ls=(0, (5, 2.2)), zorder=4,
            label="best lower of satisfaction and damage detection")
    ax.scatter([585], [0.8180], s=22, c="#0A5A3C", zorder=6)
    # 两条曲线只差 0.118,标签各自贴着自己那条线的外侧,免得两个字叠在一起
    ax.text(598, 0.688, "Set 3", fontsize=9, color=BLU, va="top")
    ax.text(598, 0.830, "Set 4", fontsize=9, color="#0A5A3C", va="bottom")

    # ---- attempt strip: each round's own gate verdict
    # 通过的轮次单独占一行:此前它与 "failed its gate" 共用同一条 y,
    # 绿色三角就贴在红色标签左侧,读起来像那条标签的图例键。
    # 泳道间距按标签字号定:0.15 个数据单位 = 0.31 cm > 7.4 pt 的行高,
    # 四条右侧标签因此不再叠在一起(此前 0.08 个单位只有 0.18 cm)。
    # 只画真正有记号落进去的泳道:带标签的空泳道等于指着一片空白。
    # 泳道映射:fig1_attempts.csv 的 outcome 写法与泳道名对齐。
    # 明确不打记号的三类,合计 175 行,理由写在 SI Note S1 的作图约定里:
    #   not-run (blocked upstream) 131  —— 从未执行,不存在自己的判决
    #   unmarked                    37  —— 归档里没有可核对的判决
    #   measurement                  7  —— 是测量而非过闸尝试
    YPASS = -0.085
    LANES = {"fail": (-0.250, "#D6564C", "missed its criterion"),
             "nogain": (-0.400, "#35A7D8", "no gain"),
             "aband": (-0.550, "#8A8C8E", "abandoned")}
    GROUP = {"fail-gate": "fail", "no-eligible": "fail",
             "probe-pass": "nogain", "diagnostic-only": "nogain",
             "no-gain": "nogain",
             "mixed: D2 accepted, certified tree refuted out of distribution":
                 "nogain",
             "abandoned-family": "aband",
             "family-abandoned (redundancy stop)": "aband",
             "not-run (blocked upstream)": None,
             "measurement": None, "unmarked": None, "pass": "pass"}
    unmapped = sorted(set(at.outcome.astype(str).str.strip())
                      - set(GROUP)) if "outcome" in at else []
    if unmapped:                                   # 新增写法不得静默消失
        raise ValueError(f"fig1 strip: unmapped outcomes {unmapped}")
    for _, row in at.iterrows():
        g = GROUP.get(str(row.outcome).strip(), None)
        x = row.unified_round
        if g in LANES:
            y, c, _ = LANES[g]
            ax.plot([x, x], [y - 0.046, y + 0.046], color=c, lw=0.6,
                    alpha=0.85, zorder=3)
        elif g == "pass":
            ax.scatter([x], [YPASS], s=13, marker="^", c="#0A5A3C", zorder=5)
    ax.text(598, YPASS, "met its criterion", fontsize=7.4, color="#0A5A3C",
            va="center")
    for y, c, lab in LANES.values():
        ax.text(598, y, lab, fontsize=7.4, color=c, va="center")

    ax.set_xlim(-4, 596); ax.set_ylim(-0.64, 1.04)
    ax.set_xticks([1, 100, 200, 300, 400, 500, 594])
    ax.set_xticklabels(["1", "100", "200", "300", "400", "500", "594"],
                       fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "0.5", "1"],
                                                     fontsize=8)
    ax.set_xlabel("Investigation, unified programme index", labelpad=1.5)
    ax.set_ylabel("Best performance\n(held-out data)")
    ax.axhline(0, color="#CACCCF", lw=0.5, zorder=1)
    # 图例下移到 0.93:曲线的平台在 0.80 处,图例落在它与面板上沿之间的空白里,
    # 与上方表格 d 的末行因此拉开约 0.2 cm,不再像是浮在两个面板中间。
    ax.legend(frameon=False, fontsize=8, loc="lower left", handlelength=1.8,
              ncol=2, columnspacing=1.6, borderpad=0.1,
              bbox_to_anchor=(0.0, 0.93))
    ax.spines["left"].set_bounds(0, 1)

    # ---- inset: the first 24 investigations
    # y 轴留到 1.45:三个标记里 L3 坐在 0.70,它的标签只能放在标记正上方,
    # 而标签本身要占 0.43 个单位。抬高上界换来的顶部空白由轴题填掉。
    axi = fig.add_axes(inset_rect)
    xs, ys = steps(MS, 24)
    axi.plot(xs, ys, color=BLU, lw=1.3, zorder=4)
    # 标签一律离开自己的标记与自己的阶梯线:L1 落在 0.289 平台之上,
    # L2 贴在自己标记的左侧,L3 抬到自己标记的正上方。
    for r, v, lab, x, y, ha, va in [
            (4,  0.2890, "Set 1",  4.75, 0.400, "left",   "bottom"),
            (13, 0.6121, "Set 2", 12.20, 0.6121, "right", "center"),
            (14, 0.7004, "Set 3", 13.85, 0.870, "center", "bottom")]:
        axi.scatter([r], [v], s=14, c=BLU, ec="w", lw=0.5, zorder=6)
        axi.text(x, y, lab, fontsize=8.0, ha=ha, va=va,
                 color=BLU)
    for r in (15, 21):
        axi.scatter([r], [0.7004], s=13, facecolors="none", lw=0.9,
                    edgecolors="#35A7D8", zorder=6)
    axi.set_xlim(0, 26); axi.set_ylim(0, 1.45)
    axi.set_xticks([1, 8, 16, 24])
    axi.set_yticks([0, 0.5, 1.0]); axi.set_yticklabels(["0", "0.5", "1"])
    # 轴题收进内插图顶部的空白里:放在图外时它只能挤在刻度标签与主图的
    # 灰色 y = 0 参考线之间的那 2 mm 上。
    axi.text(0.025, 0.975, "first 24 investigations", transform=axi.transAxes,
             fontsize=6.8, color="#646668", ha="left", va="top")
    # 内插图在视觉上从属于主图:更细的框线、更浅的刻度
    axi.tick_params(labelsize=6.5, colors="#757779", width=0.4, length=2.0, pad=2.0)
    for lb in axi.get_xticklabels() + axi.get_yticklabels():
        lb.set_color("#646668")
    for sp in axi.spines.values():
        sp.set_linewidth(0.4); sp.set_color("#A8AAAD")
    ax.add_patch(mp.Rectangle((1, 0), 25, 0.74, fill=False, ec="#A8AAAD",
                              lw=0.6, ls=(0, (2, 2)), zorder=2))


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 1a — 发现回路,以及回路产出的法则拿来做什么
# 这一面板用厘米直接作图、y 向下,与 _rule_table 同一套坐标约定:面板加宽加高时
# 字号不会跟着被拉伸,多出来的尺寸全变成留白。上半是五段回路,下半是宏观的
# input/output:候选结构 -> PRIS(八条一行法则) -> 合理 / 不合理并指出违反了哪条界。
# 中间的漏斗把两半接起来 —— 下半那张卡片就是上半这条回路的产物。
# 反驳回路走在方框上方:下方留给漏斗,两条线于是不会交叉。
# 旧版顶部那条灰色横条(协议 + 数据划分)去掉了,四个划分数字在 Methods 里。
# ─────────────────────────────────────────────────────────────────────────────
A_ANION, A_CATION = "#B4C4CE", "#005B93"     # 结构示意图的阴离子 / 阳离子
A_INK, A_MUTE = "#212224", "#535557"
A_EL, A_ER = 0.32, 18.22                     # 与全幅面板共用的左右边界
H_FIG1A = 4.00                               # 面板 a 的总高(cm)
H_STAGE = 1.26                               # 五个流程框的高度


def _rbox(ax, x, y, w, h, r=0.09, **kw):
    """圆角矩形,外缘正好是 (x, y, w, h):pad=0 时 FancyBboxPatch 不向外扩。"""
    return ax.add_patch(mp.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw))


def _header_box(ax, x, y, w, h, hh, colour, r=0.09):
    """带实色标题条的卡片。标题条要上圆下方,所以先画圆角再用矩形压平下沿。"""
    _rbox(ax, x, y, w, h, r, lw=0.9, ec=colour, fc=colour + "12", zorder=2)
    _rbox(ax, x, y, w, hh, r, lw=0, fc=colour, zorder=3)
    ax.add_patch(mp.Rectangle((x, y + hh - r), w, r, lw=0, fc=colour, zorder=3))


def _arrow(ax, p0, p1, colour=A_MUTE, lw=1.0, ls="-", ms=8, z=4):
    ax.add_patch(mp.FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, ls=ls,
        color=colour, zorder=z, shrinkA=0, shrinkB=0))


def _caps(ax, x, y, s, size=6.0):
    """INPUT / OUTPUT 这一类小号疏排灰字:疏排靠插空格实现,不依赖字体特性。"""
    ax.text(x, y, " ".join(s.upper()), ha="center", va="center", fontsize=size,
            color=GRY, zorder=5)


# ---- 五段回路的图标。一律画成路径,不用符号字体:PDF/SVG 里不会缺字 ---------
def _ic_bulb(ax, x, y, s, c):
    lw = 0.8 * s / 0.24
    ax.add_patch(mp.Circle((x, y - 0.18 * s), 0.30 * s, fill=False, lw=lw, ec=c,
                           zorder=5))
    ax.plot([x - 0.16 * s, x + 0.16 * s], [y + 0.22 * s] * 2, lw=lw, color=c,
            solid_capstyle="round", zorder=5)
    ax.plot([x - 0.11 * s, x + 0.11 * s], [y + 0.40 * s] * 2, lw=lw, color=c,
            solid_capstyle="round", zorder=5)


def _ic_table(ax, x, y, s, c):
    w, h = 0.86 * s, 0.74 * s
    lw = 0.7 * s / 0.24
    ax.add_patch(mp.Rectangle((x - w / 2, y - h / 2), w, h, fill=False, lw=lw,
                              ec=c, zorder=5))
    ax.add_patch(mp.Rectangle((x - w / 2, y - h / 2), w, h / 3, lw=0, fc=c,
                              zorder=5))
    for f in (1 / 3, 2 / 3):
        ax.plot([x - w / 2, x + w / 2], [y - h / 2 + f * h] * 2, lw=lw * 0.8,
                color=c, zorder=5)
        ax.plot([x - w / 2 + f * w] * 2, [y - h / 2, y + h / 2], lw=lw * 0.8,
                color=c, zorder=5)


def _ic_lens(ax, x, y, s, c):
    lw = 0.85 * s / 0.24
    ax.add_patch(mp.Circle((x - 0.10 * s, y - 0.10 * s), 0.32 * s, fill=False,
                           lw=lw, ec=c, zorder=5))
    ax.plot([x + 0.13 * s, x + 0.40 * s], [y + 0.13 * s, y + 0.40 * s],
            lw=lw * 1.15, color=c, solid_capstyle="round", zorder=5)


def _ic_target(ax, x, y, s, c):
    lw = 0.8 * s / 0.24
    ax.add_patch(mp.Circle((x, y), 0.38 * s, fill=False, lw=lw, ec=c, zorder=5))
    ax.add_patch(mp.Circle((x, y), 0.14 * s, lw=0, fc=c, zorder=5))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([x + dx * 0.38 * s, x + dx * 0.56 * s],
                [y + dy * 0.38 * s, y + dy * 0.56 * s], lw=lw, color=c,
                solid_capstyle="round", zorder=5)


def _ic_refute(ax, x, y, s, c):
    lw = 0.85 * s / 0.24
    ax.add_patch(mp.Circle((x, y), 0.40 * s, fill=False, lw=lw, ec=c, zorder=5))
    d = 0.40 * s / np.sqrt(2)
    ax.plot([x - d, x + d], [y + d, y - d], lw=lw, color=c,
            solid_capstyle="round", zorder=5)


def _struct_glyph(ax, cx, cy, s=1.0, swap=False, n=3):
    """离子晶体的小示意:大阴离子、小阳离子、细键线,带一点斜投影。

    swap=True 时交换其中一对的身份并套紫环 —— 与 Fig. 3a 的"被换位原子"同一约定。
    """
    g = 0.20 * s
    xs = np.arange(n) - (n - 1) / 2
    pts = [(cx + i * g + 0.30 * g * j, cy + j * g) for j in xs for i in xs]
    for k, (px, py) in enumerate(pts):
        for m, (qx, qy) in enumerate(pts):
            if m <= k:
                continue
            if abs(np.hypot(qx - px, qy - py) - g) < 0.35 * g:
                ax.plot([px, qx], [py, qy], lw=0.45, color="#9AA5AC", zorder=4)
    for k, (px, py) in enumerate(pts):
        cation = (k % 2 == 1)
        if swap and k in (4, 5):
            cation = not cation
        ax.add_patch(mp.Circle((px, py), (0.062 if cation else 0.085) * s,
                               fc=A_CATION if cation else A_ANION,
                               ec="w", lw=0.4, zorder=5 + cation))
    if swap:
        for k in (4, 5):
            px, py = pts[k]
            ax.add_patch(mp.Circle((px, py), 0.135 * s, fill=False, lw=0.7,
                                   ec=ORA, zorder=7))


def _tick(ax, x, y, s, colour):
    ax.add_patch(mp.Circle((x, y), 0.115 * s, lw=0, fc=colour, zorder=5))
    ax.plot([x - 0.055 * s, x - 0.016 * s, x + 0.062 * s],
            [y + 0.004 * s, y + 0.046 * s, y - 0.048 * s],
            lw=0.95, color="w", solid_capstyle="round",
            solid_joinstyle="round", zorder=6)


def _cross(ax, x, y, s, colour):
    ax.add_patch(mp.Circle((x, y), 0.115 * s, lw=0, fc=colour, zorder=5))
    d = 0.048 * s
    ax.plot([x - d, x + d], [y - d, y + d], lw=0.95, color="w",
            solid_capstyle="round", zorder=6)
    ax.plot([x - d, x + d], [y + d, y - d], lw=0.95, color="w",
            solid_capstyle="round", zorder=6)


def _contact_glyph(ax, x, y, s=1.0):
    """把 rho 画出来:两个相切的离子,加一条 d 的尺寸线(尺寸线在下方)。"""
    ra, rc = 0.175 * s, 0.105 * s
    ax.add_patch(mp.Circle((x - rc, y), ra, fc=A_ANION, ec="w", lw=0.5, zorder=4))
    ax.add_patch(mp.Circle((x + ra, y), rc, fc=A_CATION, ec="w", lw=0.5, zorder=5))
    yb = y + ra + 0.11 * s
    ax.plot([x - rc] * 2, [y, yb], lw=0.45, color=A_INK, zorder=5)
    ax.plot([x + ra] * 2, [y, yb], lw=0.45, color=A_INK, zorder=5)
    ax.add_patch(mp.FancyArrowPatch((x - rc, yb), (x + ra, yb),
                                    arrowstyle="<|-|>", mutation_scale=4.4,
                                    lw=0.65, color=A_INK, shrinkA=0, shrinkB=0,
                                    zorder=5))
    ax.text(x + (ra - rc) / 2, yb + 0.06 * s, "$d$", ha="center", va="top",
            fontsize=6.8, color=A_INK, zorder=5)


# 五段回路。每格两行说明:三行时框宽只够 6.2 pt,两行才排得下 6.9 pt。
STAGES = [
    ("PROPOSE",  "41 literature queries", "14 agent workflows",     BLU, _ic_bulb),
    ("TABULATE", "84 descriptor tables",  "99,162 structures",      BLU, _ic_table),
    ("SEARCH",   "one-line laws and",     "fitted thresholds",      ORA, _ic_lens),
    ("TEST",     "held-out data and",     "negative controls",      GRN, _ic_target),
    ("REFUTE",   "11 claims refuted",     "and returned to search", RED, _ic_refute),
]


def _fig1_engine(ax, ytop, h=H_STAGE, hh=0.40, arc=0.30):
    """五段回路,从左到右;反驳的回路走在方框上方,给下方的漏斗让路。"""
    gap = 0.55                       # 框间距:再窄箭头就短到读不出方向
    bw = (A_ER - A_EL - 4 * gap) / 5
    cx = []
    for i, (name, l1, l2, c, icon) in enumerate(STAGES):
        x = A_EL + i * (bw + gap)
        cx.append(x + bw / 2)
        _header_box(ax, x, ytop, bw, h, hh, c)
        icon(ax, x + 0.42, ytop + hh / 2, 0.26, "w")
        # 标题右移 0.16:图标占了标题条左端,不移的话"PROPOSE"看着偏左
        ax.text(x + bw / 2 + 0.16, ytop + hh / 2 + 0.012, name, ha="center",
                va="center", color="w", fontsize=8.2, zorder=5)
        ax.text(x + bw / 2, ytop + hh + 0.30, l1, ha="center", va="center",
                fontsize=6.9, color="#323335", zorder=4)
        ax.text(x + bw / 2, ytop + hh + 0.61, l2, ha="center", va="center",
                fontsize=6.9, color="#323335", zorder=4)
        if i:
            _arrow(ax, (x - gap + 0.10, ytop + h / 2), (x - 0.10, ytop + h / 2),
                   colour="#6B6D6F", lw=1.0, ms=7.5)
    ya = ytop - arc
    ax.plot([cx[4], cx[4]], [ytop - 0.06, ya], color=RED, lw=1.0,
            ls=(0, (3.6, 2.0)), zorder=3)
    ax.plot([cx[0], cx[4]], [ya, ya], color=RED, lw=1.0, ls=(0, (3.6, 2.0)),
            zorder=3)
    _arrow(ax, (cx[0], ya), (cx[0], ytop - 0.06), colour=RED, lw=1.0,
           ls=(0, (3.6, 2.0)), ms=8, z=3)
    ax.text((cx[0] + cx[4]) / 2, ya - 0.08, "refuted claims re-enter the search",
            ha="center", va="bottom", fontsize=6.8, color=RED, zorder=4)


def _fig1_engine_io(ax):
    """面板 a 的全部内容。ax 的坐标已按 (0..18.3, H_FIG1A..0) 设好。

    方框行的上方必须留出 0.67 cm:反驳回路那条虚线加它的一行说明要占 0.62 cm,
    留不够时这行字会越过面板上沿,bbox_inches="tight" 于是把整张图裁高 —— 图在
    版面上就跟着变高,把图注末行推向页码。
    下沿则要给 b/c 的面板字母留出净空:卡片底边与字母顶边之间要有约 0.3 cm,
    0.02 cm 时"candidate structures"与字母 b 会读成同一行。
    """
    y_eng = 0.67
    _fig1_engine(ax, y_eng)
    y_eng_b = y_eng + H_STAGE

    XC, CW = 7.60, 5.90                      # PRIS 卡片
    # 漏斗带从 0.38 加宽到 0.56:"what survives refutation" 原来上下各只剩
    # 0.07 cm,读起来像贴在两排方框之间;0.56 让这行字两侧各有约 0.16 cm。
    y0 = 2.55                                # input/output 带的顶边
    # 漏斗:上沿贴着整行方框,下沿收到卡片宽度,读作"这一整条回路的产物"
    ax.add_patch(mp.Polygon([[A_EL + 0.40, y_eng_b + 0.02],
                             [A_ER - 0.40, y_eng_b + 0.02],
                             [XC + (CW - 1.60) / 2, y0 - 0.04],
                             [XC - (CW - 1.60) / 2, y0 - 0.04]],
                            closed=True, fc="#EDF1F4", ec="#DDE2E7", lw=0.5,
                            zorder=1))
    ax.text(XC, (y_eng_b + y0) / 2 + 0.01, "what survives refutation",
            ha="center", va="center", fontsize=7.0, color="#45474A", zorder=3)

    # ---- 输入:任意候选结构(其一带紫环,提示受控损伤的变体)
    _caps(ax, 1.90, y0 - 0.16, "input")
    _struct_glyph(ax, 1.20, y0 + 0.62, 1.25)
    _struct_glyph(ax, 2.61, y0 + 0.62, 1.25, swap=True)
    ax.text(1.90, y0 + 1.20, "candidate structures", ha="center", va="center",
            fontsize=7.4, color=A_INK, zorder=5)
    _arrow(ax, (3.62, y0 + 0.68), (XC - CW / 2 - 0.16, y0 + 0.68),
           colour="#6B6D6F", lw=1.1, ms=8.5)

    # ---- 回路产出的法则本身;举 Law 1 作例,画出 rho 的几何
    _rbox(ax, XC - CW / 2, y0, CW, 1.40, 0.12, lw=1.1, ec=SLATE, fc="#F5F8FA",
          zorder=2)
    ax.text(XC - CW / 2 + 0.32, y0 + 0.35, "PRIS", ha="left", va="center",
            fontsize=10.5, color=SLATE, fontweight="bold", zorder=5)
    ax.text(XC - CW / 2 + 1.44, y0 + 0.37, "eight one-line laws, five mechanisms",
            ha="left", va="center", fontsize=7.4, color=A_MUTE, zorder=5)
    ax.plot([XC - CW / 2 + 0.28, XC + CW / 2 - 0.28], [y0 + 0.60] * 2,
            color="#DDE2E7", lw=0.6, zorder=3)
    ax.text(XC - CW / 2 + 0.34, y0 + 0.86, "e.g.", ha="left", va="center",
            fontsize=6.8, color=GRY, style="italic", zorder=5)
    _contact_glyph(ax, XC - CW / 2 + 1.22, y0 + 0.86, 1.05)
    ax.text(XC - CW / 2 + 1.86, y0 + 0.98,
            r"$\rho=\dfrac{d}{r_{\mathrm{cat}}+r_{\mathrm{an}}}\ \geq\ \tau$",
            ha="left", va="center", fontsize=8.6, color=A_INK, zorder=5)

    # ---- 输出:两种判决。不合理那一格要两行,所以比合理那一格高
    XV = 12.05
    _caps(ax, (XV + A_ER) / 2, y0 - 0.16, "output")
    _arrow(ax, (XC + CW / 2 + 0.16, y0 + 0.68), (XV - 0.20, y0 + 0.68),
           colour="#6B6D6F", lw=1.1, ms=8.5)
    _rbox(ax, XV, y0, A_ER - XV, 0.56, 0.10, lw=0.85, ec=GRN, fc=GRN + "12",
          zorder=2)
    _tick(ax, XV + 0.32, y0 + 0.28, 1.0, GRN)
    ax.text(XV + 0.62, y0 + 0.28, "plausible", ha="left", va="center",
            fontsize=8.2, color=GRN, zorder=5)
    ax.text(XV + 1.86, y0 + 0.29, "— every law satisfied", ha="left",
            va="center", fontsize=7.2, color=A_MUTE, zorder=5)
    _rbox(ax, XV, y0 + 0.68, A_ER - XV, 0.72, 0.10, lw=0.85, ec=RED,
          fc=RED + "12", zorder=2)
    _cross(ax, XV + 0.32, y0 + 1.04, 1.0, RED)
    ax.text(XV + 0.62, y0 + 1.04, "implausible", ha="left", va="center",
            fontsize=8.2, color=RED, zorder=5)
    ax.text(XV + 2.24, y0 + 0.89, "— names the violated law", ha="left",
            va="center", fontsize=7.2, color=A_MUTE, zorder=5)
    ax.text(XV + 2.24, y0 + 1.19, "and the mechanism to review", ha="left",
            va="center", fontsize=7.2, color=A_MUTE, zorder=5)


def fig1():
    cal = _split("calibration")
    # 总高硬约束:这一页是 [p] 浮动体,图注紧跟在图下面,超出正文块的部分
    # 会压到页码上。裁剪后的 PDF 高度必须 <= 20.6 cm(实测 20.5 时图注末行
    # 离页码还有 0.2 cm)。a 从 2.45 长到 3.86 之后多出来的 1.41 cm,是从
    # b/c(-0.50)、d(-0.08)、e(-0.24)和三处面板间距(-0.59)里挤出来的,
    # 总高保持 20.35 不变。
    _W, _H = 18.3, 20.35
    fig = plt.figure(figsize=(W2, _H * CM))

    def _rect(x0, w, ytop, h):
        return [x0 / _W, (_H - ytop - h) / _H, w / _W, h / _H]

    # 显式面板矩形(cm)。a/b/d 左缘同为 EL、右缘同为 ER;e 的绘图区右缘留出
    # 泳道标签的位置,左缘在下面按 y 轴装饰的实际宽度回调,使其 y 装饰左端
    # 也落在 EL 上 —— 这样 stamp() 的分列逻辑会把 a/b/d/e 的字母排在同一条线,
    # 而 c(唯一有 y 类别标签的面板)自成一列。
    # 右边界推到 18.22:裁剪后的 PDF 越宽,\textwidth 缩放系数越小,
    # 同样的版面高度落到页面上就越矮 —— 图注末行与页码的间距是这么挣来的。
    EL, ER = 0.32, 18.22            # 全幅面板的左右边界
    YA = 0.02                       # a 的顶边:面板内部自己留好了上边距
    # b/c 这一行压到 3.29:b 是一团椭圆点云,少掉的 0.66 cm 全部来自它上下的
    # 空白;c 的六条横杠在 3.29 下行距仍有 0.53 cm。顶边再下移 0.06,给 a 的
    # 漏斗带让路,同时把 a 的卡片底边与 b/c 字母的净空保持在 0.18 cm 以上。
    HBC, YBC = 3.29, 4.52           # b/c 这一行的高度
    WB = 8.40                       # b 的宽度:横向铺开,散点盘成椭圆
    XC0 = 12.10                     # c 的绘图矩形左缘(类别标签右对齐到此,
                                    # 需让最长标签的左端避开 b 的右缘)
    XRT = 16.14                     # e 的绘图矩形右缘

    # ---- (a) the discovery loop, and what the loop produces
    # 面板矩形横跨整幅图(0..18.3):漏斗要从整行方框收到中间那张卡片,
    # 用 EL..ER 的窄矩形装不下漏斗的两个上角。内部坐标就是厘米,y 向下。
    axa = fig.add_axes(_rect(0.0, _W, YA, H_FIG1A))
    axa.set_xlim(0, _W); axa.set_ylim(H_FIG1A, 0)
    axa.set_xticks([]); axa.set_yticks([])
    axa.axis("off")
    _fig1_engine_io(axa)

    # ---- (b) the searched law space
    axe = fig.add_axes(_rect(EL, WB, YBC, HBC))
    _fig1_rulespace(axe)

    # ---- (c) scale of the autonomous run
    axb = fig.add_axes(_rect(XC0, ER - XC0, YBC, HBC))
    s = pd.read_csv(DATA / "fig1_agent_stats.csv").set_index("metric")["value"]
    keys = [("candidate_evaluations", "Candidate evaluations"),
            ("archived_result_files", "Result files archived"),
            ("scripts_written_total", "Analysis scripts written"),
            ("numbered_investigations", "Numbered investigations"),
            ("hypotheses_refuted", "Claims refuted by controls"),
            ("rules_surviving", "One-line laws surviving")]
    v = [float(s[k]) for k, _ in keys]; lab = [l for _, l in keys]
    col = [GRY, GRY, GRY, BLU, RED, "#0A5A3C"]
    scale_bars = axb.barh(
        np.arange(len(v))[::-1], v, color=col, height=0.62, lw=0,
    )
    style_bar_container(scale_bars, col)
    # 右端留到 1.2e8:最大的数值标签 "2,037,606" 是全图最右的一笔墨,
    # 6e7 时它伸出 a/b/d 共用的右边界 ER,并顶到裁剪框上。
    axb.set_xscale("log"); axb.set_xlim(4, 1.2e8)
    axb.set_xticks([1e1, 1e3, 1e5, 1e7])
    axb.tick_params(axis="x", labelsize=8.8)
    axb.set_yticks(np.arange(len(v))[::-1])
    axb.set_yticklabels(lab, fontsize=7.4)          # 右对齐,紧贴纵轴
    axb.tick_params(axis="y", pad=2.5, length=0)
    axb.set_ylim(-0.62, len(v) - 0.38)
    axb.set_xlabel("Count (log scale)", labelpad=2.0)
    for i, (val, c) in enumerate(zip(v, col)):
        axb.text(val * 1.30, len(v) - 1 - i, f"{int(val):,}", va="center",
                 fontsize=7.4, color=c if c != GRY else "#535557")

    # ---- (d) the complete catalogue: eight predicates, five nested law sets
    # d 让出 0.24:面板 e 的高度有测试守住(内插图刻度必须离主图 y=0 参考线
    # 至少 1 pt),不能从 e 那里挤。行高因此从 0.64 降到 0.60 cm。
    HD = 6.88
    axd = fig.add_axes(_rect(EL, ER - EL, 8.86, HD))
    _rule_table(axd, cal, ER - EL, HD)

    # ---- (e) discovery trajectory over the unified programme index
    # 内插图的矩形受两条硬约束夹住:上沿必须留在主图 0.70 平台线之下,
    # 下沿加一行刻度标签必须停在灰色 y = 0 参考线之上。
    # Keep the bottom edge fixed while making the panel 13% taller.  This
    # moves the main y=0 reference line clear of the inset's 1/8/16/24 labels.
    YE, HE = 16.49, 3.34
    axT = fig.add_axes(_rect(1.80, XRT - 1.80, YE, HE))
    _fig1_trajectory(axT, fig, _rect(6.20, 6.60, 17.62, 0.86))

    # e 的 y 轴装饰(两行轴标题 + 刻度标签)对齐到 EL:先画一次量实际左端,
    # 再把绘图矩形整体右移这个差值,stamp() 于是给 a/b/d/e 同一个字母 x。
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dx = EL / _W - _ydeco_left(axT, fig, rend)
    bb = axT.get_position()
    axT.set_position([bb.x0 + dx, bb.y0, bb.width - dx, bb.height])

    # b/c 的字母下移:默认位置正好落在 a 的漏斗上沿。
    # d 的字母再多下移 0.37 cm:表格的表头文字从面板上沿往下 0.16 cm 才开始,
    # 字母按面板矩形定位就会浮在 c 的轴题与表头之间;下移后它像 b/c 一样
    # 紧贴自己面板的第一行内容。
    DYBC, DYD = -0.012, -0.018
    stamp(fig, [(axe, "b", (0.0, DYBC)), (axb, "c", (0.0, DYBC)),
                (axd, "d", (0.0, DYD)), (axT, "e")])
    # a 的面板矩形从 x=0 起,不与 b/d/e 同列,stamp 的分列逻辑对它不适用;
    # 字母单独放,x 仍取与其余面板同一条线。
    fig.text(EL / _W - 0.013, (_H - YA - 0.04) / _H, "a", fontsize=10,
             fontweight="bold", va="top", ha="left")
    save(fig, "fig1_agentic_law_learning")



# ─────────────────────────────────────────────────────────────────────────────
# Fig. 4 — Abstention-dominant failure of Pauling's rules
# ─────────────────────────────────────────────────────────────────────────────
SHORT = {"Pauling 2 (bond-strength dev.)": "Pauling 2",
         "Pauling 3 (edge/face sharing)": "Pauling 3",
         "Pauling 4 (high-valence contact)": "Pauling 4",
         "Pauling 5 (parsimony)": "Pauling 5",
         "This work: bl_min": r"$\rho$",
         "Volume per atom": "Volume/atom",
         "Shannon packing": "Shannon packing",
         "DFT E_hull (baseline)": "DFT energy"}
SLATE = "#29445C"
RSET = ["L1", "L1'", "L2", "L3", "L4"]
RLAB = {"L1": "Set 1", "L1'": "Set 1$'$", "L2": "Set 2", "L3": "Set 3",
        "L4": "Set 4"}
CMAP = {"Pauling 2": SLATE, "Pauling 3": SLATE, "Pauling 4": SLATE, "Pauling 5": SLATE,
        r"$\rho$": BLU, "Volume/atom": GRY, "Shannon packing": GRY,
        "DFT energy": GRN}


def fig4_validation():
    """Detection, blind spots and transfer: four panels."""
    _W, _H = 18.3, 14.90
    fig = plt.figure(figsize=(W2, _H * CM))

    def _rect(x0, w, ytop, h):
        return [x0 / _W, (_H - ytop - h) / _H, w / _W, h / _H]

    # 左列右移并收窄:a 的最右一根柱子原来伸到 7.96 cm,压在 b 的行标签(自 7.71 cm
    # 起)底下 —— 现在 a 的墨迹止于 7.43 cm。两行之间原有 1.5 cm 的空白带,
    # 收掉之后同样的总高分给了更高的面板。
    XL, XR_, XC1 = 1.75, 16.80, 5.90
    XB = 9.95
    HR = 5.60
    Y1, Y2 = 0.60, 8.30

# ---- (a) deployed validity filters against the whole ladder of law sets
    axa = fig.add_axes(_rect(XL, XC1, Y1, HR))
    v = pd.read_csv(DATA / "fig6_validity.csv").set_index("criterion")
    order = ["min pair distance > 0.5 A", "min pair distance > 0.7 A",
             "min pair distance > 1.0 A", "SMACT charge neutrality",
             "L1 (D1, tau=0.735)", "L1' (D1+D2)",
             "L2 (D1,D3-D5)", "L3 (D1,D3-D6)", "L4 (D1,D3-D8)"]
    lab = [r"$d$ 0.5$\,\AA$", r"$d$ 0.7$\,\AA$", r"$d$ 1.0$\,\AA$", "SMACT",
           "Set 1", "Set 1′", "Set 2", "Set 3", "Set 4"]
    c = [GRY, GRY, GRY, "#B3B8BD",
         BLU, PUR, ORA, RED, "#0A5A3C"]
    ex = [float(v.loc[k, "exclusion_total"]) for k in order]
    sa = [float(v.loc[k, "real_satisfaction"]) for k in order]
    axa.bar(range(len(order)), ex, color=c, width=0.60, lw=0, alpha=0.9)
    axa.set_xticks(range(len(order))); axa.set_xticklabels(lab, fontsize=7.4)
    for t in axa.get_xticklabels()[:4]:
        t.set_rotation(38); t.set_ha("right"); t.set_rotation_mode("anchor")
    axa.set_ylabel("Damage detection")
    axa.set_ylim(0, 1.20)
    axa.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axa.spines["left"].set_bounds(0, 1.0)
    for i, (e, s) in enumerate(zip(ex, sa)):
        axa.text(i, e + 0.017, f"{e:.3f}", ha="center", fontsize=7.0, color=c[i],
                 fontweight="bold")
        axa.text(i, 1.055, f"{s:.2f}", ha="center", fontsize=6.9, color="#97999C")
    axa.set_xlim(-0.62, 8.62)
    axa.text(-0.55, 1.135, "satisfaction of experimental structures", fontsize=7.4,
             color="#97999C", ha="left", va="center")
    axa.axvline(2.5, color="#B9BBBE", lw=0.7, ls=":", ymax=1.0 / 1.20)
    axa.axvline(3.5, color="#86888A", lw=0.8, ls=":", ymax=1.0 / 1.20)

    # ---- (b) per-perturbation-class exclusion, same test set
    axb = fig.add_axes(_rect(XB, XR_ - XB, Y1, HR))
    horder = ["min pair distance > 0.5 A", "min pair distance > 0.7 A",
              "min pair distance > 1.0 A", "SMACT charge neutrality",
              "L1 (D1, tau=0.735)",
              "D1 alone, tau=0.804", "L1' (D1+D2)", "L2 (D1,D3-D5)", "L3 (D1,D3-D6)",
              "L4 (D1,D3-D8)"]
    SC = [c2 for c2 in v.columns if c2.startswith("S") and c2[1].isdigit()]
    M = v.loc[horder, SC].values
    rows = [r"min $d$ > 0.5 $\AA$", r"min $d$ > 0.7 $\AA$", r"min $d$ > 1.0 $\AA$",
            "SMACT (composition)",
            # 集合成分见 Fig. 1d;这里只留集合名,否则 Law/Set 全称的行标签
            # 会向左伸进面板 a 的柱子里。
            "Set 1", "Law 1 only, " + r"$\tau$=0.804",
            "Set 1′", "Set 2", "Set 3", "Set 4"]
    im = axb.imshow(M, cmap="palmatrix", vmin=0, vmax=1, aspect="auto")
    axb.set_xticks(range(5))
    axb.set_xticklabels(CLASSLAB[:5], fontsize=7.0, linespacing=1.25)
    axb.set_yticks(range(len(rows))); axb.set_yticklabels(rows, fontsize=7.3)
    for i in range(len(rows)):
        for j in range(5):
            axb.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.3,
                     color="w" if M[i, j] > 0.55 else "#212224")
    axb.axhline(2.5, color="k", lw=0.8)
    axb.axhline(3.5, color="k", lw=1.1)
    axb.add_patch(mp.Rectangle((2.5, 3.5), 1, 2, fill=False, ec="#101113", lw=1.3,
                               zorder=5))
    cax = fig.add_axes(_rect(XR_ + 0.16, 0.24, Y1, HR))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Damage detection", fontsize=8.0); cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.5)
    for sp in axb.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)

    # ---- (c) why the guard is needed
    axc = fig.add_axes(_rect(XL, XC1, Y2, HR))
    h = pd.read_csv(DATA / "fig4_rho_hist.csv")
    ctr = (h.lo + h.hi) / 2
    for col, cl, lb, al, zo in [
            ("real_ionic", SLATE, r"experimental, ionic ($f_i>$0.5)", 0.55, 3),
            ("real_nonionic", "#BDBFC2", r"experimental, non-ionic", 0.60, 2),
            ("S4", RED, "D4 expansion", 0.45, 4)]:
        axc.fill_between(ctr, h[col] / h[col].sum(), step="mid", color=cl, alpha=al,
                         lw=0, zorder=zo)
        axc.step(ctr, h[col] / h[col].sum(), where="mid", color=cl, lw=0.9, label=lb,
                 zorder=zo + 0.5)
    axc.axvline(1.05, color="k", lw=0.9, ls="--")
    axc.text(1.068, 0.180, r"$\rho$ = 1.05", fontsize=7.5, rotation=90, va="top")
    axc.set_xlim(0.45, 1.95); axc.set_ylim(0, 0.185)
    axc.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
    axc.set_yticks([0, 0.05, 0.10, 0.15])
    axc.set_xlabel(r"$\rho$"); axc.set_ylabel("Fraction of structures")
    # 图例移到数据区之外,与同一行的 d 面板一致
    legc = axc.legend(frameon=False, fontsize=7.4, loc="lower left", ncol=2,
                      handlelength=1.2, labelspacing=0.30, columnspacing=1.2,
                      borderpad=0.1, bbox_to_anchor=(-0.02, 1.005))
    legc.get_texts()[0].set_fontsize(8.8)

    # ---- (d) leave-one-perturbation-out: certified search vs one inequality
    axd = fig.add_axes(_rect(XB, XR_ - XB, Y2, HR))
    lk = pd.read_csv(DATA / "fig3_loko.csv")
    x = np.arange(len(lk)); w = 0.20
    TREE, THR = "#8A8C8E", BLU        # 与 Fig 2b 一致:认证树用灰,单一 rho 门限用蓝
    axd.bar(x - 1.5 * w, lk.tree_seen, w, color=TREE, lw=0, alpha=0.42)
    axd.bar(x - 0.5 * w, lk.tree_held, w, color=TREE, lw=0)
    axd.bar(x + 0.5 * w, lk.thr_seen, w, color=THR, lw=0, alpha=0.42)
    axd.bar(x + 1.5 * w, lk.thr_held, w, color=THR, lw=0)
    axd.set_xticks(x); axd.set_xticklabels([f"hold\n{dmglab(h2)}" for h2 in lk.held],
                                           fontsize=7.5)
    axd.set_ylabel("Damage detection")
    hs = [mp.Patch(fc=TREE, alpha=0.42, lw=0), mp.Patch(fc=TREE, lw=0),
          mp.Patch(fc=THR, alpha=0.42, lw=0), mp.Patch(fc=THR, lw=0)]
    axd.legend(hs, ["optimal tree, seen classes", "optimal tree, omitted class",
                    r"single $\rho$ threshold, seen",
                    r"single $\rho$ threshold, omitted class"],
               frameon=False, fontsize=7.4, loc="lower left", ncol=2,
               handlelength=1.4, labelspacing=0.30, columnspacing=1.2,
               borderpad=0.1, bbox_to_anchor=(-0.02, 1.005))
    axd.set_ylim(0, 1.02)
    axd.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # c/d 的图例挂在各自轴框之上；d 的字母再额外抬高，避开 c 的两列图例。
    stamp(fig, [(axa, "a"), (axb, "b"),
                (axc, "c", (0.0, 0.30 / _H)), (axd, "d", (0.0, 0.55 / _H))])
    save(fig, "fig4_validation")


def wrap(s, n):
    out, line = [], ""
    for w in str(s).split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return "\n".join(out)


def figS8_ledger():
    r = pd.read_csv(DATA / "fig5_retractions.csv")
    # 版面按“行高 = 字号 x 行距”反算图高:正文 8.2 pt、行距 1.38 -> 每行 0.399 cm
    FS, LS, HH = 8.2, 1.38, 20.4
    LINE = FS * LS / 72 * 2.54 / HH           # 一行文字占图高的比例
    PAD = 0.23 / HH                           # 每条记录的上下留白
    fig, ax = plt.subplots(figsize=(W2, HH * CM))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    xs = [0.0, 0.037, 0.408, 0.700]
    hd = ["", "Claim that did not survive", "Why it looked true",
          "Diagnostic that exposed it"]
    ytop = 0.965
    for x, h in zip(xs, hd):
        ax.text(x, ytop, h, fontsize=9.0, fontweight="bold", va="top", color="#212224")
    ax.plot([0, 1], [ytop - 0.023] * 2, color="#323335", lw=0.9)

    y = ytop - 0.045
    for i, (_, row) in enumerate(r.iterrows()):
        a = wrap(row["claim"], 45)
        b = wrap(row["why_it_looked_true"], 35)
        c = wrap(row["what_exposed_it"], 36)
        nl = max(a.count("\n"), b.count("\n"), c.count("\n")) + 1
        h = LINE * nl + PAD
        if i % 2 == 0:
            ax.add_patch(mp.Rectangle((-0.004, y - h + 0.010), 1.008, h,
                                      fc="#F2F6F9", lw=0, zorder=0))
        ax.text(xs[0], y, f"R{i+1}", fontsize=FS, fontweight="bold", va="top",
                color=RED)
        ax.text(xs[1], y, a, fontsize=FS, va="top", color="#191A1C", linespacing=LS)
        ax.text(xs[2], y, b, fontsize=FS, va="top", color="#424446", linespacing=LS)
        ax.text(xs[3], y, c, fontsize=FS, va="top", color=GRN, linespacing=LS)
        y -= h
    ax.plot([0, 1], [y + 0.010] * 2, color="#323335", lw=0.9)
    save(fig, "figS8_refutation_ledger")


if __name__ == "__main__":
    # 主图 1、2、4 与 legacy asset figS8(正文显示为 Supplementary Fig. S1,反驳账本)。
    # fig4_abstention 已从稿件撤下(其内容并入图 5),生成器保留在
    # backup-20260817-figure-rename/ 的历史版本里,不再默认重跑。
    fig1(); fig2(); fig4_validation(); figS8_ledger()
