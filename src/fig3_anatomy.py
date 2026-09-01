"""Fig. "Anatomy of implausibility": one spinel, five damaged versions, and the
physical mechanism that catches each one.

Panel a: MgAl2O4 primitive cell, experimental + D1-D5 (archive keys S1-S5),
rendered with the firing predicate annotated with its computed value.
Panel b: mechanism-by-damage-type matrix from published rows.
Panel c: the Born-Mayer wall against the observed rho density.

Illustrative only: the spinel is built from scratch, the perturbation procedures
reproduce src/make_negatives.py with a frozen seed and mid-range amplitudes.
No held-out or sealed population is touched.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figs"
DATA = ROOT / "paper" / "data"

# type, spines and page width inherit the global rcParams of the main figures
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figs as _pf                                   # noqa: E402  (sets rcParams)
from discriminate import criteria                          # noqa: E402
from elec_feat import elec_feats                           # noqa: E402
from phys_law import phys_feats                            # noqa: E402

CM = 1 / 2.54
W2 = 18.0 * CM
CUBE_ZOOM = 1.10        # leaves marker-radius clearance for expanded S4 corners

# The energy at which a short contact stops being a geometric curiosity and becomes a
# chemical cost.  Frozen in dft/PREREG-DFT.md before the calculations were run; panel c
# marks it so that the crossing the law is calibrated on can be read off the axis.
E1_COST = 0.1           # eV per atom

# ---------------------------------------------------------------- structure
def spinel():
    s = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(8.089), ["Mg", "Al", "O"],
        [[0, 0, 0], [0.625, 0.625, 0.625], [0.3873, 0.3873, 0.3873]])
    s = s.get_primitive_structure()
    s.add_oxidation_state_by_element({"Mg": 2, "Al": 3, "O": -2})
    return s

# perturbation procedures, maths from src/make_negatives.py, frozen amplitudes
def corrupt(s0, mode, rng):
    s = s0.copy()
    if mode == "S1":                                # uniaxial compression
        # scale the cubic lattice vector a (Cartesian x) by (1-u), u = 0.25;
        # on the primitive cell this is the Cartesian strain diag(1-u, 1, 1)
        strain = np.diag([1 - 0.25, 1.0, 1.0])
        m = s.lattice.matrix @ strain.T
        s = Structure(Lattice(m), s.species, s.frac_coords)
    elif mode == "S2":                              # cation-cation swap, dz>=1
        i = [k for k, sp in enumerate(s.species) if sp.symbol == "Mg"][0]
        j = [k for k, sp in enumerate(s.species) if sp.symbol == "Al"][0]
        si, sj = s[i].specie, s[j].specie
        s[i], s[j] = sj, si
        s[i].properties["hi"] = True; s[j].properties["hi"] = True
    elif mode == "S3":                              # gaussian displacement
        for k in range(len(s)):
            s.translate_sites(k, rng.normal(0, 0.55, 3), frac_coords=False)
    elif mode == "S4":                              # isotropic expansion
        s = Structure(Lattice(s.lattice.matrix * 1.30), s.species, s.frac_coords)
    elif mode == "S5":                              # cation-anion swap
        i = [k for k, sp in enumerate(s.species) if sp.symbol == "Al"][0]
        j = [k for k, sp in enumerate(s.species) if sp.symbol == "O"][-1]
        si, sj = s[i].specie, s[j].specie
        s[i], s[j] = sj, si
        s[i].properties["hi"] = True; s[j].properties["hi"] = True
    return s

# ---------------------------------------------------------------- predicates
FI = 0.605          # 1-exp(-dchi^2/4), mean cation chi 1.51 vs O 3.44

def evaluate(s):
    """Evaluate the schematic with the same feature functions as the benchmark."""
    z = np.array([sp.oxi_state for sp in s.species], float)
    pf = phys_feats(s, z)
    ef = elec_feats(s, z)
    cf = criteria(s, z)
    if pf is None or ef is None or cf is None:
        raise RuntimeError("production feature calculation failed for Fig. 3 schematic")
    # Wyckoff economy at symprec 0.01
    try:
        sga = SpacegroupAnalyzer(s, symprec=0.01)
        orbits = len(set(sga.get_symmetry_dataset().equivalent_atoms))
    except Exception:
        orbits = len(s)
    econ = orbits / len(s)
    return dict(rho=float(pf["bl_min"]), bl_mean=float(pf["bl_mean"]),
                cn_an_mean=float(cf["cn_an_mean"]), like=float(pf["frac_like_bonds"]),
                d4=float(ef["madz_range"]), d5=float(ef["mad_max"]),
                econ=float(econ), d8=float(ef["bv_rel_mean"]))

PRIO = {"S1": ["Law 1", "Law 4", "Law 8"], "S2": ["Law 4"],
        "S3": ["Law 7", "Law 8", "Law 1", "Law 6", "Law 4"],
        "S4": ["Law 2", "Law 3", "Law 8"], "S5": ["Law 6", "Law 7", "Law 4", "Law 8"]}

def verdicts(v, mode):
    """Fired predicates with their computed values, mechanism-priority order.

    Reading only: the predicate name and the number it was evaluated at.  The
    thresholds themselves are carried by Fig. 2, by panel c and by the caption.
    """
    fired = {}
    if v["rho"] < 0.804:
        fired["Law 1"] = f"$\\rho$ = {v['rho']:.2f}"
    if FI > 0.50 and v["rho"] > 1.05:
        fired["Law 2"] = f"$\\rho$ = {v['rho']:.2f}"
    if v["cn_an_mean"] <= 3.333 and v["bl_mean"] > 1.081:
        fired["Law 3"] = f"mean contact = {v['bl_mean']:.2f}"
    if v["d4"] > 31.45:
        fired["Law 4"] = f"range $E_\\mathrm{{M}}/|z|$ = {v['d4']:.0f}"
    if v["d5"] > 15.17:
        fired["Law 5"] = f"max $E_\\mathrm{{M}}$ = {v['d5']:.0f} eV"
    if FI > 0.55 and v["like"] > 1e-4:
        fired["Law 6"] = f"like bonds = {v['like']:.2f}"
    if v["econ"] > 2 / 3:
        fired["Law 7"] = f"sites = {v['econ']:.2f}"
    if v["d8"] > 0.7143040821865658:
        fired["Law 8"] = f"BV = {v['d8']:.2f}"
    order = PRIO.get(mode, []) + [k for k in fired if k not in PRIO.get(mode, [])]
    return [(k, fired[k]) for k in order if k in fired]

# ---------------------------------------------------------------- rendering
COL = {"Mg": "#BD93D0", "Al": "#4178A6", "O": "#E7796D"}
# spheres are drawn semi-transparent so the exchange rings read clearly
SPHERE_ALPHA = 0.80
RAD = {"Mg": 60, "Al": 52, "O": 88}
P2C = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]     # fcc primitive -> cubic cell

def render(ax, s, scale_ref=None, zoom=1.0):
    sc = s.copy(); sc.make_supercell(P2C)
    cart = sc.cart_coords
    z = np.array([sp.oxi_state for sp in sc.species])
    syms = [sp.symbol for sp in sc.species]
    hi = [i for i, st in enumerate(sc) if st.properties.get("hi")]
    A = sc.lattice.matrix
    ctr = A.sum(0) / 2
    cart = cart - ctr
    # cell edges
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(corners[a] - corners[b])) == 1:
                p, q = corners[a] @ A - ctr, corners[b] @ A - ctr
                ax.plot(*zip(p, q), color="#646668", lw=0.55, alpha=0.85, zorder=0)
    for i in range(len(sc)):
        for nb in sc.get_neighbors(sc[i], 2.35):
            j = nb.index
            if z[i] * z[j] < 0:
                p = nb.coords - ctr
                ax.plot(*zip(cart[i], p), color="#AEB0B3", lw=0.45, alpha=0.8,
                        zorder=1)
    for i in range(len(sc)):
        ax.scatter(cart[i, 0], cart[i, 1], cart[i, 2], s=RAD[syms[i]],
                   c=COL[syms[i]], edgecolors="#212224", linewidths=0.3,
                   alpha=SPHERE_ALPHA, zorder=3, depthshade=True)
    for i in hi:
        ax.scatter(cart[i, 0], cart[i, 1], cart[i, 2], s=RAD[syms[i]] * 3.0,
                   facecolors="none", edgecolors="#783F90", linewidths=1.35,
                   zorder=5)
    L = scale_ref if scale_ref else np.abs(cart).max() * 1.02
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    ax.set_proj_type("ortho"); ax.view_init(elev=14, azim=-58)
    ax.set_axis_off(); ax.set_box_aspect([1, 1, 1], zoom=zoom)

# ---------------------------------------------------------------- build all
def main():
    rng = np.random.default_rng(20260815)
    s0 = spinel()
    variants = [("experimental MgAl$_2$O$_4$", "real", s0, [])]
    labels = {"S1": "D1 uniaxial compression",
              "S2": "D2 cation–cation swap",
              "S3": "D3 random displacement",
              "S4": "D4 isotropic expansion",
              "S5": "D5 cation–anion swap"}
    hi = {"S2": [0, 2], "S5": [2, 13]}
    for m in ["S1", "S2", "S3", "S4", "S5"]:
        variants.append((labels[m], m, corrupt(s0, m, rng), hi.get(m, [])))

    readings = {}
    for title, m, st, h in variants:
        readings[m] = evaluate(st)
    json.dump(readings, open(OUT / "fig3_anatomy_readings.json", "w"), indent=1)

    FH = 18.5 * CM
    fig = plt.figure(figsize=(W2, FH))
    # panel a is laid out by hand in inches so that every cell is a tight band
    # of [title | cube | readings]; the 3D axes rect is always squared by
    # Axes3D.apply_aspect, so the square side is what has to be budgeted.
    fx, fy = 1.0 / W2, 1.0 / FH                     # inch -> figure fraction
    SQ = 1.54                                       # cube square side (inch)
    TITLE_B, READ_B, GAP_R = 0.19, 0.32, 0.12       # bands and inter-row gap
    LINE = 0.135                                    # reading line pitch (inch)
    rows_top = [1.0 - 0.06 * fy]
    rows_top.append(rows_top[0] - (TITLE_B + SQ + READ_B + GAP_R) * fy)
    # column centres: the left and right margins are equal and wide enough to
    # carry the species key on the letter column
    xc = [0.185, 0.5, 0.815]
    # b and c share their top and bottom edges exactly
    BY0, BY1 = 0.063, 0.359
    # b's left edge leaves the two-line row labels a full gutter with room for
    # the panel letter above them
    RECT_B = [0.168, BY0, 0.470 - 0.168, BY1 - BY0]
    RECT_C = [0.585, BY0, 0.930 - 0.585, BY1 - BY0]
    # scale: all panels share the S4 spatial extent so size changes are visible
    s4 = [v for v in variants if v[1] == "S4"][0][2]
    tmp = s4.copy(); tmp.make_supercell(P2C)
    ext = np.abs(tmp.cart_coords - tmp.lattice.matrix.sum(0) / 2).max() * 1.02

    cube_axes = []
    for k, (title, m, st, h) in enumerate(variants):
        cx = xc[k % 3]
        top = rows_top[k // 3]
        cube_top = top - TITLE_B * fy
        ax = fig.add_axes([cx - SQ * fx / 2, cube_top - SQ * fy,
                           SQ * fx, SQ * fy], projection="3d")
        cube_axes.append(ax)
        render(ax, st, scale_ref=ext, zoom=CUBE_ZOOM)
        fig.text(cx, top, title, ha="center", va="top", fontsize=8.5,
                 color="#212224")
        v = verdicts(readings[m], m)
        if not v:
            # 无谓词触发:同样写出三个读数,并写明触发数为零
            lines = ["satisfies all laws",
                     (f"$\\rho$ = {readings[m]['rho']:.2f}  ·  sites = "
                      f"{readings[m]['econ']:.2f}  ·  BV = "
                      f"{readings[m]['d8']:.2f}")]
            col = "#0A5A3C"
        else:
            # two readings to a line keeps every cell to at most two lines
            items = [f"{n}  {s}" for n, s in v[:4]]
            lines = ["  ·  ".join(items[i:i + 2]) for i in range(0, len(items), 2)]
            col = "#CC4C43"
        for li, ln in enumerate(lines):
            fig.text(cx, cube_top - (SQ + 0.045 + LINE * li) * fy, ln,
                     ha="center", va="top", fontsize=7.6, color=col)

    # ---- panel b: mechanism by damage type, all rows on the held-out split
    axb = fig.add_axes(RECT_B)
    import pandas as pd
    SC = ["S1", "S2", "S3", "S4", "S5"]
    s5c = pd.read_csv(ROOT / "paper" / "si_data" / "s5_split_consistency.csv")
    cal = s5c[s5c.split == "calibration"].set_index("ruleset")
    l1 = cal.loc["L1", SC].values.astype(float)
    l1p = cal.loc["L1'", SC].values.astype(float)
    l2 = cal.loc["L2", SC].values.astype(float)
    l3 = cal.loc["L3", SC].values.astype(float)
    l4 = cal.loc["L4", SC].values.astype(float)
    # SI Note S17.1 single-addition rows (same calibration split)
    l3d7 = np.array([0.678, 0.899, 1.000, 0.788, 0.984])
    l3d8 = np.array([0.687, 0.596, 0.960, 0.915, 0.609])
    rows = [("contact floor  Law 1\nalone (Set 1)", l1),
            ("Law 2 ionic-contact law\nadded to Set 1", l1p - l1),
            ("Law 1 at 0.804 + Law 3–Law 5\nadded to Set 1′", l2 - l1p),
            ("like-charge ban  Law 6\nadded to Set 2", l3 - l2),
            ("distinct-site  Law 7\nadded to Set 3", l3d7 - l3),
            ("bond-valence  Law 8\nadded to Set 3", l3d8 - l3)]
    M = np.vstack([r[1] for r in rows] + [l4])
    im = axb.imshow(M, cmap="palseq2", vmin=0, vmax=1, aspect="auto")
    axb.set_xticks(range(5))
    # 归档列名仍是 S1--S5;图内显示 D1--D5,避免与正文的 Set 1--Set 5 混读
    axb.set_xticklabels(["D" + k[1:] for k in SC], fontsize=8)
    axb.set_xlabel("Damage type")
    axb.set_yticks(range(7))
    axb.set_yticklabels([r[0] for r in rows] + ["Set 4 (seven laws)"], fontsize=7.4)
    axb.tick_params(length=0)
    for i in range(7):
        for j in range(5):
            axb.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                     fontsize=7.4, color="w" if M[i, j] > 0.55 else "#212224")
    for j, i in enumerate(np.argmax(M[:6], axis=0)):
        axb.add_patch(mp.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   ec="#CC4C43", lw=1.3, zorder=5))
    axb.axhline(5.5, color="k", lw=0.9)
    for sp in axb.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)

    # ---- panel c: the Born-Mayer wall
    axc = fig.add_axes(RECT_C)
    h = pd.read_csv(DATA / "fig4_rho_hist.csv")
    ctr = ((h.lo + h.hi) / 2).values
    dens = (h.real_ionic + h.real_nonionic).values
    dens = dens / dens.sum()
    axc2 = axc.twinx()
    axc2.fill_between(ctr, dens, step="mid", color="#7D9BB6", alpha=0.55, lw=0)
    axc2.set_ylabel("Fraction of experimental structures", color="#0D5B90")
    axc2.tick_params(colors="#0D5B90")
    axc2.spines["right"].set_visible(True)
    axc2.spines["right"].set_color("#0D5B90")
    axc2.set_ylim(0, dens.max() * 1.6)
    # The curve here used to be a scaled Born-Mayer model.  It is now the measured
    # landscape: twenty experimental compounds compressed and expanded on the reduced
    # coordinate and evaluated with plane-wave DFT, so the wall the law encodes is read
    # off first principles rather than off a functional form chosen to have one.
    curves = json.load(open(ROOT / "dft" / "E1_rho_curve" / "curves.json"))
    gkeys = sorted(curves[0]["curve"], key=float)
    rr = np.array([float(k) for k in gkeys])
    M = np.array([[c["curve"][k] for k in gkeys] for c in curves])
    med = np.median(M, axis=0)
    q25, q75 = np.percentile(M, 25, axis=0), np.percentile(M, 75, axis=0)
    # a log axis: the excess spans 0.1 to 25 eV per atom across the grid, and on a
    # linear axis the compressed end flattens everything above the floor into the baseline
    YLO, YHI = 0.02, 60.0
    axc.set_yscale("log")
    axc.fill_between(rr, np.clip(q25, YLO, None), np.clip(q75, YLO, None),
                     color="#323335", alpha=0.16, lw=0, zorder=2)
    axc.plot(rr, np.clip(med, YLO, None), color="#323335", lw=1.2, zorder=3)
    # the level the crossing is defined at, so rho* can be read straight off the panel
    axc.axhline(E1_COST, color="#6E7276", lw=0.6, ls=(0, (1, 2)), zorder=2)
    axc.text(1.462, E1_COST * 1.16, "0.1 eV per atom", fontsize=7, ha="right",
             va="bottom", color="#6E7276")
    axc.set_xlim(0.355, 1.485)
    axc.set_ylim(YLO, YHI)
    axc.set_xlabel(r"Reduced contact ratio $\rho$")
    axc.set_ylabel("DFT energy above minimum (eV per atom)")
    # each label hugs its own threshold line from the side, so the lines stay
    # unbroken and no knocked-out box has to be punched through the bands
    for x, lab, xt, ha in [(0.735, r"$\tau$ = 0.735", 0.720, "right"),
                           (0.804, "0.804", 0.816, "left"),
                           (1.05, "1.05", 1.062, "left")]:
        axc.axvline(x, color="#CC4C43" if x < 1 else "#8C55A3", lw=0.8,
                    ls=(0, (3, 2)))
        axc.text(xt, 33.0, lab, fontsize=8, ha=ha, va="center",
                 color="#CC4C43" if x < 1 else "#8C55A3")
    axc.fill_betweenx([YLO, YHI], 0.355, 0.735, color="#CC4C43", alpha=0.06, lw=0)
    axc.fill_betweenx([YLO, YHI], 1.05, 1.485, color="#8C55A3", alpha=0.06, lw=0)
    # region names only; the Law 1 name sits below the measured curve so that no
    # knocked-out background has to be laid over the curve
    axc.text(0.497, 5.2, "Law 1 floor", fontsize=8,
             ha="center", va="center", color="#CC4C43")
    axc.text(1.300, 11.0, "Law 2 ceiling", fontsize=8,
             ha="center", va="center", color="#8C55A3")

    # Panel letters on a grid: a and b are the left column and share one x
    # (the leftmost y-axis decoration of either), b and c are one row and share
    # one y.  Everything is measured from the drawn artists, so the letters
    # cannot drift when the type or the row labels change.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    PAD_X, PAD_Y = 0.013, 0.011
    x_left = min(_pf._ydeco_left(axb, fig, rend),
                 cube_axes[0].get_position().x0) - PAD_X
    x_c = _pf._ydeco_left(axc, fig, rend) - PAD_X
    y_bc = BY1 + PAD_Y
    fig.text(x_left, rows_top[0], "a", fontsize=10, fontweight="bold",
             va="top", ha="left")
    fig.text(x_left, y_bc, "b", fontsize=10, fontweight="bold",
             va="bottom", ha="left")
    fig.text(x_c, y_bc, "c", fontsize=10, fontweight="bold",
             va="bottom", ha="left")

    # 物种色键:三种球色此前在图内没有任何标识。放在字母列的左页边,和第一
    # 行晶胞同高;这一列本来就是空的,因此不占任何图面预算,也不压任何图元。
    key_y = rows_top[0] - (TITLE_B + SQ / 2) * fy
    for ki, el in enumerate(("Mg", "Al", "O")):
        fig.text(x_left, key_y + (1 - ki) * 0.155 * fy, "●  " + el,
                 ha="left", va="center", fontsize=7.6, color=COL[el])

    for ext_ in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_anatomy.{ext_}")
    print("wrote fig3_anatomy;", {k: {kk: round(vv, 3) if isinstance(vv, float) else vv
          for kk, vv in v.items()} for k, v in readings.items()})

if __name__ == "__main__":
    main()
