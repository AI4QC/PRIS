#!/usr/bin/env python3
"""Supplementary figures for the pre-registered first-principles campaign.

Four experiments, each shown as the evidence behind a claim the main text makes with a
single number.  House style follows ``si_figs.py``: Arial, no titles, panel letters from
``stamp``.  Every axis that carries a computed quantity names the method on the axis, so a
reader never has to reach the caption to learn which numbers are DFT.

Data come from ``dft/`` -- the task packages, the extraction and the analysis are in that
directory, and the predictions they are scored against were frozen in ``dft/PREREG-DFT.md``
before any job was submitted.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paper_figs import (plt, stamp, save, CM, W1, W2,          # noqa: E402
                        BLU, RED, ORA, GRN, GRY, PUR, ROOT)
from collections import Counter  # noqa: E402

DFT = ROOT / "dft"
INK = "#323335"
SOFT = "#757779"
L4C = "#0A5A3C"

# the damage operators, in the order the main text introduces them
VARIANT = (("P0", INK, "undamaged"), ("S1", ORA, "D1 compression"),
           ("S2", GRN, "D2 cation–cation"), ("S3", RED, "D3 displacement"),
           ("S4", BLU, "D4 expansion"), ("S5", PUR, "D5 cation–anion"))
# anion families, coloured as in Fig. S10
ANION = {"O": BLU, "F": ORA, "N": GRN, "S": PUR, "Cl": RED}


def _load(name: str):
    return json.loads((DFT / name).read_text())


# ───────────────────────────────────────────────── E1: the measured contact wall
def si_e1_landscape():
    """Every compound's own energy landscape, and the hard-potential control."""
    curves = _load("E1_rho_curve/curves.json")
    shift = _load("E1b_paw_control/paw_shift.json")

    fig, axes = plt.subplots(1, 3, figsize=(W2, 7.2 * CM),
                             gridspec_kw={"width_ratios": [1.0, 0.80, 0.52]})
    fig.subplots_adjust(left=0.062, right=0.988, top=0.90, bottom=0.165, wspace=0.30)

    # (a) all twenty curves, not their average: the claim is that the wall is steep for
    # every chemistry, and an average would hide a single soft one
    ax = axes[0]
    keys = sorted(curves[0]["curve"], key=float)
    grid = np.array([float(k) for k in keys])
    for c in curves:
        y = np.clip([c["curve"][k] for k in keys], 1.5e-2, None)
        ax.plot(grid, y, color=ANION.get(c["anion"], GRY), lw=0.75, alpha=0.72,
                zorder=3)
    ax.axhline(0.1, color=SOFT, lw=0.7, ls=(0, (1, 2)), zorder=2)
    ax.text(1.40, 0.115, "0.1 eV per atom", fontsize=7, color=SOFT, ha="right",
            va="bottom")
    for x, colr in ((0.735, "#CC4C43"), (0.804, "#CC4C43"), (1.05, "#8C55A3")):
        ax.axvline(x, color=colr, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.text(0.745, 40, "0.735   0.804", fontsize=7, color="#CC4C43", ha="left",
            va="center")
    ax.text(1.06, 40, "1.05", fontsize=7, color="#8C55A3", ha="left", va="center")
    ax.set_yscale("log")
    ax.set_xlim(0.58, 1.42)
    ax.set_ylim(1.2e-2, 90)
    ax.set_xlabel(r"Reduced contact ratio $\rho$")
    ax.set_ylabel("DFT energy above minimum (eV per atom)")
    handles = [plt.Line2D([], [], color=v, lw=1.4) for v in ANION.values()]
    ax.legend(handles, list(ANION), frameon=False, fontsize=7.4, ncol=5,
              loc="lower left", bbox_to_anchor=(-0.012, 1.00), handlelength=1.2,
              handletextpad=0.4, columnspacing=1.0, borderpad=0.0)

    # (b) the frozen-core question, asked directly.  At the compressed end the PAW spheres
    # of neighbouring atoms overlap, which is where a frozen core is least defensible, so
    # eight compounds were recomputed with hard potentials and the two are plotted against
    # each other.
    ax = axes[1]
    std = np.array([r["standard_ev_per_atom"] for r in shift])
    hrd = np.array([r["hard_ev_per_atom"] for r in shift])
    lim = (0.6 * min(std.min(), hrd.min()), 1.7 * max(std.max(), hrd.max()))
    ax.plot(lim, lim, color="#B9BBBE", lw=0.7, zorder=1)
    ax.scatter(std, hrd, s=26, color=L4C, alpha=0.85, lw=0, zorder=3)
    rel = np.median([abs(r["relative_shift"]) for r in shift])
    ax.text(0.035, 0.955, f"median relative shift {rel:.1%}\n(pre-registered bound 25%)",
            transform=ax.transAxes, fontsize=7.4, color=INK, va="top", linespacing=1.35)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("DFT excess, standard potentials (eV per atom)")
    ax.set_ylabel("DFT excess, hard potentials (eV per atom)")

    # (c) why the reduced coordinate is the coordinate.  Each compound contributes the
    # contact at which compression first costs 0.1 eV per atom; both columns are divided by
    # their own median so that two different units become comparable and the only thing on
    # show is which one travels across chemistries.
    ax = axes[2]
    rho_star = np.array([c["rho_star"] for c in curves])
    d_star = np.array([c["d_star_a"] for c in curves])
    jit = np.random.default_rng(20260828).uniform(-0.11, 0.11, len(curves))
    for col, vals, colr in ((0, rho_star, BLU), (1, d_star, ORA)):
        v = vals / np.median(vals)
        ax.scatter(col + jit, v, s=13, color=colr, alpha=0.78, lw=0, zorder=3)
        q25, q75 = np.percentile(v, [25, 75])
        ax.add_patch(plt.Rectangle((col - 0.28, q25), 0.56, q75 - q25, facecolor=colr,
                                   alpha=0.15, lw=0, zorder=1))
        ax.plot([col - 0.28, col + 0.28], [1, 1], color=colr, lw=1.2, zorder=4)
        ax.text(col, 1.44, f"{(q75 - q25):.3f}", fontsize=7.2, color=colr, ha="center",
                va="top")
    ax.text(0.5, 1.53, "interquartile spread", fontsize=7.2, color=INK, ha="center",
            va="top")
    ax.text(0.5, 0.60, "1.80× tighter\nin the reduced coordinate", fontsize=7.2,
            color=INK, ha="center", va="bottom", linespacing=1.35)
    ax.set_xlim(-0.62, 1.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$\rho$", "$d$ (Å)"], fontsize=8.5)
    ax.set_ylim(0.55, 1.58)
    ax.set_yticks([0.75, 1.0, 1.25])
    ax.set_ylabel("0.1 eV crossing ÷ median")

    stamp(fig, [(axes[0], "a"), (axes[1], "b"), (axes[2], "c")])
    save(fig, "figSdft_e1_landscape")


# ───────────────────────────────────────── E4: the property screen under first principles
def si_e4_bulk():
    """Predicted against computed bulk modulus, one point per candidate."""
    rows = _load("E4_design/bulk_moduli.json")
    sweep = _load("E4_design/queue_sweep.json")
    thr = sweep["dft_threshold_gpa"]

    fig, ax = plt.subplots(figsize=(W1 * 1.16, 8.2 * CM))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.90, bottom=0.135)

    ROLE = (("control", GRY, "control"), ("screened", RED, "removed by PSS"),
            ("priority", BLU, "high-property"))
    for role, colr, lab in ROLE:
        sub = [r for r in rows if r["role"] == role]
        ax.scatter([r["uma_bulk_modulus_gpa"] for r in sub],
                   [r["dft_bulk_modulus_gpa"] for r in sub],
                   s=13, color=colr, alpha=0.72, lw=0, zorder=3,
                   label=f"{lab}  (n = {len(sub)})")
    lo, hi = 290, 450
    ax.plot([lo, hi], [lo, hi], color="#B9BBBE", lw=0.7, zorder=1)
    ax.axvline(400, color="#8C55A3", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.axhline(400, color="#8C55A3", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.axhline(thr, color=L4C, lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.text(lo + 3, 402, "400 GPa target", fontsize=7, color="#8C55A3", va="bottom")
    ax.text(lo + 3, thr + 1.5, f"{thr:.0f} GPa, the same target rescaled",
            fontsize=7, color=L4C, va="bottom")
    # the one candidate the screen removed that DFT puts highest of all: named, because a
    # single counterexample of this kind is the most informative point on the panel
    worst = max((r for r in rows if r["role"] == "screened"),
                key=lambda r: r["dft_bulk_modulus_gpa"])
    ax.annotate(f"{worst['formula']}\nremoved, and the highest here",
                (worst["uma_bulk_modulus_gpa"], worst["dft_bulk_modulus_gpa"]),
                textcoords="offset points", xytext=(9, -2), fontsize=6.8, color=RED,
                va="center", linespacing=1.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Machine-learned proxy bulk modulus (GPa)")
    ax.set_ylabel("DFT bulk modulus (GPa)")
    ax.legend(frameon=False, fontsize=7.4, loc="lower right", handletextpad=0.4,
              borderpad=0.2, labelspacing=0.3)
    save(fig, "figSdft_e4_bulk")



# ──────────────────────── E4 structures: what the design queue is actually made of
def si_e4_composition():
    """Elements, space groups, and what relaxation does to the symmetry."""
    import csv
    path = ROOT / "outputs" / "20260828_dft_supplementary_structures" / "index.csv"
    rows = list(csv.DictReader(path.open()))
    LAW7 = 2.0 / 3.0

    fig, axes = plt.subplots(1, 3, figsize=(W2, 6.6 * CM),
                             gridspec_kw={"width_ratios": [0.72, 1.15, 0.85]})
    fig.subplots_adjust(left=0.056, right=0.982, top=0.905, bottom=0.175, wspace=0.315)

    # (a) which elements the generator reached for.  The queue was conditioned on a bulk
    # modulus above 400 GPa, and the answer is a narrow corner of the periodic table.
    ax = axes[0]
    from collections import Counter
    counts = Counter(e for r in rows for e in r["elements"].split())
    names = [k for k, _ in counts.most_common()]
    vals = [counts[k] for k in names]
    ax.barh(range(len(names))[::-1], vals, color=BLU, alpha=0.85, lw=0)
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.text(v + 3, len(names) - 1 - i, str(v), fontsize=7, color=SOFT,
                ha="left", va="center")
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("Candidates containing the element")
    ax.tick_params(axis="y", length=0)

    # (b) the space groups of the relaxed cells, by crystal system so that thirty-eight
    # symbols do not have to be read one at a time
    ax = axes[1]
    SYSTEMS = (("triclinic", 1, 2), ("monoclinic", 3, 15), ("orthorhombic", 16, 74),
               ("tetragonal", 75, 142), ("trigonal", 143, 167),
               ("hexagonal", 168, 194), ("cubic", 195, 230))
    sysname = []
    for r in rows:
        n = int(r["spacegroup_number_dft_relaxed"])
        sysname.append(next(s for s, lo, hi in SYSTEMS if lo <= n <= hi))
    labels = [s for s, _, _ in SYSTEMS]
    gen_sys = []
    for r in rows:
        n = int(r["spacegroup_number_generated"])
        gen_sys.append(next(s for s, lo, hi in SYSTEMS if lo <= n <= hi))
    x = np.arange(len(labels))
    w = 0.38
    cg = Counter(gen_sys); cr = Counter(sysname)
    ax.bar(x - w / 2, [cg[k] for k in labels], w, color=GRY, alpha=0.85, lw=0,
           label="as generated")
    ax.bar(x + w / 2, [cr[k] for k in labels], w, color=L4C, alpha=0.85, lw=0,
           label="after DFT relaxation")
    ax.set_xticks(x)
    ax.set_xticklabels(["tric.", "mono.", "orth.", "tetr.", "trig.", "hex.", "cub."],
                       fontsize=7.6)
    ax.set_ylabel("Candidates")
    ax.set_xlabel("Crystal system")
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower left",
              bbox_to_anchor=(-0.015, 1.00), handlelength=1.2, handletextpad=0.45,
              columnspacing=1.1, borderpad=0.0)

    # (c) the quantity Law 7 is written on, before and after.  Every point can only move
    # down or stay: relaxation merges sites, it does not split them.
    ax = axes[2]
    ROLE = (("screened", RED, "removed by PSS"), ("control", GRY, "control"),
            ("priority", BLU, "high-property"))
    rng = np.random.default_rng(20260828)
    for role, colr, lab in ROLE:
        sub = [r for r in rows if r["role_in_E4"] == role]
        gx = np.array([float(r["site_fraction_generated"]) for r in sub])
        rx = np.array([float(r["site_fraction_dft_relaxed"]) for r in sub])
        ax.scatter(gx + rng.uniform(-0.012, 0.012, len(sub)), rx, s=11, color=colr,
                   alpha=0.70, lw=0, zorder=3, label=f"{lab}  ({len(sub)})")
    ax.plot([0.25, 1.02], [0.25, 1.02], color="#B9BBBE", lw=0.7, zorder=1)
    ax.axhline(LAW7, color="#8C55A3", lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.axvline(LAW7, color="#8C55A3", lw=0.8, ls=(0, (3, 2)), zorder=2)
    # above the diagonal is empty by construction, so the bound is named there
    ax.text(0.295, LAW7 - 0.020, "Law 7 bound, 2/3", fontsize=7, color="#8C55A3",
            ha="left", va="top")
    ax.set_xlim(0.25, 1.03); ax.set_ylim(0.25, 1.03)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Distinct-site fraction, generated")
    ax.set_ylabel("Distinct-site fraction, DFT")
    ax.legend(frameon=False, fontsize=7.0, loc="upper left", handletextpad=0.35,
              borderpad=0.25, labelspacing=0.28)

    stamp(fig, [(axes[0], "a"), (axes[1], "b"), (axes[2], "c")])
    save(fig, "figSdft_e4_composition")


# ─────────────────────────────── E3: the machine-learned potential against first principles
def si_e3_paired():
    """The same cells, both ways round."""
    pair = _load("E3_crosscheck/paired_energies.json")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 7.6 * CM),
                             gridspec_kw={"width_ratios": [1.0, 0.78]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.155, wspace=0.245)

    ax = axes[0]
    lo, hi = 1e-4, 4e1
    ax.plot([lo, hi], [lo, hi], color="#B9BBBE", lw=0.7, zorder=1)
    for v, colr, lab in VARIANT:
        pts = [(r["mattersim_release_ev_per_atom"], r["dft_release_ions_ev_per_atom"])
               for r in pair if r["variant"] == v
               and r["mattersim_release_ev_per_atom"] is not None]
        if not pts:
            continue
        ax.scatter(np.clip([q[0] for q in pts], lo, None),
                   np.clip([q[1] for q in pts], lo, None), s=11, color=colr,
                   alpha=0.72, lw=0, zorder=3, label=f"{lab}  ({len(pts)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("MatterSim energy released (eV per atom)")
    ax.set_ylabel("DFT energy released (eV per atom)")
    ax.legend(frameon=False, fontsize=6.8, ncol=1, loc="upper left",
              handletextpad=0.35, borderpad=0.3, labelspacing=0.32)

    # the aggregate the main text quotes, resolved by damage class so that a class where
    # the two methods disagree cannot hide inside a single rank correlation
    ax = axes[1]
    x = np.arange(len(VARIANT))
    w = 0.38
    for k, (v, colr, _) in enumerate(VARIANT):
        d = [r["dft_release_ions_ev_per_atom"] for r in pair if r["variant"] == v]
        m = [r["mattersim_release_ev_per_atom"] for r in pair if r["variant"] == v
             and r["mattersim_release_ev_per_atom"] is not None]
        ax.bar(k - w / 2, np.median(d), w, color=colr, alpha=0.85, lw=0)
        ax.bar(k + w / 2, np.median(m), w, facecolor="none", edgecolor=colr, lw=0.9,
               hatch="////")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] if v[0] == "P0" else "D" + v[0][1] for v in VARIANT],
                       fontsize=8.0)
    ax.set_ylim(3e-4, 12)
    ax.set_xlabel("Perturbation class")
    ax.set_ylabel("Median energy released (eV per atom)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=SOFT, alpha=0.85, lw=0),
               plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=SOFT, lw=0.9,
                             hatch="////")],
              ["DFT", "MatterSim"], frameon=False, fontsize=7.4, ncol=2,
              loc="lower left", bbox_to_anchor=(-0.015, 1.00), handlelength=1.3,
              handletextpad=0.4, columnspacing=1.0, borderpad=0.0)

    stamp(fig, [(axes[0], "a"), (axes[1], "b")])
    save(fig, "figSdft_e3_paired")


# ────────────────────────── E2: are the orderings that split the sites thermodynamically real
def si_e2_ordering():
    """Every ordering of every entry, measured against that entry's best one."""
    rows = _load("E2_ordering/ordering_energies.json")
    coll = _load("E2_ordering/collected.json")["records"]

    per_entry: dict[str, list[float]] = {}
    for r in coll:
        if r["status"] not in ("complete", "unconverged"):
            continue
        e = r["stage_results"].get("static", {}).get("energy_last_ev")
        if e is None:
            continue
        per_entry.setdefault(r["entry"], []).append(e / r["n_atoms"])
    keep = {r["entry"]: r for r in rows}

    fig, ax = plt.subplots(figsize=(W2, 8.8 * CM))
    fig.subplots_adjust(left=0.098, right=0.985, top=0.965, bottom=0.155)

    order = ([r["entry"] for r in rows if r["kind"] == "gnome"]
             + [r["entry"] for r in rows if r["kind"] == "experimental"])
    order.sort(key=lambda e: (keep[e]["kind"] != "gnome",
                              keep[e]["dE_disorder_ev_per_atom"]))
    FLOOR = 1e-6
    for i, ent in enumerate(order):
        vals = np.array(per_entry.get(ent, []))
        if not len(vals):
            continue
        rel = np.clip(vals - vals.min(), FLOOR, None)
        colr = PUR if keep[ent]["kind"] == "gnome" else L4C
        ax.scatter(np.full(len(rel), i), rel, s=7.5, color=colr, alpha=0.62, lw=0,
                   zorder=3)
        ax.plot([i - 0.34, i + 0.34],
                [keep[ent]["dE_disorder_ev_per_atom"]] * 2, color=colr, lw=1.1,
                zorder=4)
    n_g = sum(1 for r in rows if r["kind"] == "gnome")
    ax.axvline(n_g - 0.5, color="#C2C4C7", lw=0.7)
    # the line an entry has to clear for its ordering to survive room temperature
    kb300 = 8.617333262e-5 * 300
    ax.axhline(kb300, color=SOFT, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(-0.55, kb300 * 1.30, r"$k_\mathrm{B}T$ at 300 K", fontsize=7,
            color=SOFT, ha="left", va="bottom")
    ax.text((n_g - 1) / 2, 1.7, f"GNoME  (n = {n_g})", fontsize=8, color=PUR,
            ha="center", va="top")
    ax.text(n_g + (len(order) - n_g) / 2 - 0.5, 1.7,
            f"experimental controls  (n = {len(order) - n_g})", fontsize=8, color=L4C,
            ha="center", va="top")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.55, 2.6)
    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_xticks([])
    ax.set_xlabel("Merge group, ordered within each class by cost of disordering")
    ax.set_ylabel("DFT energy above best ordering (eV per atom)")
    save(fig, "figSdft_e2_ordering")


def main():
    si_e1_landscape()
    si_e4_bulk()
    si_e4_composition()
    si_e3_paired()
    si_e2_ordering()


if __name__ == "__main__":
    main()
