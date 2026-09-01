#!/usr/bin/env python3
"""Turn the collected VASP results into the pre-registered numbers, and say whether each
prediction in PREREG-DFT.md was met.

Runs locally after `collect.py`, because it needs numpy, scipy and spglib. Every estimator
and threshold below is quoted from PREREG-DFT.md rather than chosen here; this script only
evaluates them.

    python dft/analyze.py                      # all four, writes dft/RESULTS.md
    python dft/analyze.py --only E2

Nothing here silently drops a task. Tasks excluded by the pre-registered failure rules are
counted and reported alongside the numbers they were excluded from.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KB_EV = 8.617333262e-5           # eV per K

# thresholds, all quoted from PREREG-DFT.md section 2
E1_FLOOR_RHO = 0.735       # the L1 D1 floor; it lies between grid points, so interpolate
E1_FLOOR_RHO_STRICT = 0.804   # the L2-L4 D1 floor
E1_CEILING_RHO = 1.05
E1_COST_EV = 0.1           # the energy scale at which a short contact becomes chemical
E1_MIN_SPREAD_RATIO = 1.5  # how much tighter the reduced coordinate must be than angstroms
E1_RHO_STAR_RANGE = (0.70, 1.00)
E1B_MAX_RELATIVE_SHIFT = 0.25   # frozen-core error the hard potentials may reveal
E2_GNOME_T_MEDIAN_MAX = 300.0
E2_CONTROL_T_MEDIAN_MIN = 1000.0
E2_GNOME_FRACTION_MIN = 0.60
E2_CONTROL_FRACTION_MAX = 0.20
E3_MIN_SPEARMAN = 0.7
E3_S5_MIN_EXCESS_EV = 0.3
E4_MAX_SCREENED_HIGH_B = 2
E4_MIN_PRIORITY_CONFIRMED = 0.70
E4_MIN_PEARSON = 0.8
E4_TARGET_B_GPA = 400.0

EV_A3_TO_GPA = 160.21766208


def load(pkg: str, stage_b: bool = False) -> dict:
    p = HERE / pkg / ("stage_b" if stage_b else "") / "collected.json"
    if not p.exists():
        raise SystemExit(f"{p} not found — run collect.py first")
    return json.loads(p.read_text())


def usable(record: dict) -> bool:
    """Pre-registered rule: failed and incomplete tasks are excluded, unconverged kept."""
    return record["status"] in ("complete", "unconverged")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def birch_murnaghan(volumes: np.ndarray, energies: np.ndarray):
    """Third-order Birch-Murnaghan fit; returns (E0, V0, B0 in GPa, Bp, rms residual)."""
    from scipy.optimize import curve_fit

    def bm(v, e0, v0, b0, bp):
        eta = (v0 / v) ** (2.0 / 3.0)
        return e0 + 9.0 * v0 * b0 / 16.0 * (
            (eta - 1.0) ** 3 * bp + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))

    # a quadratic in V gives the starting point, which keeps the fit from wandering
    quad = np.polyfit(volumes, energies, 2)
    v0 = -quad[1] / (2 * quad[0])
    e0 = np.polyval(quad, v0)
    b0 = 2 * quad[0] * v0
    try:
        popt, _ = curve_fit(bm, volumes, energies, p0=[e0, v0, b0, 4.0], maxfev=20000)
    except Exception:
        return None
    resid = energies - bm(volumes, *popt)
    rms = float(np.sqrt((resid ** 2).mean()))
    e0, v0, b0, bp = (float(x) for x in popt)
    if not (v0 > 0 and b0 > 0 and volumes.min() < v0 < volumes.max()):
        return None
    return e0, v0, b0 * EV_A3_TO_GPA, bp, rms


def spacegroup_and_economy(contcar_text: str, symprec: float = 0.01):
    """Space-group number and distinct-site fraction of a relaxed cell, D7's quantity."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        st = Structure.from_str(contcar_text, fmt="poscar")
        prim = SpacegroupAnalyzer(st, symprec=symprec).get_primitive_standard_structure()
        ds = SpacegroupAnalyzer(prim, symprec=symprec).get_symmetry_dataset()
        eq = np.asarray(ds.equivalent_atoms if hasattr(ds, "equivalent_atoms")
                        else ds["equivalent_atoms"])
        num = int(ds.number if hasattr(ds, "number") else ds["number"])
        return num, float(len(np.unique(eq)) / len(prim))
    except Exception:
        return None, None


# ------------------------------------------------------------------ E1


def _e1_curves(pkg: str) -> tuple[list[dict], list]:
    """Excess-energy curves for one E1-style package, keyed by reduced contact."""
    from scipy.interpolate import PchipInterpolator
    data = load(pkg)
    sel = json.loads((HERE / pkg / "selection.json").read_text())
    grid = sel["rho_grid"]
    rows, excluded = [], []
    for r in data["records"]:
        if not usable(r):
            excluded.append((r["task"], r["status_reason"]))
            continue
        n = r["n_atoms"]
        e = {}
        for k, rho in enumerate(grid):
            st = r["stage_results"].get(f"v{k:02d}", {})
            if st.get("energy_last_ev") is not None:
                e[rho] = st["energy_last_ev"] / n
        if len(e) < len(grid):
            excluded.append((r["task"], f"only {len(e)} of {len(grid)} grid points"))
            continue
        emin = min(e.values())
        xs = np.array(sorted(e))
        ys = np.array([e[x] for x in xs]) - emin
        curve = PchipInterpolator(xs, ys)
        rho_min = float(xs[int(np.argmin(ys))])

        # rho*(tau): compressing from the minimum, where the cost first reaches tau
        fine = np.linspace(xs.min(), rho_min, 2000)
        vals = curve(fine)
        hits = np.flatnonzero(vals >= E1_COST_EV)
        rho_star = float(fine[hits[-1]]) if len(hits) else float("nan")

        radius_sum = r.get("radius_sum_a")
        rows.append({"task": r["task"], "source_id": r.get("source_id"),
                     "anion": r.get("anion"), "formula": r.get("formula"),
                     "potentials": r.get("potentials", "standard"),
                     "radius_sum_a": radius_sum,
                     "rho_at_minimum": rho_min,
                     "rho_star": rho_star,
                     "d_star_a": rho_star * radius_sum if radius_sum else None,
                     "excess_at_floor": float(curve(E1_FLOOR_RHO)),
                     "excess_at_strict_floor": float(curve(E1_FLOOR_RHO_STRICT)),
                     "excess_at_ceiling": float(curve(E1_CEILING_RHO)),
                     "curve": {str(k): float(v - emin) for k, v in e.items()}})
    return rows, excluded


def _rel_spread(values: np.ndarray) -> float:
    """Spread relative to the centre, robust to one outlier."""
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.median(values) == 0:
        return float("nan")
    q75, q25 = np.percentile(values, [75, 25])
    return float((q75 - q25) / np.median(values))


def analyse_e1(out: list[str]) -> list[dict]:
    rows, excluded = _e1_curves("E1_rho_curve")
    out.append("## E1 — the reduced-contact energy landscape\n")
    if not rows:
        out.append(f"No usable curves ({len(excluded)} excluded).\n")
        return []

    rho_star = np.array([r["rho_star"] for r in rows], float)
    d_star = np.array([r["d_star_a"] if r["d_star_a"] else np.nan for r in rows], float)
    s_rho, s_d = _rel_spread(rho_star), _rel_spread(d_star)
    if s_rho > 0:
        ratio = s_d / s_rho
    elif s_d > 0:
        ratio = float("inf")       # the reduced coordinate collapses the set exactly
    else:
        ratio = float("nan")       # both spreads vanish, so the comparison says nothing
    med_rho = float(np.nanmedian(rho_star))

    floor = np.array([r["excess_at_floor"] for r in rows])
    strict = np.array([r["excess_at_strict_floor"] for r in rows])
    ceil = np.array([r["excess_at_ceiling"] for r in rows])

    out.append(f"- curves fitted: {len(rows)}, excluded: {len(excluded)}")
    out.append(f"- reduced contact at which compression costs {E1_COST_EV} eV per atom: "
               f"median {med_rho:.3f}, range {np.nanmin(rho_star):.3f}-"
               f"{np.nanmax(rho_star):.3f}")
    out.append(f"- the same crossing in angstroms: median {np.nanmedian(d_star):.3f}, "
               f"range {np.nanmin(d_star):.3f}-{np.nanmax(d_star):.3f} A")
    out.append(f"- relative spread across chemistries: {s_rho:.3f} in the reduced "
               f"coordinate against {s_d:.3f} in angstroms, a factor of {ratio:.2f}")
    out.append(f"- energy excess: {np.median(floor):.3f} eV per atom at rho_c="
               f"{E1_FLOOR_RHO}, {np.median(strict):.3f} at {E1_FLOOR_RHO_STRICT}, "
               f"{np.median(ceil):.3f} at the D2 ceiling {E1_CEILING_RHO} (medians)\n")
    (HERE / "E1_rho_curve" / "curves.json").write_text(json.dumps(rows, indent=1) + "\n")

    checks = [
        {"experiment": "E1", "prediction":
         f"the reduced coordinate localises the {E1_COST_EV} eV per atom crossing at least "
         f"{E1_MIN_SPREAD_RATIO}x more tightly than angstroms",
         "value": f"{ratio:.2f}x", "met": bool(ratio >= E1_MIN_SPREAD_RATIO)},
        {"experiment": "E1", "prediction":
         f"the median crossing lies in {E1_RHO_STAR_RANGE}, bracketing the D1 floors",
         "value": f"{med_rho:.3f}",
         "met": bool(E1_RHO_STAR_RANGE[0] <= med_rho <= E1_RHO_STAR_RANGE[1])},
    ]

    # ---- E1b: does the frozen core drive any of this ---------------------------
    try:
        hard, hard_excluded = _e1_curves("E1b_paw_control")
    except SystemExit:
        out.append("E1b hard-potential control not collected yet.\n")
        return checks
    by_id = {r["source_id"]: r for r in rows}
    shifts, pairs = [], []
    for h in hard:
        base = by_id.get(h["source_id"])
        if not base or base["excess_at_floor"] <= 0:
            continue
        rel = abs(h["excess_at_floor"] - base["excess_at_floor"]) / base["excess_at_floor"]
        shifts.append(rel)
        pairs.append({"source_id": h["source_id"], "formula": h["formula"],
                      "standard_ev_per_atom": base["excess_at_floor"],
                      "hard_ev_per_atom": h["excess_at_floor"],
                      "relative_shift": rel,
                      "rho_star_standard": base["rho_star"], "rho_star_hard": h["rho_star"]})
    out.append("### E1b — hard-potential control\n")
    if shifts:
        med_shift = float(np.median(shifts))
        out.append(f"- compounds replicated: {len(pairs)}, excluded: {len(hard_excluded)}")
        out.append(f"- relative change in the excess at rho_c={E1_FLOOR_RHO} when the "
                   f"small-core potentials replace the standard ones: median "
                   f"{med_shift:.1%}, worst {max(shifts):.1%}\n")
        (HERE / "E1b_paw_control" / "paw_shift.json").write_text(
            json.dumps(pairs, indent=1) + "\n")
        checks.append({"experiment": "E1b", "prediction":
                       f"hard potentials move the excess at the floor by less than "
                       f"{E1B_MAX_RELATIVE_SHIFT:.0%}",
                       "value": f"{med_shift:.1%}",
                       "met": bool(med_shift < E1B_MAX_RELATIVE_SHIFT)})
    else:
        out.append("- no matched pairs; the control cannot be evaluated\n")
    return checks


# ------------------------------------------------------------------ E2


def analyse_e2(out: list[str]) -> list[dict]:
    data = load("E2_ordering")
    by_entry: dict[str, list[dict]] = {}
    excluded = []
    for r in data["records"]:
        if not usable(r):
            excluded.append((r["task"], r["status_reason"]))
            continue
        by_entry.setdefault(r["entry"], []).append(r)

    rows = []
    for entry, recs in sorted(by_entry.items()):
        native = [r for r in recs if r.get("is_released_ordering")]
        if not native:
            excluded.append((entry, "the released ordering did not finish"))
            continue
        # The pre-registration enumerates every ordering precisely so that the minimum is
        # not taken over a subset. A missing ordering can only raise that minimum, which
        # lowers the ordering energy and the order-disorder temperature with it - the
        # direction that favours the hypothesis. An incomplete entry is therefore dropped,
        # not analysed with what happens to have finished.
        expected = next((e["n_orderings"] for e in
                         json.loads((HERE / "E2_ordering" / "selection.json").read_text())["entries"]
                         if e["id"] == entry), None)
        if expected is not None and len(recs) < expected:
            excluded.append((entry, f"only {len(recs)} of {expected} orderings finished; "
                                    f"a partial minimum would bias the ordering energy down"))
            continue
        if len(recs) < 2:
            excluded.append((entry, "fewer than two orderings finished"))
            continue
        n = recs[0]["n_atoms"]
        energies = {r["ordering_index"]: r["stage_results"]["static"]["energy_last_ev"] / n
                    for r in recs if r["stage_results"].get("static", {}).get("energy_last_ev")}
        if native[0]["ordering_index"] not in energies or len(energies) < 2:
            excluded.append((entry, "missing static energies"))
            continue
        e_native = energies[native[0]["ordering_index"]]
        e_min = min(energies.values())
        d_e = e_native - e_min
        # The pre-registered dE is the released ordering's distance above the best one,
        # which measures whether the release picked the ground state, not whether the
        # compound is ordered. The energy that decides order against disorder is the cost
        # of leaving the ground state for a random configuration, approximated by the mean
        # over orderings minus the minimum. Both are reported; only the first enters the
        # pre-registered predictions.
        d_e_disorder = sum(energies.values()) / len(energies) - e_min
        spread = max(energies.values()) - e_min

        # configurational entropy per mixed site, from the merge group's composition
        sel = json.loads((HERE / "E2_ordering" / "selection.json").read_text())
        info = next(e for e in sel["entries"] if e["id"] == entry)
        ordering = next(o for o in info["orderings"] if o["is_released_ordering"])["ordering"]
        counts: dict[str, int] = {}
        for sym in ordering:
            counts[sym] = counts.get(sym, 0) + 1
        total = sum(counts.values())
        ds_per_site = -sum((c / total) * math.log(c / total) for c in counts.values())
        # entropy is per mixed site; energies are per atom, so scale to the same basis
        ds_per_atom = ds_per_site * total / n
        t_od = d_e / (KB_EV * ds_per_atom) if ds_per_atom > 0 else float("nan")
        t_dis = d_e_disorder / (KB_EV * ds_per_atom) if ds_per_atom > 0 else float("nan")

        sg_native, econ_native = spacegroup_and_economy(
            native[0]["stage_results"].get("relax_cell", {}).get("contcar", ""))
        # Ce and Eu keep f electrons in the valence, where PBE without U describes them
        # poorly; those entries are reported separately rather than quietly mixed in
        f_valence = bool({"Ce", "Eu"} & set(recs[0].get("species", [])))
        rows.append({"entry": entry, "kind": recs[0]["kind"],
                     "merge_class": recs[0]["merge_class"],
                     "n_orderings_done": len(energies),
                     "dE_order_ev_per_atom": d_e,
                     "dE_disorder_ev_per_atom": d_e_disorder,
                     "T_disorder_K": t_dis,
                     "dS_per_atom_kB": ds_per_atom,
                     "T_od_K": t_od,
                     "relaxed_spacegroup": sg_native,
                     "relaxed_site_economy": econ_native,
                     "f_electrons_in_valence": f_valence,
                     "spread_ev_per_atom": spread})

    out.append("## E2 — is GNoME's low-symmetry excess thermodynamically real\n")
    if not rows:
        out.append(f"No usable entries ({len(excluded)} excluded).\n")
        return []
    g = [r for r in rows if r["kind"] == "gnome"]
    c = [r for r in rows if r["kind"] == "experimental"]
    gt = np.array([r["T_od_K"] for r in g]) if g else np.array([])
    ct = np.array([r["T_od_K"] for r in c]) if c else np.array([])
    g_frac = float((gt < E2_GNOME_T_MEDIAN_MAX).mean()) if len(gt) else float("nan")
    c_frac = float((ct < E2_GNOME_T_MEDIAN_MAX).mean()) if len(ct) else float("nan")
    out.append(f"- entries analysed: {len(g)} GNoME, {len(c)} experimental controls; "
               f"excluded: {len(excluded)}")
    for ident, why in excluded[:12]:
        out.append(f"    - {ident}: {why}")
    if len(gt):
        out.append(f"- GNoME ordering energy: median {np.median([r['dE_order_ev_per_atom'] for r in g]):.4f} "
                   f"eV per atom; order-disorder temperature: median {np.median(gt):.0f} K")
    if len(ct):
        out.append(f"- control ordering energy: median {np.median([r['dE_order_ev_per_atom'] for r in c]):.4f} "
                   f"eV per atom; order-disorder temperature: median {np.median(ct):.0f} K")
    out.append(f"- below {E2_GNOME_T_MEDIAN_MAX:.0f} K: {g_frac:.1%} of GNoME entries, "
               f"{c_frac:.1%} of controls")
    gf = [r for r in g if r["f_electrons_in_valence"]]
    if gf:
        gt_nof = np.array([r["T_od_K"] for r in g if not r["f_electrons_in_valence"]])
        out.append(f"- {len(gf)} GNoME entries contain Ce or Eu, whose f electrons PBE "
                   f"treats poorly; excluding them the median is "
                   f"{np.median(gt_nof):.0f} K over {len(gt_nof)} entries")
    out.append("")
    # reported beside the pre-registered numbers, not in place of them
    gd = np.array([r["dE_disorder_ev_per_atom"] for r in g]) if g else np.array([])
    cd = np.array([r["dE_disorder_ev_per_atom"] for r in c]) if c else np.array([])
    gt2 = np.array([r["T_disorder_K"] for r in g]) if g else np.array([])
    ct2 = np.array([r["T_disorder_K"] for r in c]) if c else np.array([])
    if len(gd) and len(cd):
        out.append("")
        out.append("Reported alongside, not part of the predictions: the pre-registered dE "
                   "measures how far the released ordering sits above the best one, which "
                   "says whether the release picked the ground state rather than whether "
                   "the compound is ordered. The cost of disordering, mean over orderings "
                   "minus the minimum, is the quantity that decides that:")
        out.append(f"- disordering energy: GNoME median {np.median(gd):.5f} eV per atom, "
                   f"controls {np.median(cd):.5f}, a factor of "
                   f"{np.median(cd) / max(np.median(gd), 1e-12):.0f}")
        out.append(f"- the temperature that implies: GNoME median {np.median(gt2):.0f} K, "
                   f"controls {np.median(ct2):.0f} K")
        out.append(f"- below {E2_GNOME_T_MEDIAN_MAX:.0f} K on that measure: "
                   f"{int((gt2 < E2_GNOME_T_MEDIAN_MAX).sum())} of {len(gt2)} GNoME, "
                   f"{int((ct2 < E2_GNOME_T_MEDIAN_MAX).sum())} of {len(ct2)} controls\n")

    checks = [
        {"experiment": "E2", "prediction":
         f"GNoME median T_od below {E2_GNOME_T_MEDIAN_MAX:.0f} K",
         "value": f"{np.median(gt):.0f} K" if len(gt) else "n/a",
         "met": bool(len(gt) and np.median(gt) < E2_GNOME_T_MEDIAN_MAX)},
        {"experiment": "E2", "prediction":
         f"control median T_od above {E2_CONTROL_T_MEDIAN_MIN:.0f} K",
         "value": f"{np.median(ct):.0f} K" if len(ct) else "n/a",
         "met": bool(len(ct) and np.median(ct) > E2_CONTROL_T_MEDIAN_MIN)},
        {"experiment": "E2", "prediction":
         f"at least {E2_GNOME_FRACTION_MIN:.0%} of GNoME entries below "
         f"{E2_GNOME_T_MEDIAN_MAX:.0f} K",
         "value": f"{g_frac:.1%}", "met": bool(g_frac >= E2_GNOME_FRACTION_MIN)},
        {"experiment": "E2", "prediction":
         f"at most {E2_CONTROL_FRACTION_MAX:.0%} of controls below "
         f"{E2_GNOME_T_MEDIAN_MAX:.0f} K",
         "value": f"{c_frac:.1%}", "met": bool(c_frac <= E2_CONTROL_FRACTION_MAX)},
    ]
    (HERE / "E2_ordering" / "ordering_energies.json").write_text(
        json.dumps(rows, indent=1) + "\n")
    return checks


# ------------------------------------------------------------------ E3


def analyse_e3(out: list[str]) -> list[dict]:
    data = load("E3_crosscheck")
    ms_path = HERE / "E3_crosscheck" / "mattersim_reference.json"
    ms = {}
    if ms_path.exists():
        ms = {r["task"]: r for r in json.loads(ms_path.read_text())["records"]}

    rows, excluded = [], []
    for r in data["records"]:
        if not usable(r):
            excluded.append((r["task"], r["status_reason"]))
            continue
        ions = r["stage_results"].get("relax_ions_fixed_cell", {})
        if ions.get("relax_release_ev_per_atom") is None:
            excluded.append((r["task"], "no frozen-cell relaxation energy"))
            continue
        cell = r["stage_results"].get("relax_cell", {})
        rows.append({
            "task": r["task"], "parent": r.get("parent"), "kind": r.get("kind"),
            "variant": r.get("variant"), "formula": r.get("formula"),
            # measured on the same degrees of freedom MatterSim uses
            "dft_release_ions_ev_per_atom": ions["relax_release_ev_per_atom"],
            "dft_release_full_ev_per_atom": (
                (ions["energy_first_ev"] - cell["energy_last_ev"]) / r["n_atoms"]
                if cell.get("energy_last_ev") is not None else None),
            "mattersim_release_ev_per_atom": ms.get(r["task"], {}).get(
                "release_ev_per_atom"),
        })

    out.append("## E3 — do the controlled damages read as damage to DFT\n")
    if not rows:
        out.append(f"No usable cells ({len(excluded)} excluded).\n")
        return []
    out.append(f"- cells analysed: {len(rows)}, excluded: {len(excluded)}")
    out.append("")
    out.append("| variant | n | DFT release, median (eV per atom) | MatterSim, median |")
    out.append("|---|---|---|---|")
    med = {}
    for v in ("P0", "S1", "S2", "S3", "S4", "S5"):
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        d = np.array([r["dft_release_ions_ev_per_atom"] for r in sub])
        m = np.array([r["mattersim_release_ev_per_atom"] for r in sub
                      if r["mattersim_release_ev_per_atom"] is not None])
        med[v] = (float(np.median(d)), float(np.median(m)) if len(m) else float("nan"))
        out.append(f"| {v} | {len(sub)} | {med[v][0]:.4f} | "
                   f"{'—' if math.isnan(med[v][1]) else f'{med[v][1]:.4f}'} |")
    out.append("")

    paired = [(r["dft_release_ions_ev_per_atom"], r["mattersim_release_ev_per_atom"])
              for r in rows if r["mattersim_release_ev_per_atom"] is not None]
    rho = spearman(np.array([p[0] for p in paired]),
                   np.array([p[1] for p in paired])) if len(paired) > 2 else float("nan")
    if paired:
        out.append(f"- DFT vs MatterSim rank correlation over {len(paired)} paired cells: "
                   f"{rho:.3f}")
    else:
        out.append("- MatterSim reference absent; run `mattersim_reference.py` to pair them")

    nan = float("nan")
    s5 = med.get("S5", (nan, nan))
    p0 = med.get("P0", (nan, nan))
    # persist the paired values: the correlation is the headline, but the per-cell pairs are
    # what a figure needs, and recomputing them from two collected files is needless work
    (HERE / "E3_crosscheck" / "paired_energies.json").write_text(
        json.dumps(rows, indent=1) + "\n")

    s5_excess = s5[0] - p0[0]
    ratio_dft = s5[0] / p0[0] if p0[0] else nan
    ratio_ms = s5[1] / p0[1] if p0[1] and not math.isnan(p0[1]) else nan
    out.append(f"- S5 cation-anion exchange exceeds the undamaged parents by "
               f"{s5_excess:.4f} eV per atom\n")
    return [
        {"experiment": "E3", "prediction":
         f"DFT and MatterSim rank correlation at least {E3_MIN_SPEARMAN}",
         "value": f"{rho:.3f}", "met": bool(rho >= E3_MIN_SPEARMAN)},
        {"experiment": "E3", "prediction":
         f"S5 median exceeds the parent median by {E3_S5_MIN_EXCESS_EV} eV per atom",
         "value": f"{s5_excess:.4f}", "met": bool(s5_excess >= E3_S5_MIN_EXCESS_EV)},
        {"experiment": "E3", "prediction":
         "the S5-to-parent ratio is larger under DFT than under MatterSim",
         "value": f"DFT {ratio_dft:.2f} vs MatterSim {ratio_ms:.2f}",
         "met": bool(ratio_dft > ratio_ms)},
    ]


# ------------------------------------------------------------------ E4


def analyse_e4(out: list[str]) -> list[dict]:
    stage_b = load("E4_design", stage_b=True)
    by_parent: dict[str, list[dict]] = {}
    excluded = []
    for r in stage_b["records"]:
        if not usable(r):
            excluded.append((r["task"], r["status_reason"]))
            continue
        by_parent.setdefault(r["parent_task"], []).append(r)

    rows = []
    for parent, recs in sorted(by_parent.items()):
        pts = []
        for r in recs:
            st = r["stage_results"].get("static", {})
            cell = st.get("final_cell") or r["stage_results"].get(
                "relax_ions", {}).get("final_cell")
            if st.get("energy_last_ev") is None or not cell:
                continue
            pts.append((cell["volume_a3"], st["energy_last_ev"]))
        if len(pts) < 4:
            excluded.append((parent, f"only {len(pts)} energy-volume points"))
            continue
        pts.sort()
        v = np.array([p[0] for p in pts])
        e = np.array([p[1] for p in pts])
        fit = birch_murnaghan(v, e)
        if fit is None:
            excluded.append((parent, "Birch-Murnaghan fit did not converge"))
            continue
        e0, v0, b0, bp, rms = fit
        rows.append({"parent_task": parent, "candidate_id": recs[0].get("candidate_id"),
                     "role": recs[0].get("role"), "formula": recs[0].get("formula"),
                     "n_points": len(pts),
                     "dft_bulk_modulus_gpa": b0, "bp": bp, "v0_a3": v0,
                     "fit_rms_ev": rms,
                     "uma_bulk_modulus_gpa": recs[0].get("uma_bulk_modulus_gpa"),
                     "pss": recs[0].get("pss")})

    out.append("## E4 — does the inverse-design screen survive first principles\n")
    if not rows:
        out.append(f"No usable candidates ({len(excluded)} excluded).\n")
        return []
    screened = [r for r in rows if r["role"] == "screened"]
    priority = [r for r in rows if r["role"] == "priority"]
    n_screened_high = sum(1 for r in screened
                          if r["dft_bulk_modulus_gpa"] >= E4_TARGET_B_GPA)
    frac_priority = (sum(1 for r in priority
                         if r["dft_bulk_modulus_gpa"] >= E4_TARGET_B_GPA) / len(priority)
                     if priority else float("nan"))
    uma = np.array([r["uma_bulk_modulus_gpa"] for r in rows])
    dft = np.array([r["dft_bulk_modulus_gpa"] for r in rows])
    r_p = pearson(uma, dft) if len(rows) > 2 else float("nan")

    out.append(f"- candidates fitted: {len(rows)} "
               f"({len(screened)} screened, {len(priority)} priority, "
               f"{len(rows) - len(screened) - len(priority)} control); "
               f"excluded: {len(excluded)}")
    out.append(f"- screened candidates at or above {E4_TARGET_B_GPA:.0f} GPa under DFT: "
               f"{n_screened_high} of {len(screened)}")
    out.append(f"- priority candidates confirmed at or above {E4_TARGET_B_GPA:.0f} GPa: "
               f"{frac_priority:.1%}")
    out.append(f"- UMA vs DFT bulk modulus correlation: r = {r_p:.3f}")
    out.append(f"- DFT bulk modulus: median {np.median(dft):.0f} GPa, "
               f"max {dft.max():.0f} GPa\n")

    # Reported alongside, not in place of, the pre-registered numbers. The 400 GPa target
    # was set on the proxy's scale. If the proxy carries a systematic offset then an
    # absolute threshold carried over to DFT tests its calibration rather than the
    # screen's selection, and the second prediction can fail while every decision the
    # screen made is upheld. The selection is therefore also tested in ways that do not
    # depend on where the line falls.
    rho_s = spearman(uma, dft) if len(rows) > 2 else float("nan")
    bias = float(np.median(dft / uma))
    control = [r for r in rows if r["role"] == "control"]
    retained = [r for r in rows if r["role"] != "screened"]   # priority and control both
    out.append("Reported alongside, not part of the predictions:")
    out.append(f"- the proxy runs high: median DFT/UMA ratio {bias:.3f}, so the 400 GPa "
               f"target sits at {400 * bias:.0f} GPa on the DFT scale")
    out.append(f"- rank correlation (Spearman) {rho_s:.3f}, against Pearson {r_p:.3f}")
    if screened and priority:
        # the right statistic when the two distributions overlap: how often a candidate the
        # screen kept for its property beats one the screen removed
        wins = sum(1 for a in priority for b in screened
                   if a["dft_bulk_modulus_gpa"] > b["dft_bulk_modulus_gpa"])
        ties = sum(1 for a in priority for b in screened
                   if a["dft_bulk_modulus_gpa"] == b["dft_bulk_modulus_gpa"])
        auc = (wins + 0.5 * ties) / (len(priority) * len(screened))
        out.append(f"- a priority candidate outranks a screened one {auc:.3f} of the time "
                   f"(0.5 would be no signal)")
    if screened:
        rescaled = 400.0 * bias
        n_lost = sum(1 for r in screened if r["dft_bulk_modulus_gpa"] >= rescaled)
        n_kept_high = sum(1 for r in retained if r["dft_bulk_modulus_gpa"] >= rescaled)
        out.append(f"- at that rescaled {rescaled:.0f} GPa: {n_lost} of {len(screened)} "
                   f"screened candidates reach it, against {n_kept_high} of {len(retained)} "
                   f"the screen retained")
    # threshold-free: take as many candidates by DFT as the proxy calls high-property, and
    # ask how many of them the synthesizability screen had already thrown away
    n_high_uma = sum(1 for r in rows if (r["uma_bulk_modulus_gpa"] or 0) >= E4_TARGET_B_GPA)
    if n_high_uma:
        top = sorted(rows, key=lambda r: -r["dft_bulk_modulus_gpa"])[:n_high_uma]
        lost = [r for r in top if r["role"] == "screened"]
        out.append(f"- of the {n_high_uma} highest bulk moduli under DFT, {len(lost)} had "
                   f"been removed by the screen: it retains "
                   f"{100 * (1 - len(lost) / n_high_uma):.1f}%")
        for r in sorted(lost, key=lambda x: -x["dft_bulk_modulus_gpa"])[:3]:
            # a missing proxy value or PSS must not take down the whole analysis at its
            # last step, so they are formatted defensively rather than assumed present
            up = r.get("uma_bulk_modulus_gpa")
            ps = r.get("pss")
            out.append(f"    - lost: {r['candidate_id']} {r['formula']}, DFT "
                       f"{r['dft_bulk_modulus_gpa']:.0f} GPa, proxy "
                       f"{f'{up:.0f} GPa' if up is not None else 'absent'}, PSS "
                       f"{f'{ps:.3f}' if ps is not None else 'absent'}")
    for g in ("priority", "control", "screened"):
        b = [r["dft_bulk_modulus_gpa"] for r in rows if r["role"] == g]
        if b:
            out.append(f"- {g}: n = {len(b)}, median {np.median(b):.0f} GPa, "
                       f"IQR {np.percentile(b, 25):.0f}-{np.percentile(b, 75):.0f}")
    out.append("")
    out.append("The candidates chosen for E4 deliberately over-sample high proxy bulk "
               "moduli, so none of these percentages carry over to the full candidate "
               "pool; they describe this set only.\n")

    (HERE / "E4_design" / "bulk_moduli.json").write_text(json.dumps(rows, indent=1) + "\n")
    return [
        {"experiment": "E4", "prediction":
         f"at most {E4_MAX_SCREENED_HIGH_B} screened candidates reach "
         f"{E4_TARGET_B_GPA:.0f} GPa",
         "value": str(n_screened_high), "met": n_screened_high <= E4_MAX_SCREENED_HIGH_B},
        {"experiment": "E4", "prediction":
         f"at least {E4_MIN_PRIORITY_CONFIRMED:.0%} of priority candidates confirmed",
         "value": f"{frac_priority:.1%}",
         "met": bool(frac_priority >= E4_MIN_PRIORITY_CONFIRMED)},
        {"experiment": "E4", "prediction":
         f"UMA and DFT bulk moduli correlate with r at least {E4_MIN_PEARSON}",
         "value": f"{r_p:.3f}", "met": bool(r_p >= E4_MIN_PEARSON)},
    ]


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="E1,E2,E3,E4")   # E1 covers E1b
    ap.add_argument("--report", default=str(HERE / "RESULTS.md"))
    a = ap.parse_args()
    wanted = [w.strip() for w in a.only.split(",") if w.strip()]

    out: list[str] = ["# First-principles results (E1-E4)", "",
                      "Estimators and thresholds are quoted from `PREREG-DFT.md`; this "
                      "report only evaluates them.", ""]
    checks: list[dict] = []
    runners = {"E1": analyse_e1, "E2": analyse_e2, "E3": analyse_e3, "E4": analyse_e4}
    for key in wanted:
        try:
            checks += runners[key](out) or []
        except SystemExit as exc:
            out.append(f"## {key}\n\nNot analysed: {exc}\n")

    summary = ["## Pre-registered predictions", "",
               "| experiment | prediction | measured | met |", "|---|---|---|---|"]
    for c in checks:
        summary.append(f"| {c['experiment']} | {c['prediction']} | {c['value']} | "
                       f"{'yes' if c['met'] else '**no**'} |")
    summary.append("")
    n_met = sum(1 for c in checks if c["met"])
    summary.append(f"{n_met} of {len(checks)} predictions met.")
    summary.append("")

    Path(a.report).write_text("\n".join(out[:4] + summary + out[4:]) + "\n")
    print("\n".join(summary))
    print(f"-> {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
