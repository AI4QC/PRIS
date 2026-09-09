# -*- coding: utf-8 -*-
"""Table S1: five Pauling rules x 3 neighbour algorithms x 2 symprec x 2 oxidation-state
sources = 60 numbers.

Called by `reproduce_george.py --stage table`. The inputs are the four `george_*.parquet`
files from the compute stage.

**Granularity declaration (PREREG section 4.3 requires it explicitly; George 2020 mixes
granularities)**
| rule | granularity | does symprec matter |
|---|---|---|
| 1 radius ratio | `orbit` (cation sites de-duplicated by crystallographic orbit), `site` as an alternative | **yes** (the orbit definition) |
| 2 electrostatic valence | `orbit` (**anion** sites), `site` as an alternative | **yes** |
| 3 connection type | `pair` (George's original convention); the symprec column de-duplicates to `orbit-pair` | **yes** (only under the orbit-pair convention) |
| 4 adjacent polyhedra | `structure` | **no**; the two columns are necessarily identical, and are labelled as such |
| 5 parsimony | `structure` | **no**, as above |
| 2-5 conjunction | `structure` | no |

symprec has no structural effect on rules 4 and 5 (they do not count by site), so 24 of the
60 cells are **identical by construction**. That is not copy-paste but a genuine consequence
of the definitions, and the report has to say so.
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd

from pauling_radii import univalent_radius, predict_cn

FEAT = os.environ.get("PRIS_FEATURES", "features/")
ALGOS = ("chemenv", "crystalnn", "brunner")
OXS = ("cif", "guess")
SYMPRECS = (("s001", 0.01), ("s01", 0.1))
EPS = 0.01           # George:|Σs − 2| ≤ 0.01
R_ANION = 1.76       # Pauling univalent radius of O2- (pauling_radii.PAULING_UNIVALENT)


def wilson(k, n, z=1.959963985):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def cell(k, n):
    p, lo, hi = wilson(k, n)
    return dict(pct=p, lo=lo, hi=hi, k=int(k), n=int(n))


# ================================================================ loading
def load(smoke=False):
    suf = "_smoke" if smoke else ""
    st = pd.read_parquet(f"{FEAT}/george_struct{suf}.parquet")
    si = pd.read_parquet(f"{FEAT}/george_site{suf}.parquet")
    an = pd.read_parquet(f"{FEAT}/george_anion{suf}.parquet")
    pr = pd.read_parquet(f"{FEAT}/george_pair{suf}.parquet")
    st = st[st.status == "ok"].copy()
    ok = set(st.source_id)
    return st, si[si.source_id.isin(ok)].copy(), an[an.source_id.isin(ok)].copy(), \
        pr[pr.source_id.isin(ok)].copy()


# ================================================================ rule 1
def rule1_sites(si, tier="published"):
    """Attach Pauling univalent radii and a predicted CN to the cation sites.
    tier='published' is the main statistical convention."""
    allow_ext = (tier == "extended")
    key = si[["element", "ox_state"]].copy()
    key["oxi"] = key.ox_state.round().astype("Int64")
    uniq = key.drop_duplicates().dropna(subset=["oxi"])
    rmap, tmap = {}, {}
    for el, _, o in uniq.itertuples(index=False):
        r, t = univalent_radius(el, o, allow_extended=allow_ext)
        rmap[(el, int(o))] = r
        tmap[(el, int(o))] = t
    k = list(zip(si.element, si.ox_state.round()))
    si = si.copy()
    si["r_cat"] = [rmap.get((e, int(o)), None) if o == o else None for e, o in k]
    si["r_tier"] = [tmap.get((e, int(o)), None) if o == o else None for e, o in k]
    si["ratio"] = si.r_cat / R_ANION
    si["cn_pred"] = [predict_cn(x) for x in si.ratio]
    return si


# ================================================================ rules 4 / 5 (structure level)
def struct_rules(si, pr, algo):
    """Return the rule4/rule5 booleans for each structure. granularity=structure.

    Rule 4 (George's original convention): V = the highest oxidation state among the
    structure's cation sites, C = the smallest CN; A = {ox==V and cn==C}; a connected pair
    within A is a violation. Applies only when there are at least 2 cation species.
    Two decompositions are also given: `ox` uses only ox==V, `cn` uses only cn==C (George
    shows only the latter agrees with the data).
    Rule 5: each (element, oxidation state) species has only one CN within the structure (the
    original Fig. 5b, "only coordination numbers are considered").
    """
    cnc = f"cn_{algo}"
    s = si[["source_id", "element", "ox_state", cnc]].dropna(subset=[cnc]).copy()
    s = s[s.ox_state.notna()]
    s["oxr"] = s.ox_state.round(3)
    g = s.groupby("source_id")
    agg = pd.DataFrame({
        "V": g.oxr.max(), "C": g[cnc].min(),
        "n_species": g.apply(lambda d: d.groupby(["element", "oxr"]).ngroups, include_groups=False),
        # rule 5: any species showing more than one CN is a violation
        "r5_ok": g.apply(lambda d: d.groupby(["element", "oxr"])[cnc].nunique().max() == 1,
                         include_groups=False),
        "n_cat": g.size(),
    })
    p = pr[pr.nn_algo == algo][["source_id", "ox_i", "ox_j", "cn_i", "cn_j"]].copy()
    p["oxi"] = p.ox_i.round(3)
    p["oxj"] = p.ox_j.round(3)
    p = p.join(agg[["V", "C"]], on="source_id")
    hit_g = p[(p.oxi == p.V) & (p.oxj == p.V) & (p.cn_i == p.C) & (p.cn_j == p.C)]
    hit_o = p[(p.oxi == p.V) & (p.oxj == p.V)]
    hit_c = p[(p.cn_i == p.C) & (p.cn_j == p.C)]
    agg["viol_george"] = agg.index.isin(hit_g.source_id)
    agg["viol_ox"] = agg.index.isin(hit_o.source_id)
    agg["viol_cn"] = agg.index.isin(hit_c.source_id)
    multi = agg.n_species >= 2                     # the original: "In a crystal containing different cations"
    for tag in ("george", "ox", "cn"):
        agg[f"r4_{tag}_ok"] = ~(agg[f"viol_{tag}"] & multi)
    return agg


# ================================================================ main flow
def stage_table(smoke=False):
    st, si, an, pr = load(smoke)
    n_all = len(st)
    print("=" * 100)
    print(f"domain: provenance.oxide_strict, {n_all} entries computed successfully "
          f"(the anion is always O)")
    print("differences from George 2020: ICSD+COD (no MP) / experimental cells, unrelaxed / "
          "ordered structures only / single anion O /")
    print("                              oxidation states from cif|guess only (BVAnalyzer "
          "excluded wholesale, PREREG section 5)")
    print(st.ox_source.value_counts(dropna=False).to_string())
    print(f"cif_all_zero (native ICSD oxidation states all 0, downgraded) = {int(st.cif_all_zero.sum())} "
          f"({st.cif_all_zero.mean():.2%})")
    print("=" * 100)

    si = si.merge(st[["source_id", "ox_source"]].rename(columns={"ox_source": "oxs"}),
                  on="source_id", how="left")
    an = an.merge(st[["source_id", "ox_source"]].rename(columns={"ox_source": "oxs"}),
                  on="source_id", how="left")
    pr = pr.merge(st[["source_id", "ox_source"]], on="source_id", how="left")

    si1 = rule1_sites(si, "published")
    si1x = rule1_sites(si, "extended")
    cov = si1.r_cat.notna().mean()
    print(f"[rule 1] Pauling univalent radius coverage (published): {cov:.1%} of cation sites; "
          f"{si1x.r_cat.notna().mean():.1%} with the extrapolation")
    miss = (si1[si1.r_cat.isna()].groupby(["element"]).size().sort_values(ascending=False).head(12))
    print("         top 12 uncovered elements:", dict(miss))

    # ---- structure-level rules 4/5 (computed once per algorithm, independent of symprec)
    sr = {a: struct_rules(si, pr, a) for a in ALGOS}

    rows = []

    def add(rule, algo, sp, oxs, k, n, gran, note=""):
        c = cell(k, n)
        c.update(rule=rule, algo=algo, symprec=sp, ox_source=oxs, granularity=gran, note=note)
        rows.append(c)

    for oxs in OXS:
        sid_ox = set(st.loc[st.ox_source == oxs, "source_id"])
        for algo in ALGOS:
            cnc = f"cn_{algo}"
            sgc = f"sigma_{algo}"
            ncc = f"ncat_{algo}"
            agg = sr[algo]
            aggo = agg[agg.index.isin(sid_ox)]
            prx = pr[(pr.nn_algo == algo) & (pr.ox_source == oxs)]
            for sp, _v in SYMPRECS:
                ob = f"orbit_{sp}"
                # ---- rule 1: orbit granularity
                d = si1[(si1.oxs == oxs) & si1.r_cat.notna() & si1[cnc].notna() & si1[ob].notna()]
                d = d.drop_duplicates(subset=["source_id", ob])
                # main convention = judge only on CN values for which the hard-sphere stability
                # window is defined (George's own wording; see the comment in the cmp block
                # below). The strict all-CN version is kept as a separate row,
                # `R1_radius_ratio_allCN`, for audit.
                dw_ = d[d[cnc].isin({2, 3, 4, 6, 8, 12})]
                add("R1_radius_ratio", algo, sp, oxs,
                    (dw_.cn_pred == dw_[cnc]).sum(), len(dw_), "orbit")
                add("R1_radius_ratio_allCN", algo, sp, oxs,
                    (d.cn_pred == d[cnc]).sum(), len(d), "orbit")
                # ---- rule 2: anion orbit granularity
                e = an[(an.oxs == oxs) & (an[ncc] > 0) & an[sgc].notna() & an[ob].notna()]
                e = e.drop_duplicates(subset=["source_id", ob])
                add("R2_electrostatic_valence", algo, sp, oxs,
                    ((e[sgc] - 2.0).abs() <= EPS).sum(), len(e), "orbit")
                # ---- rule 3: de-duplicated to orbit-pair (the original pair convention is reported separately)
                q = prx.dropna(subset=[f"orbit_i_{sp}", f"orbit_j_{sp}"]).copy()
                a1 = np.minimum(q[f"orbit_i_{sp}"], q[f"orbit_j_{sp}"])
                a2 = np.maximum(q[f"orbit_i_{sp}"], q[f"orbit_j_{sp}"])
                q = q.assign(_k1=a1, _k2=a2).drop_duplicates(
                    subset=["source_id", "_k1", "_k2", "n_shared"])
                add("R3_corner_share", algo, sp, oxs,
                    (q["mode"] == "corner").sum(), len(q), "orbit-pair")
                # ---- rules 4 / 5: structure granularity, independent of symprec (the two columns are identical by construction)
                add("R4_contiguous_polyhedra", algo, sp, oxs,
                    int(aggo.r4_george_ok.sum()), len(aggo), "structure", "symprec-invariant")
                add("R5_parsimony", algo, sp, oxs,
                    int(aggo.r5_ok.sum()), len(aggo), "structure", "symprec-invariant")

    tab = pd.DataFrame(rows)
    out = f"{FEAT}/george_tableS1{'_smoke' if smoke else ''}.csv"
    tab.to_csv(out, index=False)

    # ================================================================ print Table S1
    print("\n" + "=" * 100)
    print("Table S1  satisfaction of the five rules, % [Wilson 95% CI]  --  "
          "3 neighbour algorithms x 2 symprec x 2 oxidation-state sources = 60 cells")
    print("=" * 100)
    for rule in ["R1_radius_ratio", "R2_electrostatic_valence", "R3_corner_share",
                 "R4_contiguous_polyhedra", "R5_parsimony"]:
        sub = tab[tab.rule == rule]
        gran = sub.granularity.iloc[0]
        print(f"\n--- {rule}   granularity={gran}"
              + ("   [symprec has no effect by construction]" if sub.note.iloc[0] else ""))
        piv = sub.pivot_table(index=["ox_source", "symprec"], columns="algo",
                              values="pct", aggfunc="first")[list(ALGOS)]
        nn = sub.pivot_table(index=["ox_source", "symprec"], columns="algo",
                             values="n", aggfunc="first")[list(ALGOS)]
        for idx in piv.index:
            vals = piv.loc[idx]
            spread = vals.max() - vals.min()
            cis = {a: sub[(sub.ox_source == idx[0]) & (sub.symprec == idx[1]) & (sub.algo == a)]
                   for a in ALGOS}
            s = "  ".join(f"{a[:4]}={vals[a]:5.1f}[{cis[a].lo.iloc[0]:.1f},{cis[a].hi.iloc[0]:.1f}]"
                          for a in ALGOS)
            print(f"  {idx[0]:5s} {idx[1]:5s}  {s}   n={int(nn.loc[idx].max()):>7d}"
                  f"   G6 spread={spread:5.2f} pt {'OK' if spread < 3 else '**FAIL**'}")

    # ================================================================ rule-by-rule comparison with George (main convention)
    print("\n" + "=" * 100)
    print("Rule-by-rule comparison with George 2020 (main convention: all ox_source in "
          "{cif,guess} pooled, ChemEnv, symprec=0.01)")
    print("=" * 100)
    sid_main = set(st.loc[st.ox_source.isin(OXS), "source_id"])
    cmp_rows = []

    d = si1[si1.oxs.isin(OXS) & si1.r_cat.notna() & si1.cn_chemenv.notna()
            & si1.orbit_s001.notna()].drop_duplicates(subset=["source_id", "orbit_s001"])
    cmp_rows.append(("rule 1 radius ratio (published radii)", 66.0, *wilson((d.cn_pred == d.cn_chemenv).sum(), len(d)), "orbit"))
    dx = si1x[si1x.oxs.isin(OXS) & si1x.r_cat.notna() & si1x.cn_chemenv.notna()
              & si1x.orbit_s001.notna()].drop_duplicates(subset=["source_id", "orbit_s001"])
    cmp_rows.append(("rule 1 radius ratio (with extrapolated radii, sensitivity)", 66.0, *wilson((dx.cn_pred == dx.cn_chemenv).sum(), len(dx)), "orbit"))
    ds = si1[si1.oxs.isin(OXS) & si1.r_cat.notna() & si1.cn_chemenv.notna()]
    cmp_rows.append(("rule 1 (site granularity, no orbit de-duplication)", 66.0, *wilson((ds.cn_pred == ds.cn_chemenv).sum(), len(ds)), "site"))
    cmp_rows.append(("rule 1 (tolerant version, |CNpred-CNobs|<=1)", None,
                     *wilson(((d.cn_pred - d.cn_chemenv).abs() <= 1).sum(), len(d)), "orbit"))
    # George states rule 1 as "A coordination environment is stable only if the radius ratio
    # falls within the geometrically derived stability window **of this environment**".
    # The hard-sphere stability window is defined only for CN in {2,3,4,6,8,12} (linear,
    # trigonal, tetrahedral, octahedral, cubic, cuboctahedral); sites observed at
    # CN = 5/7/9/10/11 have **no testable window at all**, and on the original wording
    # "tested local environments" they should be excluded. This layer is the one most
    # comparable with the 66%.
    WIN = {2, 3, 4, 6, 8, 12}
    dw = d[d.cn_chemenv.isin(WIN)]
    cmp_rows.append(("rule 1 (only CN in {2,3,4,6,8,12}, where the hard-sphere window is defined)", 66.0,
                     *wilson((dw.cn_pred == dw.cn_chemenv).sum(), len(dw)), "orbit"))
    dwx = dx[dx.cn_chemenv.isin(WIN)]
    cmp_rows.append(("rule 1 (hard-sphere window + extrapolated radii)", 66.0,
                     *wilson((dwx.cn_pred == dwx.cn_chemenv).sum(), len(dwx)), "orbit"))

    e = an[an.oxs.isin(OXS) & (an.ncat_chemenv > 0) & an.sigma_chemenv.notna()
           & an.orbit_s001.notna()].drop_duplicates(subset=["source_id", "orbit_s001"])
    cmp_rows.append(("rule 2, O sites with |Sum s - 2| <= 0.01", 20.0, *wilson(((e.sigma_chemenv - 2).abs() <= EPS).sum(), len(e)), "orbit(anion)"))
    e2 = an[an.oxs.isin(OXS) & (an.ncat_chemenv > 0) & an.sigma_chemenv.notna()]
    cmp_rows.append(("rule 2 (site granularity, no orbit de-duplication)", 20.0, *wilson(((e2.sigma_chemenv - 2).abs() <= EPS).sum(), len(e2)), "site(anion)"))
    # rule 2 on the "high-symmetry, undistorted subset" (right of George Fig. 2b: 221
    # materials, nearly perfect)
    e3 = e2[e2.maxcsm_chemenv <= 1.0]
    cmp_rows.append(("rule 2 (undistorted O sites only, max CSM <= 1)", None, *wilson(((e3.sigma_chemenv - 2).abs() <= EPS).sum(), len(e3)), "site(anion)"))

    P = pr[(pr.nn_algo == "chemenv") & pr.ox_source.isin(OXS)]
    for m, ref in (("corner", 62.5), ("edge", 27.2), ("face", 10.3)):
        cmp_rows.append((f"rule 3, {m} fraction (all CN)", ref, *wilson((P["mode"] == m).sum(), len(P)), "pair"))
    P8 = P[(P.cn_i <= 8) & (P.cn_j <= 8)]
    for m, ref in (("corner", 73.3), ("edge", 25.0), ("face", 1.6)):
        cmp_rows.append((f"rule 3, {m} fraction (CN<=8)", ref, *wilson((P8["mode"] == m).sum(), len(P8)), "pair"))

    agg = sr["chemenv"]
    aggm = agg[agg.index.isin(sid_main)]
    cmp_rows.append(("rule 4 violation rate (George version: ox==V and cn==C)", 40.0,
                     *wilson((~aggm.r4_george_ok).sum(), len(aggm)), "structure"))
    cmp_rows.append(("rule 4 violation rate (oxidation-state version: ox==V only)", None,
                     *wilson((~aggm.r4_ox_ok).sum(), len(aggm)), "structure"))
    cmp_rows.append(("rule 4 violation rate (CN version: cn==C only)", None,
                     *wilson((~aggm.r4_cn_ok).sum(), len(aggm)), "structure"))
    cmp_rows.append(("rule 5 satisfaction (CN criterion)", 70.3,
                     *wilson(aggm.r5_ok.sum(), len(aggm)), "structure"))
    # the ce_symbol version of rule 5 (ChemEnv only, and stricter)
    s5 = si[si.source_id.isin(sid_main) & si.ce_symbol.notna() & si.ox_state.notna()].copy()
    s5["oxr"] = s5.ox_state.round(3)
    g5 = s5.groupby("source_id").apply(
        lambda d: d.groupby(["element", "oxr"]).ce_symbol.nunique().max() == 1, include_groups=False)
    cmp_rows.append(("rule 5 satisfaction (ce_symbol criterion, stricter)", None, *wilson(g5.sum(), len(g5)), "structure"))

    # ---- conjunction of 2-5
    comb = combined(st, an, pr, sr, sid_main, si)
    cmp_rows.append(("rules 2-5 satisfied jointly", 13.0, *wilson(comb["k"], comb["n"]), "structure"))
    cmp_rows.append(("rules 2-5 satisfied jointly (structures with every cation CN<=8)", 20.0,
                     *wilson(comb["k8"], comb["n8"]), "structure"))

    print(f"{'rule':<58s}{'George':>8s}{'here':>10s}{'95% CI':>18s}{'d pt':>8s}  granularity")
    for name, ref, p, lo, hi, gran in cmp_rows:
        dlt = "" if ref is None else f"{p-ref:+7.1f}"
        rr = "" if ref is None else f"{ref:8.1f}"
        flag = ""
        if ref is not None:
            flag = "  ✓" if abs(p - ref) <= 3 else ("  ✗>3pt" if abs(p - ref) <= 10 else "  ✗✗")
        print(f"{name:<58s}{rr:>8s}{p:10.1f}{f'[{lo:.1f},{hi:.1f}]':>18s}{dlt:>8s}  {gran}{flag}")

    # ================================================================ G6
    print("\n" + "=" * 100)
    print("G6 (robustness of the bond definition; the section 4.3 hard gate: satisfaction "
          "varies by < 3 pt across the three algorithms)")
    print("=" * 100)
    g6 = (tab.groupby(["rule", "ox_source", "symprec"])
            .pct.agg(["min", "max"]).assign(spread=lambda d: d["max"] - d["min"]))
    summ = g6.groupby("rule").spread.agg(["min", "median", "max"]).round(2)
    print(summ.to_string())
    print("\nverdict:", end=" ")
    bad = summ[summ["max"] >= 3].index.tolist()
    print("all below 3 pt, G6 passes" if not bad
          else f"these rules have a spread >= 3 pt: {bad}")

    # site-level agreement across the three algorithms (what G6 actually needs, not the
    # aggregate statistics)
    sm = si.dropna(subset=["cn_chemenv", "cn_crystalnn", "cn_brunner"])
    print(f"site-level CN agreement: ce==cnn {np.mean(sm.cn_chemenv == sm.cn_crystalnn):.1%}; "
          f"ce==bru {np.mean(sm.cn_chemenv == sm.cn_brunner):.1%}; "
          f"all three equal {np.mean((sm.cn_chemenv == sm.cn_crystalnn) & (sm.cn_chemenv == sm.cn_brunner)):.1%}"
          f"  (n={len(sm)})")

    # ================================ the epsilon-satisfaction curve for rule 2 (section 4.3 asks for the whole curve)
    print("\nepsilon-satisfaction curve (rule 2, ChemEnv, anion orbit granularity):")
    for eps in (0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"   ε={eps:<5g} → {100*np.mean((e.sigma_chemenv-2).abs() <= eps):5.1f}%")
    # ============================================ George domain-alignment layer: ICSD n MP
    # George's domain is the roughly 5,000 oxides that are "in ICSD and in Materials
    # Project". `icsd_mp_link.parquet` gives the ICSD->MP structure match, and taking
    # non-null mp_id reproduces that filter -- which tells us whether the gap between us and
    # George is a bug or a domain difference. This is the key discriminator for go/no-go.
    try:
        lk = pd.read_parquet(f"{FEAT}/icsd_mp_link.parquet", columns=["source_id", "mp_id"])
        mp_ids = set(lk.loc[lk.mp_id.notna(), "source_id"])
        sid_mp = sid_main & mp_ids
        print("\n" + "=" * 100)
        print(f"domain-alignment layer: ICSD n MP n oxide_strict n ox in {{cif,guess}} = "
              f"{len(sid_mp)} entries (George: about 5,000)")
        print("=" * 100)
        dm = si1[si1.source_id.isin(sid_mp) & si1.r_cat.notna() & si1.cn_chemenv.notna()
                 & si1.orbit_s001.notna()].drop_duplicates(subset=["source_id", "orbit_s001"])
        em = an[an.source_id.isin(sid_mp) & (an.ncat_chemenv > 0) & an.sigma_chemenv.notna()
                & an.orbit_s001.notna()].drop_duplicates(subset=["source_id", "orbit_s001"])
        Pm = pr[(pr.nn_algo == "chemenv") & pr.source_id.isin(sid_mp)]
        Pm8 = Pm[(Pm.cn_i <= 8) & (Pm.cn_j <= 8)]
        am = sr["chemenv"][sr["chemenv"].index.isin(sid_mp)]
        cm2 = combined(st, an, pr, sr, sid_mp, si)
        for nm, ref, k_, n_ in [
                ("rule 1 radius ratio", 66.0, (dm.cn_pred == dm.cn_chemenv).sum(), len(dm)),
                ("rule 2 |Sum s - 2| <= 0.01", 20.0, ((em.sigma_chemenv - 2).abs() <= EPS).sum(), len(em)),
                ("rule 3 corner (all CN)", 62.5, (Pm["mode"] == "corner").sum(), len(Pm)),
                ("rule 3 edge (all CN)", 27.2, (Pm["mode"] == "edge").sum(), len(Pm)),
                ("rule 3 face (all CN)", 10.3, (Pm["mode"] == "face").sum(), len(Pm)),
                ("rule 3 corner (CN<=8)", 73.3, (Pm8["mode"] == "corner").sum(), len(Pm8)),
                ("rule 3 edge (CN<=8)", 25.0, (Pm8["mode"] == "edge").sum(), len(Pm8)),
                ("rule 3 face (CN<=8)", 1.6, (Pm8["mode"] == "face").sum(), len(Pm8)),
                ("rule 4 violation rate (George version)", 40.0, (~am.r4_george_ok).sum(), len(am)),
                ("rule 5 satisfaction", 70.3, am.r5_ok.sum(), len(am)),
                ("rules 2-5 jointly", 13.0, cm2["k"], cm2["n"]),
                ("rules 2-5 jointly (CN<=8)", 20.0, cm2["k8"], cm2["n8"])]:
            p_, lo_, hi_ = wilson(k_, n_)
            fl = "ok" if abs(p_ - ref) <= 3 else (">3pt" if abs(p_ - ref) <= 10 else "!!")
            print(f"  {nm:<40s} George {ref:5.1f}   here {p_:5.1f} [{lo_:.1f},{hi_:.1f}] "
                  f"d{p_-ref:+6.1f} pt  n={n_:<7d} {fl}")
    except FileNotFoundError:
        print("[warn] icsd_mp_link.parquet is missing; skipping the domain-alignment layer")

    # ============================================ diagnostics: rule 1 by element / confusion
    print("\nrule 1 diagnostics (ChemEnv, orbit): predicted CN x observed CN confusion "
          "(rows = predicted)")
    cm = pd.crosstab(d.cn_pred, d.cn_chemenv.astype(int), normalize=False)
    print(cm.to_string())
    per = (d.assign(hit=d.cn_pred == d.cn_chemenv)
             .groupby("element").hit.agg(["mean", "size"]).sort_values("size", ascending=False).head(15))
    per["mean"] = (100 * per["mean"]).round(1)
    print("rule 1 hit rate by element, top 15 by sample size:")
    print(per.to_string())

    # ============================================ diagnostics: rule 2 CSM threshold sweep
    print("\nrule 2 distortion dependence (right of George Fig. 2b: a high-symmetry subset "
          "of 221 materials, nearly perfect):")
    for c in (5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01):
        z = e2[e2.maxcsm_chemenv <= c]
        if len(z):
            print(f"   max CSM ≤ {c:<5g} → {100*np.mean((z.sigma_chemenv-2).abs() <= EPS):5.1f}%  (n={len(z)})")

    print(f"\n[wrote] {out}")
    return tab


def combined(st, an, pr, sr, sid_main, si=None):
    """The conjunction of 2-5, at structure granularity.

    - rule 2 at structure level = **every** cation-coordinated O site in the structure
      satisfies |Sum s - 2| <= 0.01
    - rule 3 at structure level = **no face-sharing pair** (Pauling's own "particularly of
      shared faces")
    - the CN<=8 layer = **every cation site** in the structure has CN<=8 (the blue bars of
      George Fig. 6a: excluding cations in high-coordination environments)
    """
    a = an[an.source_id.isin(sid_main) & (an.ncat_chemenv > 0) & an.sigma_chemenv.notna()]
    r2 = a.assign(ok=(a.sigma_chemenv - 2).abs() <= EPS).groupby("source_id").ok.all()
    p = pr[(pr.nn_algo == "chemenv") & pr.source_id.isin(sid_main)]
    r3 = p.assign(f=(p["mode"] == "face")).groupby("source_id").f.any().rename("has_face")
    agg = sr["chemenv"]
    df = pd.DataFrame(index=sorted(sid_main))
    df["r2"] = r2.reindex(df.index)
    df["r3"] = ~r3.reindex(df.index).fillna(False)          # no pairs -> no face sharing -> satisfied
    df["r4"] = agg.r4_george_ok.reindex(df.index)
    df["r5"] = agg.r5_ok.reindex(df.index)
    ev = df.dropna()
    k = int((ev.r2 & ev.r3 & ev.r4 & ev.r5).sum())
    # the CN<=8 structure layer: take each structure's maximum cation CN from the **site**
    # table, not the pair table -- structures made of isolated polyhedra (isolated SiO4, say)
    # have no rows in the pair table and would be dropped wholesale.
    if si is not None:
        cn8 = si[si.source_id.isin(sid_main)].groupby("source_id").cn_chemenv.max()
    else:
        cn8 = p.groupby("source_id")[["cn_i", "cn_j"]].max().max(axis=1)
    lo8 = set(cn8[cn8 <= 8].index)
    ev8 = ev[ev.index.isin(lo8)]
    return dict(k=k, n=len(ev), k8=int((ev8.r2 & ev8.r3 & ev8.r4 & ev8.r5).sum()), n8=len(ev8))
