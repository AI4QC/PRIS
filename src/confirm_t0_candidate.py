# -*- coding: utf-8 -*-
"""One-shot confirmation on calibration of the single T0 candidate that clears G1-G8.

The candidate (from the extended search A1-ext, a pairwise interaction primitive **outside**
the vocabulary frozen in PREREG section 3.2):

    IF  acid_str / charge_per_an > theta   THEN  cn_chemenv = 6

On discovery: cov 1.80%, acc 0.7857, modal look-up 0.6931, gain +9.25 pt,
tau 0.952, perm_z 19.4, G6 spread 2.08.

The script has three stages and their order may not change:
  S1  leak detection (leakguard levels A/B/C) -- reads discovery only
  S2  multiple-comparison correction (Bonferroni + Benjamini-Hochberg) -- reads discovery only
  S3  one-shot confirmation on calibration -- what is being evaluated is printed and frozen
      **before** calibration is read

Discipline:
  * thresholds, body, metric and baseline convention are all fixed in code and printed before
    S3, and S3 may not go back and change them.
  * not one byte of the lockbox is touched. BVAnalyzer is disabled throughout.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import search_rules as sr          # noqa: E402
import leakguard as lg             # noqa: E402

FEAT = sr.FEAT
SCRATCH = ("/tmp/claude-1000/-home-zhilong-workspace-newpauling/"
           "6021b5d8-e7ef-4fa5-8416-1361cb973e90/scratchpad")
OUT = os.path.join(FEAT, "confirm_t0_candidate.json")

# ============================================================ the frozen evaluation protocol
FEATURE_NUM = "acid_str"          # = ox / (r_uni + r_uni_an)      Lewis acid-strength proxy
FEATURE_DEN = "charge_per_an"     # = n_cat_at * ox_mean_cat / n_an_at
IX_NAME = f"IX_{FEATURE_NUM}/{FEATURE_DEN}"
OP = ">"
BODY_CN = 6
THETA_PRINTED = 1.69097           # the %g value printed in desc; S1 recovers it exactly from
                                  # the quantile grid
ALGO_MAIN = "chemenv"
ALGOS = ("chemenv", "crystalnn", "brunner")
BASELINE_KEY = ["element", "ox"]  # key of the modal look-up baseline (Waroquiers convention)
DEFF = 15.7                       # PREREG revision R1, T_CE compression target
N_HYP_A1 = None                   # filled in from measurement in S2
N_HYP_TOTAL_DECLARED = 66198      # A1-ext 56,198 + beam search / A2 family, about 10,000
ALPHA = 0.05

DISCOVERY_REF = {                 # the numbers the previous extended search reported; S1 must
                                  # reproduce every one of them
    "n_trig": 1297, "cov": 0.017992, "acc": 0.785659,
    "acc_mode_matched": 0.6931, "gain_vs_mode": 0.0925,
    "perm_z": 19.414115, "tau": 0.952, "G6_spread": 2.081727,
}


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


# ============================================================ data preparation (by split)
def prep_split(split):
    """A split-parameterised version of search_rules.prep(). Identical line for line apart
    from the split."""
    site = pd.read_parquet(os.path.join(FEAT, "site.parquet"), columns=[
        "source_id", "element", "ox_state", "ox_source", "cn_chemenv",
        "cn_crystalnn", "cn_brunner", "orbit_id_s01", "mult_s01",
        "proto_id", "split"])
    site = site[site.split == split]
    assert set(site.split.unique()) == {split}, f"out of bounds: rows outside {split}"
    assert "lockbox" not in set(site.split.unique())
    site = site[site.ox_state.notna() & site.orbit_id_s01.notna()]
    site = site[site.ox_state > 0]
    site = site[site.cn_chemenv.notna() & site.cn_crystalnn.notna()
                & site.cn_brunner.notna()]
    orb = (site.sort_values(["source_id", "orbit_id_s01"])
               .groupby(["source_id", "orbit_id_s01"], as_index=False)
               .agg(element=("element", "first"), ox=("ox_state", "first"),
                    ox_source=("ox_source", "first"), mult=("mult_s01", "first"),
                    proto_id=("proto_id", "first"),
                    cn_chemenv=("cn_chemenv", "median"),
                    cn_crystalnn=("cn_crystalnn", "median"),
                    cn_brunner=("cn_brunner", "median")))
    for a in ALGOS:
        orb["cn_" + a] = orb["cn_" + a].round().astype(int)
    prov = pd.read_parquet(os.path.join(FEAT, "provenance.parquet"), columns=[
        "source_id", "formula", "anion", "n_elements", "source"])
    prov = prov[prov.source_id.isin(set(orb.source_id))].copy()
    prov["comp"] = prov.formula.map(sr.parse_formula)
    orb = orb.merge(prov[["source_id", "formula", "anion", "comp", "source"]],
                    on="source_id", how="left")
    orb = orb[orb.anion.notna()].reset_index(drop=True)
    return orb


def ix_values(X):
    a = X[FEATURE_NUM].values.astype(np.float64)
    b = X[FEATURE_DEN].values.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / np.where(np.abs(b) < 1e-9, np.nan, b)
    return r


def rule_mask(v, theta):
    m = (v > theta) if OP == ">" else (v <= theta)
    return m & np.isfinite(v)


# ============================================================ the modal look-up baseline
def mode_table(meta, key_cols, cn_col):
    tab = (meta.groupby(key_cols)[cn_col]
               .agg(lambda s: s.value_counts().idxmax()))
    glob = int(meta[cn_col].value_counts().idxmax())
    return tab, glob


def apply_mode_table(meta, tab, glob, key_cols):
    pred = meta.set_index(key_cols).index.map(tab)
    return pd.Series(pred, index=meta.index).fillna(glob).astype(int).values


# ============================================================ statistics
def mcnemar_exact(b, c):
    """Exact binomial McNemar (two-sided). b = rule right / baseline wrong, c = rule wrong /
    baseline right."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def mcnemar_cluster(b, c, deff):
    """Cluster correction: shrink the number of discordant pairs by deff before the binomial
    test (conservative)."""
    from scipy.stats import binomtest
    n_eff = int(round((b + c) / deff))
    if n_eff < 1:
        return 1.0, 0, 0
    k_eff = int(round(min(b, c) / deff))
    k_eff = min(k_eff, n_eff // 2)
    return (float(binomtest(k_eff, n_eff, 0.5, alternative="two-sided").pvalue),
            k_eff, n_eff)


def bh_reject(pvals, alpha):
    """Benjamini-Hochberg: returns (critical p, number rejected, array of q values)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    o = np.argsort(p)
    ps = p[o]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ps <= thresh
    k = int(np.flatnonzero(below).max() + 1) if below.any() else 0
    pcrit = float(ps[k - 1]) if k else 0.0
    q = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.clip(q, 0, 1)
    qq = np.empty(m)
    qq[o] = q
    return pcrit, k, qq


def block_perm_z(y, pred, mask, block, n_perm=2000, seed=20260728):
    """z against the null distribution from permuting labels within proto blocks (same
    convention as search_rules, with B raised to 2000)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    order0 = np.argsort(block, kind="stable")
    inv = np.argsort(order0, kind="stable")
    acc0 = float((pred[mask] == y[mask]).mean())
    null = np.empty(n_perm)
    for b in range(n_perm):
        key = rng.random(n)
        idx = order0[np.lexsort((key[order0], block[order0]))]
        yp = y[idx][inv]
        null[b] = float((pred[mask] == yp[mask]).mean())
    sd = null.std(ddof=1)
    z = (acc0 - null.mean()) / sd if sd > 1e-12 else 99.0
    p_emp = float((null >= acc0).sum() + 1) / (n_perm + 1)
    return float(z), float(null.mean()), float(sd), p_emp


def eval_rule(v, theta, meta, y3, mode_pred3, block, tag, n_perm=2000):
    m = rule_mask(v, theta)
    n = int(m.sum())
    y = y3[ALGO_MAIN]
    pred = np.where(m, BODY_CN, -1)
    accs = {a: float((pred[m] == y3[a][m]).mean()) for a in ALGOS}
    base = float((mode_pred3[ALGO_MAIN][m] == y[m]).mean())
    hit_r = (pred[m] == y[m])
    hit_b = (mode_pred3[ALGO_MAIN][m] == y[m])
    b = int((hit_r & ~hit_b).sum())
    c = int((~hit_r & hit_b).sum())
    p_mc = mcnemar_exact(b, c)
    p_cl, k_eff, n_eff = mcnemar_cluster(b, c, DEFF)
    z, nm, nsd, p_emp = block_perm_z(y, pred, m, block, n_perm=n_perm)
    ne = pd.Series(meta.element.values[m]).value_counts()
    na = pd.Series(meta.anion.values[m]).value_counts()
    taus = []
    for an in pd.unique(meta.anion.values):
        te = m & (meta.anion.values == an)
        tr = m & (meta.anion.values != an)
        if te.sum() < 50 or tr.sum() < 200:
            continue
        taus.append(float((pred[te] == y[te]).mean()
                          / max((pred[tr] == y[tr]).mean(), 1e-9)))
    n_eff_sites = n / DEFF
    dL = n_eff_sites * (sr._H(base) - sr._H(accs[ALGO_MAIN])) - 28.0
    return {
        "tag": tag, "theta": float(theta), "n_all": int(len(meta)),
        "n_trig": n, "cov": n / len(meta),
        **{f"acc_{a}": round(accs[a], 6) for a in ALGOS},
        "acc": round(accs[ALGO_MAIN], 6),
        "G6_spread_pt": round((max(accs.values()) - min(accs.values())) * 100, 4),
        "acc_mode_matched": round(base, 6),
        "gain_pt": round((accs[ALGO_MAIN] - base) * 100, 4),
        "mcnemar_b": b, "mcnemar_c": c, "p_mcnemar": p_mc,
        "p_mcnemar_deff": p_cl, "mcnemar_eff": [k_eff, n_eff],
        "perm_z": round(z, 3), "perm_null_mean": round(nm, 6),
        "perm_p_emp": p_emp,
        "tau_worst": round(min(taus), 4) if taus else None,
        "n_folds": len(taus),
        "n_cat_elem": int((ne >= 20).sum()), "n_anion_fam": int((na >= 20).sum()),
        "dL_bits": round(float(dL), 1),
    }


# ============================================================ S0 freeze the protocol
def stage0():
    banner("S0  the frozen evaluation protocol (printed before calibration is read; not "
          "revisable afterwards)")
    proto = {
        "rule": f"IF ({FEATURE_NUM}) / ({FEATURE_DEN}) {OP} theta  THEN cn_{ALGO_MAIN} = {BODY_CN}",
        "feature_def": {
            FEATURE_NUM: "ox / (r_uni + r_uni_an)   -- composition-level Lewis acid-strength proxy",
            FEATURE_DEN: "n_cat_at * ox_mean_cat / n_an_at  -- cation charge shared per anion",
        },
        "theta": ("recovered exactly by S1 from discovery's 60-point quantile grid, then "
                  "frozen and applied unchanged in S3"),
        "theta_printed": THETA_PRINTED,
        "body": f"cn = {BODY_CN} (a constant body, no free parameters)",
        "primary_metric": "top-1 accuracy at matched coverage, against the modal look-up table",
        "baseline": {
            "name": "mode_lookup(element, ox) -> the most common cn_chemenv",
            "fit_split": "discovery (the frozen look-up), applied unchanged to calibration in S3",
            "secondary": ("a look-up refitted on calibration (more favourable to the "
                          "baseline; a sensitivity check)"),
        },
        "decision_rule": (
            "gain_pt > 0 on calibration and cluster-corrected McNemar p < 0.05 "
            "=> confirmed; otherwise recorded as search-optimism bias on discovery"),
        "gates_rechecked_on_calibration": ["G4 tau>=0.90", "G5 gain>0 and perm_z>=5",
                                           "G6 spread<3.0 pt", "G8 >=2 elements / >=2 anion families"],
        "one_shot": ("this script evaluates calibration exactly once and never reruns after "
                     "adjusting a threshold or a convention"),
        "lockbox": "not read",
    }
    print(json.dumps(proto, ensure_ascii=False, indent=2))
    return proto


# ============================================================ S1 leak detection
def stage1():
    banner("S1  discovery: reproduce the candidate + three levels of leak detection")
    orb = pd.read_parquet(os.path.join(FEAT, "_t4_orbits_raw.parquet"))
    orb["comp"] = orb.formula.map(sr.parse_formula)
    X, meta, feat_cols, rnd_cols = sr.build_features(orb)
    v = ix_values(X)
    y3 = {a: meta["cn_" + a].values.astype(int) for a in ALGOS}
    N = len(X)
    print(f"[S1] discovery orbits {N}, structures {meta.source_id.nunique()}")

    # ---- recover theta exactly (the 60-point quantile grid A1-ext used, rounded to 6 dp)
    ok = np.isfinite(v)
    qs = np.unique(np.round(np.nanquantile(v[ok], np.linspace(0.02, 0.98, 60)), 6))
    theta = float(qs[np.argmin(np.abs(qs - THETA_PRINTED))])
    print(f"[S1] theta recovered = {theta!r}  (printed value {THETA_PRINTED})")

    tab, glob = mode_table(meta, BASELINE_KEY, "cn_" + ALGO_MAIN)
    mode3 = {a: sr.mode_lookup(meta, BASELINE_KEY, "cn_" + a) for a in ALGOS}
    block = pd.factorize(meta.proto_id.values)[0]
    res = eval_rule(v, theta, meta, y3, mode3, block, "discovery")
    print("[S1] reproduction:")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    chk = {k: (res[{"cov": "cov", "acc": "acc", "n_trig": "n_trig",
                    "acc_mode_matched": "acc_mode_matched"}.get(k, k)]
               if k in res else None) for k in DISCOVERY_REF}
    ok_rep = (res["n_trig"] == DISCOVERY_REF["n_trig"]
              and abs(res["acc"] - DISCOVERY_REF["acc"]) < 1e-5
              and abs(res["gain_pt"] / 100 - DISCOVERY_REF["gain_vs_mode"]) < 5e-4)
    print(f"[S1] matches the extended search: {ok_rep}   (n_trig {res['n_trig']} vs "
          f"{DISCOVERY_REF['n_trig']}, acc {res['acc']:.6f} vs {DISCOVERY_REF['acc']:.6f})")

    # ---------------------------------------------------------- level A
    banner("S1-A  empirical determinism / zero-error one-sided region")
    tgt = (y3[ALGO_MAIN] == BODY_CN).astype(np.int64)   # target = 1[cn == 6]
    okall = np.ones(N, bool)
    a_ix = lg.level_a(v, tgt, okall)
    a_num = lg.level_a(X[FEATURE_NUM].values, tgt, okall)
    a_den = lg.level_a(X[FEATURE_DEN].values, tgt, okall)
    for nm, a in (("IX (candidate)", a_ix), (FEATURE_NUM, a_num), (FEATURE_DEN, a_den)):
        print(f"  {nm:16s} err_thresh={a['err_thresh']:.4f} "
              f"err_value={a['err_value']}  ({a.get('mean_per_value',0):.1f} samples per value) "
              f"tail_purity={a['tail_purity']:.4f}@cov={a['tail_cov']:.3f} "
              f"pure_side={a['pure_side']}")
    # the reverse: does cn determine IX (the other half of an identity)
    dfv = pd.DataFrame({"cn": y3[ALGO_MAIN], "v": np.round(v, 6)})
    g = dfv.groupby("cn").v.nunique()
    print(f"  reverse check, |{{IX}} | cn| distinct values = {dict(g)}  "
          f"(a cn matching exactly 1 IX value would be an identity)")
    # negative control: run the random features down the same path
    negs = {}
    for c in rnd_cols[:6]:
        an = lg.level_a(X[c].values, tgt, okall)
        negs[c] = lg.combine(an, None, None)[0]
    print(f"  negative control (6 random features), level-A verdicts: {negs}")

    # ---------------------------------------------------------- level B
    banner("S1-B  single-feature grouped-CV ceiling (grouped by proto_id)")
    b_ix = lg.level_b(v, tgt, okall, meta.proto_id.values)
    print(f"  IX: cv_acc={b_ix['cv_acc']:.4f} (±{b_ix['cv_sd']:.4f}) "
          f"majority class={b_ix['cv_maj']:.4f} gain={b_ix['cv_gain']:+.4f}  "
          f"[BLOCK threshold {lg.CEIL_BLOCK} / REVIEW threshold {lg.CEIL_REVIEW}]")

    # ---------------------------------------------------------- level C
    banner("S1-C  construction provenance (the features.yaml declarations + a new T0 target T_CN)")
    import yaml
    cfg = yaml.safe_load(open(lg.YAML_SRC))
    # the new target: cn itself. The inputs of its defining expression are the bond incidence
    # table of the cation sites.
    cfg["targets"]["T_CN"] = {
        "desc": "coordination number cn_chemenv of a cation orbit (the primary T0 target)",
        "scope": "cation_orbit", "base": "cn",
        "forbidden_atoms": ["bond_inc_cat", "cn_orbit", "bond_inc_an",
                            "an_cn_site", "pair_graph", "s_prior",
                            "ox_bonded_cat"],
        "inherit": ["t0"],
    }
    # declaring the candidate interaction term: the ratio of two already-declared T0 features;
    # take the union of their atoms and the coarser of their domains
    cfg["features"]["T_CN"] = {
        IX_NAME: {"atoms": ["elem_const", "ox_formal", "comp"], "op": "ratio",
                  "scope": "structure", "base": "elem_over_ox",
                  "note": ("acid_str / charge_per_an, an interaction primitive outside the "
                           "PREREG section 3.2 vocabulary")},
    }
    c_all = lg.level_c(cfg, "T_CN")
    c_ix = c_all[IX_NAME]
    print(f"  {IX_NAME}")
    print(f"    atom closure = {c_ix['atoms']}")
    print(f"    R1 (hits a forbidden atom of the target) = {c_ix['R1']}  hits={c_ix['R1_atom_hit']}")
    print(f"    R2 (same base + envelope operator + domain containment) = {c_ix['R2']}")
    print(f"    R3 (same domain, same base)                             = {c_ix['R3']}")
    print(f"    verdict_C = {c_ix['verdict_c']}")
    for f in (FEATURE_NUM, FEATURE_DEN):
        print(f"  {f:16s} verdict_C={c_all[f]['verdict_c']} "
              f"atoms={c_all[f]['atoms']}")
    n_block = sum(1 for r in c_all.values() if r["verdict_c"] == "BLOCK")
    print(f"  under target T_CN, {n_block} of the 41 T0 base features are BLOCKed at level C "
          f"(should be 0: they are all composition level)")

    verdict, reasons = lg.combine(a_ix, b_ix, c_ix)
    print(f"\n  >>> combined LeakGuard verdict across the three levels: {verdict}   reasons={reasons}")

    return dict(orb=orb, X=X, meta=meta, v=v, y3=y3, theta=theta,
                tab=tab, glob=glob, block=block, feat_cols=feat_cols,
                rnd_cols=rnd_cols, res_disc=res,
                leak={"verdict": verdict, "reasons": reasons,
                      "A_ix": {k: a_ix[k] for k in ("err_thresh", "err_value",
                                                    "tail_purity", "tail_cov",
                                                    "pure_side", "mean_per_value")},
                      "B_ix": b_ix, "C_ix": {k: c_ix[k] for k in
                                             ("atoms", "R1", "R2", "R3",
                                              "R1_atom_hit", "verdict_c")},
                      "negctl": negs})


# ============================================================ S2 multiple comparisons
def stage2(S):
    banner("S2  multiple-comparison correction: Bonferroni + Benjamini-Hochberg")
    import itertools
    X, meta, y3 = S["X"], S["meta"], S["y3"]
    y = y3[ALGO_MAIN]
    N = len(X)
    # rebuild the extended search's A1-ext hypothesis family (line for line the same
    # convention as v5_t0_expand.py)
    CORE = ["ox", "ratio_uni", "ratio_calc", "r_uni", "r_uni_an", "X", "dX",
            "ion_pot", "ion_pot2", "acid_str", "ionicity", "x_in_cat",
            "cat_an_ratio", "ox_mean_cat", "n_cat_species", "bv_budget", "ie1",
            "nd_ion", "r_sum", "charge_per_an", "ox_rel", "period", "group", "Z"]
    inter = {}
    for a, b in itertools.combinations(CORE, 2):
        va, vb = X[a].values.astype(np.float64), X[b].values.astype(np.float64)
        inter[f"IX_{a}*{b}"] = (va * vb).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            inter[f"IX_{a}/{b}"] = (va / np.where(np.abs(vb) < 1e-9, np.nan,
                                                  vb)).astype(np.float32)
    XA = pd.concat([X[S["feat_cols"] + S["rnd_cols"]], pd.DataFrame(inter)], axis=1)
    cols = S["feat_cols"] + S["rnd_cols"] + list(inter)
    mode_main = sr.mode_lookup(meta, BASELINE_KEY, "cn_" + ALGO_MAIN)
    base_hit = (mode_main == y)
    classes = np.arange(1, 14)
    from scipy.stats import binomtest
    rows = []
    t0 = time.time()
    for f in cols:
        vv = XA[f].values.astype(np.float64)
        okf = np.isfinite(vv)
        if okf.sum() < 500:
            continue
        vo = vv[okf]
        qs = np.unique(np.round(np.nanquantile(vo, np.linspace(0.02, 0.98, 60)), 6))
        for th in qs:
            for op in ("<=", ">"):
                m = (vv <= th) if op == "<=" else (vv > th)
                m &= okf
                n = int(m.sum())
                if n < 0.005 * N:
                    continue
                cnt = np.bincount(y[m], minlength=14)[1:14]
                c = int(classes[int(np.argmax(cnt))])
                hit_r = (y[m] == c)
                hb = base_hit[m]
                bb = int((hit_r & ~hb).sum())
                cc = int((~hit_r & hb).sum())
                rows.append((f, op, float(th), n, float(hit_r.mean()),
                             float(hb.mean()), bb, cc))
    A = pd.DataFrame(rows, columns=["feature", "op", "theta", "n", "acc",
                                    "acc_mode", "b", "c"])
    print(f"[S2] rebuilt the A1-ext hypothesis family: {len(A)} entries "
          f"({time.time()-t0:.0f}s); the extended-search log records 56198")
    A["gain_pt"] = (A.acc - A.acc_mode) * 100
    # one-sided exact McNemar (rule better than baseline)
    A["p_raw"] = [binomtest(int(c_), int(b_ + c_), 0.5, alternative="less").pvalue
                  if (b_ + c_) > 0 else 1.0 for b_, c_ in zip(A.b, A.c)]
    # the cluster-corrected version
    pd_eff = []
    for b_, c_ in zip(A.b, A.c):
        ne = int(round((b_ + c_) / DEFF))
        ke = int(round(c_ / DEFF))
        pd_eff.append(binomtest(min(ke, ne), ne, 0.5, alternative="less").pvalue
                      if ne >= 1 else 1.0)
    A["p_deff"] = pd_eff

    sel = A[(A.feature == IX_NAME) & (A.op == OP)
            & (np.abs(A.theta - S["theta"]) < 1e-9)]
    assert len(sel) == 1, f"the candidate is not uniquely located in the family: {len(sel)}"
    r0 = sel.iloc[0]
    m_a1 = len(A)
    m_tot = N_HYP_TOTAL_DECLARED
    out = {"m_A1_rebuilt": int(m_a1), "m_A1_logged": 56198,
           "m_total_declared": int(m_tot)}
    for nm, pcol in (("naive", "p_raw"), ("deff", "p_deff")):
        p0 = float(r0[pcol])
        bonf_a1 = min(1.0, p0 * m_a1)
        bonf_tot = min(1.0, p0 * m_tot)
        pcrit, k, q = bh_reject(A[pcol].values, ALPHA)
        q0 = float(q[sel.index[0] - A.index[0]]) if False else float(
            q[np.flatnonzero(A.index == sel.index[0])[0]])
        out[nm] = {"p_raw": p0,
                   "bonferroni_p_A1": bonf_a1, "bonferroni_p_total": bonf_tot,
                   "bonferroni_sig_total": bool(bonf_tot < ALPHA),
                   "BH_pcrit": pcrit, "BH_n_reject": int(k),
                   "BH_q_candidate": q0, "BH_sig": bool(q0 < ALPHA)}
        print(f"[S2/{nm}] p_raw={p0:.3e}  Bonferroni(m={m_a1})={bonf_a1:.3e}  "
              f"Bonferroni(m={m_tot})={bonf_tot:.3e}  "
              f"BH q={q0:.3e} ({k}/{m_a1} rejected, p_crit={pcrit:.3e})")
    # Bonferroni on perm_z: the two-sided normal p corresponding to z=19.41
    from scipy.stats import norm
    z = S["res_disc"]["perm_z"]
    p_perm = float(2 * norm.sf(abs(z)))
    out["perm_z"] = {"z": z, "p_two_sided": p_perm,
                     "bonferroni_total": min(1.0, p_perm * m_tot),
                     "note": ("the null of the permutation test is 'the labels are unrelated "
                              "to the rule', not 'the rule is no better than the look-up "
                              "table'; clearing Bonferroni here is not the same as beating "
                              "the baseline")}
    print(f"[S2] perm_z={z} -> p={p_perm:.3e}, Bonferroni(m={m_tot}) = "
          f"{out['perm_z']['bonferroni_total']:.3e}")
    # how many members of the family have a larger gain than the candidate
    out["n_gain_ge_candidate"] = int((A.gain_pt >= r0.gain_pt).sum())
    out["n_gain_gt_0"] = int((A.gain_pt > 0).sum())
    out["candidate_gain_rank"] = int((A.gain_pt > r0.gain_pt).sum() + 1)
    out["rnd_family"] = {
        "n": int(A.feature.str.startswith("rnd_").sum()),
        "n_BH_reject": int(((A.feature.str.startswith("rnd_"))
                            & (bh_reject(A["p_deff"].values, ALPHA)[2] < ALPHA)).sum()),
        "max_gain_pt": float(A[A.feature.str.startswith("rnd_")].gain_pt.max()),
    }
    print(f"[S2] the candidate gain of {r0.gain_pt:.2f} pt ranks "
          f"{out['candidate_gain_rank']} of {m_a1} hypotheses; {out['n_gain_gt_0']} have a "
          f"gain above 0")
    print(f"[S2] negative-control random-feature family: {out['rnd_family']}")
    return out


# ============================================================ S3 calibration
def stage3(S, proto):
    banner("S3  *** the first and only read of calibration ***")
    print(f"[S3] frozen theta = {S['theta']!r}, body cn = {BODY_CN}, "
          f"baseline = mode_lookup{BASELINE_KEY} fitted on discovery")
    orbc = prep_split("calibration")
    Xc, metac, _, _ = sr.build_features(orbc)
    vc = ix_values(Xc)
    y3c = {a: metac["cn_" + a].values.astype(int) for a in ALGOS}
    print(f"[S3] calibration orbits {len(Xc)}, structures {metac.source_id.nunique()}")

    # baseline one: the look-up frozen on discovery (the main convention)
    frozen3 = {}
    for a in ALGOS:
        tab, glob = mode_table(S["meta"], BASELINE_KEY, "cn_" + a)
        frozen3[a] = apply_mode_table(metac, tab, glob, BASELINE_KEY)
    # baseline two: refitted on calibration (more favourable to the baseline)
    self3 = {a: sr.mode_lookup(metac, BASELINE_KEY, "cn_" + a) for a in ALGOS}

    blockc = pd.factorize(metac.proto_id.values)[0]
    r_frozen = eval_rule(vc, S["theta"], metac, y3c, frozen3, blockc,
                         "calibration/frozen-baseline")
    r_self = eval_rule(vc, S["theta"], metac, y3c, self3, blockc,
                       "calibration/self-fit-baseline")
    for r in (r_frozen, r_self):
        print("\n" + json.dumps(r, ensure_ascii=False, indent=2))

    d = S["res_disc"]
    print(f"\n[S3] discovery -> calibration comparison")
    print(f"     cov      {d['cov']*100:6.2f}%  ->  {r_frozen['cov']*100:6.2f}%")
    print(f"     acc      {d['acc']:.4f}  ->  {r_frozen['acc']:.4f}")
    print(f"     look-up  {d['acc_mode_matched']:.4f}  ->  "
          f"{r_frozen['acc_mode_matched']:.4f} (frozen) / "
          f"{r_self['acc_mode_matched']:.4f} (refitted)")
    print(f"     gain     {d['gain_pt']:+.2f} pt  ->  {r_frozen['gain_pt']:+.2f} pt"
          f" (frozen) / {r_self['gain_pt']:+.2f} pt (refitted)")
    shrink = (r_frozen["gain_pt"] / d["gain_pt"]) if d["gain_pt"] else float("nan")
    print(f"     gain retained {shrink:.2%}")

    gates = {
        "G4_tau>=0.90": (r_frozen["tau_worst"] is not None
                         and r_frozen["tau_worst"] >= 0.90),
        "G5_gain>0": r_frozen["gain_pt"] > 0,
        "G5_perm_z>=5": r_frozen["perm_z"] >= 5,
        "G6_spread<3pt": r_frozen["G6_spread_pt"] < 3.0,
        "G8_elem>=2&anion>=2": (r_frozen["n_cat_elem"] >= 2
                                and r_frozen["n_anion_fam"] >= 2),
        "decision_p_deff<0.05": r_frozen["p_mcnemar_deff"] < ALPHA,
    }
    print("\n[S3] gates rerun on calibration:")
    for k, v in gates.items():
        print(f"     {k:24s} {v}")
    confirmed = bool(gates["G5_gain>0"] and gates["decision_p_deff<0.05"])
    print(f"\n[S3] >>> verdict under the frozen decision rule: "
          f"{'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
    S["_cal"] = (Xc, metac, vc)
    return {"frozen": r_frozen, "self_fit": r_self, "gates": gates,
            "gain_retention": shrink, "confirmed": confirmed}


# ============================================ S3b appendix: whole-cluster bootstrap
def cluster_boot(meta, m, pred_rule, pred_base, y, key, B=4000, seed=20260728):
    sub = pd.DataFrame({"k": meta[key].values[m].astype(str),
                        "r": (pred_rule[m] == y[m]).astype(float),
                        "b": (pred_base[m] == y[m]).astype(float)})
    codes, uniq = pd.factorize(sub.k)
    K = len(uniq)
    n_in = np.bincount(codes, minlength=K).astype(float)
    r_in = np.bincount(codes, weights=sub.r.values, minlength=K)
    b_in = np.bincount(codes, weights=sub.b.values, minlength=K)
    rng = np.random.default_rng(seed)
    g = np.empty(B)
    for i in range(B):
        s = rng.integers(0, K, K)
        g[i] = (r_in[s].sum() - b_in[s].sum()) / n_in[s].sum()
    obs = (sub.r.sum() - sub.b.sum()) / len(sub)
    # deff measured over the matched region (on the difference indicator
    # d = 1[rule right] - 1[baseline right])
    d = sub.r.values - sub.b.values
    nk = n_in
    mk = float((nk ** 2).sum() / nk.sum())
    gm = d.mean()
    gmean = np.bincount(codes, weights=d, minlength=K) / nk
    msb = float(((gmean - gm) ** 2 * nk).sum() / max(K - 1, 1))
    msw = float(((d - gmean[codes]) ** 2).sum() / max(len(d) - K, 1))
    n0 = (nk.sum() - (nk ** 2).sum() / nk.sum()) / max(K - 1, 1)
    den = msb + (n0 - 1) * msw
    icc = (msb - msw) / den if den > 0 else 0.0
    return {"key": key, "n_sites": int(m.sum()), "n_clusters": int(K),
            "obs_gain_pt": round(obs * 100, 3),
            "se_pt": round(float(g.std(ddof=1)) * 100, 3),
            "ci95_pt": [round(float(np.percentile(g, 2.5)) * 100, 3),
                        round(float(np.percentile(g, 97.5)) * 100, 3)],
            "p_one_sided": float((g <= 0).mean()),
            "m_kish": round(mk, 2), "icc": round(float(icc), 4),
            "deff_emp": round(1 + (mk - 1) * float(icc), 2)}


def stage3b(S):
    banner("S3b  appendix: whole-cluster bootstrap standard errors (the rule, threshold and "
          "baseline are unchanged; only the SE estimator differs)")
    print("  The `discordant pairs / deff=15.7` S3 decides on is the overall deff of T_CE\n"
          "  across the whole store, which is over-conservative on a narrow 1.8% trigger\n"
          "  region; this resamples whole clusters directly.")
    out = {}
    # discovery
    y = S["meta"].cn_chemenv.values.astype(int)
    m = rule_mask(S["v"], S["theta"])
    pb = sr.mode_lookup(S["meta"], BASELINE_KEY, "cn_" + ALGO_MAIN)
    out["discovery"] = {k: cluster_boot(S["meta"], m, np.where(m, BODY_CN, -1),
                                        pb, y, k)
                        for k in ("proto_id", "source_id")}
    # calibration (using the objects S3 already built)
    Xc, metac, vc = S["_cal"]
    mc = rule_mask(vc, S["theta"])
    yc = metac.cn_chemenv.values.astype(int)
    tab, glob = mode_table(S["meta"], BASELINE_KEY, "cn_" + ALGO_MAIN)
    pbc = apply_mode_table(metac, tab, glob, BASELINE_KEY)
    out["calibration"] = {k: cluster_boot(metac, mc, np.where(mc, BODY_CN, -1),
                                          pbc, yc, k)
                          for k in ("proto_id", "source_id")}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    ok = all(out["calibration"][k]["p_one_sided"] < ALPHA
             for k in ("proto_id", "source_id"))
    print(f"\n[S3b] >>> calibration verdict under the whole-cluster bootstrap: "
          f"{'CONFIRMED (alpha=0.05)' if ok else 'NOT CONFIRMED'}")
    out["confirmed_bootstrap"] = bool(ok)
    return out


def main():
    t0 = time.time()
    proto = stage0()
    S = stage1()
    mc = stage2(S)
    cal = stage3(S, proto)
    cal_b = stage3b(S)
    out = {"protocol": proto, "discovery": S["res_disc"], "leakguard": S["leak"],
           "multiple_comparison": mc, "calibration": cal,
           "calibration_cluster_bootstrap": cal_b,
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] wrote {OUT}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
