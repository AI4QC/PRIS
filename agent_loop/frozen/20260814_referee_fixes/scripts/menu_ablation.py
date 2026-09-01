#!/usr/bin/env python3
"""Referee fix: menu ablation of the L4 greedy predicate search.

Variant "no_pauling": candidate pool excludes all Pauling-mechanism families
  (wyckoff_econ*, sg_num*, csys*, bv_*, gii, p2_*, p4_*, p5_*).
Variant "no_symmetry": excludes only wyckoff_econ*, sg_num*, csys*.

DISCOVERY split only; calibration is never touched. Same greedy rule as
src/l4_search.py: satisfaction floor 0.81, min exclusion gain 0.005, <=4 steps.
Writes outputs/20260814_referee_fixes/menu_ablation.json.
"""
from __future__ import annotations
import itertools, json, os, re
import numpy as np
import pandas as pd

F = "$PRIS_FEATURES/"
B = "$PRIS_LAW_TABLES/"
OUT = "<repo>/outputs/20260814_referee_fixes"

QGRID = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
         0.60, 0.70, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
EXCLUDE = {"source_id", "anion", "split", "psplit", "sid", "kind", "parent", "n_el",
           "n_sites", "p2_n_bad_020", "p2_sum_dev", "p3_n_pairs", "p3_n_face",
           "p3_n_edge", "p4_n_viol", "nsites"}

PAULING_PAT = re.compile(r"^(wyckoff_econ|sg_num|csys|bv_|p2_|p4_|p5_)|^gii$")
SYMMETRY_PAT = re.compile(r"^(wyckoff_econ|sg_num|csys)")


def load_tables():
    lr = pd.read_parquet(B + "law_real.parquet")
    lb = pd.read_parquet(B + "law_bad.parquet")
    ar = pd.read_parquet(F + "law_real_aug.parquet")
    ab = pd.read_parquet(F + "law_bad_aug.parquet")
    lr = lr.merge(ar, on="source_id", how="left")
    lb = lb.merge(ab, on=["parent", "kind"], how="left")
    return lr.reset_index(drop=True), lb.reset_index(drop=True)


def _le(v, th):
    return np.where(np.isfinite(v), v <= th, True)


def _ge(v, th):
    return np.where(np.isfinite(v), v >= th, True)


def l3_mask(d):
    bl, blm = d.bl_min.values, d.bl_mean.values
    cn, mz, mx = d.cn_an_mean.values, d.madz_range.values, d.mad_max.values
    lk, fi = d.frac_like_bonds.values, d.fi.values
    d1 = _ge(bl, 0.804)
    d3 = np.where(np.isfinite(cn) & (cn <= 3.333), _le(blm, 1.081), True)
    d4, d5 = _le(mz, 31.45), _le(mx, 15.17)
    d6 = np.where(np.isfinite(fi) & (fi > 0.55), _le(lk, 1e-4), True)
    return d1 & d3 & d4 & d5 & d6


def pred_mask(d, feat, direction, th, guard, _cache={}):
    key = id(d)
    cols = _cache.setdefault(key, {})
    if feat not in cols:
        cols[feat] = d[feat].values.astype(float)
    v = cols[feat]
    m = _le(v, th) if direction == "le" else _ge(v, th)
    if guard:
        if "fi" not in cols:
            cols["fi"] = d.fi.values.astype(float)
        fi = cols["fi"]
        m = np.where(np.isfinite(fi) & (fi > guard), m, True)
    return m


def candidates(lr_disc, ban_pat):
    feats = [c for c in lr_disc.columns
             if c not in EXCLUDE and pd.api.types.is_numeric_dtype(lr_disc[c])
             and not ban_pat.match(c)]
    cands, banned = [], sorted(
        c for c in lr_disc.columns
        if c not in EXCLUDE and pd.api.types.is_numeric_dtype(lr_disc[c])
        and ban_pat.match(c))
    for f in feats:
        v = lr_disc[f].values.astype(float)
        v = v[np.isfinite(v)]
        if v.size < 100 or np.std(v) == 0:
            continue
        ths = sorted(set(np.quantile(v, QGRID).tolist()))
        for th, dr, g in itertools.product(ths, ("le", "ge"), (None, 0.50, 0.55)):
            cands.append((f, dr, float(th), g))
    return cands, banned


def greedy(lrd, lbd, ban_pat, log):
    cands, banned = candidates(lrd, ban_pat)
    log.append(f"{len(cands)} candidate predicates; banned features: {banned}")
    Sr, Sb = l3_mask(lrd), l3_mask(lbd)
    chosen, steps = [], []
    for step in range(4):
        base_exc = 1 - Sb.mean()
        best = None
        for c in cands:
            mr = pred_mask(lrd, *c)
            sat = (Sr & mr).mean()
            if sat < 0.81:
                continue
            mb = pred_mask(lbd, *c)
            exc = 1 - (Sb & mb).mean()
            if best is None or exc > best[0]:
                best = (exc, sat, c)
        if best is None or best[0] - base_exc < 0.005:
            log.append(f"stop at step {step+1}: best gain "
                       f"{(best[0]-base_exc) if best else float('nan'):+.4f}")
            break
        exc, sat, c = best
        chosen.append(c)
        Sr &= pred_mask(lrd, *c)
        Sb &= pred_mask(lbd, *c)
        steps.append(dict(step=step + 1, predicate=list(c),
                          disc_sat=float(sat), disc_exc=float(exc)))
        log.append(f"step {step+1}: {c} -> disc sat {sat:.4f} exc {exc:.4f}")
    kinds = lbd.kind.values
    per = {k: float(1 - Sb[kinds == k].mean()) for k in sorted(set(kinds))}
    return dict(banned_features=banned, n_candidates=len(cands),
                predicates=[list(c) for c in chosen], steps=steps,
                disc_sat=float(Sr.mean()), disc_exc=float(1 - Sb.mean()),
                per_class_disc_exc=per)


def main():
    lr, lb = load_tables()
    lrd = lr[lr.split == "discovery"].reset_index(drop=True)
    lbd = lb[lb.psplit == "discovery"].reset_index(drop=True)
    log = []
    sat_d = l3_mask(lrd).mean(); exc_d = 1 - l3_mask(lbd).mean()
    log.append(f"L3 reproduction on discovery: {sat_d:.4f}/{exc_d:.4f} (pub 0.9071/0.7052)")
    print(log[-1])
    assert abs(sat_d - 0.9071) <= 0.002 and abs(exc_d - 0.7052) <= 0.002, "L3 mismatch"

    kinds = lbd.kind.values
    Sb3 = l3_mask(lbd)
    l3_per = {k: float(1 - Sb3[kinds == k].mean()) for k in sorted(set(kinds))}

    res = dict(
        note="Menu ablation of L4 greedy search, discovery split only; "
             "same rule as src/l4_search.py (sat floor 0.81, min gain 0.005, <=4 steps). "
             "Calibration never evaluated.",
        l3_baseline=dict(disc_sat=float(sat_d), disc_exc=float(exc_d),
                         per_class_disc_exc=l3_per),
        archived_pauling_menu=dict(
            predicates=[["wyckoff_econ_001", "le", 0.6666666666666666, None],
                        ["bv_rel_mean", "le", 0.7143040821865658, None]],
            disc_sat=0.8107188093730209, disc_exc=0.9051222351571595),
        variants={})
    for name, pat in (("no_pauling_menu", PAULING_PAT),
                      ("no_symmetry_only", SYMMETRY_PAT)):
        log.append(f"=== variant {name} ===")
        print(log[-1], flush=True)
        r = greedy(lrd, lbd, pat, log)
        for line in log[-(len(r["steps"]) + 2):]:
            pass
        print(json.dumps({k: r[k] for k in ("predicates", "disc_sat", "disc_exc",
                                            "per_class_disc_exc")}, indent=1),
              flush=True)
        res["variants"][name] = r
    res["log"] = log
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "menu_ablation.json")
    json.dump(res, open(p, "w"), indent=1)
    print("saved", p)


if __name__ == "__main__":
    main()
