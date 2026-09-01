#!/usr/bin/env python3
"""把四个法则集当作**二值判据**放到同组成排序任务上。

主文图 4a/4c/4d 原来只有 rho 一个连续量。读者会问:L1–L3 呢?
答案本身是结果:法则集作为二值判据在这个任务上大量弃权,和泡林规则一样 ——
rho 之所以从不弃权,是因为它被当作**连续量**用,而不是被当作阈值用。

口径与已发表的行完全一致(本脚本先复现 Pauling 5 / bl_min / vol_per_atom /
e_hull 四行到小数点后四位再输出新行):
  commit  = 组内一对结构被赋予**不同**取值
  accuracy= 只在 commit 的对上算,按组等权
  top-1   = 判据最优集合中已合成相的期望占比,基线为该组已合成占比
  tie     = 判据无法挑出单一结构的组占比
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# external feature store: gigabytes of derived tables, not in this repository.
# Override with PRIS_FEATURES; see README "Reproducing the figures".
F = os.environ.get("PRIS_FEATURES",
                   "features/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "data")


def load():
    d = pd.read_parquet(F + "synth_rank.parquet")
    aug = F + "synth_rank_aug.parquet"
    if os.path.exists(aug):
        d = d.merge(pd.read_parquet(aug), on="mp_id", how="left")
    g = d.groupby("rk").synth.agg(["sum", "count"])
    keep = set(g[(g["sum"] > 0) & (g["sum"] < g["count"])].index)
    return d[d.rk.isin(keep)].reset_index(drop=True)


def ionicity_col(d):
    """f_i from the reduced formula, mirroring apply_rules.ionicity exactly."""
    from t0_guard import parse
    ep = pd.read_parquet(F + "_elem_props.parquet")
    X = dict(zip(ep.element, ep.X))
    ANI = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As", "H", "C"}
    cache = {}

    def fi(rk):
        if rk in cache:
            return cache[rk]
        try:
            c = parse(rk)
            cand = [e for e in c if e in ANI and e in X and np.isfinite(X[e])]
            if not cand:
                raise ValueError
            an = max(cand, key=lambda e: X[e])
            cats = {e: n for e, n in c.items()
                    if e != an and e in X and np.isfinite(X[e])}
            if not cats:
                raise ValueError
            dx = sum(n * (X[an] - X[e]) for e, n in cats.items()) / sum(cats.values())
            v = float(1 - np.exp(-0.25 * dx * dx))
        except Exception:
            v = np.nan
        cache[rk] = v
        return v

    return d.rk.map(fi)


def _le(v, th):
    return np.where(np.isfinite(v), v <= th, True)


def _ge(v, th):
    return np.where(np.isfinite(v), v >= th, True)


def rulesets(d):
    """四个集合的论文评估口径。缺失值按满足处理，以复现已发表统计。

    公开 CLI 不沿用这个二值缺失约定；`apply_rules.judge()` 会返回“无法判定”。
    """
    bl, blm = d.bl_min.values, d.bl_mean.values
    cn, mz, mx = d.cn_an_mean.values, d.madz_range.values, d.mad_max.values
    lk, fi = d.frac_like_bonds.values, d["fi"].values

    d1a, d1b = _ge(bl, 0.735), _ge(bl, 0.804)
    d2 = np.where(np.isfinite(fi) & (fi > 0.50), _le(bl, 1.05), True)
    d3 = np.where(np.isfinite(cn) & (cn <= 3.333), _le(blm, 1.081), True)
    d4, d5 = _le(mz, 31.45), _le(mx, 15.17)
    d6 = np.where(np.isfinite(fi) & (fi > 0.55), _le(lk, 1e-4), True)
    out = {"L1": d1a.astype(float),
           "L1'": (d1a & d2).astype(float),
           "L2": (d1b & d3 & d4 & d5).astype(float),
           "L3": (d1b & d3 & d4 & d5 & d6).astype(float)}
    if "wyckoff_econ_001" in d.columns and "bv_rel_mean" in d.columns:
        d7 = _le(d.wyckoff_econ_001.values, 2.0 / 3.0)
        d8 = _le(d.bv_rel_mean.values, 0.7143040821865658)
        out["L4"] = (d1b & d3 & d4 & d5 & d6 & d7 & d8).astype(float)
    return out


def evaluate(d, vals, higher_better=True):
    """commitment profile + top-1 + tie rate + accuracy where e_hull is wrong."""
    cov, acc, accw, t1, base, ties = [], [], [], [], [], []
    eh = d.e_hull.values
    for _, sub in d.groupby("rk"):
        idx = sub.index.values
        v, y = np.asarray(vals)[idx], sub.synth.values.astype(bool)
        e = eh[idx]
        a, b = np.where(y)[0], np.where(~y)[0]
        if not len(a) or not len(b):
            continue
        va, vb = v[a][:, None], v[b][None, :]
        fin = np.isfinite(va) & np.isfinite(vb)
        if fin.sum() == 0:
            continue
        diff = fin & (va != vb)
        win = ((va > vb) if higher_better else (va < vb)) & diff
        cov.append(diff.sum() / fin.sum())
        if diff.sum():
            acc.append(win.sum() / diff.sum())
        # pairs the DFT hull energy ranks wrongly: the unmade phase sits lower
        ew = (e[a][:, None] > e[b][None, :]) & diff
        if ew.sum():
            accw.append((win & ew).sum() / ew.sum())
        # top-1 over the criterion's best set
        vf = v[np.isfinite(v)]
        if vf.size:
            best = v.max() if higher_better else v.min()
            sel = np.isfinite(v) & (v == best)
            t1.append(y[sel].mean())
            base.append(y.mean())
            ties.append(1.0 if sel.sum() > 1 else 0.0)
    return dict(coverage=np.mean(cov), accuracy=np.mean(acc),
                acc_energy_wrong=np.mean(accw), n_groups=len(acc),
                top1=np.mean(t1), random_baseline=np.mean(base),
                lift=np.mean(t1) - np.mean(base), tie_rate=np.mean(ties),
                n_groups_top1=len(t1))


def main():
    d = load()
    d["fi"] = ionicity_col(d)

    # --- reproduce published rows before trusting any new one
    checks = [("Pauling 5 (parsimony)", d.p5_n_distinct.values, False, 0.222951, 0.655287),
              ("This work: bl_min", d.bl_min.values, True, 1.0, 0.568837),
              ("Volume per atom", d.vol_per_atom.values, False, 0.986266, 0.643537),
              ("DFT E_hull (baseline)", d.e_hull.values, False, 0.992982, 0.750133)]
    print("reproduction check (must match paper/data/fig3_coverage_accuracy.csv):")
    ok = True
    for name, v, hb, pc, pa in checks:
        r = evaluate(d, v, hb)
        good = abs(r["coverage"] - pc) < 5e-4 and abs(r["accuracy"] - pa) < 5e-4
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name:24s} cov {r['coverage']:.4f} "
              f"(pub {pc:.4f})  acc {r['accuracy']:.4f} (pub {pa:.4f})")
    if not ok:
        print("\nconventions do not match; refusing to emit new rows")
        return 1

    rows = []
    for name, v in rulesets(d).items():
        r = evaluate(d, v, higher_better=True)
        r["rule"] = name
        rows.append(r)
        print(f"\n{name}: commit {r['coverage']:.4f}  acc {r['accuracy']:.4f}  "
              f"acc|e_wrong {r['acc_energy_wrong']:.4f}  top1 {r['top1']:.4f}  "
              f"lift {r['lift']:+.4f}  tie {r['tie_rate']:.4f}  n {r['n_groups']}")
    out = pd.DataFrame(rows)[["rule", "coverage", "accuracy", "acc_energy_wrong",
                              "n_groups", "top1", "random_baseline", "lift",
                              "tie_rate", "n_groups_top1"]]
    out.to_csv(os.path.join(OUT, "rank_rulesets.csv"), index=False)
    print("\nwrote paper/data/rank_rulesets.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
