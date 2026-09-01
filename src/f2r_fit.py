#!/usr/bin/env python3
"""PREREG-F2R: 实验域 e_hull 排序公式重建。

  audit  discovery 单特征稳定性排序准确率
  fit    discovery 贪心前向 + 冻结(含 F2 重述基线的冻结 z 统计)
  calib  calibration 一次性求值(F2R vs F2重述 vs 单量;配对聚类自助)

冻结文档 docs/plans/2026-08-14-f2r-stability-prereg.md
(sha256 e292af8d...,见 outputs/20260814_f3_synth/PREREG_SHA256)。
"""
from __future__ import annotations
import os
import argparse, hashlib, json, os, sys, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

F = os.environ.get("PRIS_FEATURES", "features/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "20260814_f2r_stability")

DROP = {"source_id", "rk", "e_hull", "split", "anion", "sid", "parent", "kind"}

# 论文 F2 的七项(系数为已发表值;列名映射按 S3 特征字典)
F2_TERMS = [("min_opp_frac", -0.322), ("econ_mean", 0.149), ("mef_mean", 0.137),
            ("angvar_mean", 0.060), ("econ_max", 0.039), ("dist_rsd", -0.030),
            ("p5_n_distinct", 0.002)]


def load_all() -> pd.DataFrame:
    d = pd.read_parquet(F + "real_rank.parquet")
    for f_ in ("phys_real.parquet", "elec_real.parquet", "geom_real.parquet"):
        if os.path.exists(F + f_):
            d = d.merge(pd.read_parquet(F + f_), on="source_id", how="inner")
    bad = d.split.isna() | (d.split == "lockbox")
    d = d[~bad]
    aug = pd.read_parquet(F + "real_rank_aug.parquet")
    d = d.merge(aug, on="source_id", how="left")
    # 去重列名冲突(merge 后缀)
    d = d.loc[:, ~d.columns.duplicated()]
    return d.reset_index(drop=True)


def admissible(d):
    bad = DROP | {"nsites", "n_sites", "p2_n_bad_020", "p2_sum_dev", "p3_n_pairs",
                  "p3_n_face", "p3_n_edge", "p4_n_viol", "ewald_real", "ewald_recip",
                  "ewald_point", "bv_param_cov"}
    return sorted(c for c in d.columns
                  if c not in bad and pd.api.types.is_numeric_dtype(d[c]))


def eval_stability(d: pd.DataFrame, vals, higher_more_stable=True):
    """组等权:配对 = 组内 e_hull 严格不同的对;commit = 分数不同。"""
    accs, covs = [], []
    v = np.asarray(vals, float)
    for _, sub in d.groupby("rk"):
        idx = sub.index.values
        e = sub.e_hull.values
        de = e[:, None] - e[None, :]
        pair = np.triu(de != 0, 1)
        if not pair.any():
            continue
        dv = v[idx][:, None] - v[idx][None, :]
        fin = np.isfinite(dv)
        commit = pair & fin & (dv != 0)
        covs.append(commit.sum() / max(1, (pair & fin).sum()))
        if commit.sum():
            more_stable = de < 0  # a 的 hull 更低
            win = (dv > 0) == more_stable if higher_more_stable else (dv < 0) == more_stable
            accs.append(win[commit].mean())
    return dict(coverage=float(np.mean(covs)), accuracy=float(np.mean(accs)),
                n_groups=len(accs))


def prep(dv, cols):
    med = dv[cols].median()
    mu = dv[cols].fillna(med).mean()
    sd = dv[cols].fillna(med).std().replace(0, 1.0)
    return med, mu, sd


def zmat(d, cols, med, mu, sd):
    return ((d[cols].fillna(med) - mu) / sd).values


def make_pairs(d, Z):
    X, w = [], []
    for _, sub in d.groupby("rk"):
        idx = sub.index.values
        e = sub.e_hull.values
        ii, jj = np.where(np.triu(e[:, None] - e[None, :] != 0, 1))
        if not len(ii):
            continue
        lo = np.where(e[ii] < e[jj], ii, jj)
        hi = np.where(e[ii] < e[jj], jj, ii)
        X.append(Z[idx[lo]] - Z[idx[hi]])
        w.append(np.full(len(lo), 1.0 / len(lo)))
    return np.vstack(X), np.concatenate(w)


def fit_logistic(X, w, C=1e6):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(fit_intercept=False, C=C, max_iter=2000)
    X2 = np.vstack([X, -X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    m.fit(X2, y2, sample_weight=np.concatenate([w, w]))
    return m.coef_[0]


def cmd_audit(d, cols):
    dv = d[d.split == "discovery"].reset_index(drop=True)
    rows = []
    for c in cols:
        v = dv[c].values
        if np.isfinite(v).sum() == 0 or np.nanstd(v) == 0:
            continue
        for hb in (True, False):
            r = eval_stability(dv, v, higher_more_stable=hb)
            rows.append(dict(feature=c, higher_more_stable=hb, **r))
    t = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, "discovery_single_feature_audit.csv"), index=False)
    print(t.head(25).to_string(index=False))
    return 0


def cmd_fit(d, cols):
    dv = d[d.split == "discovery"].reset_index(drop=True)
    med, mu, sd = prep(dv, cols)
    Z = zmat(dv, cols, med, mu, sd)
    glab = dv.rk.values
    from sklearn.model_selection import GroupKFold
    folds = []
    for tr_g, _ in GroupKFold(n_splits=5).split(Z, groups=glab):
        tr_rk = set(glab[tr_g])
        m_tr = dv.rk.isin(tr_rk).values
        dtr = dv[m_tr].reset_index(drop=True)
        Xp, wp = make_pairs(dtr, Z[m_tr])
        dte = dv[~m_tr].reset_index(drop=True)
        folds.append((Xp.astype(np.float32), wp, dte, Z[~m_tr]))

    def cv(ci):
        accs, covs = [], []
        for Xp, wp, dte, Zte in folds:
            beta = fit_logistic(Xp[:, ci], wp)
            r = eval_stability(dte, Zte[:, ci] @ beta)
            accs.append(r["accuracy"]); covs.append(r["coverage"])
        # PREREG-F2R 修订1:coverage < 0.99 直接拒绝(同 PREREG-F3 修订1)。
        if np.mean(covs) < 0.99:
            return -1.0
        return float(np.mean(accs))

    chosen, best, hist = [], -1.0, []
    while len(chosen) < 8:
        sc, j = max((cv(chosen + [j]), j) for j in range(len(cols)) if j not in chosen)
        gain = sc - best
        print(f"step {len(chosen)+1}: +{cols[j]:22s} cv {sc:.4f} ({gain:+.4f})", flush=True)
        if best > 0 and gain < 0.002:
            print("stop: gain < 0.002")
            break
        chosen.append(j); best = sc
        hist.append(dict(step=len(chosen), feature=cols[j], cv_acc=sc))

    Xp, wp = make_pairs(dv, Z[:, chosen])
    beta = fit_logistic(Xp, wp)
    disc_acc = eval_stability(dv, Z[:, chosen] @ beta)["accuracy"]

    bestC, bestCV = None, -1
    for C in (0.01, 0.1, 1.0, 10.0):
        accs = [eval_stability(dte, Zte @ fit_logistic(Xp2, wp2, C=C))["accuracy"]
                for Xp2, wp2, dte, Zte in folds]
        m = float(np.mean(accs))
        if m > bestCV:
            bestCV, bestC = m, C
    Xq, wq = make_pairs(dv, Z)
    beta_full = fit_logistic(Xq, wq, C=bestC)

    # F2 重述:已发表系数 + discovery 冻结 z 统计;orientation 在 discovery 上确定
    f2cols = [c for c, _ in F2_TERMS]
    s_f2 = np.zeros(len(dv))
    for c, coef in F2_TERMS:
        s_f2 += coef * zmat(dv, [c], med[[c]], mu[[c]], sd[[c]])[:, 0]
    a_hi = eval_stability(dv, s_f2, True)["accuracy"]
    a_lo = eval_stability(dv, s_f2, False)["accuracy"]
    f2_orient = bool(a_hi >= a_lo)
    print(f"F2 restated on discovery: higher {a_hi:.4f} / lower {a_lo:.4f} "
          f"-> orient higher_better={f2_orient}")

    frozen = dict(
        prereg_sha256="e292af8ddff453cfca66bd7a04c4dd048fde6c65ec130cde774969d331203161",
        features=[cols[j] for j in chosen], beta=[float(b) for b in beta],
        impute_median={c: float(med[c]) for c in cols},
        mu={c: float(mu[c]) for c in cols}, sd={c: float(sd[c]) for c in cols},
        all_cols=cols, cv_history=hist, cv_best=best, disc_acc=disc_acc,
        full_C=bestC, full_cv=bestCV, beta_full=[float(b) for b in beta_full],
        f2_terms=F2_TERMS, f2_orient_higher=f2_orient,
        f2_disc_acc=max(a_hi, a_lo),
    )
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "F2R_frozen.json")
    json.dump(frozen, open(p, "w"), indent=1)
    open(os.path.join(OUT, "F2R_frozen.sha256"), "w").write(
        hashlib.sha256(open(p, "rb").read()).hexdigest() + "\n")
    print(f"F2R frozen: {len(chosen)} terms, cv {best:.4f}, discovery acc {disc_acc:.4f}")
    return 0


def cmd_calib(d, cols):
    contact = os.path.join(OUT, "CALIB_CONTACT.log")
    if os.path.exists(contact):
        print("calibration already contacted; refusing")
        return 1
    fz = json.load(open(os.path.join(OUT, "F2R_frozen.json")))
    cols = fz["all_cols"]
    med = pd.Series(fz["impute_median"]); mu = pd.Series(fz["mu"]); sd = pd.Series(fz["sd"])
    ca = d[d.split == "calibration"].reset_index(drop=True)
    Z = zmat(ca, cols, med, mu, sd)
    ji = [cols.index(f) for f in fz["features"]]
    s_new = Z[:, ji] @ np.array(fz["beta"])
    s_full = Z @ np.array(fz["beta_full"])
    s_f2 = np.zeros(len(ca))
    for c, coef in fz["f2_terms"]:
        s_f2 += coef * zmat(ca, [c], med[[c]], mu[[c]], sd[[c]])[:, 0]
    if not fz["f2_orient_higher"]:
        s_f2 = -s_f2

    res = {"F2R": eval_stability(ca, s_new), "F2R_full": eval_stability(ca, s_full),
           "F2_restated": eval_stability(ca, s_f2),
           "bl_min": eval_stability(ca, ca.bl_min.values, True),
           "vol_per_atom": eval_stability(ca, ca.vol_per_atom.values, False)}

    # 配对聚类自助 acc(F2R) − acc(F2_restated)
    rng = np.random.default_rng(20260728)
    pg = []
    for _, sub in ca.groupby("rk"):
        idx = sub.index.values
        e = sub.e_hull.values
        de = e[:, None] - e[None, :]
        pair = np.triu(de != 0, 1)
        if not pair.any():
            continue
        row = []
        for s in (s_new, s_f2):
            dvv = s[idx][:, None] - s[idx][None, :]
            commit = pair & np.isfinite(dvv) & (dvv != 0)
            row.append(((dvv > 0) == (de < 0))[commit].mean() if commit.any() else np.nan)
        pg.append(row)
    pg = np.array(pg, float)
    both = pg[np.isfinite(pg).all(1)]
    diffs = [both[rng.integers(0, len(both), len(both))].mean(0) for _ in range(2000)]
    diffs = [a - b for a, b in diffs]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    delta = both[:, 0].mean() - both[:, 1].mean()

    out = dict(results=res, delta=float(delta), ci=[float(lo), float(hi)],
               H1=bool(lo > 0),
               H2=bool(res["F2R"]["accuracy"] > max(res["bl_min"]["accuracy"],
                                                    res["vol_per_atom"]["accuracy"])),
               n_groups_paired=int(len(both)))
    json.dump(out, open(os.path.join(OUT, "calibration_result.json"), "w"), indent=1)
    open(contact, "w").write("calibration contacted once by f2r_fit.py calib\n")
    for k, r in res.items():
        print(f"{k:12s} cov {r['coverage']:.4f} acc {r['accuracy']:.4f} n {r['n_groups']}")
    print(f"Delta(F2R - F2_restated) = {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "fit", "calib"])
    a = ap.parse_args()
    d = load_all()
    cols = admissible(d)
    return {"audit": cmd_audit, "fit": cmd_fit, "calib": cmd_calib}[a.cmd](d, cols)


if __name__ == "__main__":
    raise SystemExit(main())
