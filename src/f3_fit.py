#!/usr/bin/env python3
"""PREREG-F3 §5: dev 审计与 F3 拟合。§6: 一次性 holdout 求值(需显式 --holdout)。

程序冻结于 docs/plans/2026-08-14-f3-synthesizability-prereg.md
(sha256 7e7a9c0b...,见 outputs/20260814_f3_synth/PREREG_SHA256)。

子命令:
  audit    dev 单特征 group-equal 准确率表(两方向)
  fit      贪心前向选择 + 全 dev 重拟合,冻结系数到 JSON
  holdout  一次性求值(拒绝在冻结文件缺失时运行;写接触记录)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, zlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_rulesets import evaluate, load  # noqa: E402  (published conventions)

F = os.environ.get("PRIS_FEATURES",
                   "features/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "20260814_f3_synth")

EXCLUDE = {"synth", "e_hull", "mp_id", "rk", "nsites", "n_sites",
           "p2_n_bad_020", "p2_sum_dev", "p3_n_pairs", "p3_n_face", "p3_n_edge",
           "p4_n_viol", "ewald_real", "ewald_recip", "ewald_point", "bv_param_cov"}


def load_all() -> pd.DataFrame:
    d = load()  # keep-filtered synth_rank (1,508 groups)
    aug = pd.read_parquet(F + "synth_rank_aug.parquet")
    d = d.merge(aug, on="mp_id", how="left")
    return d.reset_index(drop=True)


def is_dev(rk: str) -> bool:
    return zlib.crc32(f"{rk}|synthsplit20260814".encode()) % 10 < 6


def admissible(d: pd.DataFrame) -> list[str]:
    cols = [c for c in d.columns
            if c not in EXCLUDE and pd.api.types.is_numeric_dtype(d[c])]
    return sorted(cols)


def prep(dev: pd.DataFrame, cols: list[str]):
    """dev 中位数插补 + dev 标准化统计(冻结后同样应用于 holdout)。"""
    med = dev[cols].median()
    mu = dev[cols].fillna(med).mean()
    sd = dev[cols].fillna(med).std().replace(0, 1.0)
    return med, mu, sd


def zmat(d: pd.DataFrame, cols, med, mu, sd) -> np.ndarray:
    return ((d[cols].fillna(med) - mu) / sd).values


def make_pairs(d: pd.DataFrame, Z: np.ndarray):
    """组内 synth1×synth0 全配对,X=z1−z0,权重 1/组内对数。"""
    X, w = [], []
    for _, sub in d.groupby("rk"):
        idx = sub.index.values
        y = sub.synth.values.astype(bool)
        a, b = idx[y], idx[~y]
        if not len(a) or not len(b):
            continue
        diff = Z[a][:, None, :] - Z[b][None, :, :]
        diff = diff.reshape(-1, Z.shape[1])
        X.append(diff)
        w.append(np.full(len(diff), 1.0 / len(diff)))
    return np.vstack(X), np.concatenate(w)


def fit_logistic(X, w, C=1e6):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(fit_intercept=False, C=C, max_iter=2000)
    # 显式镜像 (X,1)/(−X,0):与单方向无截距 logistic 恒等,满足 sklearn 双类要求。
    X2 = np.vstack([X, -X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    m.fit(X2, y2, sample_weight=np.concatenate([w, w]))
    return m.coef_[0]


def cv_score(d, Z, cols_idx, groups, n_splits=5, seed=20260728):
    """5 折 GroupKFold;返回验证折 group-equal accuracy 的均值。"""
    from sklearn.model_selection import GroupKFold
    rks = d.rk.values
    accs = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr_g, te_g in gkf.split(np.zeros(len(groups)), groups=groups):
        tr_rk = set(groups[tr_g])
        m_tr = d.rk.isin(tr_rk).values
        dtr = d[m_tr].reset_index(drop=True)
        Ztr = Z[m_tr][:, cols_idx]
        Xp, wp = make_pairs(dtr, Ztr)
        beta = fit_logistic(Xp, wp)
        dte = d[~m_tr].reset_index(drop=True)
        s = Z[~m_tr][:, cols_idx] @ beta
        accs.append(evaluate(dte, s, higher_better=True)["accuracy"])
    return float(np.mean(accs))


def cmd_audit(d, cols):
    dev = d[d.rk.map(is_dev)].reset_index(drop=True)
    rows = []
    for c in cols:
        v = dev[c].values
        if np.isfinite(v).sum() == 0 or np.nanstd(v) == 0:
            continue
        for hb in (True, False):
            r = evaluate(dev, v, higher_better=hb)
            rows.append(dict(feature=c, higher_better=hb, **{k: r[k] for k in
                        ("coverage", "accuracy", "acc_energy_wrong", "n_groups")}))
    t = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    t.to_csv(os.path.join(OUT, "dev_single_feature_audit.csv"), index=False)
    # reference rows on dev
    for name, v, hb in [("DFT e_hull", dev.e_hull.values, False)]:
        r = evaluate(dev, v, higher_better=hb)
        print(f"dev reference {name}: cov {r['coverage']:.4f} acc {r['accuracy']:.4f}")
    print(t.head(25).to_string(index=False))
    return 0


def cmd_fit(d, cols):
    dev = d[d.rk.map(is_dev)].reset_index(drop=True)
    med, mu, sd = prep(dev, cols)
    Z = zmat(dev, cols, med, mu, sd)
    glab = dev.rk.values
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)

    # 预计算每折的全宽配对矩阵与验证侧;每个候选只取列子集(等价,快两个量级)。
    folds = []
    for tr_g, te_g in gkf.split(Z, groups=glab):
        tr_rk = set(glab[tr_g])
        m_tr = dev.rk.isin(tr_rk).values
        dtr = dev[m_tr].reset_index(drop=True)
        Xp, wp = make_pairs(dtr, Z[m_tr])
        dte = dev[~m_tr].reset_index(drop=True)
        folds.append((Xp.astype(np.float32), wp, dte, Z[~m_tr]))

    def cv(cols_idx):
        accs, covs = [], []
        for Xp, wp, dte, Zte in folds:
            beta = fit_logistic(Xp[:, cols_idx], wp)
            s = Zte[:, cols_idx] @ beta
            r = evaluate(dte, s, higher_better=True)
            accs.append(r["accuracy"]); covs.append(r["coverage"])
        # PREREG-F3 修订1:coverage < 0.99 的候选集直接拒绝,堵死
        # "弃权换准确率"路径(论文 §sec:pauling 诊断过的 Pauling-5 模式)。
        if np.mean(covs) < 0.99:
            return -1.0
        return float(np.mean(accs))

    chosen: list[int] = []
    best = -1.0
    hist = []
    while len(chosen) < 8:
        cand_scores = []
        for j in range(len(cols)):
            if j in chosen:
                continue
            sc = cv(chosen + [j])
            cand_scores.append((sc, j))
        sc, j = max(cand_scores)
        gain = sc - best
        print(f"step {len(chosen)+1}: +{cols[j]:22s} cv {sc:.4f} (gain {gain:+.4f})",
              flush=True)
        if best > 0 and gain < 0.002:
            print("stop: gain < 0.002")
            break
        chosen.append(j)
        best = sc
        hist.append(dict(step=len(chosen), feature=cols[j], cv_acc=sc))

    Xp, wp = make_pairs(dev, Z[:, chosen])
    beta = fit_logistic(Xp, wp)
    dev_acc = evaluate(dev, Z[:, chosen] @ beta, higher_better=True)["accuracy"]

    # F3-full ceiling (L2, C by CV over a small grid)
    bestC, bestCV = None, -1
    for C in (0.01, 0.1, 1.0, 10.0):
        accs = []
        for Xp, wp, dte, Zte in folds:
            b = fit_logistic(Xp, wp, C=C)
            accs.append(evaluate(dte, Zte @ b, higher_better=True)["accuracy"])
        m = float(np.mean(accs))
        if m > bestCV:
            bestCV, bestC = m, C
    Xq, wq = make_pairs(dev, Z)
    beta_full = fit_logistic(Xq, wq, C=bestC)
    dev_acc_full = evaluate(dev, Z @ beta_full, higher_better=True)["accuracy"]

    frozen = dict(
        prereg_sha256="7e7a9c0b4587d3a65ef824fe6538c54bda05acd13e5e187da754d494177abce5",
        features=[cols[j] for j in chosen],
        beta=[float(b) for b in beta],
        impute_median={c: float(med[c]) for c in cols},
        mu={c: float(mu[c]) for c in cols},
        sd={c: float(sd[c]) for c in cols},
        all_cols=cols,
        cv_history=hist, cv_best=best, dev_acc=dev_acc,
        full_C=bestC, full_cv=bestCV, dev_acc_full=dev_acc_full,
        beta_full=[float(b) for b in beta_full],
    )
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "F3_frozen.json")
    with open(p, "w") as fh:
        json.dump(frozen, fh, indent=1)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    open(os.path.join(OUT, "F3_frozen.sha256"), "w").write(h + "\n")
    print(f"\nF3 frozen: {len(chosen)} terms, cv {best:.4f}, dev acc {dev_acc:.4f}")
    print(f"F3-full  : C={bestC}, cv {bestCV:.4f}, dev acc {dev_acc_full:.4f}")
    print(f"sha256 {h}")
    return 0


def cmd_holdout(d, cols):
    contact = os.path.join(OUT, "HOLDOUT_CONTACT.log")
    if os.path.exists(contact):
        print("holdout already contacted once; refusing (PREREG-F3 §2)")
        return 1
    fz = json.load(open(os.path.join(OUT, "F3_frozen.json")))
    cols = fz["all_cols"]
    med = pd.Series(fz["impute_median"])
    mu, sd = pd.Series(fz["mu"]), pd.Series(fz["sd"])
    hold = d[~d.rk.map(is_dev)].reset_index(drop=True)
    Z = zmat(hold, cols, med, mu, sd)
    ji = [cols.index(f) for f in fz["features"]]
    s3 = Z[:, ji] @ np.array(fz["beta"])
    sfull = Z @ np.array(fz["beta_full"])

    res = {}
    res["F3"] = evaluate(hold, s3, higher_better=True)
    res["F3_full"] = evaluate(hold, sfull, higher_better=True)
    res["e_hull"] = evaluate(hold, hold.e_hull.values, higher_better=False)
    res["vol_per_atom"] = evaluate(hold, hold.vol_per_atom.values, higher_better=False)
    res["sh_pack"] = evaluate(hold, hold.sh_pack.values, higher_better=True)
    res["bl_min"] = evaluate(hold, hold.bl_min.values, higher_better=True)

    # PREREG-F3 修订2: F3H 混合(同一次接触)
    sH = None
    fH = os.path.join(OUT, "F3H_frozen.json")
    if os.path.exists(fH):
        hz = json.load(open(fH))
        eh = hold.e_hull.values
        H = np.column_stack([(s3 - hz["s3_mu"]) / hz["s3_sd"],
                             (eh - hz["eh_mu"]) / hz["eh_sd"]])
        sH = H @ np.array(hz["beta"])
        res["F3H"] = evaluate(hold, sH, higher_better=True)

    # paired cluster bootstrap of acc(F3) - acc(e_hull) [and F3H - e_hull], B=2000
    rng = np.random.default_rng(20260728)
    per_group = []
    ehv = hold.e_hull.values
    for _, sub in hold.groupby("rk"):
        idx = sub.index.values
        y = sub.synth.values.astype(bool)
        a, b = idx[y], idx[~y]
        if not len(a) or not len(b):
            continue
        row = []
        for v, hb in ((s3, True), (ehv, False), (sH, True) if sH is not None else (None, True)):
            if v is None:
                row.append(np.nan)
                continue
            dv = v[a][:, None] - v[b][None, :]
            c = (dv != 0) & np.isfinite(dv)
            row.append(((dv > 0) if hb else (dv < 0))[c].mean() if c.any() else np.nan)
        per_group.append(row)
    pg = np.array(per_group, float)
    both = pg[np.isfinite(pg[:, :2]).all(1)]
    idxs = [rng.integers(0, len(both), len(both)) for _ in range(2000)]
    diffs = [both[i, 0].mean() - both[i, 1].mean() for i in idxs]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    delta = both[:, 0].mean() - both[:, 1].mean()
    hyb = None
    if sH is not None:
        bh = pg[np.isfinite(pg).all(1)]
        idxs = [rng.integers(0, len(bh), len(bh)) for _ in range(2000)]
        dh = [bh[i, 2].mean() - bh[i, 1].mean() for i in idxs]
        hyb = dict(delta=float(bh[:, 2].mean() - bh[:, 1].mean()),
                   ci=[float(np.percentile(dh, 2.5)), float(np.percentile(dh, 97.5))])
        hyb["G6"] = bool(hyb["ci"][0] > 0)

    gates = dict(
        G1=bool(lo > 0), delta=float(delta), ci=[float(lo), float(hi)],
        G2=bool(res["F3"]["coverage"] >= 0.99),
        G3=bool(res["F3"]["lift"] >= res["e_hull"]["lift"]),
        G4=bool(all(res["F3"]["accuracy"] > res[k]["accuracy"]
                    for k in ("vol_per_atom", "sh_pack", "bl_min"))),
        n_groups_paired=int(len(both)),
    )
    out = dict(results={k: {m: float(v) for m, v in r.items()} for k, r in res.items()},
               gates=gates, hybrid=hyb)
    with open(os.path.join(OUT, "holdout_result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(contact, "w") as fh:
        fh.write("holdout contacted once by f3_fit.py holdout\n")
    for k, r in res.items():
        print(f"{k:12s} cov {r['coverage']:.4f} acc {r['accuracy']:.4f} "
              f"top1 {r['top1']:.4f} lift {r['lift']:+.4f}")
    print(f"\nDelta acc(F3 - e_hull) = {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    if hyb:
        print(f"Delta acc(F3H - e_hull) = {hyb['delta']:+.4f}  "
              f"95% CI [{hyb['ci'][0]:+.4f}, {hyb['ci'][1]:+.4f}]  G6={hyb['G6']}")
    print("gates:", {k: v for k, v in gates.items() if k.startswith('G')})
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "fit", "holdout"])
    a = ap.parse_args()
    d = load_all()
    cols = admissible(d)
    return {"audit": cmd_audit, "fit": cmd_fit, "holdout": cmd_holdout}[a.cmd](d, cols)


if __name__ == "__main__":
    raise SystemExit(main())
