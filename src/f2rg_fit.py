#!/usr/bin/env python3
"""F2R-G: 能隙 ≥25 meV 配对上的稳定性公式(PREREG-F2R 追加链条)。

与 f2r_fit 同一机器,仅配对限制 |dE|>=0.025 eV;fit 与 calib 两个子命令。
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f2r_fit as F2  # noqa: E402

GAP = 0.025
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "20260814_f2rg_gap25")


def eval_gap(d, vals):
    accs = []
    v = np.asarray(vals, float)
    for _, sub in d.groupby("rk"):
        idx = sub.index.values
        e = sub.e_hull.values
        de = e[:, None] - e[None, :]
        pair = np.triu(np.abs(de) >= GAP, 1)
        if not pair.any():
            continue
        dv = v[idx][:, None] - v[idx][None, :]
        commit = pair & np.isfinite(dv) & (dv != 0)
        if commit.sum():
            accs.append(((dv > 0) == (de < 0))[commit].mean())
    return dict(accuracy=float(np.mean(accs)), n_groups=len(accs))


def make_pairs_gap(d, Z):
    X, w = [], []
    for _, sub in d.groupby("rk"):
        idx = sub.index.values
        e = sub.e_hull.values
        ii, jj = np.where(np.triu(np.abs(e[:, None] - e[None, :]) >= GAP, 1))
        if not len(ii):
            continue
        lo = np.where(e[ii] < e[jj], ii, jj)
        hi = np.where(e[ii] < e[jj], jj, ii)
        X.append(Z[idx[lo]] - Z[idx[hi]])
        w.append(np.full(len(lo), 1.0 / len(lo)))
    return np.vstack(X), np.concatenate(w)


def cmd_fit(d, cols):
    dv = d[d.split == "discovery"].reset_index(drop=True)
    med, mu, sd = F2.prep(dv, cols)
    Z = F2.zmat(dv, cols, med, mu, sd)
    glab = dv.rk.values
    from sklearn.model_selection import GroupKFold
    folds = []
    for tr_g, _ in GroupKFold(n_splits=5).split(Z, groups=glab):
        tr_rk = set(glab[tr_g])
        m_tr = dv.rk.isin(tr_rk).values
        dtr = dv[m_tr].reset_index(drop=True)
        try:
            Xp, wp = make_pairs_gap(dtr, Z[m_tr])
        except ValueError:
            continue
        dte = dv[~m_tr].reset_index(drop=True)
        folds.append((Xp.astype(np.float32), wp, dte, Z[~m_tr]))

    def cv(ci):
        accs, covs = [], []
        for Xp, wp, dte, Zte in folds:
            beta = F2.fit_logistic(Xp[:, ci], wp)
            s = Zte[:, ci] @ beta
            r = eval_gap(dte, s)
            # commitment floor: reuse full-pair coverage from F2 convention
            c = F2.eval_stability(dte, s)
            accs.append(r["accuracy"]); covs.append(c["coverage"])
        if np.mean(covs) < 0.99:
            return -1.0
        return float(np.mean(accs))

    chosen, best = [], -1.0
    while len(chosen) < 8:
        sc, j = max((cv(chosen + [j]), j) for j in range(len(cols)) if j not in chosen)
        gain = sc - best
        print(f"step {len(chosen)+1}: +{cols[j]:22s} cv {sc:.4f} ({gain:+.4f})", flush=True)
        if best > 0 and gain < 0.002:
            break
        chosen.append(j); best = sc

    Xp, wp = make_pairs_gap(dv, Z[:, chosen])
    beta = F2.fit_logistic(Xp, wp)
    frozen = dict(features=[cols[j] for j in chosen], beta=[float(b) for b in beta],
                  impute_median={c: float(med[c]) for c in cols},
                  mu={c: float(mu[c]) for c in cols}, sd={c: float(sd[c]) for c in cols},
                  all_cols=cols, cv_best=best, gap=GAP,
                  disc_acc=eval_gap(dv, Z[:, chosen] @ beta)["accuracy"])
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "F2RG_frozen.json")
    json.dump(frozen, open(p, "w"), indent=1)
    open(p + ".sha256", "w").write(hashlib.sha256(open(p, "rb").read()).hexdigest() + "\n")
    print(f"F2RG frozen: {len(chosen)} terms cv {best:.4f} disc {frozen['disc_acc']:.4f}")
    return 0


def cmd_calib(d, cols):
    contact = os.path.join(OUT, "CALIB_CONTACT.log")
    if os.path.exists(contact):
        print("already contacted; refusing")
        return 1
    fz = json.load(open(os.path.join(OUT, "F2RG_frozen.json")))
    cols = fz["all_cols"]
    med = pd.Series(fz["impute_median"]); mu = pd.Series(fz["mu"]); sd = pd.Series(fz["sd"])
    ca = d[d.split == "calibration"].reset_index(drop=True)
    Z = F2.zmat(ca, cols, med, mu, sd)
    s = Z[:, [cols.index(f) for f in fz["features"]]] @ np.array(fz["beta"])
    r = eval_gap(ca, s)
    # cluster bootstrap
    per = []
    for _, sub in ca.groupby("rk"):
        idx = sub.index.values
        e = sub.e_hull.values
        de = e[:, None] - e[None, :]
        pair = np.triu(np.abs(de) >= GAP, 1)
        if not pair.any():
            continue
        dv = s[idx][:, None] - s[idx][None, :]
        commit = pair & np.isfinite(dv) & (dv != 0)
        if commit.sum():
            per.append(((dv > 0) == (de < 0))[commit].mean())
    per = np.array(per)
    rng = np.random.default_rng(20260728)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out = dict(result=r, ci=[float(lo), float(hi)], gate=bool(r["accuracy"] >= 0.80))
    json.dump(out, open(os.path.join(OUT, "calib_result.json"), "w"), indent=1)
    open(contact, "w").write("contacted once\n")
    print(f"F2RG calibration (|dE|>={GAP*1000:.0f} meV): acc {r['accuracy']:.4f} "
          f"[{lo:.4f},{hi:.4f}] n_groups {r['n_groups']} gate>=0.80: {out['gate']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "calib"])
    a = ap.parse_args()
    d = F2.load_all()
    cols = F2.admissible(d)
    return {"fit": cmd_fit, "calib": cmd_calib}[a.cmd](d, cols)


if __name__ == "__main__":
    raise SystemExit(main())
