#!/usr/bin/env python3
"""PREREG-F3 修订2: 次级冻结模型 F3H = 反对称 logistic([s_F3, e_hull])。

dev 上拟合,冻结到 F3H_frozen.json;holdout 求值由 f3_fit.py holdout 一并执行。
拒绝在 HOLDOUT_CONTACT.log 存在时运行。
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f3_fit import (OUT, load_all, is_dev, zmat, make_pairs, fit_logistic)  # noqa: E402


def main() -> int:
    if os.path.exists(os.path.join(OUT, "HOLDOUT_CONTACT.log")):
        print("holdout already contacted; refusing")
        return 1
    fz = json.load(open(os.path.join(OUT, "F3_frozen.json")))
    cols = fz["all_cols"]
    med = pd.Series(fz["impute_median"]); mu = pd.Series(fz["mu"]); sd = pd.Series(fz["sd"])
    d = load_all()
    dev = d[d.rk.map(is_dev)].reset_index(drop=True)
    Z = zmat(dev, cols, med, mu, sd)
    ji = [cols.index(f) for f in fz["features"]]
    s3 = Z[:, ji] @ np.array(fz["beta"])
    # e_hull 的 dev 标准化(冻结)
    eh = dev.e_hull.values
    eh_mu, eh_sd = float(np.mean(eh)), float(np.std(eh) or 1.0)
    s3_mu, s3_sd = float(np.mean(s3)), float(np.std(s3) or 1.0)
    H = np.column_stack([(s3 - s3_mu) / s3_sd, (eh - eh_mu) / eh_sd])
    Xp, wp = make_pairs(dev, H)
    beta = fit_logistic(Xp, wp)
    frozen = dict(beta=[float(b) for b in beta],
                  s3_mu=s3_mu, s3_sd=s3_sd, eh_mu=eh_mu, eh_sd=eh_sd)
    p = os.path.join(OUT, "F3H_frozen.json")
    json.dump(frozen, open(p, "w"), indent=1)
    open(os.path.join(OUT, "F3H_frozen.sha256"), "w").write(
        hashlib.sha256(open(p, "rb").read()).hexdigest() + "\n")
    from rank_rulesets import evaluate
    r = evaluate(dev, H @ beta, higher_better=True)
    print(f"F3H frozen: beta={beta}, dev acc {r['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
