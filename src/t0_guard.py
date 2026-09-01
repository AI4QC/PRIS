#!/usr/bin/env python3
"""纯组分(T0)量 —— 不当法则用,当**前提**用。

# 为什么

本工作先前的结论:T0 量在这两个靶上**恒为零权重**,因为法则的排除力算在同组成的
破坏样本上、公式在同组成组内配对,组分量两边精确抵消。那是靶设计的必然结果。

但"当法则无效"不等于"没用"。直接证据来自 `frac_like_bonds`(同号离子成键占比):

  - 它单独能排除 **96%** 的 S5(阴阳离子互换),是唯一对这一类有效的量
  - 但它在真实结构上只有 **79.3%** 满足,进不了 0.95 的满足率下限
  - 而那 20.7% 的违例**不是随机分布的**:磷化物 66.8% / 碲化物 53.0% /
    硒化物 33.8% / 硫化物 28.1% / 氮化物 25.9%,而氟化物只有 16.0% ——
    **集中在最不离子的那些化学**,那里阳离子-阳离子成键本来就是真实的

所以"不得有同号成键"是一条**离子晶体**的定律,它缺的是一个前提:
**"若该化合物足够离子性"**。而"离子性"恰好是纯组分量。

# 算什么

  dchi  = 化学计量加权的 (电负性(阴离子) - 电负性(阳离子)) 平均
  fi    = 1 - exp(-0.25 * dchi^2)      泡林离子性分数

两者都只需化学式,不需结构 —— 所以对同组成的破坏样本取值完全相同,
当法则用排除力恒为 0,当前提用则把法则限制到它真正成立的化学域。
"""
from __future__ import annotations
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
F = os.environ.get("PRIS_FEATURES", "features/")


def parse(f):
    c = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(f)):
        if el:
            c[el] += float(n) if n else 1.0
    return c


def main() -> int:
    prov = pd.read_parquet(F + "provenance.parquet",
                           columns=["source_id", "formula", "anion", "n_elements"])
    ep = pd.read_parquet(F + "_elem_props.parquet")[["element", "X"]]
    X = dict(zip(ep.element, ep.X))

    rows = []
    for t in prov.itertuples():
        an = t.anion
        if not isinstance(an, str) or an not in X or not np.isfinite(X[an]):
            continue
        c = parse(t.formula)
        cats = {e: n for e, n in c.items()
                if e != an and e in X and np.isfinite(X[e])}
        if not cats:
            continue
        tot = sum(cats.values())
        d = sum(n * (X[an] - X[e]) for e, n in cats.items()) / tot
        rows.append({"source_id": t.source_id, "dchi": float(d),
                     "fi": float(1 - np.exp(-0.25 * d * d)),
                     "dchi_min": float(min(X[an] - X[e] for e in cats))})
    d = pd.DataFrame(rows)
    d.to_parquet(F + "t0_guard.parquet", index=False)
    print(f"写出 {len(d):,} 行")
    print(d[["dchi", "fi", "dchi_min"]].describe().round(3).to_string())

    # 与 frac_like_bonds 的关系 —— 验证"离子性越强,同号成键越少"
    import os
    if os.path.exists(F + "phys_real.parquet"):
        pr = pd.read_parquet(F + "phys_real.parquet")[["source_id", "frac_like_bonds"]]
        m = d.merge(pr, on="source_id", how="inner").dropna()
        print(f"\n可对照 {len(m):,} 条。按离子性分数分箱:")
        m["bin"] = pd.qcut(m.fi, 5, duplicates="drop")
        g = m.groupby("bin").agg(n=("fi", "size"),
                                 fi中位=("fi", "median"),
                                 无同号键占比=("frac_like_bonds", lambda x: float((x == 0).mean())))
        print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
