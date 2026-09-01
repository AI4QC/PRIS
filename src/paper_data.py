#!/usr/bin/env python3
"""生成论文全部主图所需的数据表 —— 一次算齐,存 CSV,图从 CSV 画。

论文主线(经三次自我推翻后定下的):
  泡林规则的失效主要不是**精度**问题,而是**适用性**问题 ——
  它们在四分之三的情形下不表态。本文给出覆盖率 100% 的替代判据。

图 1  泡林规则在 9.9 万实验结构上的满足率 + 对"键"定义的敏感性
图 2  满足率-排除力前沿:本文法则 vs 泡林,含分化学层
图 3  覆盖率-准确率平面:泡林各条 vs bl_min vs DFT 能量(合成靶)
图 4  四库同能量区间对照 + 组大小集中度诊断
图 5  自我推翻清单:九个看似成立的结论及其证伪检验
"""
from __future__ import annotations
import os
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATURES = os.environ.get("PRIS_FEATURES")
OUT = ROOT / "paper" / "data"


def feature_file(name):
    """返回外部特征文件路径；只有需要结构级数据的函数才调用。"""
    if not FEATURES:
        raise RuntimeError("请先设置 PRIS_FEATURES=/path/to/features")
    return pathlib.Path(FEATURES) / name


def fig3_coverage_accuracy():
    """图 3:覆盖率-准确率平面。每个判据一个点。"""
    d = pd.read_parquet(feature_file("synth_rank.parquet")).reset_index(drop=True)
    ok = d.groupby("rk").synth.agg(["size", "sum"])
    ok = ok[(ok["sum"] >= 1) & (ok["size"] - ok["sum"] >= 1)].index
    d = d[d.rk.isin(ok)].reset_index(drop=True)
    V = {c: d[c].values for c in d.columns if d[c].dtype.kind == "f"}
    E, S = d.e_hull.values, d.synth.values
    RULES = [("Pauling 2 (bond-strength dev.)", "p2_mean_dev", +1),
             ("Pauling 3 (edge/face sharing)", "p3_frac_edge_face", +1),
             ("Pauling 4 (high-valence contact)", "p4_violate", +1),
             ("Pauling 5 (parsimony)", "p5_n_distinct", +1),
             ("This work: bl_min", "bl_min", -1),
             ("Volume per atom", "vol_per_atom", +1),
             ("Shannon packing", "sh_pack", -1),
             ("DFT E_hull (baseline)", "e_hull", +1)]
    rows = []
    groups = d.groupby("rk").groups
    for nm, col, sgn in RULES:
        if col not in V:
            continue
        rec = []
        for rk, idx in groups.items():
            ii = d.index.get_indexer(np.asarray(idx))
            n = tie = w = 0
            nw = ww = 0
            for p in [i for i in ii if S[i] == 1]:
                for q in [i for i in ii if S[i] == 0]:
                    if not (np.isfinite(V[col][p]) and np.isfinite(V[col][q])
                            and np.isfinite(E[p]) and np.isfinite(E[q])):
                        continue
                    dv = sgn * (V[col][p] - V[col][q])
                    n += 1
                    if abs(dv) <= 1e-9:
                        tie += 1
                        continue
                    w += int(dv < 0)
                    if E[p] >= E[q]:            # 能量判错的配对
                        nw += 1
                        ww += int(dv < 0)
            if n:
                rec.append((1 - tie / n, w / (n - tie) if n > tie else np.nan,
                            ww / nw if nw else np.nan))
        r = pd.DataFrame(rec, columns=["coverage", "acc", "acc_ewrong"])
        rows.append(dict(rule=nm, coverage=r.coverage.mean(),
                         accuracy=r.acc.mean(), acc_energy_wrong=r.acc_ewrong.mean(),
                         n_groups=int(r.acc.notna().sum())))
    out = pd.DataFrame(rows)
    path = OUT / "fig3_coverage_accuracy.csv"
    out.to_csv(path, index=False)
    print("图3:", path)
    print(out.round(4).to_string(index=False))
    return out


def fig4_db_concentration():
    """图 4b:组大小集中度 —— 每库最大组占配对的比例。"""
    rows = []
    for f, nm, ec in [("real_rank.parquet", "MP experimental", "e_hull"),
                      ("alex_rank.parquet", "Alexandria", "epa"),
                      ("elem_rank.parquet", "ELEMENTA", "epa"),
                      ("lemat_rank.parquet", "LeMat", "e_per_atom"),
                      ("synth_rank.parquet", "MP synthesis target", None)]:
        path = feature_file(f)
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        if "split" in d.columns:
            d = d[d.split != "lockbox"]
        if ec is None:
            ok = d.groupby("rk").synth.agg(["size", "sum"])
            ok = ok[(ok["sum"] >= 1) & (ok["size"] - ok["sum"] >= 1)].index
            d = d[d.rk.isin(ok)]
            npair = d.groupby("rk").synth.agg(lambda x: (x == 1).sum() * (x == 0).sum())
            span = np.nan
        else:
            sz = d.groupby("rk").size()
            d = d[d.rk.isin(sz[sz >= 2].index)]
            npair = d.groupby("rk").size().map(lambda n: n * (n - 1) // 2)
            span = d.groupby("rk")[ec].agg(lambda x: x.max() - x.min()).median() * 1000
        npair = npair.sort_values(ascending=False)
        rows.append(dict(database=nm, n_groups=int(len(npair)),
                         n_structures=int(len(d)), n_pairs=int(npair.sum()),
                         largest_group=str(npair.index[0]),
                         frac_top1=float(npair.iloc[0] / npair.sum()),
                         frac_top5=float(npair.head(5).sum() / npair.sum()),
                         median_span_meV=span))
    out = pd.DataFrame(rows)
    path = OUT / "fig4_db_concentration.csv"
    out.to_csv(path, index=False)
    print("\n图4b:", path)
    print(out.round(4).to_string(index=False))
    return out


def fig1_pauling_audit():
    """图 1:泡林规则在实验结构上的满足率 + 对"键"定义的敏感性。

    数据来自本工作的 George 复现(4.1)与 G6 算法敏感性分析(4.2)。
    """
    rows = [
        dict(rule="Pauling 1 (radius ratio)", satisfaction=0.617, sat_hi=0.648,
             algo_spread_pp=4.05, george=0.66),
        dict(rule="Pauling 2 (bond strength)", satisfaction=0.179, sat_hi=0.202,
             algo_spread_pp=0.83, george=0.20),
        dict(rule="Pauling 3 (edge/face)", satisfaction=0.734, sat_hi=0.734,
             algo_spread_pp=3.93, george=0.733),
        dict(rule="Pauling 4 (high-valence)", satisfaction=0.347, sat_hi=0.403,
             algo_spread_pp=5.74, george=0.40),
        dict(rule="Pauling 5 (parsimony)", satisfaction=0.678, sat_hi=0.678,
             algo_spread_pp=1.81, george=0.703),
        dict(rule="Rules 2-5 jointly", satisfaction=0.0651, sat_hi=0.116,
             algo_spread_pp=float("nan"), george=0.13),
        dict(rule="This work: rho>=0.735", satisfaction=0.9919, sat_hi=0.9919,
             algo_spread_pp=0.18, george=float("nan")),
    ]
    out = pd.DataFrame(rows)
    path = OUT / "fig1_pauling_audit.csv"
    out.to_csv(path, index=False)
    print("图1:", path)
    print(out.round(4).to_string(index=False))
    return out


def fig2_frontier():
    """图 2:满足率-排除力前沿。discovery 值，与 FACTS §16 口径一致。"""
    rows = [
        dict(setting="Pauling rules 2-5 (joint)", n_rules=4, satisfaction=0.0651,
             exclusion=float("nan"), worst_chem=float("nan")),
        dict(setting="floor 0.99", n_rules=5, satisfaction=0.9917, exclusion=0.3040,
             worst_chem=0.986),
        dict(setting="floor 0.98", n_rules=3, satisfaction=0.9813, exclusion=0.4161,
             worst_chem=0.973),
        dict(setting="floor 0.95 (recommended)", n_rules=5, satisfaction=0.9539,
             exclusion=0.6224, worst_chem=0.928),
        dict(setting="floor 0.95 + geometric", n_rules=6, satisfaction=0.9536,
             exclusion=0.6254, worst_chem=0.928),
        dict(setting="trusted core (no CN rules)", n_rules=4, satisfaction=0.9511,
             exclusion=0.6173, worst_chem=float("nan")),
        dict(setting="core + ionicity-guarded charge rule", n_rules=5,
             satisfaction=0.9071, exclusion=0.7052, worst_chem=float("nan")),
    ]
    out = pd.DataFrame(rows)
    path = OUT / "fig2_frontier.csv"
    out.to_csv(path, index=False)
    print("\n图2:", path)
    print(out.round(4).to_string(index=False))
    return out


def main():
    # 先验证外部数据，避免前两张硬编码表已覆盖、到第三张才失败的半成品状态。
    feature_file("synth_rank.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_pauling_audit()
    fig2_frontier()
    fig3_coverage_accuracy()
    fig4_db_concentration()


if __name__ == "__main__":
    main()
