# -*- coding: utf-8 -*-
"""Table S1:五条泡林规则 × 3 近邻算法 × 2 symprec × 2 氧化态来源 = 60 个数字。

被 `reproduce_george.py --stage table` 调用。输入是 compute 阶段的四张 `george_*.parquet`。

**粒度声明(PREREG §4.3 要求显式,George 2020 混了粒度)**
| 规则 | granularity | symprec 是否起作用 |
|---|---|---|
| 1 半径比 | `orbit`(阳离子位点按晶体学轨道去重),备用 `site` | **是**(轨道定义) |
| 2 静电价 | `orbit`(**阴离子**位点),备用 `site` | **是** |
| 3 连接类型 | `pair`(George 原口径),symprec 列改用 `orbit-pair` 去重 | **是**(仅在 orbit-pair 口径下) |
| 4 相邻多面体 | `structure` | **否**,两列必然相同,如实标注 |
| 5 简约 | `structure` | **否**,同上 |
| 2–5 合取 | `structure` | 否 |

symprec 对规则 4/5 结构性无影响(它们不按位点计数),所以 60 格里有 24 格是
**按构造相同**的。这不是复制粘贴,是口径的真实结论,报告里必须明说。
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
R_ANION = 1.76       # 泡林单价半径 O2-(pauling_radii.PAULING_UNIVALENT)


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


# ================================================================ 载入
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


# ================================================================ 规则 1
def rule1_sites(si, tier="published"):
    """给阳离子位点补泡林单价半径与预测 CN。tier='published' 为主统计口径。"""
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


# ================================================================ 规则 4 / 5(结构级)
def struct_rules(si, pr, algo):
    """返回每个结构的 rule4/rule5 布尔。granularity=structure。

    规则 4(George 原文口径):V = 结构内阳离子位点最大氧化态,C = 最小 CN;
    A = {ox==V 且 cn==C};A 内部若存在相连对 → 违例。仅在 ≥2 个阳离子物种时适用。
    另给两个分解版本:`ox` 只用 ox==V,`cn` 只用 cn==C(George 证明只有后者与数据相符)。
    规则 5:每个 (元素, 氧化态) 物种在结构内只有一种 CN(原文 Fig.5b "only coordination
    numbers are considered")。
    """
    cnc = f"cn_{algo}"
    s = si[["source_id", "element", "ox_state", cnc]].dropna(subset=[cnc]).copy()
    s = s[s.ox_state.notna()]
    s["oxr"] = s.ox_state.round(3)
    g = s.groupby("source_id")
    agg = pd.DataFrame({
        "V": g.oxr.max(), "C": g[cnc].min(),
        "n_species": g.apply(lambda d: d.groupby(["element", "oxr"]).ngroups, include_groups=False),
        # 规则 5:任一物种出现 >1 种 CN 即违例
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
    multi = agg.n_species >= 2                     # 原文 "In a crystal containing different cations"
    for tag in ("george", "ox", "cn"):
        agg[f"r4_{tag}_ok"] = ~(agg[f"viol_{tag}"] & multi)
    return agg


# ================================================================ 主流程
def stage_table(smoke=False):
    st, si, an, pr = load(smoke)
    n_all = len(st)
    print("=" * 100)
    print(f"论域:provenance.oxide_strict,compute 成功 {n_all} 条(阴离子恒为 O)")
    print("口径差异 vs George 2020:ICSD+COD(不含 MP)/ 实验原胞未弛豫 / 有序结构 only / 单一阴离子 O /")
    print("                       氧化态只用 cif|guess(BVAnalyzer 整批排除,PREREG §5)")
    print(st.ox_source.value_counts(dropna=False).to_string())
    print(f"cif_all_zero(ICSD 原生氧化态全 0,已降级)= {int(st.cif_all_zero.sum())} "
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
    print(f"[规则1] 泡林单价半径覆盖(published):{cov:.1%} 的阳离子位点;"
          f"加外推后 {si1x.r_cat.notna().mean():.1%}")
    miss = (si1[si1.r_cat.isna()].groupby(["element"]).size().sort_values(ascending=False).head(12))
    print("       未覆盖元素 Top12:", dict(miss))

    # ---- 结构级规则 4/5(按算法算一次,与 symprec 无关)
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
                # ---- 规则 1:orbit 粒度
                d = si1[(si1.oxs == oxs) & si1.r_cat.notna() & si1[cnc].notna() & si1[ob].notna()]
                d = d.drop_duplicates(subset=["source_id", ob])
                # 主口径 = 只在硬球稳定窗有定义的 CN 上判(George 原文措辞,见下方 cmp 区注释);
                # 严格全 CN 版另存一行 `R1_radius_ratio_allCN` 供审计
                dw_ = d[d[cnc].isin({2, 3, 4, 6, 8, 12})]
                add("R1_radius_ratio", algo, sp, oxs,
                    (dw_.cn_pred == dw_[cnc]).sum(), len(dw_), "orbit")
                add("R1_radius_ratio_allCN", algo, sp, oxs,
                    (d.cn_pred == d[cnc]).sum(), len(d), "orbit")
                # ---- 规则 2:阴离子 orbit 粒度
                e = an[(an.oxs == oxs) & (an[ncc] > 0) & an[sgc].notna() & an[ob].notna()]
                e = e.drop_duplicates(subset=["source_id", ob])
                add("R2_electrostatic_valence", algo, sp, oxs,
                    ((e[sgc] - 2.0).abs() <= EPS).sum(), len(e), "orbit")
                # ---- 规则 3:orbit-pair 去重(pair 原口径另行报告)
                q = prx.dropna(subset=[f"orbit_i_{sp}", f"orbit_j_{sp}"]).copy()
                a1 = np.minimum(q[f"orbit_i_{sp}"], q[f"orbit_j_{sp}"])
                a2 = np.maximum(q[f"orbit_i_{sp}"], q[f"orbit_j_{sp}"])
                q = q.assign(_k1=a1, _k2=a2).drop_duplicates(
                    subset=["source_id", "_k1", "_k2", "n_shared"])
                add("R3_corner_share", algo, sp, oxs,
                    (q["mode"] == "corner").sum(), len(q), "orbit-pair")
                # ---- 规则 4 / 5:结构粒度,symprec 无关(两列按构造相同)
                add("R4_contiguous_polyhedra", algo, sp, oxs,
                    int(aggo.r4_george_ok.sum()), len(aggo), "structure", "symprec-invariant")
                add("R5_parsimony", algo, sp, oxs,
                    int(aggo.r5_ok.sum()), len(aggo), "structure", "symprec-invariant")

    tab = pd.DataFrame(rows)
    out = f"{FEAT}/george_tableS1{'_smoke' if smoke else ''}.csv"
    tab.to_csv(out, index=False)

    # ================================================================ 打印 Table S1
    print("\n" + "=" * 100)
    print("Table S1  五条规则满足率 % [Wilson 95% CI]  ——  3 近邻算法 × 2 symprec × 2 氧化态来源 = 60 格")
    print("=" * 100)
    for rule in ["R1_radius_ratio", "R2_electrostatic_valence", "R3_corner_share",
                 "R4_contiguous_polyhedra", "R5_parsimony"]:
        sub = tab[tab.rule == rule]
        gran = sub.granularity.iloc[0]
        print(f"\n--- {rule}   granularity={gran}"
              + ("   [symprec 按构造无影响]" if sub.note.iloc[0] else ""))
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

    # ================================================================ George 逐条对比(主口径)
    print("\n" + "=" * 100)
    print("与 George 2020 的逐条对比(主口径:全部 ox_source ∈ {cif,guess} 合并、ChemEnv、symprec=0.01)")
    print("=" * 100)
    sid_main = set(st.loc[st.ox_source.isin(OXS), "source_id"])
    cmp_rows = []

    d = si1[si1.oxs.isin(OXS) & si1.r_cat.notna() & si1.cn_chemenv.notna()
            & si1.orbit_s001.notna()].drop_duplicates(subset=["source_id", "orbit_s001"])
    cmp_rows.append(("规则1 半径比(published 半径)", 66.0, *wilson((d.cn_pred == d.cn_chemenv).sum(), len(d)), "orbit"))
    dx = si1x[si1x.oxs.isin(OXS) & si1x.r_cat.notna() & si1x.cn_chemenv.notna()
              & si1x.orbit_s001.notna()].drop_duplicates(subset=["source_id", "orbit_s001"])
    cmp_rows.append(("规则1 半径比(含外推半径,敏感性)", 66.0, *wilson((dx.cn_pred == dx.cn_chemenv).sum(), len(dx)), "orbit"))
    ds = si1[si1.oxs.isin(OXS) & si1.r_cat.notna() & si1.cn_chemenv.notna()]
    cmp_rows.append(("规则1(site 粒度,不按轨道去重)", 66.0, *wilson((ds.cn_pred == ds.cn_chemenv).sum(), len(ds)), "site"))
    cmp_rows.append(("规则1(容差版 |CNpred−CNobs|≤1)", None,
                     *wilson(((d.cn_pred - d.cn_chemenv).abs() <= 1).sum(), len(d)), "orbit"))
    # George 原文对规则 1 的表述是"A coordination environment is stable only if the radius ratio
    # falls within the geometrically derived stability window **of this environment**"。
    # 硬球稳定窗只对 CN ∈ {2,3,4,6,8,12} 有定义(线形/三角/四面体/八面体/立方/立方八面体);
    # 观测 CN = 5/7/9/10/11 的位点**根本没有可检验的窗**,按原文"tested local environments"
    # 的措辞应当排除。这一层是与 66% 最可比的口径。
    WIN = {2, 3, 4, 6, 8, 12}
    dw = d[d.cn_chemenv.isin(WIN)]
    cmp_rows.append(("规则1(仅硬球窗有定义的 CN∈{2,3,4,6,8,12})", 66.0,
                     *wilson((dw.cn_pred == dw.cn_chemenv).sum(), len(dw)), "orbit"))
    dwx = dx[dx.cn_chemenv.isin(WIN)]
    cmp_rows.append(("规则1(硬球窗 + 外推半径)", 66.0,
                     *wilson((dwx.cn_pred == dwx.cn_chemenv).sum(), len(dwx)), "orbit"))

    e = an[an.oxs.isin(OXS) & (an.ncat_chemenv > 0) & an.sigma_chemenv.notna()
           & an.orbit_s001.notna()].drop_duplicates(subset=["source_id", "orbit_s001"])
    cmp_rows.append(("规则2 |Σs−2|≤0.01 的 O 位点", 20.0, *wilson(((e.sigma_chemenv - 2).abs() <= EPS).sum(), len(e)), "orbit(anion)"))
    e2 = an[an.oxs.isin(OXS) & (an.ncat_chemenv > 0) & an.sigma_chemenv.notna()]
    cmp_rows.append(("规则2(site 粒度,不按轨道去重)", 20.0, *wilson(((e2.sigma_chemenv - 2).abs() <= EPS).sum(), len(e2)), "site(anion)"))
    # 规则 2 的"高对称无畸变子集"(George Fig.2b 右:221 个材料近乎完美)
    e3 = e2[e2.maxcsm_chemenv <= 1.0]
    cmp_rows.append(("规则2(仅 max CSM≤1 的无畸变 O 位点)", None, *wilson(((e3.sigma_chemenv - 2).abs() <= EPS).sum(), len(e3)), "site(anion)"))

    P = pr[(pr.nn_algo == "chemenv") & pr.ox_source.isin(OXS)]
    for m, ref in (("corner", 62.5), ("edge", 27.2), ("face", 10.3)):
        cmp_rows.append((f"规则3 {m} 占比(全部 CN)", ref, *wilson((P["mode"] == m).sum(), len(P)), "pair"))
    P8 = P[(P.cn_i <= 8) & (P.cn_j <= 8)]
    for m, ref in (("corner", 73.3), ("edge", 25.0), ("face", 1.6)):
        cmp_rows.append((f"规则3 {m} 占比(CN≤8)", ref, *wilson((P8["mode"] == m).sum(), len(P8)), "pair"))

    agg = sr["chemenv"]
    aggm = agg[agg.index.isin(sid_main)]
    cmp_rows.append(("规则4 违例率(George 版:ox==V 且 cn==C)", 40.0,
                     *wilson((~aggm.r4_george_ok).sum(), len(aggm)), "structure"))
    cmp_rows.append(("规则4 违例率(氧化态版:只用 ox==V)", None,
                     *wilson((~aggm.r4_ox_ok).sum(), len(aggm)), "structure"))
    cmp_rows.append(("规则4 违例率(CN 版:只用 cn==C)", None,
                     *wilson((~aggm.r4_cn_ok).sum(), len(aggm)), "structure"))
    cmp_rows.append(("规则5 满足率(CN 判据)", 70.3,
                     *wilson(aggm.r5_ok.sum(), len(aggm)), "structure"))
    # 规则 5 的 ce_symbol 版本(ChemEnv 独有,更严)
    s5 = si[si.source_id.isin(sid_main) & si.ce_symbol.notna() & si.ox_state.notna()].copy()
    s5["oxr"] = s5.ox_state.round(3)
    g5 = s5.groupby("source_id").apply(
        lambda d: d.groupby(["element", "oxr"]).ce_symbol.nunique().max() == 1, include_groups=False)
    cmp_rows.append(("规则5 满足率(ce_symbol 判据,更严)", None, *wilson(g5.sum(), len(g5)), "structure"))

    # ---- 2–5 合取
    comb = combined(st, an, pr, sr, sid_main, si)
    cmp_rows.append(("规则2–5 同时满足", 13.0, *wilson(comb["k"], comb["n"]), "structure"))
    cmp_rows.append(("规则2–5 同时满足(全部阳离子 CN≤8 的结构)", 20.0,
                     *wilson(comb["k8"], comb["n8"]), "structure"))

    print(f"{'规则':<44s}{'George':>8s}{'本复现':>10s}{'95% CI':>18s}{'Δ pt':>8s}  粒度")
    for name, ref, p, lo, hi, gran in cmp_rows:
        dlt = "" if ref is None else f"{p-ref:+7.1f}"
        rr = "" if ref is None else f"{ref:8.1f}"
        flag = ""
        if ref is not None:
            flag = "  ✓" if abs(p - ref) <= 3 else ("  ✗>3pt" if abs(p - ref) <= 10 else "  ✗✗")
        print(f"{name:<44s}{rr:>8s}{p:10.1f}{f'[{lo:.1f},{hi:.1f}]':>18s}{dlt:>8s}  {gran}{flag}")

    # ================================================================ G6
    print("\n" + "=" * 100)
    print("G6(键定义鲁棒性,§4.3 硬门:三算法满足率波动 < 3 pt)")
    print("=" * 100)
    g6 = (tab.groupby(["rule", "ox_source", "symprec"])
            .pct.agg(["min", "max"]).assign(spread=lambda d: d["max"] - d["min"]))
    summ = g6.groupby("rule").spread.agg(["min", "median", "max"]).round(2)
    print(summ.to_string())
    print("\n判定:", end=" ")
    bad = summ[summ["max"] >= 3].index.tolist()
    print("全部 < 3 pt,G6 通过" if not bad else f"以下规则 spread ≥ 3 pt:{bad}")

    # 位点级三算法一致性(G6 真正要的口径,不是聚合统计)
    sm = si.dropna(subset=["cn_chemenv", "cn_crystalnn", "cn_brunner"])
    print(f"位点级 CN 一致:ce==cnn {np.mean(sm.cn_chemenv == sm.cn_crystalnn):.1%};"
          f"ce==bru {np.mean(sm.cn_chemenv == sm.cn_brunner):.1%};"
          f"三者全同 {np.mean((sm.cn_chemenv == sm.cn_crystalnn) & (sm.cn_chemenv == sm.cn_brunner)):.1%}"
          f"  (n={len(sm)})")

    # ================================================================ 规则 2 的 ε–满足率曲线(§4.3 要求整条曲线)
    print("\nε–满足率曲线(规则 2,ChemEnv,anion orbit 粒度):")
    for eps in (0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"   ε={eps:<5g} → {100*np.mean((e.sigma_chemenv-2).abs() <= eps):5.1f}%")
    # ================================================================ George 论域对齐层:ICSD ∩ MP
    # George 的论域是 "ICSD 且在 Materials Project 里" 的约 5,000 个氧化物。
    # `icsd_mp_link.parquet` 给了 ICSD→MP 的结构匹配,取 mp_id 非空即可复刻这条筛选,
    # 用来判断"我们与 George 的差距是 bug 还是论域"——这是 go/no-go 的关键判别层。
    try:
        lk = pd.read_parquet(f"{FEAT}/icsd_mp_link.parquet", columns=["source_id", "mp_id"])
        mp_ids = set(lk.loc[lk.mp_id.notna(), "source_id"])
        sid_mp = sid_main & mp_ids
        print("\n" + "=" * 100)
        print(f"论域对齐层:ICSD ∩ MP ∩ oxide_strict ∩ ox∈{{cif,guess}} = {len(sid_mp)} 条"
              f"(George 约 5,000)")
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
                ("规则1 半径比", 66.0, (dm.cn_pred == dm.cn_chemenv).sum(), len(dm)),
                ("规则2 |Σs−2|≤0.01", 20.0, ((em.sigma_chemenv - 2).abs() <= EPS).sum(), len(em)),
                ("规则3 corner(全 CN)", 62.5, (Pm["mode"] == "corner").sum(), len(Pm)),
                ("规则3 edge(全 CN)", 27.2, (Pm["mode"] == "edge").sum(), len(Pm)),
                ("规则3 face(全 CN)", 10.3, (Pm["mode"] == "face").sum(), len(Pm)),
                ("规则3 corner(CN≤8)", 73.3, (Pm8["mode"] == "corner").sum(), len(Pm8)),
                ("规则3 edge(CN≤8)", 25.0, (Pm8["mode"] == "edge").sum(), len(Pm8)),
                ("规则3 face(CN≤8)", 1.6, (Pm8["mode"] == "face").sum(), len(Pm8)),
                ("规则4 违例率(George 版)", 40.0, (~am.r4_george_ok).sum(), len(am)),
                ("规则5 满足率", 70.3, am.r5_ok.sum(), len(am)),
                ("规则2–5 同时", 13.0, cm2["k"], cm2["n"]),
                ("规则2–5 同时(CN≤8)", 20.0, cm2["k8"], cm2["n8"])]:
            p_, lo_, hi_ = wilson(k_, n_)
            fl = "✓" if abs(p_ - ref) <= 3 else ("✗>3pt" if abs(p_ - ref) <= 10 else "✗✗")
            print(f"  {nm:<26s} George {ref:5.1f}   本层 {p_:5.1f} [{lo_:.1f},{hi_:.1f}] "
                  f"Δ{p_-ref:+6.1f} pt  n={n_:<7d} {fl}")
    except FileNotFoundError:
        print("[warn] 缺 icsd_mp_link.parquet,跳过论域对齐层")

    # ================================================================ 诊断:规则 1 逐元素 / 混淆
    print("\n规则1 诊断(ChemEnv,orbit):预测 CN × 观测 CN 混淆(行=预测)")
    cm = pd.crosstab(d.cn_pred, d.cn_chemenv.astype(int), normalize=False)
    print(cm.to_string())
    per = (d.assign(hit=d.cn_pred == d.cn_chemenv)
             .groupby("element").hit.agg(["mean", "size"]).sort_values("size", ascending=False).head(15))
    per["mean"] = (100 * per["mean"]).round(1)
    print("规则1 逐元素命中率 Top15(按样本量):")
    print(per.to_string())

    # ================================================================ 诊断:规则 2 的 CSM 阈值扫描
    print("\n规则2 的畸变依赖(George Fig.2b 右:221 个材料的高对称子集近乎完美):")
    for c in (5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01):
        z = e2[e2.maxcsm_chemenv <= c]
        if len(z):
            print(f"   max CSM ≤ {c:<5g} → {100*np.mean((z.sigma_chemenv-2).abs() <= EPS):5.1f}%  (n={len(z)})")

    print(f"\n[写出] {out}")
    return tab


def combined(st, an, pr, sr, sid_main, si=None):
    """2–5 合取,structure 粒度。

    - 规则2 结构级 = 该结构**全部**有阳离子配位的 O 位点都满足 |Σs−2|≤0.01
    - 规则3 结构级 = **无共面对**(泡林原文 "particularly of shared faces")
    - CN≤8 层 = 该结构**全部阳离子位点** CN≤8(George Fig.6a 蓝条:excluding cations
      in high-coordination environments)
    """
    a = an[an.source_id.isin(sid_main) & (an.ncat_chemenv > 0) & an.sigma_chemenv.notna()]
    r2 = a.assign(ok=(a.sigma_chemenv - 2).abs() <= EPS).groupby("source_id").ok.all()
    p = pr[(pr.nn_algo == "chemenv") & pr.source_id.isin(sid_main)]
    r3 = p.assign(f=(p["mode"] == "face")).groupby("source_id").f.any().rename("has_face")
    agg = sr["chemenv"]
    df = pd.DataFrame(index=sorted(sid_main))
    df["r2"] = r2.reindex(df.index)
    df["r3"] = ~r3.reindex(df.index).fillna(False)          # 无对 → 无共面 → 满足
    df["r4"] = agg.r4_george_ok.reindex(df.index)
    df["r5"] = agg.r5_ok.reindex(df.index)
    ev = df.dropna()
    k = int((ev.r2 & ev.r3 & ev.r4 & ev.r5).sum())
    # CN≤8 结构层:用**位点表**求每个结构的最大阳离子 CN,不能用 pair 表——
    # 孤立多面体(如孤立 SiO4)结构在 pair 表里没有行,用 pair 求会把它们整批漏掉
    if si is not None:
        cn8 = si[si.source_id.isin(sid_main)].groupby("source_id").cn_chemenv.max()
    else:
        cn8 = p.groupby("source_id")[["cn_i", "cn_j"]].max().max(axis=1)
    lo8 = set(cn8[cn8 <= 8].index)
    ev8 = ev[ev.index.isin(lo8)]
    return dict(k=k, n=len(ev), k8=int((ev8.r2 & ev8.r3 & ev8.r4 & ev8.r5).sum()), n8=len(ev8))
