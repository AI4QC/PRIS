#!/usr/bin/env python3
"""把法则集应用到任意结构 —— 交付给使用者的入口。

用法:
    python src/apply_rules.py POSCAR CONTCAR foo.cif ...        # 默认五条
    python src/apply_rules.py --set core4 *.cif                 # 可信核心四条
    python src/apply_rules.py --set single *.cif                # 只用核心单条
    python src/apply_rules.py --verbose foo.cif                 # 逐条给出判定与实测值

法则集(全部在 discovery 上搜出、calibration 上验证、lockbox 未触碰):

留出 calibration 集(5,297 真实 / 3,612 扰动)上的表现,取自 paper/FACTS.md §16:

  single  核心单条        满足 0.9919 / 排除 0.2890   (LeMat DFT 弛豫假阳性差 -0.2 pt)
  core4   可信核心四条    满足 0.9579 / 排除 0.6121
  five    推荐五条(默认) 满足 0.9171 / 排除 0.7004

对照:泡林第 2-5 条同时满足率 0.0651。

阈值用 FACTS §16 规定的发表值 0.735 / 0.804。早先版本这里写的是 0.7353 /
0.8044——与规则描述里印出来的数字不一致,且按 FACTS 会让 calibration 满足率
偏移 0.0019。

选哪一套:
  - 复核已有的实验/计算结构      -> core4
  - 筛生成模型的输出(失效模式未知)-> five(第 5 条专抓"电荷放错位点",
                                    那是键长类判据完全看不见的一类)
  - 只要一条最稳的               -> single

**价态从组成的形式电荷推定(guess_oxi),不用 BVAnalyzer** ——
后者从键长反推价态再去检验键价定律,是循环论证。

# 适用范围

整数价态定不出来时,自动回退到**非整数平均价态**(如 Fe3O4 的 Fe^2.67+):
阴离子取常见价,余下电荷按化学计量分给变价阳离子。

实测(calibration,737 条):
  仅整数价态       覆盖 64.5%
  加非整数回退     覆盖 **80.9%**,新人群上四条合取满足率 0.9256
                  (整数人群 0.9577,121 条样本 95% CI 约 ±4.7 pt,差异不显著)

**阈值一个都没改就迁移过去了** —— 因为它们是物理量的分位数,不是拟合参数。

仍有约 19% 判不了(多阴离子、含复杂分子基团、无变价元素而整数组合无解)。
它们不是"不合理",是本方法**判不了**。**不要把"跳过"读成"通过"。**
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminate import guess_oxi          # noqa: E402
from phys_law import phys_feats             # noqa: E402
from elec_feat import elec_feats            # noqa: E402
from geom_feat import geom_feats            # noqa: E402
from t0_guard import parse                  # noqa: E402

# (描述, 特征, 方向, 阈值, 前提or None)
R_SINGLE = [("最短阳-阴键 >= 0.735 x (Shannon 半径和)",
             "bl_min", "lo", 0.735, None)]

R_CORE4 = [
    ("最短阳-阴键 >= 0.804 x (Shannon 半径和)", "bl_min", "lo", 0.804, None),
    ("若 平均阴离子CN <= 3.33,则 平均键长比 <= 1.081",
     "bl_mean", "hi", 1.081, ("cn_an_mean", "lo", 3.333)),
    ("位点马德隆能/电荷 的极差 <= 31.45", "madz_range", "hi", 31.45, None),
    ("最高位点马德隆能 <= 15.17", "mad_max", "hi", 15.17, None),
]

R_FIVE = R_CORE4 + [
    ("若 离子性分数 fi > 0.55,则 不得存在同号离子成键",
     "frac_like_bonds", "hi", 1e-4, ("fi", "hi", 0.55)),
]

SETS = {"single": R_SINGLE, "core4": R_CORE4, "five": R_FIVE}
ANI = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As", "H", "C"}


def ionicity(st, X):
    """泡林离子性分数 fi = 1 - exp(-0.25 dchi^2)。纯组成量。"""
    c = parse(st.composition.reduced_formula.replace(" ", ""))
    cand = [e for e in c if e in ANI and e in X and np.isfinite(X[e])]
    if not cand:
        return np.nan
    an = max(cand, key=lambda e: X[e])
    cats = {e: n for e, n in c.items()
            if e != an and e in X and np.isfinite(X[e])}
    if not cats:
        return np.nan
    d = sum(n * (X[an] - X[e]) for e, n in cats.items()) / sum(cats.values())
    return float(1 - np.exp(-0.25 * d * d))


COMMON = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Ag": 1, "Mg": 2, "Ca": 2,
          "Sr": 2, "Ba": 2, "Zn": 2, "Cd": 2, "Al": 3, "Ga": 3, "In": 3, "Sc": 3,
          "Y": 3, "La": 3, "Si": 4, "Ge": 4, "Zr": 4, "Hf": 4, "B": 3, "P": 5}
ANI_Q = {"O": -2, "S": -2, "Se": -2, "Te": -2, "F": -1, "Cl": -1,
         "Br": -1, "I": -1, "N": -3}


def frac_oxi(st):
    """非整数平均价态回退。阴离子取常见价,余下电荷分给变价阳离子。

    整数价态组合无解的结构里,约七成是混合价态(Fe3O4 的 Fe^2.67+ 之类)。
    Ewald 原生支持分数电荷,Shannon 半径查表时取整即可。
    实测:覆盖率 64.5% -> 80.9%,而**阈值一个都不用改**(新人群满足率 0.9256)。
    """
    from collections import Counter
    syms = [s.specie.symbol for s in st]
    ans = [i for i, e in enumerate(syms) if e in ANI_Q]
    cats = [i for i, e in enumerate(syms) if e not in ANI_Q]
    if not ans or not cats:
        return None
    qa = sum(ANI_Q[syms[i]] for i in ans)
    cc = Counter(syms[i] for i in cats)
    val = [0.0] * len(st)
    for i in ans:
        val[i] = ANI_Q[syms[i]]
    if len(cc) == 1:
        q = -qa / len(cats)
        for i in cats:
            val[i] = q
        return val
    var = [e for e in cc if e not in COMMON]
    if len(var) != 1:          # 两个以上变价元素时分配不唯一,不猜
        return None
    qf = sum(COMMON[e] * cc[e] for e in cc if e in COMMON)
    qv = (-qa - qf) / cc[var[0]]
    if not (0 < qv <= 8):
        return None
    for i in cats:
        val[i] = COMMON.get(syms[i], qv)
    return val


def features(st):
    """算出法则集需要的全部量。返回 (dict, 错误原因)。"""
    val, ok = guess_oxi(st)
    if not ok:
        val = frac_oxi(st)     # 回退到非整数平均价态
        if val is None:
            return None, "无法给出电荷平衡的价态(整数与非整数回退都失败)"
    f, errors = {}, []
    for fn in (phys_feats, elec_feats, geom_feats):
        try:
            g = fn(st, val)
            if g:
                f.update(g)
        except Exception as exc:
            errors.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
    # 平均阴离子配位数(前提用),与 criteria() 同口径
    try:
        from discriminate import criteria
        c = criteria(st, val)
        if c:
            f.update(c)
    except Exception as exc:
        errors.append(f"criteria: {type(exc).__name__}: {exc}")

    # fi 是纯组成量，直接从 pymatgen 元素表取电负性。公开判定入口
    # 不应为这一个量依赖外部数 GB 特征库中的 _elem_props.parquet。
    try:
        from pymatgen.core.periodic_table import Element
        comp = parse(st.composition.reduced_formula.replace(" ", ""))
        X = {}
        for symbol in comp:
            value = Element(symbol).X
            if value is not None and np.isfinite(value):
                X[symbol] = float(value)
        f["fi"] = ionicity(st, X)
    except Exception as exc:
        errors.append(f"ionicity: {type(exc).__name__}: {exc}")
    return f, "; ".join(errors) if errors else None


def judge(f, rules, verbose=False):
    """逐条判定，返回 True / False / None(信息不足)。

    一条已计算的违例足以判为 False；若没有违例但任一适用性或目标量
    算不出，必须返回 None，不得把“未知”报成“合理”。
    """
    lines, failed, unknown = [], False, False
    for desc, col, side, th, g in rules:
        v = f.get(col, np.nan)
        applies = True
        if g:
            gc, gs, gt = g
            gv = f.get(gc, np.nan)
            if not np.isfinite(gv):
                unknown = True
                lines.append(("?", desc, f"前提 {gc} 算不出，无法判定是否适用"))
                continue
            else:
                applies = (gv > gt) if gs == "hi" else (gv <= gt)
        if not applies:
            lines.append(("—", desc, f"前提不成立({g[0]}={f.get(g[0], float('nan')):.3f})"))
            continue
        if not np.isfinite(v):
            unknown = True
            lines.append(("?", desc, f"{col} 算不出，无法判定"))
            continue
        ok = bool((v <= th) if side == "hi" else (v >= th))
        failed |= not ok
        lines.append(("✓" if ok else "✗", desc, f"{col} = {v:.4f}"))
    if verbose:
        for mark, desc, detail in lines:
            print(f"    {mark} {desc}")
            print(f"        {detail}")
    if failed:
        return False
    if unknown:
        return None
    return True


# F2 实验域紧凑公式(7 项)。z 用 real_all+各特征表的全样本均值/标准差,
# 与拟合时的折内标准化略有差别 —— **S 只在同组成候选之间比较,绝对值无意义**。
F2 = [("min_opp_frac", -0.3222), ("econ_mean", +0.1491), ("mef_mean", +0.1369),
      ("angvar_mean", +0.0599), ("econ_max", +0.0390), ("dist_rsd", -0.0300),
      ("p5_n_distinct", +0.0016)]
_STAT = {}


def f2_score(f):
    """F2 判据分。S 小者更稳定;缺任一项则返回 nan。"""
    import os
    import pandas as pd
    if not _STAT:
        base = os.environ.get("PRIS_FEATURES",
                              "features/")
        frames = []
        for fn in ("real_all.parquet", "phys_real.parquet",
                   "elec_real.parquet", "geom_real.parquet"):
            if os.path.exists(base + fn):
                frames.append(pd.read_parquet(base + fn))
        for fr in frames:
            for c, _ in F2:
                if c in fr.columns and c not in _STAT:
                    v = fr[c].dropna()
                    _STAT[c] = (float(v.mean()), float(v.std()) or 1.0)
    s = 0.0
    for c, w in F2:
        v = f.get(c, np.nan)
        if c not in _STAT or not np.isfinite(v):
            return np.nan
        mu, sd = _STAT[c]
        s += w * (v - mu) / sd
    return float(s)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="用 PRIS 法则判定离子晶体结构是否合理")
    ap.add_argument("files", nargs="+", help="结构文件(CIF / POSCAR / 任何 pymatgen 能读的格式)")
    ap.add_argument("--set", choices=list(SETS), default="five",
                    help="法则集:single / core4 / five(默认)")
    ap.add_argument("--verbose", action="store_true", help="逐条给出判定与实测值")
    ap.add_argument("--formula", action="store_true",
                    help="同时给出 F2 判据分 S(**只在同组成候选之间可比**,S 小者更稳定)")
    a = ap.parse_args()
    rules = SETS[a.set]
    print(f"法则集 [{a.set}],共 {len(rules)} 条\n")

    from pymatgen.core import Structure
    npass = nfail = nskip = 0
    for path in a.files:
        try:
            st = Structure.from_file(path)
        except Exception as e:
            print(f"[跳过] {path}: 读不出结构({e})")
            nskip += 1
            continue
        f, err = features(st)
        if f is None:
            print(f"[跳过] {path}: {err}")
            nskip += 1
            continue
        verdict = judge(f, rules, a.verbose)
        if err and a.verbose:
            print(f"    ! 部分特征计算失败: {err}")
        if verdict is None:
            detail = f" ({err})" if err else ""
            print(f"[跳过] {path}: 法则所需特征不完整，无法判定{detail}")
            nskip += 1
            continue
        extra = ""
        if a.formula:
            sv = f2_score(f)
            extra = f"   S = {sv:+.3f}" if np.isfinite(sv) else "   S = 算不出"
        print(f"[{'合理' if verdict else '不合理'}] {path}  "
              f"({st.composition.reduced_formula}, {len(st)} 原子){extra}")
        npass += verdict
        nfail += (not verdict)
    print(f"\n合理 {npass} / 不合理 {nfail} / 跳过 {nskip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
