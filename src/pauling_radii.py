# -*- coding: utf-8 -*-
"""泡林单价半径(Pauling univalent radii)表 —— 泡林第一定律专用。

为什么不用 Shannon:Shannon 半径按配位数查表,拿它去预测配位数是循环论证
(研究计划 §4.3 G7 硬门点名要挡的就是这一条)。George 2020 原文同样用
"Pauling's univalent radii"(ref [7] = Pauling, *The Nature of the Chemical Bond*)。

半径来源分两级,`radius_tier` 会如实标注:

* `tier='published'` —— 泡林本人给出的单价半径(闭壳层离子:惰性气体构型 + 18 电子构型)。
  这是主统计口径。
* `tier='extended'` —— 我们按泡林自己的屏蔽公式 R1 = C_n / (Z − S) 外推到开壳层
  d 区离子。**这是我们的外推,不是泡林的表**,只作敏感性层报告,绝不当主数。

屏蔽公式的标定方式(每一个常数都可复核):
    泡林 1927 年的原始定义 R1 = C_n /(Z − S),同一等电子系列共用 (C_n, S)。
    我们对每个闭壳层用该系列**两个已发表的锚点**反解 (C_n, S),再检验系列内其余值。
    例:Ne 壳层用 Na+ 0.95 与 Mg2+ 0.82 →  S=4.69, C=5.995;
        回代 Al3+ 0.721(published 0.72)、Si4+ 0.644(0.65)、P5+ 0.582(0.59),
        O2- 1.811(published 1.76,差 2.9%)。→ 已发表值优先,公式只作兜底。

开壳层外推(tier='extended')的构造:
    同一主量子数的 d 壳层里,屏蔽常数 S 在 n_d=0(惰性气体锚点)与 n_d=10(18 电子锚点)
    之间**按 d 电子数线性插值**,C_n 同样线性插值。两端严格复现泡林的已发表值。
    3d:S 从 11.13(Ar 壳层)到 18.00(Ni 壳层)= 0.687/电子;C 从 10.47 到 10.56。
    4d:S 从 28.75(Kr)到 37.50(Pd);C 从 12.21 到 11.97。
    5d/4f:S 从 45.40(Xe)到 68.58(Pt),按 (n_f + n_d) 共 24 个电子线性插值;C 从 16.22 到 14.28。
"""
from __future__ import annotations

# ---------------------------------------------------------------- 已发表的泡林单价半径(Å)
# 键:(元素符号, 整数氧化态)。数值取自 Pauling《The Nature of the Chemical Bond》
# 第三版 Table 13-3(单价半径 univalent radii),即 George 2020 引用的 ref [7]。
PAULING_UNIVALENT = {
    # ---- He 壳层 (2 e)
    ("H", -1): 2.08, ("Li", 1): 0.60, ("Be", 2): 0.44, ("B", 3): 0.35,
    ("C", 4): 0.29, ("N", 5): 0.25,
    # ---- Ne 壳层 (10 e)
    ("N", -3): 2.47, ("O", -2): 1.76, ("F", -1): 1.36,
    ("Na", 1): 0.95, ("Mg", 2): 0.82, ("Al", 3): 0.72, ("Si", 4): 0.65,
    ("P", 5): 0.59, ("S", 6): 0.53, ("Cl", 7): 0.49,
    # ---- Ar 壳层 (18 e)
    ("S", -2): 2.19, ("Cl", -1): 1.81,
    ("K", 1): 1.33, ("Ca", 2): 1.18, ("Sc", 3): 1.06, ("Ti", 4): 0.96,
    ("V", 5): 0.88, ("Cr", 6): 0.81, ("Mn", 7): 0.75,
    # ---- Ni(3d10)18 电子壳层
    ("Cu", 1): 0.96, ("Zn", 2): 0.88, ("Ga", 3): 0.81, ("Ge", 4): 0.76,
    ("As", 5): 0.71, ("Se", 6): 0.66, ("Br", 7): 0.62,
    # ---- Kr 壳层 (36 e)
    ("Se", -2): 2.32, ("Br", -1): 1.95,
    ("Rb", 1): 1.48, ("Sr", 2): 1.32, ("Y", 3): 1.20, ("Zr", 4): 1.09,
    ("Nb", 5): 1.00, ("Mo", 6): 0.93, ("Tc", 7): 0.87,
    # ---- Pd(4d10)18 电子壳层
    ("Ag", 1): 1.26, ("Cd", 2): 1.14, ("In", 3): 1.04, ("Sn", 4): 0.96,
    ("Sb", 5): 0.89, ("Te", 6): 0.82, ("I", 7): 0.77,
    # ---- Xe 壳层 (54 e)
    ("Te", -2): 2.50, ("I", -1): 2.16,
    ("Cs", 1): 1.69, ("Ba", 2): 1.53, ("La", 3): 1.39, ("Ce", 4): 1.27,
    # ---- Pt(5d10)18 电子壳层
    ("Au", 1): 1.37, ("Hg", 2): 1.25, ("Tl", 3): 1.15, ("Pb", 4): 1.06,
    ("Bi", 5): 0.98,
}

# ---------------------------------------------------------------- 外推用的壳层标定常数
# (C_n, S) 两端锚点;见模块 docstring 的标定过程
_SHELL_ANCHOR = {
    "3d": dict(Z0=18, C0=10.47, S0=11.13, C1=10.56, S1=18.00, nmax=10),   # Ar → Ni
    "4d": dict(Z0=36, C0=12.21, S0=28.75, C1=11.97, S1=37.50, nmax=10),   # Kr → Pd
    "5d": dict(Z0=54, C0=16.22, S0=45.40, C1=14.28, S1=68.58, nmax=24),   # Xe → Pt(含 4f14)
}

_Z = {}  # 元素 → 原子序数(懒加载 pymatgen,避免本模块被 worker 反复导入时的开销)


def _atomic_number(el: str) -> int:
    if not _Z:
        from pymatgen.core.periodic_table import Element
        for e in Element:
            _Z[e.symbol] = e.Z
    return _Z[el]


def _extended_radius(el: str, ox: int):
    """开壳层 d 区离子的外推单价半径。返回 None 表示无法外推。"""
    z = _atomic_number(el)
    ne = z - ox                      # 离子的电子数
    if ne < 2:
        return None
    for shell, a in _SHELL_ANCHOR.items():
        k = ne - a["Z0"]             # 超出惰性气体芯的电子数
        if 0 < k < a["nmax"]:        # 端点(k=0 / k=nmax)已在已发表表里
            f = k / a["nmax"]
            C = a["C0"] + (a["C1"] - a["C0"]) * f
            S = a["S0"] + (a["S1"] - a["S0"]) * f
            den = z - S
            return C / den if den > 0.5 else None
    return None


def univalent_radius(el: str, ox, allow_extended: bool = False):
    """返回 (半径 Å, tier) ;查不到返回 (None, None)。

    ox 会被四舍五入到整数——泡林的表本身只有整数价;分数价(如 Fe2.5+)按最近整数取,
    这在特征库里通过 `frac_ox` 旗标可事后剔除。
    """
    if ox is None or ox != ox:
        return None, None
    o = int(round(float(ox)))
    r = PAULING_UNIVALENT.get((el, o))
    if r is not None:
        return r, "published"
    if allow_extended:
        r = _extended_radius(el, o)
        if r is not None:
            return r, "extended"
    return None, None


# ---------------------------------------------------------------- 半径比 → 配位数
# 泡林第一定律的临界半径比(硬球几何推导,见 George 2020 Fig.1a):
#   <0.155 → CN 2(直线) / 0.155–0.225 → CN 3(平面三角) / 0.225–0.414 → CN 4(四面体)
#   0.414–0.732 → CN 6(八面体) / 0.732–1.000 → CN 8(立方) / ≥1.000 → CN 12(立方八面体)
RATIO_BOUNDS = [(0.0, 0.155, 2), (0.155, 0.225, 3), (0.225, 0.414, 4),
                (0.414, 0.732, 6), (0.732, 1.000, 8), (1.000, 1e9, 12)]


def predict_cn(ratio):
    if ratio is None or ratio != ratio:
        return None
    for lo, hi, cn in RATIO_BOUNDS:
        if lo <= ratio < hi:
            return cn
    return None


if __name__ == "__main__":
    # 自检:公式回代已发表值的偏差(用于 README / 报告)
    import statistics
    errs = []
    for (el, o), r in sorted(PAULING_UNIVALENT.items()):
        rx = _extended_radius(el, o)
        if rx:
            errs.append(abs(rx - r) / r)
    print(f"已发表离子 {len(PAULING_UNIVALENT)} 个;公式可回代 {len(errs)} 个,"
          f"中位相对误差 {statistics.median(errs):.3%}" if errs else "无重叠")
    for t in [(("Fe", 3)), (("Fe", 2)), (("Mn", 2)), (("Cu", 2)), (("W", 6)), (("Ta", 5)), (("Nd", 3))]:
        print(t, univalent_radius(*t, allow_extended=True))
