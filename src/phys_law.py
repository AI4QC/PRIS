#!/usr/bin/env python3
"""有物理含义的合理性特征 —— 用来替掉法则集里那条纯经验的 `vol_per_atom`。

# 为什么要换

上一版推荐的 3 条法则里,"每原子体积 <= 50.8 A^3"扛了很大一部分排除力,
但它是纯标定阈值:没有化学含义,阈值随数据库组成漂移,也没法推广到新体系。

# 换成什么

全部基于 Shannon 配位相关半径 r(元素, 氧化态, CN):

  sh_pack     = sum(4/3 pi r^3) / V_cell        堆积分数 —— 离子当硬球,不能互相穿透
  bl_min      = min over 阳-阴键 of d/(r_c+r_a) 最短键相对半径和 —— **Born 排斥**
  bl_cat_max  = max over 阳离子 of (它自己最短的 d/(r_c+r_a))  最"悬空"的阳离子
  bl_rsd_max  = max over 多面体 of 键长相对标准差              多面体畸变

`bl_min` 这一条特别重要:**泡林的静电框架里没有短程排斥项**,所以它在原理上
论证不了"压缩后不合理"(实测 S1 单轴压缩的马德隆能反而更低,见结果总结 5.2b)。
d/(r_c+r_a) 正是把 Born 排斥显式写进判据 —— 也正是 Hawthorne 2026 提出的
"用半径**和**而非半径**比**"那条替代规则的可计算形式。

# 复现性

make_negatives.py 原本用 `abs(hash(sid))` 播种。Python 的字符串 hash 受
PYTHONHASHSEED 随机化,**跨进程不同** —— 意味着此前的 negatives.parquet 无法重建,
新算的特征按 sid 合并会配到"同一母体、同一类型、但不同随机实现"的另一个结构上。
这里统一改用 crc32,确定性播种。
"""
from __future__ import annotations
import os
import argparse
import sys
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminate import criteria, guess_oxi, read_blob_cif  # noqa: E402
from phys_feat import R as RFALL  # noqa: E402

F = os.environ.get("PRIS_FEATURES", "features/")
_ROMAN = {2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
          9: "IX", 10: "X", 11: "XI", 12: "XII"}
_CACHE = {}


def seed_of(sid: str) -> int:
    """确定性种子。不用 hash() —— 它被 PYTHONHASHSEED 随机化,跨进程不可复现。"""
    return zlib.crc32(sid.encode()) & 0x7FFFFFFF


def shannon(sym, ox, cn):
    """Shannon 半径,按 (元素, 氧化态, 配位数) 查表;邻近 CN 回退,再回退代表值。"""
    # 符号必须进 key:回退半径按 ox 的原始符号取值,而 int(round(ox)) 会把
    # ±0.5 这类分数电荷都折到 0,两者共用缓存项后 ρ 就依赖于进程内的处理顺序。
    key = (sym, int(round(ox)), int(round(cn)), ox > 0)
    if key in _CACHE:
        return _CACHE[key]
    r = None
    try:
        from pymatgen.core import Species
        sp = Species(sym, int(round(ox)))
        c0 = int(round(cn))
        for c in (c0, c0 - 1, c0 + 1, 6):
            try:
                r = float(sp.get_shannon_radius(_ROMAN.get(c, "VI")))
                if r:
                    break
            except Exception:
                continue
    except Exception:
        pass
    if not r:
        r = RFALL.get(sym, 1.0 if ox > 0 else 1.4)
    _CACHE[key] = r
    return r


def phys_feats(st, val, nn=None):
    """Shannon 半径类物理特征。nn 可传入以复用 CrystalNN 的一次计算。"""
    if nn is None:
        from pymatgen.analysis.local_env import CrystalNN
        cnn = CrystalNN(weighted_cn=False, x_diff_weight=0.0)
        nn = []
        for i in range(len(st)):
            try:
                nn.append(cnn.get_nn_info(st, i))
            except Exception:
                nn.append([])
    cn = [len(x) for x in nn]
    vol = 0.0
    for i in range(len(st)):
        vol += 4 / 3 * np.pi * shannon(st[i].specie.symbol, val[i], max(cn[i], 1)) ** 3

    # 同号离子成键的比例。真实离子晶体里阳离子周围应当是阴离子,反之亦然;
    # 这个量对"电荷放错位点"(S5 阴阳离子互换)直接敏感,而且是 T1 级、100% 可算,
    # 不像 GII 那样受键价参数表覆盖(仅 85%)限制。
    like = tot_b = 0
    worst_op = 1.0
    for i in range(len(st)):
        if not nn[i] or abs(val[i]) < 1e-9:
            continue
        opp = 0
        for nb in nn[i]:
            j = nb["site_index"]
            tot_b += 1
            if val[i] * val[j] > 0:
                like += 1
            elif val[i] * val[j] < 0:
                opp += 1
        worst_op = min(worst_op, opp / len(nn[i]))
    out_like = like / tot_b if tot_b else np.nan

    ratios_all, cat_short, rsd = [], [], []
    for i in range(len(st)):
        if val[i] <= 0:
            continue
        rt, ds = [], []
        ri = shannon(st[i].specie.symbol, val[i], max(cn[i], 1))
        for nb in nn[i]:
            j = nb["site_index"]
            if val[j] >= 0:
                continue
            d = st[i].distance(st[j], jimage=nb.get("image"))
            rj = shannon(st[j].specie.symbol, val[j], max(cn[j], 1))
            rs = ri + rj
            if rs <= 0:
                continue
            rt.append(d / rs)
            ds.append(d)
        if not rt:
            continue
        ratios_all.extend(rt)
        cat_short.append(min(rt))       # 该阳离子最近的阴离子有多近
        if len(ds) > 1:
            rsd.append(float(np.std(ds) / np.mean(ds)))
    if not ratios_all:
        return None
    return {
        "frac_like_bonds": float(out_like),   # 同号成键占比,真实离子晶体应接近 0
        "min_opp_frac": float(worst_op),      # 最"不合群"的离子:异号邻居占比
        "sh_pack": float(vol / st.volume),
        "bl_min": float(min(ratios_all)),        # 全局最短键 —— Born 排斥
        "bl_cat_max": float(max(cat_short)),     # 最悬空的阳离子
        "bl_mean": float(np.mean(ratios_all)),
        "bl_rsd_max": float(max(rsd)) if rsd else 0.0,
    }


# ---------------------------------------------------------------- 驱动

def _real(r):
    from pymatgen.core import Structure
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > 150:
            return None
        val, ok = guess_oxi(st)
        if not ok:
            return None
        f = phys_feats(st, val)
        if f is None:
            return None
        f["source_id"] = r["sid"]
        return f
    except Exception:
        return None


def _bad(r):
    """重造 4 类扰动(确定性种子),同时算 criteria + 物理特征,保证两者同源。"""
    from pymatgen.core import Structure
    from make_negatives import perturb, swapped_val
    out = []
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > 80:
            return out
        val, ok = guess_oxi(st)
        if not ok:
            return out
        rng = np.random.default_rng(seed_of(r["sid"]))
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            p = perturb(st, kind, rng, val)
            if p is None:
                continue
            pv = swapped_val(p, val)
            try:
                c = criteria(p, pv)
            except Exception:
                continue
            if c is None:
                continue
            f = phys_feats(p, pv)
            if f is None:
                continue
            c.update(f)
            c.update(sid=r["sid"] + "_" + kind, kind=kind, parent=r["sid"])
            out.append(c)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "bad"])
    ap.add_argument("--n", type=int, default=6000, help="bad 模式的母体数")
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()

    prov = pd.read_parquet(F + "provenance.parquet",
                           columns=["source_id", "in_analysis_set", "blob_offset",
                                    "blob_length", "n_elements"])
    if a.mode == "real":
        d = prov[prov.n_elements >= 2]
        fn, outf = _real, F + "phys_real.parquet"
    else:
        # 与 make_negatives.py 完全相同的母体抽样(random_state=0),便于对照
        d = prov[prov.in_analysis_set & (prov.n_elements >= 2)].sample(
            n=min(a.n, len(prov)), random_state=0)
        fn, outf = _bad, F + "phys_bad.parquet"
    recs = [{"sid": t.source_id, "off": int(t.blob_offset), "ln": int(t.blob_length)}
            for t in d.itertuples()]
    print(f"{a.mode}: {len(recs):,} 条", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in enumerate(ex.map(fn, recs, chunksize=8)):
            if a.mode == "real":
                if x:
                    rows.append(x)
            else:
                rows.extend(x)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1:,}/{len(recs):,} -> {len(rows):,}", flush=True)
    pd.DataFrame(rows).to_parquet(outf, index=False)
    print(f"写出 {outf} {len(rows):,} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
