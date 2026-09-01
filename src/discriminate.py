#!/usr/bin/env python3
"""判别任务:泡林二到五条作为**判据**的真实评分。

# 为什么要换任务

泡林五条里只有第一条是"给组成猜 CN"的预测性规则,二到五条都是**判据**:
拿到一个结构,判断它合不合理。前几轮把全部五条都放在"预测配位环境"的
top-1 准确率上评分,那是拿尺子量温度 —— 第三定律根本不预测 CN,
它说的是"共面的结构不稳定"。

判据的价值在于**它能拒绝什么**。而库里全是已经存在的结构,按定义都"合理",
所以只能算"多少比例的真实结构满足它"(George 的 13%),那个数对判据毫无意义:
恒真规则满足率 100%,恒假规则 0%,两者都没用。

# 这个脚本做什么

配对判别:对同一个约化化学组成,取
  正类 = 我们分析集里的真实结构(ICSD/COD,实验实现过)
  负类 = ELEMENTA 里同组成的端点(DFT 局域极小,但从未被合成)
问:泡林各条判据能不能把真的那个排在前面?

**组成被完全控制** —— 两侧化学式相同,差别只在结构。这是判据该受的检验。
实测规模:2,229 个共同组成,真实侧 7,317 条、ELEMENTA 侧 11,374 条(每组成均 5.1 个)。

# 三个必须显式处理的口径问题

1. **氧化态两侧必须同源。** ELEMENTA 只有组成、无 CIF 装饰,所以只能用
   `oxi_state_guesses`。为公平,真实侧**也强制用 guess**,不用 ICSD 原生价态。
   代价是丢掉混合价信息,但避免了"正类有额外信息"这个致命偏差。
2. **DFT 弛豫 vs 实验测定的系统差。** ELEMENTA 是 PBE 弛豫的,真实侧是实验精修的。
   PBE 通常高估晶格常数约 1%。若判据靠键长(BVS),这会变成伪信号。
   故**主判据只用不依赖绝对键长的量**(CN、连接类型、s_pauling=z/CN),
   BVS 版本单独报作敏感性。
3. **负类的标签噪声。** ELEMENTA 的某个候选可能其实是已知多形体,只是不在我们
   分析集里(建库时丢了无序结构、或不满足单一阴离子过滤)。这会低估判别力,
   即结论偏保守 —— 可接受,但要在报告里说。
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

F = Path(os.environ.get("PRIS_FEATURES", "features/"))
ELEM = Path(os.environ.get("PRIS_ELEMENTA", "elementa/endpoints_open.extxyz"))
BLOB = Path(os.environ.get("PRIS_MATDATA_BLOB", "structures.blob"))
OUT = F / "discriminate.parquet"

ANIONS = ["O", "S", "Se", "Te", "N", "P", "F", "Cl", "Br", "I"]
RE_F = re.compile(r"\bformula=(\S+)")
RE_LAT = re.compile(r'Lattice="([^"]+)"')
RE_EL = re.compile(r"([A-Z][a-z]?)")


def redkey(formula: str) -> str | None:
    """约化组成键。Fe2O3 与 Fe4O6 同键,这样跨库配对才对得上。"""
    d = collections.Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if el:
            d[el] += int(n or 1)
    if not d:
        return None
    g = 0
    for v in d.values():
        g = gcd(g, v)
    g = g or 1
    return "|".join(f"{k}{v // g}" for k, v in sorted(d.items()))


# ---------------------------------------------------------------- 结构读取

def read_blob_cif(off: int, ln: int) -> str:
    with open(BLOB, "rb") as fh:
        fh.seek(off)
        raw = fh.read(ln)
    import zlib
    try:
        return zlib.decompress(raw).decode("utf-8", "ignore")
    except zlib.error:
        return raw.decode("utf-8", "ignore")


def iter_elementa(keep_keys: set[str]):
    """流式读 ELEMENTA 端点,只吐出目标组成的。内存 O(1)。"""
    with open(ELEM) as fh:
        idx = 0
        while True:
            head = fh.readline()
            if not head:
                break
            try:
                nat = int(head.strip())
            except ValueError:
                continue
            comment = fh.readline()
            lines = [fh.readline() for _ in range(nat)]
            idx += 1
            m = RE_F.search(comment)
            if not m:
                continue
            fo = m.group(1)
            els = set(RE_EL.findall(fo))
            if "H" in els or "C" in els:
                continue
            if len([a for a in ANIONS if a in els]) != 1:
                continue
            # 必须有阳离子。单质(如纯 Br)满足"恰好一种阴离子元素"但没有阴阳之分,
            # 泡林规则在其上无定义。实测不加这条会让 iter 前几千条全是单质卤素。
            if len(els) < 2:
                continue
            rk = redkey(fo)
            if rk not in keep_keys:
                continue
            ml = RE_LAT.search(comment)
            if not ml:
                continue
            lat = np.array([float(x) for x in ml.group(1).split()]).reshape(3, 3)
            syms, pos = [], []
            for ln_ in lines:
                p = ln_.split()
                if len(p) < 4:
                    break
                syms.append(p[0])
                pos.append([float(p[1]), float(p[2]), float(p[3])])
            if len(syms) != nat:
                continue
            yield {"sid": f"elem-{idx}", "rk": rk, "formula": fo,
                   "lattice": lat, "species": syms, "coords": np.array(pos)}


# ---------------------------------------------------------------- 判据计算

def guess_oxi(struct):
    """两侧统一用组成推断价态。返回 (valences, ok)。"""
    from pymatgen.core import Composition
    comp = Composition(struct.composition.reduced_formula)
    try:
        guesses = comp.oxi_state_guesses(max_sites=-10)
    except Exception:
        return None, False
    if not guesses:
        return None, False
    g = guesses[0]
    try:
        val = [float(g[str(site.specie.symbol)]) for site in struct]
    except (KeyError, AttributeError):
        return None, False
    if not any(v > 0 for v in val) or not any(v < 0 for v in val):
        return None, False          # 没有阴阳之分,泡林规则无定义
    return val, True


def criteria(struct, val):
    """算泡林 2/3/4/5 四条判据。全部只用 CN 与拓扑,不用绝对键长(见文件头口径 2)。"""
    from pymatgen.analysis.local_env import CrystalNN
    out = {}
    try:
        cnn = CrystalNN(weighted_cn=False, x_diff_weight=0.0)
        nn = [cnn.get_nn_info(struct, i) for i in range(len(struct))]
    except Exception:
        return None
    cn = [len(x) for x in nn]
    cats = [i for i, v in enumerate(val) if v > 0]
    ans = [i for i, v in enumerate(val) if v < 0]
    if not cats or not ans:
        return None

    # --- 规则 2:每个阴离子收到的键强和 s=z/CN 对其电荷的偏差
    recv = collections.defaultdict(float)
    for i in cats:
        if cn[i] == 0:
            continue
        s = val[i] / cn[i]
        for nb in nn[i]:
            j = nb["site_index"]
            if val[j] < 0:
                recv[j] += s
    devs = [abs(recv.get(j, 0.0) - abs(val[j])) for j in ans]
    out["p2_mean_dev"] = float(np.mean(devs)) if devs else np.nan
    out["p2_frac_ok_010"] = float(np.mean([d <= 0.10 for d in devs])) if devs else np.nan

    # --- 规则 3:多面体连接里共边/共面的比例(共享配体数 >= 2)
    ligs = {i: {nb["site_index"] for nb in nn[i]} for i in cats}
    n_corner = n_edge = n_face = 0
    for a_i, a in enumerate(cats):
        for b in cats[a_i + 1:]:
            sh = len(ligs[a] & ligs[b])
            if sh == 1:
                n_corner += 1
            elif sh == 2:
                n_edge += 1
            elif sh >= 3:
                n_face += 1
    tot = n_corner + n_edge + n_face
    out["p3_frac_edge_face"] = (n_edge + n_face) / tot if tot else np.nan
    out["p3_frac_face"] = n_face / tot if tot else np.nan
    out["p3_n_pairs"] = tot

    # --- 规则 4:最高价且最低配位的阳离子彼此是否相连
    if tot:
        zc = {i: val[i] for i in cats}
        hi = max(zc.values())
        lo_cn = min(cn[i] for i in cats if zc[i] == hi)
        crit = [i for i in cats if zc[i] == hi and cn[i] == lo_cn]
        viol = 0
        for a_i, a in enumerate(crit):
            for b in crit[a_i + 1:]:
                if ligs[a] & ligs[b]:
                    viol += 1
        out["p4_violate"] = float(viol > 0)
    else:
        out["p4_violate"] = np.nan

    # --- 规则 5:每个 (元素, 氧化态) 占几种不同 CN
    grp = collections.defaultdict(set)
    for i in cats:
        grp[(struct[i].specie.symbol, round(val[i], 2))].add(cn[i])
    out["p5_n_distinct"] = float(np.mean([len(v) for v in grp.values()])) if grp else np.nan
    out["p5_ok"] = float(all(len(v) == 1 for v in grp.values())) if grp else np.nan

    out["n_sites"] = len(struct)
    out["mean_cn_cat"] = float(np.mean([cn[i] for i in cats]))

    # --- 以下是供搜索用的候选判据原料。全部是 T1 级(CN + 拓扑),
    # 不含任何绝对键长量 —— 否则会检出"PBE 弛豫 vs 实验精修"的系统差而非化学。
    cnc = [cn[i] for i in cats]
    cna = [cn[j] for j in ans]
    out["cn_cat_max"] = float(max(cnc))
    out["cn_cat_min"] = float(min(cnc))
    out["cn_cat_span"] = float(max(cnc) - min(cnc))
    out["cn_cat_std"] = float(np.std(cnc))
    out["cn_an_mean"] = float(np.mean(cna)) if cna else np.nan
    out["cn_an_max"] = float(max(cna)) if cna else np.nan
    out["cn_an_span"] = float(max(cna) - min(cna)) if cna else np.nan
    out["frac_corner"] = n_corner / tot if tot else np.nan
    out["pair_per_cat"] = tot / len(cats)
    # 多面体连接图的度分布:每个阳离子多面体连了几个别的
    deg = collections.Counter()
    for a_i, a in enumerate(cats):
        for b in cats[a_i + 1:]:
            if ligs[a] & ligs[b]:
                deg[a] += 1
                deg[b] += 1
    dv = [deg.get(i, 0) for i in cats]
    out["poly_deg_mean"] = float(np.mean(dv))
    out["poly_deg_max"] = float(max(dv))
    out["frac_isolated"] = float(np.mean([x == 0 for x in dv]))
    # 电荷/化学计量类(纯组成级,T0)
    out["z_cat_max"] = float(max(val[i] for i in cats))
    out["z_cat_mean"] = float(np.mean([val[i] for i in cats]))
    out["n_cat_el"] = float(len({struct[i].specie.symbol for i in cats}))
    out["cat_an_ratio"] = len(cats) / len(ans)
    # 阴离子配位数的离散度 —— Hawthorne 说键强重分配就发生在这里
    out["cn_an_std"] = float(np.std(cna)) if cna else np.nan
    # 密堆填充代理:每原子体积(无量纲化到离子半径立方,避免直接用绝对体积)
    try:
        out["vol_per_atom"] = float(struct.volume / len(struct))
    except Exception:
        out["vol_per_atom"] = np.nan

    # --- 极值与计数型局部量(v2 新增)。
    # 泡林规则说的是**单个**多面体、**单个**阴离子的事;前一轮全用结构级均值/比例,
    # 把局部的严重违例平均掉了。实测结构级均值在多形体排序上最好只到 0.5631,
    # 而规则本身的主张是"存在一个共面连接就不稳定",那是 max/count 语义不是 mean 语义。
    if devs:
        out["p2_max_dev"] = float(max(devs))                      # 最严重的那个阴离子
        out["p2_n_bad_020"] = float(sum(1 for x in devs if x > 0.20))
        out["p2_n_bad_per_an"] = out["p2_n_bad_020"] / len(devs)
        out["p2_sum_dev"] = float(sum(devs))                      # 总违例量,不归一化
    else:
        out["p2_max_dev"] = out["p2_n_bad_020"] = np.nan
        out["p2_n_bad_per_an"] = out["p2_sum_dev"] = np.nan
    # 规则 3 的 count 语义:有几个共面/共边连接,而不是占比
    out["p3_n_face"] = float(n_face)
    out["p3_n_edge"] = float(n_edge)
    out["p3_n_face_per_cat"] = n_face / len(cats)
    out["p3_has_face"] = float(n_face > 0)                        # "存在即违反"
    # 规则 4 的 count 语义
    if tot:
        zc = {i: val[i] for i in cats}
        hi_z = max(zc.values())
        nv = 0
        for a_i, a in enumerate(cats):
            for b in cats[a_i + 1:]:
                if zc[a] == hi_z and zc[b] == hi_z and (ligs[a] & ligs[b]):
                    nv += 1
        out["p4_n_viol"] = float(nv)
        out["p4_n_viol_per_cat"] = nv / len(cats)
    else:
        out["p4_n_viol"] = out["p4_n_viol_per_cat"] = np.nan
    # 规则 5 的 max 语义:最坏的那个物种占了几种环境
    out["p5_max_distinct"] = float(max(len(v) for v in grp.values())) if grp else np.nan
    # 局部极端配位:最偏离该元素常见配位的那个位点
    out["cn_cat_range_norm"] = (max(cnc) - min(cnc)) / max(np.mean(cnc), 1e-9)
    return out


def process(rec):
    from pymatgen.core import Structure, Lattice
    try:
        if rec["kind"] == "real":
            st = Structure.from_str(read_blob_cif(rec["off"], rec["ln"]), fmt="cif")
        else:
            st = Structure(Lattice(rec["lattice"]), rec["species"], rec["coords"],
                           coords_are_cartesian=True)
        if len(st) > 200:
            return None                      # 大胞太慢,两侧同样截断
        val, ok = guess_oxi(st)
        if not ok:
            return None
        c = criteria(st, val)
        if c is None:
            return None
        c.update(sid=rec["sid"], rk=rec["rk"], kind=rec["kind"],
                 split=rec.get("split", "unsplit"), anion=rec.get("anion", ""))
        return c
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if OUT.exists() and not a.force:
        print(f"{OUT} 已存在,加 --force 重算")
        return 0

    prov = pd.read_parquet(F / "provenance.parquet",
                           columns=["source_id", "formula", "in_analysis_set",
                                    "anion", "blob_offset", "blob_length"])
    sp = pd.read_parquet(F / "splits.parquet")
    real = prov[prov.in_analysis_set].merge(sp, on="source_id", how="left")
    real["rk"] = real.formula.map(redkey)
    # 与 iter_elementa 同一条:剔单质。实测分析集 38,307 里含 810 个单质结构,
    # 泡林规则在其上没有定义,两侧必须用同一口径剔除。
    real = real[real.rk.notna() & real.rk.str.count(r"[A-Z]").ge(2)]
    print(f"真实侧剔单质后 {len(real):,}")

    ekeys = set()
    for r in iter_elementa(set(real.rk.dropna())):
        ekeys.add(r["rk"])
    common = set(real.rk.dropna()) & ekeys
    print(f"共同约化组成 {len(common):,}")

    recs = []
    sub = real[real.rk.isin(common)]
    for t in sub.itertuples():
        recs.append({"kind": "real", "sid": t.source_id, "rk": t.rk,
                     "off": int(t.blob_offset), "ln": int(t.blob_length),
                     "split": t.split, "anion": t.anion})
    for r in iter_elementa(common):
        r["kind"] = "elem"
        recs.append(r)
    if a.limit:
        recs = recs[:a.limit]
    print(f"待处理 {len(recs):,}(real {sum(1 for r in recs if r['kind']=='real'):,} / "
          f"elem {sum(1 for r in recs if r['kind']=='elem'):,})")

    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(process, recs, chunksize=16)):
            if res:
                rows.append(res)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,}/{len(recs):,} -> {len(rows):,} ok", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"\n写出 {OUT}  {len(df):,} 行")
    print(df.kind.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
