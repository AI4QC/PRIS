#!/usr/bin/env python3
"""ELEMENTA 内部多形体排序 v2:先在组成层面筛电荷可平衡的离子化合物。

# v1 的瓶颈

v1 对每个结构调 oxi_state_guesses,成功率只有 7%。诊断发现**不是 bug**:
ELEMENTA 是"全部一元二元 + 系数<=4 的三元"的系统扫描,绝大多数组成
根本配不平电荷(如 Li2BeBr:+4 vs -1),不是离子化合物,泡林规则无定义。

修法:**电荷平衡是组成的性质,不是结构的性质**。同一组成的 5 个多形体
共享同一套价态。所以在组成层面判一次、缓存,而不是每个结构判一次。
吞吐因此提高一个量级,可用组从 633 扩到数千。

# 任务(与 v1 相同,零指纹)

同一组成、同一生成器、同一套 DFT 设置下的多形体组内排序:
判据要把能量更低的排在前面。生成器/组成/DFT 设置在组内恒定,指纹在构造上不存在。
"""
from __future__ import annotations
import os
import argparse
import collections
import itertools
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminate import criteria, redkey, ANIONS  # noqa: E402

F = os.environ.get("PRIS_FEATURES", "features/")
ELEM = os.environ.get("PRIS_ELEMENTA", "elementa/endpoints_open.extxyz")
RE_F = re.compile(r"\bformula=(\S+)")
RE_LAT = re.compile(r'Lattice="([^"]+)"')
RE_E = re.compile(r"\benergy=(\S+)")

# 阴离子的标准电荷。ELEMENTA 都是简单二元三元,不考虑过氧/多硫等异常价态。
AN_Z = {"O": -2, "S": -2, "Se": -2, "Te": -2, "N": -3, "P": -3,
        "F": -1, "Cl": -1, "Br": -1, "I": -1}
# 常见阳离子价态。取正值,按常见度排序 —— 平衡解不唯一时取第一个可行组合。
CAT_Z = {
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1], "Ag": [1], "Tl": [1, 3],
    "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2], "Zn": [2], "Cd": [2],
    "Cu": [2, 1], "Ni": [2, 3], "Co": [2, 3], "Fe": [3, 2], "Mn": [2, 4, 3, 7],
    "Cr": [3, 6, 2], "V": [5, 3, 4, 2], "Ti": [4, 3], "Zr": [4], "Hf": [4],
    "Al": [3], "Ga": [3], "In": [3, 1], "Sc": [3], "Y": [3], "La": [3],
    "B": [3], "Si": [4], "Ge": [4, 2], "Sn": [4, 2], "Pb": [2, 4],
    "As": [5, 3], "Sb": [5, 3], "Bi": [3, 5], "Nb": [5], "Ta": [5],
    "Mo": [6, 4], "W": [6, 4], "Re": [7, 4], "Ru": [4, 3], "Rh": [3], "Pd": [2],
    "Pt": [4, 2], "Au": [3, 1], "Hg": [2, 1], "Th": [4], "U": [6, 4],
    "Ce": [3, 4], "Pr": [3], "Nd": [3], "Sm": [3], "Eu": [3, 2], "Gd": [3],
    "Tb": [3], "Dy": [3], "Ho": [3], "Er": [3], "Tm": [3], "Yb": [3, 2], "Lu": [3],
}


def balance(formula: str):
    """组成层面解电荷平衡。返回 {元素: 价态} 或 None。

    只解一次并缓存 —— 同组成的多形体共享同一套价态,这是 v2 提速的关键。
    """
    d = collections.Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            d[el] += int(n or 1)
    ans = [e for e in d if e in AN_Z]
    if len(ans) != 1:
        return None
    an = ans[0]
    cats = [e for e in d if e != an]
    if not cats or any(c not in CAT_Z for c in cats):
        return None
    need = -AN_Z[an] * d[an]                    # 阳离子总正电荷必须等于这个
    for combo in itertools.product(*[CAT_Z[c] for c in cats]):
        if sum(z * d[c] for z, c in zip(combo, cats)) == need:
            out = {c: z for c, z in zip(cats, combo)}
            out[an] = AN_Z[an]
            return out
    return None


def scan(max_groups: int, verbose=True):
    """流式扫全库。组成层面先判平衡(带缓存),不平衡的整组跳过。"""
    bal_cache: dict[str, dict | None] = {}
    groups = collections.defaultdict(list)
    n_seen = n_bal = 0
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
            com = fh.readline()
            lines = [fh.readline() for _ in range(nat)]
            idx += 1
            if nat > 60:
                continue
            mf = RE_F.search(com)
            if not mf:
                continue
            fo = mf.group(1)
            n_seen += 1
            if fo not in bal_cache:
                bal_cache[fo] = balance(fo)
            val_map = bal_cache[fo]
            if val_map is None:
                continue
            n_bal += 1
            ml, me = RE_LAT.search(com), RE_E.search(com)
            if not (ml and me):
                continue
            rk = redkey(fo)
            lat = np.array([float(x) for x in ml.group(1).split()]).reshape(3, 3)
            syms, pos = [], []
            for ln in lines:
                p = ln.split()
                if len(p) < 4:
                    break
                syms.append(p[0])
                pos.append([float(p[1]), float(p[2]), float(p[3])])
            if len(syms) != nat:
                continue
            groups[rk].append({"sid": f"elem-{idx}", "rk": rk, "nat": nat,
                               "e_per_atom": float(me.group(1)) / nat,
                               "lattice": lat, "species": syms,
                               "coords": np.array(pos), "vmap": val_map})
            if len(groups) >= max_groups * 2 and n_bal > max_groups * 8:
                break
    if verbose:
        print(f"扫描 {n_seen:,} 个端点,电荷可平衡 {n_bal:,} ({100*n_bal/max(n_seen,1):.1f}%)")
    out = {k: v for k, v in groups.items() if len(v) >= 2}
    keys = sorted(out, key=lambda k: -len(out[k]))[:max_groups]
    return {k: out[k] for k in keys}


def one(rec):
    from pymatgen.core import Structure, Lattice
    try:
        st = Structure(Lattice(rec["lattice"]), rec["species"], rec["coords"],
                       coords_are_cartesian=True)
        val = [float(rec["vmap"][s.specie.symbol]) for s in st]
        c = criteria(st, val)
        if c is None:
            return None
        c.update(sid=rec["sid"], rk=rec["rk"], e_per_atom=rec["e_per_atom"], nat=rec["nat"])
        return c
    except Exception:
        return None


def rank_eval(d, crits, tag):
    """组内配对:判据能否把能量更低的排在前面。整簇自助给 CI。"""
    print(f"\n=== {tag}(随机=0.5)===")
    res = {}
    for name, col, sgn in crits:
        w = t = n = 0
        per = collections.defaultdict(list)
        for rk, gg in d.groupby("rk"):
            v = gg[[col, "e_per_atom"]].dropna().values
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    ci, ei = v[i]; cj, ej = v[j]
                    if ei == ej:
                        continue
                    n += 1
                    if ci == cj:
                        s = 0.5; t += 1
                    else:
                        s = 1.0 if (sgn * ci > sgn * cj) == (ei < ej) else 0.0
                        w += s
                    per[rk].append(s)
        if n < 200:
            print(f"  {name:20s} 配对不足 ({n})"); continue
        wr = (w + 0.5 * t) / n
        rng = np.random.default_rng(0); ks = list(per); bs = []
        for _ in range(2000):
            pick = rng.choice(len(ks), len(ks), replace=True)
            bs.append(np.mean([x for i in pick for x in per[ks[i]]]))
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        sig = "★" if lo > 0.5 else ("✗反向" if hi < 0.5 else "")
        print(f"  {name:20s} 胜率={wr:.4f} [{lo:.4f},{hi:.4f}] 平局={t/n:.3f} n={n:,} {sig}")
        res[name] = {"winrate": wr, "ci": [lo, hi], "tie": t / n, "n": n}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()

    g = scan(a.groups)
    recs = [r for v in g.values() for r in v]
    print(f"多形体组 {len(g):,},端点 {len(recs):,}(每组均 {len(recs)/max(len(g),1):.1f})")

    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, recs, chunksize=16)):
            if r:
                rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1:,}/{len(recs):,} -> {len(rows):,} ok", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(F + "polymorph_rank2.parquet", index=False)
    gs = d.groupby("rk")
    keep = [k for k, gg in gs if len(gg) >= 2 and gg.e_per_atom.nunique() > 1]
    d = d[d.rk.isin(keep)]
    print(f"特征化成功 {len(rows):,};可用组 {len(keep):,},端点 {len(d):,}")

    PAUL = [("泡林2 键强偏差", "p2_mean_dev", -1),
            ("泡林3 共边共面", "p3_frac_edge_face", -1),
            ("泡林3 共面", "p3_frac_face", -1),
            ("泡林4 违例", "p4_violate", -1),
            ("泡林5 CN种类数", "p5_n_distinct", -1)]
    POOL = [("每原子体积", "vol_per_atom", -1), ("平均阳离子CN", "mean_cn_cat", +1),
            ("平均阴离子CN", "cn_an_mean", +1), ("最大阳离子CN", "cn_cat_max", +1),
            ("最小阳离子CN", "cn_cat_min", +1), ("CN跨度", "cn_cat_span", -1),
            ("阴离子CN离散度", "cn_an_std", -1), ("多面体度均值", "poly_deg_mean", +1),
            ("多面体度最大", "poly_deg_max", +1), ("孤立多面体比例", "frac_isolated", -1),
            ("共角比例", "frac_corner", +1), ("每阳离子连接对数", "pair_per_cat", +1),
            ("阳离子CN标准差", "cn_cat_std", -1), ("阴阳离子数比", "cat_an_ratio", +1)]

    res = {"n_groups": len(keep), "n_endpoints": len(d)}
    res["pauling"] = rank_eval(d, PAUL, "泡林五条作为判据")
    res["pool"] = rank_eval(d, POOL, "候选判据池")
    best_p = max((v["winrate"] for v in res["pauling"].values()), default=0.5)
    print(f"\n泡林最好:{best_p:.4f}")
    win = {k: v for k, v in res["pool"].items() if v["ci"][0] > max(0.5, best_p)}
    print(f"显著打过泡林全部五条的候选:{len(win)} 条")
    for k, v in sorted(win.items(), key=lambda x: -x[1]["winrate"]):
        print(f"  ★ {k:16s} {v['winrate']:.4f} [{v['ci'][0]:.4f},{v['ci'][1]:.4f}]")
    res["winners"] = win
    with open(F + "polymorph_rank2.json", "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=float)
    print(f"\n写出 {F}polymorph_rank2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
