#!/usr/bin/env python3
"""静电 + 键价特征 —— 补上实验结构域最大的缺口。

# 为什么是最大的缺口

计算域(LeMat 多形体)的判据公式里,权重最大的三项全是静电量:
  1.852*z(ewald_point) + 0.951*z(ewald_real) ... -0.718*z(mad_min) -0.687*z(mad_range)
而实验结构域的特征表里**一个静电量都没有** —— real_rank.parquet 只有配位数统计
和泡林 2-5 的量。实验域公式因此只能靠 cn_an_max / cn_cat_max / vol_per_atom,
全覆盖准确率只有 0.54-0.61。

# 算什么

静电(EwaldSummation,形式电荷):
  ewald_per_atom / ewald_real / ewald_recip / ewald_point   每原子能量分解
  mad_std / mad_max / mad_min / mad_range                   位点马德隆能的分布
  madz_*                                                    位点能除以该位点电荷 —— **尺度无关**
                                                            (原始 ewald 正比于 z^2,跨化学不可比,
                                                             当"法则"用会退化成"高价化合物"探测器)

有效配位数(Hoppe ECoN):w_i = exp(1 - (d_i/d_min)^6),ECoN = sum w_i
  连续版配位数,对键长分布敏感,不像整数 CN 那样在阈值附近跳变。

键价(Brown-Altermatt,**形式电荷**):
  gii / bv_max_abs / bv_frac_bad / bv_param_cov
  **不用 BVAnalyzer** —— 它从键长反推价态再去检验键价定律,是循环论证。
  这里价态一律来自组成的形式电荷(guess_oxi)。

# 复现性

扰动种子用 crc32(sid),与 phys_law.py 完全一致,所以 elec_bad 与 phys_bad 的
每一行都对应**同一个**扰动结构,可按 sid 安全合并。
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminate import guess_oxi, read_blob_cif  # noqa: E402
from phys_law import seed_of  # noqa: E402

F = os.environ.get("PRIS_FEATURES", "features/")
BVPARM = os.environ.get(
    "NEWPAULING_BVPARM",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bvparm2020.cif"),
)
_BV = None


def bv_table():
    """IUCr bvparm2020。坑:loop 头行有前导空格,必须先 strip 再 split。"""
    global _BV
    if _BV is not None:
        return _BV
    tab = {}
    try:
        with open(BVPARM, errors="ignore") as fh:
            for ln in fh:
                p = ln.strip().split()
                if len(p) < 6:
                    continue
                try:
                    key = (p[0], int(p[1]), p[2], int(p[3]))
                    r0, b = float(p[4]), float(p[5])
                except (ValueError, IndexError):
                    continue
                if b > 0 and r0 > 0:
                    tab.setdefault(key, (r0, b))
    except OSError:
        pass
    _BV = tab
    return tab


def elec_feats(st, val):
    """静电 + ECoN + 键价。返回 dict,失败返回 None。"""
    from pymatgen.analysis.ewald import EwaldSummation
    from pymatgen.analysis.local_env import CrystalNN

    n = len(st)
    out = {}
    # ---- 静电
    try:
        sd = st.copy()
        sd.add_oxidation_state_by_site(val)
        ew = EwaldSummation(sd, compute_forces=False)
        out["ewald_per_atom"] = float(ew.total_energy) / n
        out["ewald_real"] = float(ew.real_space_energy) / n
        out["ewald_recip"] = float(ew.reciprocal_space_energy) / n
        out["ewald_point"] = float(ew.point_energy) / n
        sm = np.array([ew.get_site_energy(i) for i in range(n)])
        out["mad_std"] = float(np.std(sm))
        out["mad_max"] = float(np.max(sm))
        out["mad_min"] = float(np.min(sm))
        out["mad_range"] = float(np.max(sm) - np.min(sm))
        # 尺度无关版:位点能 / 电荷。原始位点能 ~ z^2,跨化学不可比,
        # 当阈值型"法则"用会退化成"高价化合物"探测器。
        z = np.array([v if abs(v) > 1e-9 else np.nan for v in val], float)
        mz = sm / np.abs(z)
        mz = mz[np.isfinite(mz)]
        if len(mz):
            out["madz_mean"] = float(np.mean(mz))
            out["madz_std"] = float(np.std(mz))
            out["madz_range"] = float(np.max(mz) - np.min(mz))
        # 阴阳离子位点能是否分离(阳离子应为负、阴离子应为负,但深浅不同)
        cs = sm[np.asarray(val) > 0]
        as_ = sm[np.asarray(val) < 0]
        if len(cs) and len(as_):
            out["mad_cat_mean"] = float(np.mean(cs))
            out["mad_an_mean"] = float(np.mean(as_))
    except Exception:
        return None

    # ---- 近邻:ECoN 与键价共用一次 CrystalNN
    try:
        cnn = CrystalNN(weighted_cn=False, x_diff_weight=0.0)
        nn = []
        for i in range(n):
            try:
                nn.append(cnn.get_nn_info(st, i))
            except Exception:
                nn.append([])
    except Exception:
        return out

    BV = bv_table()
    econ, rsd = [], []
    bvs = np.zeros(n)
    hit = miss = 0
    for i in range(n):
        nbs = nn[i]
        if not nbs:
            continue
        ds = np.array([st[i].distance(st[nb["site_index"]], jimage=nb.get("image"))
                       for nb in nbs])
        if len(ds) == 0 or ds.min() <= 0:
            continue
        # Hoppe 有效配位数:近邻按 (d/dmin)^6 指数衰减加权
        w = np.exp(1 - (ds / ds.min()) ** 6)
        econ.append(float(w.sum()))
        if len(ds) > 1:
            rsd.append(float(np.std(ds) / np.mean(ds)))
        # 键价:只累加阳-阴键,价态用形式电荷
        for nb, d in zip(nbs, ds):
            j = nb["site_index"]
            if val[i] * val[j] >= 0 or val[i] <= 0:
                continue
            par = BV.get((st[i].specie.symbol, int(round(val[i])),
                          st[j].specie.symbol, int(round(val[j]))))
            if par is None:
                miss += 1
                continue
            hit += 1
            r0, b = par
            s = np.exp((r0 - d) / b)
            bvs[i] += s
            bvs[j] += s
    if econ:
        out["econ_mean"] = float(np.mean(econ))
        out["econ_std"] = float(np.std(econ))
        out["econ_max"] = float(np.max(econ))
        out["econ_min"] = float(np.min(econ))
        out["dist_rsd"] = float(np.mean(rsd)) if rsd else 0.0
        out["dist_rsd_max"] = float(np.max(rsd)) if rsd else 0.0
    if hit >= 2:
        dev = np.array([bvs[i] - abs(val[i]) for i in range(n) if abs(val[i]) > 1e-9])
        if len(dev):
            out["gii"] = float(np.sqrt(np.mean(dev ** 2)))
            out["bv_max_abs"] = float(np.max(np.abs(dev)))
            out["bv_mean_abs"] = float(np.mean(np.abs(dev)))
            out["bv_frac_bad"] = float(np.mean(np.abs(dev) > 0.20))
            # 相对失配:除以形式电荷,跨价态可比
            rel = np.array([abs(bvs[i] - abs(val[i])) / abs(val[i])
                            for i in range(n) if abs(val[i]) > 1e-9])
            out["bv_rel_max"] = float(np.max(rel))
            out["bv_rel_mean"] = float(np.mean(rel))
        out["bv_param_cov"] = hit / max(hit + miss, 1)
    return out


# ---------------------------------------------------------------- 驱动

MAXN = 80          # Ewald 是 O(N^2) 级,超过 80 原子代价陡增


def _real(r):
    from pymatgen.core import Structure
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > MAXN:
            return None
        val, ok = guess_oxi(st)
        if not ok:
            return None
        f = elec_feats(st, val)
        if not f:
            return None
        f["source_id"] = r["sid"]
        return f
    except Exception:
        return None


def _bad(r):
    from pymatgen.core import Structure
    from make_negatives import perturb, swapped_val
    out = []
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > MAXN:
            return out
        val, ok = guess_oxi(st)
        if not ok:
            return out
        rng = np.random.default_rng(seed_of(r["sid"]))   # 与 phys_law.py 同种子
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            p = perturb(st, kind, rng, val)
            if p is None:
                continue
            f = elec_feats(p, swapped_val(p, val))
            if not f:
                continue
            f.update(sid=r["sid"] + "_" + kind, kind=kind, parent=r["sid"])
            out.append(f)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "bad"])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0, help="调试用,只跑前 N 条")
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()

    prov = pd.read_parquet(F + "provenance.parquet",
                           columns=["source_id", "in_analysis_set", "blob_offset",
                                    "blob_length", "n_elements"])
    if a.mode == "real":
        d = prov[prov.n_elements >= 2]
        fn, outf = _real, F + "elec_real.parquet"
    else:
        d = prov[prov.in_analysis_set & (prov.n_elements >= 2)].sample(
            n=min(a.n, len(prov)), random_state=0)
        fn, outf = _bad, F + "elec_bad.parquet"
    recs = [{"sid": t.source_id, "off": int(t.blob_offset), "ln": int(t.blob_length)}
            for t in d.itertuples()]
    if a.limit:
        recs = recs[:a.limit]
        outf = outf.replace(".parquet", "_smoke.parquet")
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
