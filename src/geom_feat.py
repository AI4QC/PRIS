#!/usr/bin/env python3
"""几何配位特征 —— 多智能体调研筛出、并已给出实测排除力的三个量。

调研(5 个特征族并行 + 对抗性核查)在 40 余个候选里,这三个是"有化学含义 +
可算 + 实测有效"三条同时满足的:

  aa_min  配体-配体接触比 = min over 配体对 of d(L_i,L_j)/(r_Li+r_Lj)
          **这是泡林第一定律的几何内核** —— 半径比判据的原始论证就是
          "阳离子太小则配体互相接触,多面体不稳"。泡林把它写成了 r_c/r_a 的
          区间表(据 George 2020 只有 66% 满足),这里直接算配体到底碰没碰上。
          调研实测:抓 S1(压缩轴上配体被挤到一起)与 S3(位移撞入)。

  phi     多面体凸包填充率 = Vol(配体凸包) / ((4/3)pi R^3),R = 平均阳-配体距离
          度量配位多面体有多"塌"。调研实测:真实 0.293 -> S2 0.123 / S3 0.096。
          **S2 阳离子换位是四类里最难排的一类**,phi 是少数对它有效的几何量。

  mef     Hoppe MEFIR 有效离子半径失配(有符号)
          MEFIR = sum_i w_i (d_i - r_anion_i) / sum_i w_i,w 用 ECoN 权重;
          失配 = (MEFIR - r_shannon(阳离子)) / r_shannon(阳离子)。
          调研实测四类全抓,S2 最显著(-0.47 对真实的 -0.06)——
          因为换位后大阳离子被塞进小位点,MEFIR 会明显偏离表列半径。

调研同时给出三条**负结果**,已据此不做:
  - rho_aniso(键长压缩方向各向异性):均值排除力仅 0.080
  - Ward Voronoi 堆积分数:**尺度不变量**,对 S4 整体膨胀排除力恒为 0
  - Lewis 酸碱强度匹配:在其实现下没抓到任何东西
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
from phys_law import seed_of, shannon  # noqa: E402

F = os.environ.get("PRIS_FEATURES", "features/")
MAXN = 80


def geom_feats(st, val, nn=None):
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

    aa, phis, mefs, eccs, angvars = [], [], [], [], []
    for i in range(len(st)):
        if val[i] <= 0 or not nn[i]:
            continue
        # 只取阴离子配体;配体的笛卡尔坐标要用**镜像**位置,否则跨周期边界会算错
        pos, rad, dist = [], [], []
        for nb in nn[i]:
            j = nb["site_index"]
            if val[j] >= 0:
                continue
            img = nb.get("image")
            fc = st[j].frac_coords + (np.asarray(img) if img is not None else 0)
            pos.append(st.lattice.get_cartesian_coords(fc))
            rad.append(shannon(st[j].specie.symbol, val[j], max(cn[j], 1)))
            dist.append(np.linalg.norm(
                st.lattice.get_cartesian_coords(fc) - st[i].coords))
        if len(pos) < 3:
            continue
        pos = np.asarray(pos)
        rad = np.asarray(rad)
        dist = np.asarray(dist)
        if not np.all(np.isfinite(dist)) or dist.min() <= 0:
            continue

        # --- aa_min:最近的一对配体,距离 / 半径和。< 1 表示配体壳层互相穿透
        best = np.inf
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                rs = rad[a] + rad[b]
                if rs > 0:
                    best = min(best, float(np.linalg.norm(pos[a] - pos[b]) / rs))
        if np.isfinite(best):
            aa.append(best)

        # --- phi:配体凸包体积 / 等效球体积。多面体塌陷时显著变小
        if len(pos) >= 4:
            try:
                from scipy.spatial import ConvexHull
                hv = float(ConvexHull(pos - st[i].coords).volume)
                R = float(dist.mean())
                if R > 0:
                    phis.append(hv / (4.0 / 3.0 * np.pi * R ** 3))
            except Exception:
                pass

        # --- ecc / angvar:**角度**畸变。现有全部特征都是长度类
        # (bl_rsd_max 是键长相对标准差),没有一条约束键角。
        # ecc  = |sum û_i| / n,û 为阳离子指向各配体的单位矢量。
        #        中心对称多面体应接近 0;阳离子偏离多面体中心时变大。
        # angvar = 配体-阳离子-配体张角的方差(度^2),Robinson 键角方差的简化。
        # 这两个量是 ChemEnv 连续对称性度量(CSM)的廉价替代 ——
        # 调研实测 CSM 真实中位 1.75 → S3 位移 6.56 / S1 压缩 2.99 / S2 换位 2.83,
        # 但 CSM 需要对 54 种理想多面体做配准,代价太高。
        u = (pos - st[i].coords) / dist[:, None]
        eccs.append(float(np.linalg.norm(u.sum(axis=0)) / len(u)))
        if len(u) >= 3:
            cosang = np.clip(u @ u.T, -1, 1)
            iu = np.triu_indices(len(u), 1)
            angs = np.degrees(np.arccos(cosang[iu]))
            angvars.append(float(np.var(angs)))

        # --- mef:Hoppe MEFIR 相对失配(有符号)
        w = np.exp(1 - (dist / dist.min()) ** 6)
        if w.sum() > 0:
            mefir = float((w * (dist - rad)).sum() / w.sum())
            rc = shannon(st[i].specie.symbol, val[i], max(cn[i], 1))
            if rc > 0:
                mefs.append((mefir - rc) / rc)

    if not aa and not phis and not mefs and not eccs:
        return None
    out = {}
    if aa:
        out["aa_min"] = float(min(aa))
        out["aa_mean"] = float(np.mean(aa))
    if phis:
        out["phi_min"] = float(min(phis))
        out["phi_mean"] = float(np.mean(phis))
    if eccs:
        out["ecc_max"] = float(max(eccs))
        out["ecc_mean"] = float(np.mean(eccs))
    if angvars:
        out["angvar_max"] = float(max(angvars))
        out["angvar_mean"] = float(np.mean(angvars))
    if mefs:
        m = np.asarray(mefs)
        out["mef_mean"] = float(m.mean())
        out["mef_min"] = float(m.min())
        out["mef_max"] = float(m.max())
        out["mef_absmax"] = float(np.abs(m).max())
    return out


def _real(r):
    from pymatgen.core import Structure
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > MAXN:
            return None
        val, ok = guess_oxi(st)
        if not ok:
            return None
        f = geom_feats(st, val)
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
        rng = np.random.default_rng(seed_of(r["sid"]))   # 与 phys_law/elec_feat 同种子
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            p = perturb(st, kind, rng, val)
            if p is None:
                continue
            f = geom_feats(p, swapped_val(p, val))
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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()

    prov = pd.read_parquet(F + "provenance.parquet",
                           columns=["source_id", "in_analysis_set", "blob_offset",
                                    "blob_length", "n_elements"])
    if a.mode == "real":
        d = prov[prov.n_elements >= 2]
        fn, outf = _real, F + "geom_real.parquet"
    else:
        d = prov[prov.in_analysis_set & (prov.n_elements >= 2)].sample(
            n=min(a.n, len(prov)), random_state=0)
        fn, outf = _bad, F + "geom_bad.parquet"
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
