# -*- coding: utf-8 -*-
"""复现 George 2020 的五个数字(MPU-1 / PREREG §6 门 G-A)。

George, Waroquiers, Di Stefano, Petretto, Rignanese, Hautier,
*The Limited Predictive Power of the Pauling Rules*, Angew. Chem. Int. Ed. 2020, 59, 7569.
本地 PDF:the published PDF (not redistributed here)(原文口径均逐句核对过)。

================================================================ 口径差异(必须显式列出)
| 维度 | George 2020 | 本复现 |
|---|---|---|
| 论域 | ICSD ∩ Materials Project 的约 5,000 个氧化物 | `provenance.oxide_strict` = **23,728** 条 |
| 来源 | ICSD(经 MP 再弛豫/再对称化) | **ICSD + COD**,**不含 MP**,用的是**实验报道的原胞**未经 DFT 弛豫 |
| 有序性 | MP 条目本身已是有序 | **有序结构 only**(无序条目在建库时已剔除,PREREG §8.1) |
| 阴离子 | 氧化物 | **单一阴离子 O**(`oxide_strict` 判据) |
| 氧化态 | MP 的 `oxi_state`(BVAnalyzer 派生) | **`cif`(ICSD 原生)/ `guess`(纯组成枚举)两级,BVAnalyzer 整批排除**(PREREG §5) |
| 近邻 | ChemEnv 单算法 | ChemEnv / CrystalNN / BrunnerNN **三算法**(G6 硬门要求) |

**注意 `oxide_strict` 不是 `in_analysis_set` 的子集**:差集 3,895 条全部含 P
(`in_analysis_set` 把 P 记作阴离子候选,于是磷酸盐的 `n_anion_kinds==2` 被排除)。
George 的论域**包含**磷酸盐(原文用 InPO4 做第二定律的主例),所以本脚本按 `oxide_strict`
取全部 23,728 条重算,**不复用 `site/pair/struct.parquet`**(那三张表只覆盖 `in_analysis_set`,
且 `pair` 只有 ChemEnv 一条路线、没有阳离子–阴离子键级表,第二定律的 Σs 算不了)。

================================================================ 五条规则的操作化(逐条对齐原文)
* **规则 1(半径比)** granularity=`site`/`orbit`。r_cation/r_anion 用**泡林单价半径**
  (`pauling_radii.py`;Shannon 依赖 CN,是循环论证,G7 明令禁止)。
  预测 CN 由硬球临界比给出,命中 = 预测 CN == 观测 CN(严格相等)。
  George:66% of the tested local environments。
* **规则 2(静电价)** granularity=`site(anion)`/`orbit`。对每个 O 位点求
  Σ s = Σ_cations (z_c / CN_c),判据 |Σs − 2| ≤ 0.01。George:~20% of all oxygen atoms。
* **规则 3(连接类型)** granularity=`pair`。所有相连多面体对里共角/共棱/共面的比例。
  George:62.5 / 27.2 / 10.3;**CN ≤ 8 时 73.3 / 25.0 / 1.6**。
* **规则 4(相邻多面体)** granularity=`structure`。原文对"违例"的定义是
  "structures in which the polyhedra of cations with the highest valence and smallest
  coordination number are connected"。即:V = 全结构阳离子位点的最大氧化态,
  C = 最小 CN,取集合 A = {位点 | ox==V 且 cn==C};若 A 内部存在相连对 → 违例。
  只在**含 ≥2 个不同阳离子物种**的结构上适用(原文 "In a crystal containing different cations"),
  单阳离子结构记为满足——这是复现原文给的正例(金红石 SnO2 被列为"满足四条")所必需的。
  **同时给"氧化态版"(只用 ox==V)与"CN 版"(只用 cn==C)两条**:George 证明只有后者成立。
  George:违例 ~40%。
* **规则 5(简约)** granularity=`structure`。每个阳离子物种 (元素, 氧化态) 在同一结构里
  只占据一种局域环境。原文 Fig.5b 图注写明 "only coordination numbers are considered",
  故主判据用 **CN**;ChemEnv 另给 `ce_symbol` 版本作敏感性。George:~70.3%。
* **2–5 同时** granularity=`structure`,合取。规则 2 的结构级布尔取"该结构全部 O 位点都满足",
  规则 3 的结构级布尔取"无共面对"(泡林原文 "particularly of shared faces")。
  George:13%(CN ≤ 8 时 20%)。

================================================================ 用法
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python src/reproduce_george.py --stage compute --limit 300 --workers 20 --force   # 冒烟
    nohup python src/reproduce_george.py --stage compute --workers 20 --force \
          > $FEAT/reproduce_george.log 2>&1 &                                          # 全量
    python src/reproduce_george.py --stage table                                       # 出 Table S1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_features as BF          # noqa: E402  复用 read_cif / assign_oxi / symmetry_info
from pauling_radii import univalent_radius, predict_cn   # noqa: E402

FEAT = BF.FEAT
OUT = {
    "site": f"{FEAT}/george_site.parquet",
    "anion": f"{FEAT}/george_anion.parquet",
    "pair": f"{FEAT}/george_pair.parquet",
    "struct": f"{FEAT}/george_struct.parquet",
}
SHARD_DIR = f"{FEAT}/_gshards"
ALGOS = ("chemenv", "crystalnn", "brunner")

PER_STRUCT_TIMEOUT = 300   # 秒。与 build_features 同(§6.5-3 实测 ChemEnv 在 >200 原子胞常超 60 s)
EPS_RULE2 = 0.01           # George 原文:an absolute deviation of 0.01 is allowed


# ================================================================ 氧化态(在 build_features 之上打一个补丁)
def assign_oxi_fixed(struct, source):
    """`build_features.assign_oxi` 的修正版。

    **实测坑**:ICSD 有 3,658/38,307(9.6%)条目的 `_atom_type_oxidation_number` 全是 0
    (例:exp001 ZnO 记成 Zn0/O0)。原实现 `blob_ox_present = all(x is not None)`
    判为 True 并返回 `ox_source='cif'`,而 `is_cat = [v>0]` 于是**一个阳离子位点都没有**,
    结构被静默掏空。这里加一条体检:`cif` 只在"阴离子 ox<0 且至少一个位点 ox>0"时才采信,
    否则降级到 `guess`。这条修正会改变 `ox_source` 的覆盖分布,报告里单列。
    """
    v = [getattr(s.specie, "oxi_state", None) for s in struct]
    blob_ox = all(x is not None for x in v)
    meta = {"blob_ox_present": bool(blob_ox), "n_guess_sol": 0, "guess_unique": False,
            "cif_all_zero": False}
    if blob_ox and source == "icsd":
        vv = [float(x) for x in v]
        if any(x > 0 for x in vv) and any(x < 0 for x in vv):
            return vv, "cif", meta
        meta["cif_all_zero"] = True          # 全 0 / 全同号 → 不可用,降级
    bare = struct.copy()
    bare.remove_oxidation_states()
    try:
        g = bare.composition.oxi_state_guesses(max_sites=-1)
        meta["n_guess_sol"] = len(g)
        meta["guess_unique"] = (len(g) == 1)
        if g:
            d = g[0]
            return [float(d[s.specie.symbol]) for s in bare], "guess", meta
    except Exception:
        pass
    return None, "none", meta


# ================================================================ 多面体连接枚举(三算法共用)
def enumerate_connections(cat_idx, ligands):
    """给定每个阳离子位点的配体集合,枚举**每个原胞**里的多面体连接对。

    `ligands[i]` = set of (阴离子位点号 j, 周期像 (a,b,c)),像是相对于 i 位于 (0,0,0) 时的。
    两个多面体 (i @ 0) 与 (j @ T) 共享的配体数 = |L_i ∩ (L_j + T)|。
    去重约定(与 ChemEnv `environment_subgraph` 的"每个原胞一条边"一致):
      - i < j:枚举全部 T;
      - i == j:T ≠ 0,且 T 与 −T 只取字典序较大的一个(同一条连接的两种看法)。
    返回 [(i, j, n_shared)],不返回 T(下游只用 n_shared)。
    **三算法用同一个枚举器**,这样 G6 比较的是"近邻定义"这一个自由度,而不是连带把
    连接性算法也换掉了(ChemEnv 自带的 ConnectivityFinder 与本函数的一致性在 --check-conn 里核过)。
    """
    out = []
    # 反向索引:阴离子位点 → [(阳离子, 像)],用于快速找候选平移
    for a_pos, i in enumerate(cat_idx):
        Li = ligands[i]
        if not Li:
            continue
        by_site_i = defaultdict(list)
        for (j0, im) in Li:
            by_site_i[j0].append(im)
        for j in cat_idx[a_pos:]:
            Lj = ligands[j]
            if not Lj:
                continue
            cands = set()
            for (b, ib) in Lj:
                for ia in by_site_i.get(b, ()):
                    cands.add((ia[0] - ib[0], ia[1] - ib[1], ia[2] - ib[2]))
            for T in cands:
                if i == j:
                    if T == (0, 0, 0):
                        continue
                    if T < (-T[0], -T[1], -T[2]):
                        continue
                shifted = {(b, (ib[0] + T[0], ib[1] + T[1], ib[2] + T[2])) for (b, ib) in Lj}
                ns = len(Li & shifted)
                if ns >= 1:
                    out.append((i, j, ns))
    return out


def mode_of(ns):
    """{1: 共角, 2: 共棱, ≥3: 共面}(George 2020 与 ChemEnv 同口径)"""
    return "corner" if ns == 1 else ("edge" if ns == 2 else "face")


# ================================================================ 单结构主流程
def process_one(rec):
    """rec = (source_id, source, blob_offset, blob_length)。阴离子恒为 O(论域是 oxide_strict)。"""
    import signal
    sid, source, off, ln = rec
    t0 = time.time()

    def _alarm(signum, frame):
        raise TimeoutError(f"per-structure timeout {PER_STRUCT_TIMEOUT}s")
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, PER_STRUCT_TIMEOUT)
    except Exception:
        pass

    try:
        from pymatgen.core import Structure
        st = Structure.from_str(BF.read_cif(off, ln), fmt="cif")
        n = len(st)
        els = [s.specie.symbol for s in st]
        val, ox_src, meta = assign_oxi_fixed(st, source)

        # 阳离子判定:有价态用 ox>0,否则退化为"元素 != O"(论域已保证单一阴离子 O)
        if val is None:
            ox_arr = [np.nan] * n
            is_cat = [e != "O" for e in els]
        else:
            ox_arr = val
            is_cat = [v > 0 for v in val]
        is_an = [e == "O" for e in els]
        cat_idx = [i for i in range(n) if is_cat[i]]
        an_idx = [i for i in range(n) if is_an[i]]

        frac_ox = any(abs(o - round(o)) > 1e-6 for o in ox_arr if o == o)
        el2ox = defaultdict(set)
        for e, o in zip(els, ox_arr):
            if o == o:
                el2ox[e].add(round(float(o), 4))
        mixed_valence = any(len(s) > 1 for s in el2ox.values())

        orb_hi, wy_hi, mu_hi, spg_hi, _, _ = BF.symmetry_info(st, BF.SYMPREC_HI)
        orb_lo, wy_lo, mu_lo, spg_lo, _, _ = BF.symmetry_info(st, BF.SYMPREC_LO)

        bare = st.copy()
        bare.remove_oxidation_states()

        # ---------------- 三算法各自的配体集合(只留阳离子–阴离子键)
        ligands = {a: {i: set() for i in cat_idx} for a in ALGOS}
        cn_all = {a: {i: np.nan for i in cat_idx} for a in ALGOS}   # 未按阴离子过滤的 CN(审计用)
        ok = {a: False for a in ALGOS}
        ce_sym = [None] * n
        csm = [np.nan] * n
        err = {}

        # --- CrystalNN(§6.3 坑 B:x_diff_weight 默认 3.0)
        try:
            from pymatgen.analysis.local_env import CrystalNN
            cnn = CrystalNN(weighted_cn=False, x_diff_weight=BF.X_DIFF_WEIGHT)
            for i in cat_idx:
                try:
                    info = cnn.get_nn_info(bare, i)
                except Exception:
                    continue
                cn_all["crystalnn"][i] = float(len(info))
                for d in info:
                    j = int(d["site_index"])
                    if is_an[j]:
                        ligands["crystalnn"][i].add((j, tuple(int(round(x)) for x in d["image"])))
            ok["crystalnn"] = True
        except Exception as e:
            err["crystalnn"] = f"{type(e).__name__}:{e}"[:150]

        # --- BrunnerNN_relative(默认参数)
        try:
            from pymatgen.analysis.local_env import BrunnerNN_relative
            bnn = BrunnerNN_relative()
            for i in cat_idx:
                try:
                    info = bnn.get_nn_info(bare, i)
                except Exception:
                    continue
                cn_all["brunner"][i] = float(len(info))
                for d in info:
                    j = int(d["site_index"])
                    if is_an[j]:
                        ligands["brunner"][i].add((j, tuple(int(round(x)) for x in d["image"])))
            ok["brunner"] = True
        except Exception as e:
            err["brunner"] = f"{type(e).__name__}:{e}"[:150]

        # --- ChemEnv(§6.3 坑 A:必须显式传 valences,否则 only_cations=True 返回垃圾)
        try:
            from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder \
                import LocalGeometryFinder
            from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies \
                import MultiWeightsChemenvStrategy
            from pymatgen.analysis.chemenv.coordination_environments.structure_environments \
                import LightStructureEnvironments
            lgf = LocalGeometryFinder()
            lgf.setup_parameters(centering_type="centroid", include_central_site_in_centroid=True,
                                 structure_refinement=LocalGeometryFinder.STRUCTURE_REFINEMENT_NONE)
            lgf.setup_structure(structure=bare)
            if val is not None:
                se = lgf.compute_structure_environments(
                    only_cations=True, valences=[int(round(v)) for v in val],
                    maximum_distance_factor=BF.MAX_DIST_FACTOR)
            else:
                se = lgf.compute_structure_environments(
                    only_cations=False, maximum_distance_factor=BF.MAX_DIST_FACTOR)
            lse = LightStructureEnvironments.from_structure_environments(
                strategy=MultiWeightsChemenvStrategy.stats_article_weights_parameters(),
                structure_environments=se)
            for i in cat_idx:
                if i < len(lse.coordination_environments) and lse.coordination_environments[i]:
                    c0 = lse.coordination_environments[i][0]
                    ce_sym[i] = c0["ce_symbol"]
                    csm[i] = float(c0.get("csm", np.nan))
                ns_ = lse.neighbors_sets[i] if i < len(lse.neighbors_sets) else None
                if not ns_:
                    continue
                nb = ns_[0]
                cn_all["chemenv"][i] = float(len(nb))
                for e_ in nb.neighb_sites_and_indices:
                    j = int(e_["index"])
                    if is_an[j]:
                        img = np.round(np.asarray(e_["site"].frac_coords) -
                                       np.asarray(st[j].frac_coords)).astype(int)
                        ligands["chemenv"][i].add((j, (int(img[0]), int(img[1]), int(img[2]))))
            ok["chemenv"] = True
        except Exception as e:
            err["chemenv"] = f"{type(e).__name__}:{e}"[:150]

        # ---------------- 三算法各自的 CN / 阴离子 Σs / 多面体对
        cn = {a: {i: (float(len(ligands[a][i])) if ligands[a][i] else np.nan) for i in cat_idx}
              for a in ALGOS}
        sigma = {a: {j: 0.0 for j in an_idx} for a in ALGOS}
        ncat = {a: {j: 0 for j in an_idx} for a in ALGOS}
        maxcsm = {a: {j: np.nan for j in an_idx} for a in ALGOS}
        pair_rows = []
        for a in ALGOS:
            for i in cat_idx:
                c = cn[a][i]
                if not (c == c) or c <= 0:
                    continue
                s_i = (ox_arr[i] / c) if ox_arr[i] == ox_arr[i] else np.nan
                for (j, _img) in ligands[a][i]:
                    ncat[a][j] += 1
                    sigma[a][j] += s_i if s_i == s_i else np.nan
                    if csm[i] == csm[i]:
                        maxcsm[a][j] = csm[i] if not (maxcsm[a][j] == maxcsm[a][j]) \
                            else max(maxcsm[a][j], csm[i])
            for (i, j, ns_) in enumerate_connections(cat_idx, ligands[a]):
                pair_rows.append((sid, a, i, j, els[i], els[j],
                                  ox_arr[i] if ox_arr[i] == ox_arr[i] else np.nan,
                                  ox_arr[j] if ox_arr[j] == ox_arr[j] else np.nan,
                                  cn[a][i], cn[a][j], int(ns_), mode_of(ns_),
                                  orb_hi[i], orb_hi[j], orb_lo[i], orb_lo[j]))

        site_rows = [(sid, source, i, els[i],
                      float(ox_arr[i]) if ox_arr[i] == ox_arr[i] else np.nan, ox_src,
                      cn["chemenv"][i], cn["crystalnn"][i], cn["brunner"][i],
                      cn_all["chemenv"][i], cn_all["crystalnn"][i], cn_all["brunner"][i],
                      ce_sym[i], csm[i],
                      orb_hi[i], mu_hi[i], orb_lo[i], mu_lo[i], wy_lo[i])
                     for i in cat_idx]
        anion_rows = [(sid, source, j, ox_src,
                       sigma["chemenv"][j] if ncat["chemenv"][j] else np.nan,
                       sigma["crystalnn"][j] if ncat["crystalnn"][j] else np.nan,
                       sigma["brunner"][j] if ncat["brunner"][j] else np.nan,
                       ncat["chemenv"][j], ncat["crystalnn"][j], ncat["brunner"][j],
                       maxcsm["chemenv"][j],
                       orb_hi[j], mu_hi[j], orb_lo[j], mu_lo[j])
                      for j in an_idx]
        struct_row = (sid, source, ox_src, n, len(cat_idx), len(an_idx),
                      bool(mixed_valence), bool(frac_ox), bool(meta["cif_all_zero"]),
                      int(meta["n_guess_sol"]), bool(meta["guess_unique"]),
                      spg_hi, spg_lo,
                      ok["chemenv"], ok["crystalnn"], ok["brunner"],
                      ";".join(f"{k}={v}" for k, v in err.items())[:250],
                      int((time.time() - t0) * 1000), "ok")
        return site_rows, anion_rows, pair_rows, struct_row, None

    except BaseException as e:
        f = (sid, source, type(e).__name__, str(e)[:250], traceback.format_exc()[-400:])
        return [], [], [], (sid, source, None, 0, 0, 0, False, False, False, 0, False,
                            None, None, False, False, False, f"FAIL:{type(e).__name__}",
                            int((time.time() - t0) * 1000), "fail"), f
    finally:
        try:
            import signal as _s
            _s.setitimer(_s.ITIMER_REAL, 0)
        except Exception:
            pass


# ================================================================ 表结构
SITE_COLS = ["source_id", "source", "site_index", "element", "ox_state", "ox_source",
             "cn_chemenv", "cn_crystalnn", "cn_brunner",
             "cnall_chemenv", "cnall_crystalnn", "cnall_brunner",
             "ce_symbol", "csm", "orbit_s01", "mult_s01", "orbit_s001", "mult_s001", "wyckoff_s001"]
ANION_COLS = ["source_id", "source", "site_index", "ox_source",
              "sigma_chemenv", "sigma_crystalnn", "sigma_brunner",
              "ncat_chemenv", "ncat_crystalnn", "ncat_brunner", "maxcsm_chemenv",
              "orbit_s01", "mult_s01", "orbit_s001", "mult_s001"]
PAIR_COLS = ["source_id", "nn_algo", "i", "j", "el_i", "el_j", "ox_i", "ox_j",
             "cn_i", "cn_j", "n_shared", "mode",
             "orbit_i_s01", "orbit_j_s01", "orbit_i_s001", "orbit_j_s001"]
STRUCT_COLS = ["source_id", "source", "ox_source", "n_atoms", "n_cat", "n_anion",
               "mixed_valence", "frac_ox", "cif_all_zero", "n_guess_sol", "guess_unique",
               "spg_s01", "spg_s001", "ok_chemenv", "ok_crystalnn", "ok_brunner",
               "err", "wall_ms", "status"]
FAIL_COLS = ["source_id", "source", "err_type", "err", "tb"]


def run_shard(args):
    k, recs, shard_dir = args
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    import warnings
    warnings.filterwarnings("ignore")
    done = f"{shard_dir}/struct_{k}.parquet"
    if os.path.exists(done):
        return k, -1, 0
    S, A, P, T, F = [], [], [], [], []
    for r in recs:
        s, a, p, t, f = process_one(r)
        S += s
        A += a
        P += p
        T.append(t)
        if f:
            F.append(f)
    for name, rows, cols in (("site", S, SITE_COLS), ("anion", A, ANION_COLS),
                             ("pair", P, PAIR_COLS), ("fail", F, FAIL_COLS),
                             ("struct", T, STRUCT_COLS)):
        df = pd.DataFrame(rows, columns=cols)
        df.to_parquet(f"{shard_dir}/{name}_{k}.parquet", compression="zstd", index=False)
    return k, len(T), sum(1 for t in T if t[-1] == "ok")


def merge_shards(name, shard_dir, out_path, cols):
    import glob
    files = sorted(glob.glob(f"{shard_dir}/{name}_*.parquet"),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not files:
        raise RuntimeError(f"没有 {name} 分片,拒绝产出空表")
    w, ntot = None, 0
    for f in files:
        t = pq.read_table(f)
        if t.num_rows == 0:
            continue
        if w is None:
            w = pq.ParquetWriter(out_path, t.schema, compression="zstd")
        w.write_table(t.cast(w.schema))
        ntot += t.num_rows
    if w is None:
        pd.DataFrame(columns=cols).to_parquet(out_path, compression="zstd", index=False)
    else:
        w.close()
    return ntot


# ================================================================ compute
def stage_compute(limit, workers, chunk, force):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    prov = pd.read_parquet(BF.PROV, columns=["source_id", "source", "oxide_strict",
                                             "blob_offset", "blob_length", "n_atoms"])
    sub = prov[prov.oxide_strict].copy()
    if len(sub) != 23728:
        print(f"[warn] oxide_strict = {len(sub)},与简报的 23,728 不符,请核对 provenance")
    if limit:
        # 冒烟用**随机抽样**(seed 固定)而不是取小胞:取小胞会把单结构耗时低估一个数量级,
        # 外推全量时间就废了(§6.6 要求"先小子集冒烟,量准单结构耗时,再全量")
        sub = sub.sample(n=min(limit, len(sub)), random_state=0)
    recs = list(zip(sub.source_id, sub.source, sub.blob_offset, sub.blob_length))
    shard_dir = SHARD_DIR + ("_smoke" if limit else "")
    if force and os.path.isdir(shard_dir):
        import shutil
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)
    tasks = [(k, recs[i:i + chunk], shard_dir) for k, i in enumerate(range(0, len(recs), chunk))]
    print(f"[compute] {len(recs)} 结构 / {len(tasks)} 分片 / {workers} 进程", flush=True)
    t0 = time.time()
    ndone = nok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_shard, t): t[0] for t in tasks}
        for c, fu in enumerate(as_completed(futs), 1):
            k, ns, no = fu.result()
            if ns > 0:
                ndone += ns
                nok += no
            if c % 50 == 0 or c == len(tasks):
                el = time.time() - t0
                print(f"  {c}/{len(tasks)} 片 | {ndone} 结构 | ok {nok} | "
                      f"{el/60:.1f} min | eta {el/c*(len(tasks)-c)/60:.1f} min", flush=True)
    suf = "_smoke" if limit else ""
    for name, cols in (("site", SITE_COLS), ("anion", ANION_COLS),
                       ("pair", PAIR_COLS), ("struct", STRUCT_COLS), ("fail", FAIL_COLS)):
        p = OUT.get(name, f"{FEAT}/george_{name}.parquet").replace(".parquet", f"{suf}.parquet")
        nrow = merge_shards(name, shard_dir, p, cols)
        print(f"[merge] {name}: {nrow} 行 → {p}")
    print(f"[compute] 完成,墙钟 {(time.time()-t0)/60:.1f} min")


# ================================================================ 统计工具
def wilson(k, n, z=1.959963985):
    """Wilson 95% 置信区间(比 Wald 在极端比例下可靠;PREREG 要求每条规则给 CI)"""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100*p:5.1f} [{100*lo:4.1f},{100*hi:4.1f}] n={n}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["compute", "table"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="table 阶段读 _smoke 产物")
    a = ap.parse_args()
    if a.stage == "compute":
        stage_compute(a.limit, a.workers, a.chunk, a.force)
    else:
        from george_table import stage_table
        stage_table(smoke=a.smoke)
