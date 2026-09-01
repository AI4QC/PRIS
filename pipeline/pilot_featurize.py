import os
#!/usr/bin/env python
"""newpauling 数据层 pilot: CIF -> 位点/键/连接对 特征库 (parquet)
用法: python pilot_featurize.py --n 2000 --subset exp_oxide --workers 24 --out $PRIS_FEATURES
环境: 见 requirements.txt
"""
import argparse, itertools, os, sqlite3, sys, time, warnings, zlib, math
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")
import logging; logging.getLogger().setLevel(logging.CRITICAL)

DB   = os.environ.get("PRIS_MATDATA_SQLITE", "materials.sqlite")
BLOB = os.environ.get("PRIS_MATDATA_BLOB", "structures.blob")

# ---------------------------------------------------------------- 取数
SUBSETS = {
 "exp_oxide":  "SELECT m.pk,m.material_id,m.formula,m.chemical_system,m.dataset,m.n_atoms,m.blob_offset,m.blob_length "
               "FROM materials m JOIN material_elements e ON e.material_pk=m.pk AND e.element='O' "
               "WHERE m.dataset='experimental'",
 "cand_oxide": "SELECT m.pk,m.material_id,m.formula,m.chemical_system,m.dataset,m.n_atoms,m.blob_offset,m.blob_length "
               "FROM materials m JOIN material_elements e ON e.material_pk=m.pk AND e.element='O' "
               "WHERE m.dataset='candidate'",
 "exp_all":    "SELECT pk,material_id,formula,chemical_system,dataset,n_atoms,blob_offset,blob_length "
               "FROM materials WHERE dataset='experimental'",
}

def fetch_rows(subset, n, seed=0):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(SUBSETS[subset]).fetchall()
    con.close()
    if n and n < len(rows):
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in rng.choice(len(rows), n, replace=False)]
    return rows

_BLOBF = None
def read_cif(off, ln):
    global _BLOBF
    if _BLOBF is None: _BLOBF = open(BLOB, "rb")
    _BLOBF.seek(off)
    return zlib.decompress(_BLOBF.read(ln)).decode("utf-8")

# ---------------------------------------------------------------- 氧化态
def assign_oxi(struct):
    """返回 (valences, source, ok)。source ∈ {cif, bva, guess, eneg, none}"""
    from pymatgen.analysis.bond_valence import BVAnalyzer
    v = [getattr(s.specie, "oxi_state", None) for s in struct]
    if all(x is not None for x in v):
        return [float(x) for x in v], "cif", True
    bare = struct.copy(); bare.remove_oxidation_states()
    try:
        return [float(x) for x in BVAnalyzer().get_valences(bare)], "bva", True
    except Exception:
        pass
    try:
        g = bare.composition.oxi_state_guesses(max_sites=-20)
        if g:
            d = g[0]
            return [float(d[s.specie.symbol]) for s in bare], "guess", True
    except Exception:
        pass
    return [np.nan]*len(struct), "none", False

# ---------------------------------------------------------------- 键价
from functools import lru_cache
BVPARM = os.environ.get("BVPARM_CIF",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bvparm2020.cif"))
_BVTAB = None
_REF_PREF = {"a": 0, "b": 1, "c": 2, "e": 3}   # Brown&Altermatt > Brese&O'Keeffe > Adams > Brown priv.
def bv_table():
    """解析 IUCr bvparm2020.cif -> {(el1,v1,el2,v2): (R0,B)}。
    注意: loop 头行有前导空格,必须 strip 后再判断,否则整表解析为空(踩过坑)。"""
    global _BVTAB
    if _BVTAB is not None: return _BVTAB
    tab = {}
    started = False
    for line in open(BVPARM, errors="ignore"):
        ls = line.strip()
        if ls.startswith("_valence_param_details"): started = True; continue
        if not started: continue
        p = ls.split()
        if len(p) < 7: continue
        try:
            e1, v1, e2, v2, R0, B, ref = p[0], int(p[1]), p[2], int(p[3]), float(p[4]), float(p[5]), p[6]
        except Exception: continue
        pr = _REF_PREF.get(ref, 9)
        for k in ((e1, v1, e2, v2), (e2, v2, e1, v1)):
            if k not in tab or pr < tab[k][2]: tab[k] = (R0, B, pr)
    _BVTAB = {k: (v[0], v[1]) for k, v in tab.items()}
    return _BVTAB

@lru_cache(maxsize=None)
def bv_param(el1, v1, el2, v2):
    t = bv_table()
    key = (el1, v1, el2, v2)
    if key in t: return t[key]
    # 回退1: 同元素对、价态最接近
    cands = [(abs(k[1]-v1)+abs(k[3]-v2), v) for k, v in t.items() if k[0]==el1 and k[2]==el2]
    if cands: return min(cands, key=lambda x: x[0])[1]
    # 回退2: pymatgen Brown 通用式 (b=0.37)
    try:
        from pymatgen.analysis.bond_valence import BV_PARAMS
        from pymatgen.core import Element
        a, b_ = Element(el1), Element(el2)
        r1, c1 = BV_PARAMS[a]["r"], BV_PARAMS[a]["c"]
        r2, c2 = BV_PARAMS[b_]["r"], BV_PARAMS[b_]["c"]
        R0 = r1 + r2 - r1*r2*(math.sqrt(c1)-math.sqrt(c2))**2/(c1*r1+c2*r2)
        return R0, 0.37
    except Exception:
        return None

def bond_valence(el1, o1, el2, o2, d):
    p = bv_param(el1, int(round(o1)), el2, int(round(o2)))
    if p is None: return float("nan")
    R0, b = p
    return math.exp((R0 - d)/b)

# ------------------------------------------------- Hawthorne 先验键强
def a_priori_bond_strengths(nodes_charge, edges):
    """nodes_charge: dict site_idx -> 形式电荷(阳正阴负)
       edges: list of (i, j, image_tuple) —— 标记商图(labelled quotient graph)
       返回 dict edge_idx -> 先验键强 s (vu), 以及诊断信息
       方程: (1) 每个位点 Σ s = |z|   (2) 每个独立回路 Σ (-1)^k s = 0 (等价键强规则)
    """
    import networkx as nx
    E = len(edges)
    if E == 0: return None, {"status": "no_edges"}
    idx = {v: k for k, v in enumerate(sorted(nodes_charge))}
    V = len(idx)
    rows, rhs = [], []
    # (1) 电荷守恒
    for site, z in nodes_charge.items():
        r = np.zeros(E)
        for e, (i, j, im) in enumerate(edges):
            if i == site: r[e] += 1
            if j == site: r[e] += 1
        rows.append(r); rhs.append(abs(z))
    # (2) 回路方程 —— 在标记商图上取 cycle basis
    G = nx.MultiGraph()
    for e, (i, j, im) in enumerate(edges):
        G.add_edge(i, j, key=e)
    T = nx.minimum_spanning_tree(nx.Graph(G))
    tree_edges = set()
    for u, v in T.edges():
        for k in G[u][v]:
            tree_edges.add(k); break
    for e, (i, j, im) in enumerate(edges):
        if e in tree_edges: continue
        try:
            path = nx.shortest_path(T, i, j)
        except Exception:
            continue
        r = np.zeros(E); r[e] = 1.0
        sign = -1.0
        for a, b in zip(path[:-1], path[1:]):
            for k in G[a][b]:
                if k in tree_edges:
                    r[k] += sign; sign *= -1; break
        rows.append(r); rhs.append(0.0)
    A = np.array(rows); y = np.array(rhs)
    s, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    diag = {"n_eq": A.shape[0], "n_unk": E, "rank": int(rank),
            "underdet": bool(rank < E),
            "cond": float(sv[0]/sv[-1]) if len(sv) and sv[-1] > 0 else np.inf,
            "resid": float(np.linalg.norm(A@s - y))}
    return s, diag

# ---------------------------------------------------------------- 主流程
def featurize(row, use_chemenv=True):
    pk, mid, formula, chemsys, dataset, natoms, off, ln = row
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN
    out = {"pk": pk, "material_id": mid, "formula": formula, "dataset": dataset}
    try:
        st = Structure.from_str(read_cif(off, ln), fmt="cif")
    except Exception as e:
        out["status"] = "parse_fail"; return out, [], [], []
    val, vsrc, vok = assign_oxi(st)
    out["oxi_source"] = vsrc
    if not vok:
        out["status"] = "no_oxi"; return out, [], [], []
    bare = st.copy(); bare.remove_oxidation_states()
    # --- 快通道: CrystalNN 邻接
    try:
        cnn = CrystalNN(weighted_cn=False, cation_anion=False)
        nninfo = [cnn.get_nn_info(bare, i) for i in range(len(bare))]
    except Exception as e:
        out["status"] = "nn_fail"; return out, [], [], []
    sym = [s.specie.symbol for s in bare]
    cations = [i for i in range(len(bare)) if val[i] > 0]
    anions  = [i for i in range(len(bare)) if val[i] < 0]
    # --- 键表(只保留阳-阴键)
    bonds = []          # (i, j, image, d)
    for i in cations:
        for nb in nninfo[i]:
            j = nb["site_index"]
            if val[j] >= 0: continue
            im = tuple(int(x) for x in nb["image"])
            d = float(np.linalg.norm(bare.lattice.get_cartesian_coords(
                    bare[j].frac_coords + np.array(im) - bare[i].frac_coords)))
            bonds.append((i, j, im, d))
    if not bonds:
        out["status"] = "no_cation_anion_bond"; return out, [], [], []
    # --- BVS: 用固定截断(3.5 Å)独立于 CN 算法求和;与 CN 解耦是关键(见设计文档 §5)
    bvs_cut = defaultdict(float); bv_missing = 0; bv_tot = 0
    cs_, ns_, _off, ds_ = bare.get_neighbor_list(3.5)
    for a_, b_, d_ in zip(cs_, ns_, ds_):
        if val[a_]*val[b_] >= 0: continue
        bv_tot += 1
        sv = bond_valence(sym[a_], val[a_], sym[b_], val[b_], float(d_))
        if np.isnan(sv): bv_missing += 1; continue
        bvs_cut[a_] += sv
    out["bv_param_missing_frac"] = bv_missing/max(bv_tot, 1)
    # --- 键级 BV(沿 CN 邻接)
    bvs = defaultdict(float); cn = Counter()
    bond_rows = []
    for bi, (i, j, im, d) in enumerate(bonds):
        s = bond_valence(sym[i], val[i], sym[j], val[j], d)
        if not np.isnan(s):
            bvs[i] += s; bvs[j] += s
        cn[i] += 1; cn[j] += 1
        bond_rows.append(dict(pk=pk, bond_idx=bi, i=i, j=j, el_i=sym[i], el_j=sym[j],
                              oxi_i=val[i], oxi_j=val[j], dist=d, bv=s,
                              img_a=im[0], img_b=im[1], img_c=im[2]))
    # --- Hawthorne 先验键强
    charges = {i: val[i] for i in set([b[0] for b in bonds]) | set([b[1] for b in bonds])}
    apri, diag = a_priori_bond_strengths(charges, [(b[0], b[1], b[2]) for b in bonds])
    if apri is not None:
        for bi, s in enumerate(apri): bond_rows[bi]["bv_apriori"] = float(s)
        out.update({"apri_"+k: v for k, v in diag.items()})
    # --- 位点表
    site_rows = []
    for i in range(len(bare)):
        z = val[i]; c = cn.get(i, 0)
        site_rows.append(dict(pk=pk, site=i, el=sym[i], oxi=z, cn=c,
            is_cation=bool(z > 0), bvs=float(bvs.get(i, np.nan)),
            bvs_cut=float(bvs_cut.get(i, np.nan)),
            bvs_dev=float(bvs_cut[i] - abs(z)) if bvs_cut.get(i) else np.nan,
            lewis=float(abs(z)/c) if c else np.nan,          # Lewis 酸/碱强度 = |z|/CN
            fx=float(bare[i].frac_coords[0]), fy=float(bare[i].frac_coords[1]),
            fz=float(bare[i].frac_coords[2])))
    # --- 多面体连接(共角/共边/共面)
    lig = defaultdict(set)
    for i, j, im, d in bonds: lig[i].add((j, im))
    conn_rows = []
    keys = sorted(lig)
    for a, b in itertools.combinations_with_replacement(keys, 2):
        for shift in itertools.product((-1, 0, 1), repeat=3):
            if a == b and shift <= (0, 0, 0): continue
            sb = {(jj, tuple(np.array(imm)+np.array(shift))) for jj, imm in lig[b]}
            k = len(lig[a] & sb)
            if k:
                conn_rows.append(dict(pk=pk, i=a, j=b, el_i=sym[a], el_j=sym[b],
                    oxi_i=val[a], oxi_j=val[b], cn_i=cn[a], cn_j=cn[b],
                    n_shared=k, mode={1: "corner", 2: "edge"}.get(k, "face"),
                    sx=shift[0], sy=shift[1], sz=shift[2]))
    # --- 结构级汇总
    nb = len(bond_rows)
    devs = [r["bvs_dev"] for r in site_rows if not np.isnan(r["bvs_dev"])]
    gii = float(np.sqrt(np.mean(np.square(devs)))) if devs else float("nan")
    mc = Counter(r["mode"] for r in conn_rows); tc = sum(mc.values()) or 1
    out.update(status="ok", n_sites=len(bare), n_bonds=nb, n_cations=len(cations),
               gii=gii, f_corner=mc["corner"]/tc, f_edge=mc["edge"]/tc, f_face=mc["face"]/tc,
               n_conn=tc)
    return out, site_rows, bond_rows, conn_rows

def worker(rows):
    S, SI, B, C = [], [], [], []
    for r in rows:
        try:
            o, s, b, c = featurize(r)
        except Exception as e:
            o, s, b, c = {"pk": r[0], "status": "err:"+type(e).__name__}, [], [], []
        S.append(o); SI += s; B += b; C += c
    return S, SI, B, C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--subset", default="exp_oxide")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="/tmp/newpauling_feat")
    a = ap.parse_args()
    rows = fetch_rows(a.subset, a.n)
    print(f"{a.subset}: {len(rows)} 条", flush=True)
    t0 = time.time()
    import multiprocessing as mp
    chunks = [rows[i::a.workers] for i in range(a.workers)]
    with mp.Pool(a.workers) as p:
        res = p.map(worker, chunks)
    S = [x for r in res for x in r[0]]; SI = [x for r in res for x in r[1]]
    B = [x for r in res for x in r[2]]; C = [x for r in res for x in r[3]]
    dt = time.time()-t0
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(a.out, exist_ok=True)
    for name, data in [("structures", S), ("sites", SI), ("bonds", B), ("polyconn", C)]:
        if data:
            df = pd.DataFrame(data)
            pq.write_table(pa.Table.from_pandas(df), f"{a.out}/{name}.parquet", compression="zstd")
            print(f"  {name}: {len(df)} 行 -> {os.path.getsize(a.out+'/'+name+'.parquet')/1e6:.1f} MB")
    st = Counter(x.get("status") for x in S); ox = Counter(x.get("oxi_source") for x in S)
    print("status:", dict(st)); print("oxi_source:", dict(ox))
    print(f"耗时 {dt:.1f}s, {dt/len(rows)*1e3:.1f} ms/struct(墙钟), "
          f"{dt*a.workers/len(rows)*1e3:.1f} ms/struct(核)")

if __name__ == "__main__":
    main()
