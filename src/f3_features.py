#!/usr/bin/env python3
"""PREREG-F3 §4: 对称性 + 经典 Born 项特征,写入 synth_rank_aug.parquet(新文件)。

定义冻结于 docs/plans/2026-08-14-f3-synthesizability-prereg.md。
计算不读取 synth 标签;对 dev/holdout 全部 6,878 行统一计算。
"""
from __future__ import annotations
import os
import glob, gzip, json, math, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SNAP = os.environ.get("PRIS_MP_SNAPSHOT", "mp-summary/")
F = os.environ.get("PRIS_FEATURES", "features/")

CSYS_RANK = {"triclinic": 1, "monoclinic": 2, "orthorhombic": 3, "tetragonal": 4,
             "trigonal": 5, "hexagonal": 6, "cubic": 7}
RHO_BM = 0.345  # Born–Mayer softness, Å (frozen)
CUT = 1.25      # contact window in units of r_sum (frozen)


def _feats(st, val):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.local_env import CrystalNN
    from phys_law import shannon

    out = {}
    n = len(st)
    for tag, sp in (("001", 0.01), ("01", 0.1)):
        try:
            sga = SpacegroupAnalyzer(st, symprec=sp)
            ds = sga.get_symmetry_dataset()
            out[f"sg_num_{tag}"] = float(ds.number)
            out[f"wyckoff_econ_{tag}"] = float(len(set(ds.equivalent_atoms)) / n)
            if tag == "001":
                out["csys_rank_001"] = float(CSYS_RANK.get(sga.get_crystal_system(), np.nan))
        except Exception:
            out[f"sg_num_{tag}"] = np.nan
            out[f"wyckoff_econ_{tag}"] = np.nan
            if tag == "001":
                out["csys_rank_001"] = np.nan

    cnn = CrystalNN(weighted_cn=False, x_diff_weight=0.0)
    cn = []
    for i in range(n):
        try:
            cn.append(len(cnn.get_nn_info(st, i)))
        except Exception:
            cn.append(6)
    rad = [shannon(st[i].specie.symbol, val[i], max(cn[i], 1)) for i in range(n)]

    acc = {"rep9_ca_pa": 0.0, "rep9_aa_pa": 0.0, "rep9_cc_pa": 0.0,
           "repexp_ca_pa": 0.0, "repexp_aa_pa": 0.0, "repexp_cc_pa": 0.0,
           "strain2_ca_pa": 0.0}
    try:
        nbrs = st.get_all_neighbors(5.5)
    except Exception:
        return None
    for i in range(n):
        for nb in nbrs[i]:
            j = nb.index
            d = nb.nn_distance
            if d < 1e-6:
                continue
            rs = rad[i] + rad[j]
            if rs <= 0 or d >= CUT * rs:
                continue
            s = val[i] * val[j]
            if s < 0:
                k = "ca"
                acc["strain2_ca_pa"] += 0.5 * ((d - rs) / rs) ** 2
            elif val[i] < 0 and val[j] < 0:
                k = "aa"
            elif val[i] > 0 and val[j] > 0:
                k = "cc"
            else:
                continue
            acc[f"rep9_{k}_pa"] += 0.5 * (rs / d) ** 9
            acc[f"repexp_{k}_pa"] += 0.5 * math.exp((rs - d) / RHO_BM)
    for k, v in acc.items():
        out[k] = float(v / n)
    out["density"] = float(st.density)
    return out


def one_page(job):
    path, want = job
    out = []
    try:
        with gzip.open(path, "rt") as fh:
            d = json.load(fh)
    except Exception:
        return out
    from pymatgen.core import Structure
    from polymorph_rank2 import balance
    for k in d.get("data", []):
        mid = k.get("material_id")
        if mid not in want:
            continue
        try:
            st = Structure.from_dict(k["structure"])
            vm = balance(st.composition.reduced_formula.replace(" ", ""))
            if vm is None:
                continue
            val = [float(vm[s.specie.symbol]) for s in st]
            f = _feats(st, val)
            if f is None:
                continue
            f["mp_id"] = mid
            out.append(f)
        except Exception:
            continue
    return out


def main() -> int:
    want = set(pd.read_parquet(F + "synth_rank.parquet").mp_id)
    pages = sorted(glob.glob(os.path.join(SNAP, "page-*.json.gz")))
    print(f"{len(want)} target ids, {len(pages)} pages", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, chunk in enumerate(ex.map(one_page, [(p, want) for p in pages])):
            rows.extend(chunk)
            if (i + 1) % 20 == 0:
                print(f"  page {i+1}/{len(pages)}  rows {len(rows)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(F + "synth_rank_aug.parquet", index=False)
    print(f"wrote {len(d)} rows -> synth_rank_aug.parquet")
    print(d.describe().T[["count", "mean", "std"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
