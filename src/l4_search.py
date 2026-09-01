#!/usr/bin/env python3
"""PREREG-L4: 贪心谓词搜索(L3 + ≤4 新谓词)。

  search  先复现 L3 已发表点位,后在 discovery 搜索并冻结 L4
  calib   calibration 一次求值 + 各类分解
  lopo    新谓词逐条 LOPO(删目标类重选,在删除类上求值)

冻结文档 docs/plans/2026-08-14-l4-plausibility-prereg.md(sha256 199788c4...)。
"""
from __future__ import annotations
import os
import argparse, hashlib, itertools, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

F = os.environ.get("PRIS_FEATURES", "features/")
B = os.environ.get("PRIS_LAW_TABLES", "law_tables/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "20260814_l4_plausibility")

QGRID = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
         0.60, 0.70, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
EXCLUDE = {"source_id", "anion", "split", "psplit", "sid", "kind", "parent", "n_el",
           "n_sites", "p2_n_bad_020", "p2_sum_dev", "p3_n_pairs", "p3_n_face",
           "p3_n_edge", "p4_n_viol", "nsites"}


def load_tables():
    lr = pd.read_parquet(B + "law_real.parquet")
    lb = pd.read_parquet(B + "law_bad.parquet")
    ar = pd.read_parquet(F + "law_real_aug.parquet")
    ab = pd.read_parquet(F + "law_bad_aug.parquet")
    lr = lr.merge(ar, on="source_id", how="left")
    lb = lb.merge(ab, on=["parent", "kind"], how="left")
    return lr.reset_index(drop=True), lb.reset_index(drop=True)


def _le(v, th):
    return np.where(np.isfinite(v), v <= th, True)


def _ge(v, th):
    return np.where(np.isfinite(v), v >= th, True)


def l3_mask(d):
    bl, blm = d.bl_min.values, d.bl_mean.values
    cn, mz, mx = d.cn_an_mean.values, d.madz_range.values, d.mad_max.values
    lk, fi = d.frac_like_bonds.values, d.fi.values
    d1 = _ge(bl, 0.804)
    d3 = np.where(np.isfinite(cn) & (cn <= 3.333), _le(blm, 1.081), True)
    d4, d5 = _le(mz, 31.45), _le(mx, 15.17)
    d6 = np.where(np.isfinite(fi) & (fi > 0.55), _le(lk, 1e-4), True)
    return d1 & d3 & d4 & d5 & d6


def pred_mask(d, feat, direction, th, guard, _cache={}):
    key = id(d)
    cols = _cache.setdefault(key, {})
    if feat not in cols:
        cols[feat] = d[feat].values.astype(float)
    v = cols[feat]
    m = _le(v, th) if direction == "le" else _ge(v, th)
    if guard:
        if "fi" not in cols:
            cols["fi"] = d.fi.values.astype(float)
        fi = cols["fi"]
        m = np.where(np.isfinite(fi) & (fi > guard), m, True)
    return m


def candidates(lr_disc):
    feats = [c for c in lr_disc.columns
             if c not in EXCLUDE and pd.api.types.is_numeric_dtype(lr_disc[c])]
    cands = []
    for f in feats:
        v = lr_disc[f].values.astype(float)
        v = v[np.isfinite(v)]
        if v.size < 100 or np.std(v) == 0:
            continue
        ths = sorted(set(np.quantile(v, QGRID).tolist()))
        for th, dr, g in itertools.product(ths, ("le", "ge"), (None, 0.50, 0.55)):
            cands.append((f, dr, float(th), g))
    return cands


def cmd_search():
    lr, lb = load_tables()
    lrd = lr[lr.split == "discovery"].reset_index(drop=True)
    lbd = lb[lb.psplit == "discovery"].reset_index(drop=True)
    lrc = lr[lr.split == "calibration"].reset_index(drop=True)
    lbc = lb[lb.psplit == "calibration"].reset_index(drop=True)

    # 复现检查:L3 已发表点位
    sat_d = l3_mask(lrd).mean(); exc_d = 1 - l3_mask(lbd).mean()
    sat_c = l3_mask(lrc).mean(); exc_c = 1 - l3_mask(lbc).mean()
    print(f"L3 reproduction: discovery {sat_d:.4f}/{exc_d:.4f} (pub 0.9071/0.7052)  "
          f"calibration {sat_c:.4f}/{exc_c:.4f} (pub 0.9171/0.7004)")
    if abs(sat_d - 0.9071) > 0.002 or abs(exc_d - 0.7052) > 0.002:
        print("L3 does not reproduce; refusing to search")
        return 1

    cands = candidates(lrd)
    print(f"{len(cands)} candidate predicates")
    Sr, Sb = l3_mask(lrd), l3_mask(lbd)
    chosen = []
    for step in range(4):
        base_sat = Sr.mean(); base_exc = 1 - Sb.mean()
        best = None
        for c in cands:
            mr = pred_mask(lrd, *c)
            sat = (Sr & mr).mean()
            if sat < 0.81:
                continue
            mb = pred_mask(lbd, *c)
            exc = 1 - (Sb & mb).mean()
            if best is None or exc > best[0]:
                best = (exc, sat, c)
        if best is None or best[0] - base_exc < 0.005:
            print(f"stop at step {step+1}: best gain "
                  f"{(best[0]-base_exc) if best else float('nan'):+.4f}")
            break
        exc, sat, c = best
        chosen.append(c)
        Sr &= pred_mask(lrd, *c)
        Sb &= pred_mask(lbd, *c)
        print(f"step {step+1}: {c}  -> disc sat {sat:.4f} exc {exc:.4f}", flush=True)

    frozen = dict(prereg_sha256="199788c4988da1b25d1a8c960f200e92fad91ebe84c85f707d0a2be8fa0c2ba1",
                  predicates=chosen,
                  disc_sat=float(Sr.mean()), disc_exc=float(1 - Sb.mean()))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "L4_frozen.json")
    json.dump(frozen, open(p, "w"), indent=1)
    open(p + ".sha256", "w").write(hashlib.sha256(open(p, "rb").read()).hexdigest() + "\n")
    print(f"L4 frozen: {len(chosen)} new predicates, discovery "
          f"{frozen['disc_sat']:.4f}/{frozen['disc_exc']:.4f}")
    return 0


def apply_l4(d, fz):
    m = l3_mask(d)
    for c in fz["predicates"]:
        m &= pred_mask(d, *c)
    return m


def cmd_calib():
    contact = os.path.join(OUT, "CALIB_CONTACT.log")
    if os.path.exists(contact):
        print("already contacted; refusing")
        return 1
    fz = json.load(open(os.path.join(OUT, "L4_frozen.json")))
    lr, lb = load_tables()
    lrc = lr[lr.split == "calibration"].reset_index(drop=True)
    lbc = lb[lb.psplit == "calibration"].reset_index(drop=True)
    sat = apply_l4(lrc, fz).mean()
    mb = apply_l4(lbc, fz)
    exc = 1 - mb.mean()
    per = {k: float(1 - mb[lbc.kind.values == k].mean())
           for k in ("S1", "S2", "S3", "S4", "S5")}
    res = dict(calib_sat=float(sat), calib_exc=float(exc), per_class=per,
               gates=dict(C1=bool(sat >= 0.80), C2=bool(exc >= 0.80),
                          C3=bool(min(per.values()) >= 0.55)))
    json.dump(res, open(os.path.join(OUT, "calib_result.json"), "w"), indent=1)
    open(contact, "w").write("contacted once\n")
    print(f"L4 calibration: sat {sat:.4f} exc {exc:.4f}")
    print("per class:", {k: round(v, 4) for k, v in per.items()})
    print("gates:", res["gates"])
    return 0


def cmd_lopo():
    fz = json.load(open(os.path.join(OUT, "L4_frozen.json")))
    lr, lb = load_tables()
    lrd = lr[lr.split == "discovery"].reset_index(drop=True)
    lbd = lb[lb.psplit == "discovery"].reset_index(drop=True)
    cands = candidates(lrd)
    out = {}
    for k in ("S1", "S2", "S3", "S4", "S5"):
        keep = lbd.kind.values != k
        lbk = lbd[keep].reset_index(drop=True)
        Sr, Sb = l3_mask(lrd), l3_mask(lbk)
        chosen = []
        for step in range(4):
            base_exc = 1 - Sb.mean()
            best = None
            for c in cands:
                mr = pred_mask(lrd, *c)
                sat = (Sr & mr).mean()
                if sat < 0.81:
                    continue
                mb = pred_mask(lbk, *c)
                exc = 1 - (Sb & mb).mean()
                if best is None or exc > best[0]:
                    best = (exc, sat, c)
            if best is None or best[0] - base_exc < 0.005:
                break
            chosen.append(best[2])
            Sr &= pred_mask(lrd, *best[2])
            Sb &= pred_mask(lbk, *best[2])
        held = lbd[lbd.kind.values == k].reset_index(drop=True)
        m = l3_mask(held)
        for c in chosen:
            m &= pred_mask(held, *c)
        out[k] = dict(excl_on_held=float(1 - m.mean()),
                      n_new=len(chosen), predicates=chosen)
        print(f"LOPO {k}: excl on held class {out[k]['excl_on_held']:.4f} "
              f"({len(chosen)} new preds)", flush=True)
    json.dump(out, open(os.path.join(OUT, "lopo_result.json"), "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "calib", "lopo"])
    a = ap.parse_args()
    return {"search": cmd_search, "calib": cmd_calib, "lopo": cmd_lopo}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
