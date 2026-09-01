#!/usr/bin/env python3
"""Referee fixes 2026-08-14: (1) discovery-only floor sensitivity of the frozen
L4 greedy; (2) D7/D8 isolation diagnostics. Read-only w.r.t. repo; writes only
under outputs/20260814_referee_fixes/."""
import json, os, sys
import numpy as np

sys.path.insert(0, "<repo>/src")
from l4_search import load_tables, l3_mask, pred_mask, candidates

OUT = "<repo>/outputs/20260814_referee_fixes"
KINDS = ("S1", "S2", "S3", "S4", "S5")

D7 = ("wyckoff_econ_001", "le", 0.6666666666666666, None)
D8 = ("bv_rel_mean", "le", 0.7143040821865658, None)

lr, lb = load_tables()
lrd = lr[lr.split == "discovery"].reset_index(drop=True)
lbd = lb[lb.psplit == "discovery"].reset_index(drop=True)
lrc = lr[lr.split == "calibration"].reset_index(drop=True)
lbc = lb[lb.psplit == "calibration"].reset_index(drop=True)

def per_class(mask_bad, lbx):
    return {k: float(1 - mask_bad[lbx.kind.values == k].mean()) for k in KINDS}

# ---------- 1. floor sensitivity (discovery only, frozen greedy rule) ----------
cands = candidates(lrd)
print(f"{len(cands)} candidate predicates")
floors = [0.75, 0.78, 0.81, 0.84, 0.87]
floor_res = {}
for floor in floors:
    Sr, Sb = l3_mask(lrd), l3_mask(lbd)
    chosen, trace = [], []
    for step in range(4):
        base_exc = 1 - Sb.mean()
        best = None
        for c in cands:
            mr = pred_mask(lrd, *c)
            sat = (Sr & mr).mean()
            if sat < floor:
                continue
            mb = pred_mask(lbd, *c)
            exc = 1 - (Sb & mb).mean()
            if best is None or exc > best[0]:
                best = (exc, sat, c)
        if best is None or best[0] - base_exc < 0.005:
            stop = ("no candidate above floor" if best is None
                    else f"best gain {best[0]-base_exc:+.4f} < 0.005")
            trace.append({"step": step + 1, "stop": stop})
            break
        exc, sat, c = best
        chosen.append(list(c))
        trace.append({"step": step + 1, "predicate": list(c),
                      "disc_sat": float(sat), "disc_exc": float(exc)})
        Sr &= pred_mask(lrd, *c)
        Sb &= pred_mask(lbd, *c)
        print(f"floor {floor}: step {step+1}: {c} -> sat {sat:.4f} exc {exc:.4f}",
              flush=True)
    names = [c[0] for c in chosen]
    floor_res[f"{floor:.2f}"] = dict(
        floor=floor,
        predicates=chosen,
        n_predicates=len(chosen),
        contains_D7=list(D7) in chosen,
        contains_D8=list(D8) in chosen,
        disc_sat=float(Sr.mean()),
        disc_exc=float(1 - Sb.mean()),
        per_class_exclusion_discovery=per_class(Sb, lbd),
        trace=trace)

l3_ref = dict(disc_sat=float(l3_mask(lrd).mean()),
              disc_exc=float(1 - l3_mask(lbd).mean()))

out1 = dict(
    note=("Discovery-only rerun of the frozen greedy of src/l4_search.py "
          "(same candidate pool from discovery real rows, same gain rule: "
          "max 4 steps, min exclusion gain 0.005) at varying satisfaction "
          "floors. The published L4 used floor 0.81. Frozen predicates: "
          "D7 = wyckoff_econ_001 <= 2/3, D8 = bv_rel_mean <= 0.714304."),
    frozen_D7=list(D7), frozen_D8=list(D8),
    l3_baseline_discovery=l3_ref,
    n_candidates=len(cands),
    n_real_discovery=int(len(lrd)), n_bad_discovery=int(len(lbd)),
    floors=floor_res)
os.makedirs(OUT, exist_ok=True)
json.dump(out1, open(os.path.join(OUT, "floor_sensitivity.json"), "w"), indent=1)
print("wrote floor_sensitivity.json")

# ---------- 2. D7/D8 isolation ----------
# (a) provenance of D8 calibration: quantiles of bv_rel_mean over REAL
# discovery rows; median GII over real discovery rows.
bv = lrd["bv_rel_mean"].values.astype(float)
bvf = bv[np.isfinite(bv)]
gii = lrd["gii"].values.astype(float)
giif = gii[np.isfinite(gii)]
qs = {"median": 0.50, "q75": 0.75, "q90": 0.90, "q95": 0.95}
bv_q = {k: float(np.quantile(bvf, q)) for k, q in qs.items()}
# which QGRID quantile matches the frozen D8 threshold
from l4_search import QGRID
grid_q = {f"q{q}": float(np.quantile(bvf, q)) for q in QGRID}
match = min(grid_q.items(), key=lambda kv: abs(kv[1] - D8[2]))
prov = dict(
    feature="bv_rel_mean",
    frozen_threshold=D8[2],
    n_real_discovery=int(len(lrd)),
    n_finite_bv_rel_mean=int(bvf.size),
    quantiles_over_real_discovery=bv_q,
    closest_qgrid_quantile=dict(quantile=match[0], value=match[1],
                                abs_diff=float(abs(match[1] - D8[2]))),
    gii_median_real_discovery=float(np.median(giif)),
    n_finite_gii=int(giif.size))

def iso(lrx, lbx, pred):
    m_r = l3_mask(lrx) & pred_mask(lrx, *pred)
    m_b = l3_mask(lbx) & pred_mask(lbx, *pred)
    return dict(sat=float(m_r.mean()), exc=float(1 - m_b.mean()),
                per_class_exclusion=per_class(m_b, lbx))

calib = dict(
    label=("DESCRIPTIVE: post-hoc decomposition of the already-published L4 "
           "calibration evaluation; predicates were frozen on discovery and "
           "are applied singly on top of L3. Not a new confirmatory gate."),
    L3_only=dict(sat=float(l3_mask(lrc).mean()),
                 exc=float(1 - l3_mask(lbc).mean()),
                 per_class_exclusion=per_class(l3_mask(lbc), lbc)),
    L3_plus_D7_only=iso(lrc, lbc, D7),
    L3_plus_D8_only=iso(lrc, lbc, D8))

disc = dict(
    L3_only=dict(sat=l3_ref["disc_sat"], exc=l3_ref["disc_exc"],
                 per_class_exclusion=per_class(l3_mask(lbd), lbd)),
    L3_plus_D7_only=iso(lrd, lbd, D7),
    L3_plus_D8_only=iso(lrd, lbd, D8))

out2 = dict(
    frozen_D7=list(D7), frozen_D8=list(D8),
    a_D8_calibration_provenance=prov,
    b_calibration_descriptive=calib,
    c_discovery=disc)
json.dump(out2, open(os.path.join(OUT, "d7_d8_isolation.json"), "w"), indent=1)
print("wrote d7_d8_isolation.json")
