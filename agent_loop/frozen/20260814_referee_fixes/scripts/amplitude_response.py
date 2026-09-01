#!/usr/bin/env python3
"""Graded-amplitude corruption response of the rule sets (referee fix).

Referee: "the corruption suite only tests gross damage". Here we replace the
fixed-amplitude make_negatives.perturb classes with GRADED amplitudes:

  iso_expansion   linear strain {2,5,10,20,30}%   (lattice * (1+e))
  uni_compression linear strain {2,5,10,20,30}%   (random axis, 1-e)
  gauss_disp      sigma {0.05,0.1,0.2,0.4,0.8} A  (per-site Cartesian noise)

Population and feature pipeline are identical to src/validity_rulesets.py:
same provenance sample (n=900, random_state=5), first N parents passing
len(st)<=50 and guess_oxi; apply_rules.features + aug() symmetry/bond-valence
features; same L1/L1'/L2/L3/L4 sets_of logic at published thresholds.
Deterministic per-parent seeding via phys_law.seed_of.
"""
from __future__ import annotations
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SRC = "<repo>/src"
sys.path.insert(0, SRC)

F = os.environ.get("NEWPAULING_FEATURES",
                   "$PRIS_FEATURES/")
OUT = "<repo>/outputs/20260814_referee_fixes"

STRAINS = (0.02, 0.05, 0.10, 0.20, 0.30)
SIGMAS = (0.05, 0.10, 0.20, 0.40, 0.80)
CORRUPTIONS = (("iso_expansion", STRAINS),
               ("uni_compression", STRAINS),
               ("gauss_disp", SIGMAS))
RULES = ("L1", "L1'", "L2", "L3", "L4")


# ---- rule sets at published thresholds (verbatim from validity_rulesets) ---
def _get(f, k):
    v = f.get(k, np.nan)
    return v if v is not None and np.isfinite(v) else np.nan


def _le(v, th):                       # missing counts as satisfying
    return True if not np.isfinite(v) else bool(v <= th)


def _ge(v, th):
    return True if not np.isfinite(v) else bool(v >= th)


def sets_of(f):
    bl, blm = _get(f, "bl_min"), _get(f, "bl_mean")
    cn, mz = _get(f, "cn_an_mean"), _get(f, "madz_range")
    mx, lk = _get(f, "mad_max"), _get(f, "frac_like_bonds")
    fi = _get(f, "fi")

    d1_735, d1_804 = _ge(bl, 0.735), _ge(bl, 0.804)
    d2 = True if (not np.isfinite(fi) or fi <= 0.50) else _le(bl, 1.05)
    d3 = True if (not np.isfinite(cn) or cn > 3.333) else _le(blm, 1.081)
    d4, d5 = _le(mz, 31.45), _le(mx, 15.17)
    d6 = True if (not np.isfinite(fi) or fi <= 0.55) else _le(lk, 1e-4)
    d7 = _le(_get(f, "wyckoff_econ"), 2.0 / 3.0)
    d8 = _le(_get(f, "bv_rel_mean"), 0.714)
    l3 = d1_804 and d3 and d4 and d5 and d6
    return {"L1": d1_735,
            "L1'": d1_735 and d2,
            "L2": d1_804 and d3 and d4 and d5,
            "L3": l3,
            "L4": l3 and d7 and d8}


def min_pair_dist(st):
    try:
        dm = st.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        return min(float(dm.min()), min(st.lattice.abc))
    except Exception:
        return np.nan


# ---- graded perturbations --------------------------------------------------
def graded_perturb(st, ctype, amp, rng):
    s = st.copy()
    if ctype == "iso_expansion":
        s.lattice = type(s.lattice)(s.lattice.matrix * (1.0 + amp))
    elif ctype == "uni_compression":
        f = np.eye(3)
        ax = rng.integers(0, 3)
        f[ax, ax] = 1.0 - amp
        s.lattice = type(s.lattice)(np.dot(f, s.lattice.matrix))
    elif ctype == "gauss_disp":
        for i in range(len(s)):
            s.translate_sites(i, rng.normal(0.0, amp, 3), frac_coords=False)
    else:
        raise ValueError(ctype)
    return s


def one_parent(rec):
    from pymatgen.core import Structure
    from discriminate import guess_oxi, read_blob_cif
    from phys_law import seed_of
    import apply_rules as AR
    from elec_feat import elec_feats
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    def aug(st_, val_):
        out_ = {}
        try:
            ds = SpacegroupAnalyzer(st_, symprec=0.01).get_symmetry_dataset()
            out_["wyckoff_econ"] = len(set(ds.equivalent_atoms)) / len(st_)
        except Exception:
            pass
        try:
            e = elec_feats(st_, val_)
            if e and "bv_rel_mean" in e:
                out_["bv_rel_mean"] = e["bv_rel_mean"]
        except Exception:
            pass
        return out_

    sid, off, ln = rec
    out = []
    try:
        st = Structure.from_str(read_blob_cif(int(off), int(ln)), fmt="cif")
        val, ok = guess_oxi(st)
        if not ok:
            return out
        f, _ = AR.features(st)
        if f is None:
            return out
        f.update(aug(st, val))
        row = dict(sid=sid, ctype="real", amp=0.0, md=min_pair_dist(st))
        row.update(sets_of(f))
        out.append(row)
        for ci, (ctype, amps) in enumerate(CORRUPTIONS):
            for ai, amp in enumerate(amps):
                # deterministic per (parent, corruption, amplitude)
                rng = np.random.default_rng(
                    [seed_of(sid), ci, ai])
                p = graded_perturb(st, ctype, amp, rng)
                pf, _ = AR.features(p)
                if pf is None:
                    continue
                pf.update(aug(p, val))       # geometric only: valences unchanged
                r = dict(sid=sid, ctype=ctype, amp=amp, md=min_pair_dist(p))
                r.update(sets_of(pf))
                out.append(r)
    except Exception:
        pass
    return out


def parents(n_parents):
    from pymatgen.core import Structure
    from discriminate import guess_oxi, read_blob_cif
    prov = pd.read_parquet(F + "provenance.parquet",
                           columns=["source_id", "in_analysis_set",
                                    "blob_offset", "blob_length", "n_elements"])
    d = prov[prov.in_analysis_set & (prov.n_elements >= 2)].sample(
        n=900, random_state=5)
    keep = []
    for t in d.itertuples():
        try:
            st = Structure.from_str(
                read_blob_cif(int(t.blob_offset), int(t.blob_length)), fmt="cif")
            if len(st) > 50:
                continue
            _, ok = guess_oxi(st)
            if not ok:
                continue
            keep.append((t.source_id, int(t.blob_offset), int(t.blob_length)))
        except Exception:
            continue
        if len(keep) >= n_parents:
            break
    return keep


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-parents", type=int, default=440)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--raw-only", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    ps = parents(a.n_parents)
    print(f"{len(ps)} parent structures  ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    with Pool(a.workers) as pool:
        chunks = pool.map(one_parent, ps, chunksize=2)
    r = pd.DataFrame([x for c in chunks for x in c])
    print(f"feature pass: {time.time()-t0:.0f}s, {len(r):,} rows", flush=True)
    r.to_csv(os.path.join(OUT, "amplitude_response_raw.csv"), index=False)

    real = r[r.ctype == "real"]
    print(f"parents with features: {real.sid.nunique()}", flush=True)

    rows = []
    for ctype, amps in CORRUPTIONS:
        for amp in amps:
            g = r[(r.ctype == ctype) & (np.isclose(r.amp, amp))]
            if not len(g):
                continue
            d = dict(corruption=ctype, amplitude=amp, n=len(g))
            for col in RULES:
                d[f"excl_{col}"] = float(1 - g[col].mean())
            d["excl_md_0.5"] = float(1 - (g.md > 0.5).mean())
            d["excl_md_0.7"] = float(1 - (g.md > 0.7).mean())
            rows.append(d)
    # reference row: false-positive rate on unperturbed parents
    d = dict(corruption="real", amplitude=0.0, n=len(real))
    for col in RULES:
        d[f"excl_{col}"] = float(1 - real[col].mean())
    d["excl_md_0.5"] = float(1 - (real.md > 0.5).mean())
    d["excl_md_0.7"] = float(1 - (real.md > 0.7).mean())
    rows.append(d)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "amplitude_response.csv"), index=False)
    print(out.round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
