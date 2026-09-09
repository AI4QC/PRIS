#!/usr/bin/env python3
"""Physically meaningful plausibility features -- to replace the purely empirical
`vol_per_atom` in the law set.

# Why it had to go

Among the three laws recommended in the previous version, "volume per atom <= 50.8 A^3"
carried a large share of the exclusion power, but it is a purely calibrated threshold: no
chemical meaning, a threshold that drifts with the composition of the database, and no way to
generalise to a new system.

# What replaces it

Everything here is built on the Shannon coordination-dependent radius r(element, oxidation
state, CN):

  sh_pack     = sum(4/3 pi r^3) / V_cell         packing fraction -- ions as hard spheres,
                                                 which may not interpenetrate
  bl_min      = min over cation-anion bonds of d/(r_c+r_a)
                                                 shortest bond relative to the radius sum --
                                                 **Born repulsion**
  bl_cat_max  = max over cations of (that cation's own shortest d/(r_c+r_a))
                                                 the most "dangling" cation
  bl_rsd_max  = max over polyhedra of the relative standard deviation of bond length
                                                 polyhedron distortion

`bl_min` matters particularly: **Pauling's electrostatic framework contains no short-range
repulsion term**, so in principle it cannot argue that a compressed structure is implausible
(measured, the Madelung energy of S1 uniaxial compression is actually lower; see the results
summary, 5.2b).
d/(r_c+r_a) writes Born repulsion into the criterion explicitly -- and it is the computable
form of the replacement rule Hawthorne 2026 proposes, "use the **sum** of the radii rather
than their **ratio**".

# Reproducibility

make_negatives.py originally seeded from `abs(hash(sid))`. Python's string hash is randomised
by PYTHONHASHSEED and **differs across processes** -- which means the earlier
negatives.parquet cannot be rebuilt, and merging newly computed features on sid would match
them to a different structure with "the same parent and the same class but a different random
realisation". Everything here uses crc32 instead, for deterministic seeding.
"""
from __future__ import annotations
import os
import argparse
import sys
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminate import criteria, guess_oxi, read_blob_cif  # noqa: E402
from phys_feat import R as RFALL  # noqa: E402

F = os.environ.get("PRIS_FEATURES", "features/")
_ROMAN = {2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
          9: "IX", 10: "X", 11: "XI", 12: "XII"}
_CACHE = {}


def seed_of(sid: str) -> int:
    """A deterministic seed. Not hash() -- that is randomised by PYTHONHASHSEED and is not
    reproducible across processes."""
    return zlib.crc32(sid.encode()) & 0x7FFFFFFF


def shannon(sym, ox, cn):
    """Shannon radius, looked up by (element, oxidation state, coordination number); falls
    back to a neighbouring CN, then to a representative value."""
    # The sign has to be part of the key: the fallback radius depends on the sign of the
    # original ox, while int(round(ox)) folds fractional charges such as +/-0.5 to 0. If the
    # two share a cache entry, rho comes to depend on the processing order within the process.
    key = (sym, int(round(ox)), int(round(cn)), ox > 0)
    if key in _CACHE:
        return _CACHE[key]
    r = None
    try:
        from pymatgen.core import Species
        sp = Species(sym, int(round(ox)))
        c0 = int(round(cn))
        for c in (c0, c0 - 1, c0 + 1, 6):
            try:
                r = float(sp.get_shannon_radius(_ROMAN.get(c, "VI")))
                if r:
                    break
            except Exception:
                continue
    except Exception:
        pass
    if not r:
        r = RFALL.get(sym, 1.0 if ox > 0 else 1.4)
    _CACHE[key] = r
    return r


def phys_feats(st, val, nn=None):
    """Shannon-radius physical features. Pass nn to reuse a single CrystalNN computation."""
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
    vol = 0.0
    for i in range(len(st)):
        vol += 4 / 3 * np.pi * shannon(st[i].specie.symbol, val[i], max(cn[i], 1)) ** 3

    # Fraction of bonds between like-charge ions. In a real ionic crystal a cation should be
    # surrounded by anions and vice versa; this quantity is directly sensitive to "charge on
    # the wrong site" (S5, cation-anion swap), and it is T1 and 100% computable, unlike GII,
    # which is limited by the coverage of the bond-valence parameter table (only 85%).
    like = tot_b = 0
    worst_op = 1.0
    for i in range(len(st)):
        if not nn[i] or abs(val[i]) < 1e-9:
            continue
        opp = 0
        for nb in nn[i]:
            j = nb["site_index"]
            tot_b += 1
            if val[i] * val[j] > 0:
                like += 1
            elif val[i] * val[j] < 0:
                opp += 1
        worst_op = min(worst_op, opp / len(nn[i]))
    out_like = like / tot_b if tot_b else np.nan

    ratios_all, cat_short, rsd = [], [], []
    for i in range(len(st)):
        if val[i] <= 0:
            continue
        rt, ds = [], []
        ri = shannon(st[i].specie.symbol, val[i], max(cn[i], 1))
        for nb in nn[i]:
            j = nb["site_index"]
            if val[j] >= 0:
                continue
            d = st[i].distance(st[j], jimage=nb.get("image"))
            rj = shannon(st[j].specie.symbol, val[j], max(cn[j], 1))
            rs = ri + rj
            if rs <= 0:
                continue
            rt.append(d / rs)
            ds.append(d)
        if not rt:
            continue
        ratios_all.extend(rt)
        cat_short.append(min(rt))       # how close that cation's nearest anion is
        if len(ds) > 1:
            rsd.append(float(np.std(ds) / np.mean(ds)))
    if not ratios_all:
        return None
    return {
        "frac_like_bonds": float(out_like),   # like-charge bond fraction; should be near 0
                                              # in a real ionic crystal
        "min_opp_frac": float(worst_op),      # the most out-of-place ion: its fraction of
                                              # opposite-charge neighbours
        "sh_pack": float(vol / st.volume),
        "bl_min": float(min(ratios_all)),        # globally shortest bond -- Born repulsion
        "bl_cat_max": float(max(cat_short)),     # the most dangling cation
        "bl_mean": float(np.mean(ratios_all)),
        "bl_rsd_max": float(max(rsd)) if rsd else 0.0,
    }


# ---------------------------------------------------------------- driver

def _real(r):
    from pymatgen.core import Structure
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > 150:
            return None
        val, ok = guess_oxi(st)
        if not ok:
            return None
        f = phys_feats(st, val)
        if f is None:
            return None
        f["source_id"] = r["sid"]
        return f
    except Exception:
        return None


def _bad(r):
    """Rebuild the 4 perturbation classes (deterministic seed) and compute criteria and the
    physical features in the same pass, so the two share an origin."""
    from pymatgen.core import Structure
    from make_negatives import perturb, swapped_val
    out = []
    try:
        st = Structure.from_str(read_blob_cif(r["off"], r["ln"]), fmt="cif")
        if len(st) > 80:
            return out
        val, ok = guess_oxi(st)
        if not ok:
            return out
        rng = np.random.default_rng(seed_of(r["sid"]))
        for kind in ("S1", "S2", "S3", "S4", "S5"):
            p = perturb(st, kind, rng, val)
            if p is None:
                continue
            pv = swapped_val(p, val)
            try:
                c = criteria(p, pv)
            except Exception:
                continue
            if c is None:
                continue
            f = phys_feats(p, pv)
            if f is None:
                continue
            c.update(f)
            c.update(sid=r["sid"] + "_" + kind, kind=kind, parent=r["sid"])
            out.append(c)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "bad"])
    ap.add_argument("--n", type=int, default=6000,
                    help="number of parent structures in bad mode")
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()

    prov = pd.read_parquet(F + "provenance.parquet",
                           columns=["source_id", "in_analysis_set", "blob_offset",
                                    "blob_length", "n_elements"])
    if a.mode == "real":
        d = prov[prov.n_elements >= 2]
        fn, outf = _real, F + "phys_real.parquet"
    else:
        # exactly the same parent sample as make_negatives.py (random_state=0), for comparison
        d = prov[prov.in_analysis_set & (prov.n_elements >= 2)].sample(
            n=min(a.n, len(prov)), random_state=0)
        fn, outf = _bad, F + "phys_bad.parquet"
    recs = [{"sid": t.source_id, "off": int(t.blob_offset), "ln": int(t.blob_length)}
            for t in d.itertuples()]
    print(f"{a.mode}: {len(recs):,} entries", flush=True)

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
    print(f"wrote {outf}, {len(rows):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
