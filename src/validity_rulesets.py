#!/usr/bin/env python3
"""The generative-model validity benchmark -- putting **all four law sets** on the same
batch of structures the current thresholds use.

The original validity.py computed only two quantities, min pair distance and bl_min, so
main-text Fig. 3a/3b could show only the L1 floor and the L2 floor, with no L1', L2 or L3.
This computes apply_rules.py's full feature set and judges all four sets law by law, over a
population identical to the original benchmark:

  the same sample (n=900, random_state=5), the same len(st) <= 50 and guess_oxi-succeeds
  thresholds, taking the first 440 parent structures that pass (matching the published
  440 real / 1,964 perturbed).

The thresholds are the published values 0.735 / 0.804 laid down in FACTS (not 0.7353 /
0.8044).
Missing values are treated as satisfied; that is the evaluation convention used to reproduce
the paper's set statistics. The public CLI `apply_rules.judge()` instead returns
"undecidable" for unknown features, so that a missing value is never reported as plausible.
"""
from __future__ import annotations
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# external feature store: gigabytes of derived tables, not in this repository.
# Override with PRIS_FEATURES; see README "Reproducing the figures".
F = os.environ.get("PRIS_FEATURES",
                   "features/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "data")
N_PARENTS = 440
CLASSES = ("S1", "S2", "S3", "S4", "S5")


# ---- the rule sets, at the published thresholds ---------------------------
def _get(f, k):
    v = f.get(k, np.nan)
    return v if v is not None and np.isfinite(v) else np.nan


def _le(v, th):                       # missing counts as satisfying
    return True if not np.isfinite(v) else bool(v <= th)


def _ge(v, th):
    return True if not np.isfinite(v) else bool(v >= th)


def sets_of(f):
    """Verdicts for the four law sets. Returns dict[set name] -> bool."""
    bl, blm = _get(f, "bl_min"), _get(f, "bl_mean")
    cn, mz = _get(f, "cn_an_mean"), _get(f, "madz_range")
    mx, lk = _get(f, "mad_max"), _get(f, "frac_like_bonds")
    fi = _get(f, "fi")

    d1_735, d1_804 = _ge(bl, 0.735), _ge(bl, 0.804)
    # D2: fi > 0.50  =>  bl_min <= 1.05     (guard false => vacuously satisfied)
    d2 = True if (not np.isfinite(fi) or fi <= 0.50) else _le(bl, 1.05)
    # D3: cn_an_mean <= 3.333  =>  bl_mean <= 1.081
    d3 = True if (not np.isfinite(cn) or cn > 3.333) else _le(blm, 1.081)
    d4, d5 = _le(mz, 31.45), _le(mx, 15.17)
    # D6: fi > 0.55  =>  no like-charge bonds
    d6 = True if (not np.isfinite(fi) or fi <= 0.55) else _le(lk, 1e-4)

    # D7/D8 (PREREG-L4): Wyckoff economy <= 2/3; mean rel. bond-valence dev <= 0.7143
    d7 = _le(_get(f, "wyckoff_econ"), 2.0 / 3.0)
    d8 = _le(_get(f, "bv_rel_mean"), 0.7143040821865658)
    l3 = d1_804 and d3 and d4 and d5 and d6
    return {"L1": d1_735,
            "D1_804": d1_804,
            "L1'": d1_735 and d2,
            "L2": d1_804 and d3 and d4 and d5,
            "L3": l3,
            "L4": l3 and d7 and d8}


def min_pair_dist(st):
    """Shortest distance between any pair of atoms (including periodic images) -- the
    quantity generative-model validity uses."""
    try:
        dm = st.distance_matrix.copy()
        np.fill_diagonal(dm, np.inf)
        return min(float(dm.min()), min(st.lattice.abc))
    except Exception:
        return np.nan


def one_parent(rec):
    """The parent plus the five perturbation classes; compute features and judge each."""
    from pymatgen.core import Structure
    from discriminate import guess_oxi, read_blob_cif
    from make_negatives import perturb, swapped_val
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
        row = dict(sid=sid, kind="real", md=min_pair_dist(st))
        row.update(sets_of(f))
        out.append(row)
        rng = np.random.default_rng(seed_of(sid))
        for k in CLASSES:
            p = perturb(st, k, rng, val)
            if p is None:
                continue
            pf, _ = AR.features(p)
            if pf is None:
                continue
            pf.update(aug(p, swapped_val(p, val)))
            r = dict(sid=sid, kind=k, md=min_pair_dist(p))
            r.update(sets_of(pf))
            out.append(r)
    except Exception:
        pass
    return out


def parents():
    """Reproduce the original benchmark's parent set: same sample, same thresholds, the
    first 440 structures that pass."""
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
        if len(keep) >= N_PARENTS:
            break
    return keep


def main():
    ps = parents()
    print(f"{len(ps)} parent structures", flush=True)
    nproc = max(1, min(14, (os.cpu_count() or 4) - 2))
    with Pool(nproc) as pool:
        chunks = pool.map(one_parent, ps, chunksize=2)
    r = pd.DataFrame([x for c in chunks for x in c])
    r.to_csv(os.path.join(OUT, "fig6_validity_raw.csv"), index=False)

    real, bad = r[r.kind == "real"], r[r.kind != "real"]
    print(f"real {len(real):,}  perturbed {len(bad):,}\n", flush=True)

    rows = []

    def add(name, kind, used_by, mr, mb):
        d = dict(criterion=name, kind=kind, used_by=used_by,
                 real_satisfaction=float(np.mean(mr)),
                 exclusion_total=float(1 - np.mean(mb)))
        for k, col in zip(CLASSES, ["S1_uniaxial_compression", "S2_cation_cation_swap",
                                    "S3_random_displacement", "S4_isotropic_expansion",
                                    "S5_cation_anion_swap"]):
            sel = bad.kind.values == k
            d[col] = float(1 - np.mean(mb[sel])) if sel.any() else np.nan
        rows.append(d)

    for th, who in ((0.5, "CDVAE; DiffCSP; FlowMM"), (0.7, "LeMat-GenBench"),
                    (1.0, "(stricter variant)")):
        add(f"min pair distance > {th} A", "absolute", who,
            (real.md > th).values, (bad.md > th).values)
    for col, name, who in (("L1", "L1 (D1, tau=0.735)", "this work"),
                           ("D1_804", "D1 alone, tau=0.804", "this work"),
                           ("L1'", "L1' (D1+D2)", "this work"),
                           ("L2", "L2 (D1,D3-D5)", "this work"),
                           ("L3", "L3 (D1,D3-D6)", "this work"),
                           ("L4", "L4 (D1,D3-D8)", "this work")):
        add(name, "relative", who, real[col].values, bad[col].values)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "fig6_validity.csv"), index=False)
    print(out.round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
