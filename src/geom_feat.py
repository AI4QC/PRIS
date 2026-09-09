#!/usr/bin/env python3
"""Geometric coordination features -- the three quantities a multi-agent survey selected
and for which exclusion power has actually been measured.

Out of more than 40 candidates, the survey (five feature families in parallel plus an
adversarial check) found these three to satisfy all of "chemically meaningful + computable +
measurably effective":

  aa_min  ligand-ligand contact ratio = min over ligand pairs of d(L_i,L_j)/(r_Li+r_Lj)
          **This is the geometric kernel of Pauling's first rule** -- the original argument
          behind the radius-ratio criterion is exactly "if the cation is too small the
          ligands touch each other and the polyhedron is unstable". Pauling wrote it as a
          table of r_c/r_a intervals (satisfied by only 66% according to George 2020); this
          computes directly whether the ligands actually touch.
          Measured in the survey: catches S1 (ligands squeezed together along the
          compression axis) and S3 (displacement driving them into each other).

  phi     convex-hull packing fraction of the polyhedron
          = Vol(ligand convex hull) / ((4/3)pi R^3), R = mean cation-ligand distance.
          A measure of how far the coordination polyhedron has collapsed. Measured in the
          survey: real 0.293 -> S2 0.123 / S3 0.096.
          **S2, cation transposition, is the hardest of the four classes to exclude**, and
          phi is one of the few geometric quantities effective against it.

  mef     Hoppe MEFIR effective-ionic-radius mismatch (signed)
          MEFIR = sum_i w_i (d_i - r_anion_i) / sum_i w_i, with ECoN weights for w;
          mismatch = (MEFIR - r_shannon(cation)) / r_shannon(cation).
          The survey measured it catching all four classes, S2 most strongly (-0.47 against
          -0.06 for real) -- because after transposition a large cation is forced into a
          small site and MEFIR departs noticeably from the tabulated radius.

The survey also produced three **negative results**, which are accordingly not implemented:
  - rho_aniso (directional anisotropy of bond-length compression): mean exclusion power only 0.080
  - Ward Voronoi packing fraction: **scale-invariant**, so its exclusion power against S4
    uniform expansion is identically 0
  - Lewis acid-base strength matching: caught nothing under its implementation
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
        # anion ligands only; the ligand Cartesian coordinates must use the **image**
        # positions, otherwise anything crossing a periodic boundary comes out wrong
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

        # --- aa_min: the closest pair of ligands, distance / radius sum. Below 1 means the
        # ligand shells interpenetrate
        best = np.inf
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                rs = rad[a] + rad[b]
                if rs > 0:
                    best = min(best, float(np.linalg.norm(pos[a] - pos[b]) / rs))
        if np.isfinite(best):
            aa.append(best)

        # --- phi: ligand convex-hull volume / equivalent sphere volume. Drops sharply when
        # the polyhedron collapses
        if len(pos) >= 4:
            try:
                from scipy.spatial import ConvexHull
                hv = float(ConvexHull(pos - st[i].coords).volume)
                R = float(dist.mean())
                if R > 0:
                    phis.append(hv / (4.0 / 3.0 * np.pi * R ** 3))
            except Exception:
                pass

        # --- ecc / angvar: **angular** distortion. Every existing feature is a length
        # (bl_rsd_max is the relative standard deviation of bond length); none constrains a
        # bond angle.
        # ecc  = |sum u_i| / n, with u the unit vector from the cation to each ligand.
        #        Near 0 for a centrosymmetric polyhedron; grows as the cation moves off the
        #        polyhedron centre.
        # angvar = variance of the ligand-cation-ligand angles (deg^2), a simplification of
        #        the Robinson bond-angle variance.
        # The two are a cheap substitute for the ChemEnv continuous symmetry measure (CSM) --
        # the survey measured CSM at a real median of 1.75 -> S3 displacement 6.56 / S1
        # compression 2.99 / S2 transposition 2.83, but CSM has to register against 54 ideal
        # polyhedra, which is too expensive.
        u = (pos - st[i].coords) / dist[:, None]
        eccs.append(float(np.linalg.norm(u.sum(axis=0)) / len(u)))
        if len(u) >= 3:
            cosang = np.clip(u @ u.T, -1, 1)
            iu = np.triu_indices(len(u), 1)
            angs = np.degrees(np.arccos(cosang[iu]))
            angvars.append(float(np.var(angs)))

        # --- mef: Hoppe MEFIR relative mismatch (signed)
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
        rng = np.random.default_rng(seed_of(r["sid"]))   # same seed as phys_law/elec_feat
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
