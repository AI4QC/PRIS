# -*- coding: utf-8 -*-
"""Reproduce the five numbers of George 2020 (MPU-1 / PREREG section 6, gate G-A).

George, Waroquiers, Di Stefano, Petretto, Rignanese, Hautier,
*The Limited Predictive Power of the Pauling Rules*, Angew. Chem. Int. Ed. 2020, 59, 7569.
Local PDF: the published PDF (not redistributed here). Every convention below was checked
against the original sentence by sentence.

============================================== differences in convention (stated explicitly)
| dimension | George 2020 | this reproduction |
|---|---|---|
| domain | the roughly 5,000 oxides in ICSD n Materials Project | `provenance.oxide_strict` = **23,728** entries |
| source | ICSD (re-relaxed and re-symmetrised through MP) | **ICSD + COD**, **no MP**, using the **experimentally reported cell** with no DFT relaxation |
| order | MP entries are already ordered | **ordered structures only** (disordered entries were dropped when the store was built, PREREG section 8.1) |
| anion | oxides | **single anion O** (the `oxide_strict` criterion) |
| oxidation state | MP's `oxi_state` (BVAnalyzer-derived) | **two levels, `cif` (native ICSD) and `guess` (pure compositional enumeration); BVAnalyzer excluded wholesale** (PREREG section 5) |
| neighbours | ChemEnv alone | ChemEnv / CrystalNN / BrunnerNN, **three algorithms** (required by the G6 hard gate) |

**Note that `oxide_strict` is not a subset of `in_analysis_set`**: the 3,895 entries in the
difference all contain P (`in_analysis_set` counts P as an anion candidate, so phosphates are
excluded with `n_anion_kinds==2`).
George's domain **includes** phosphates (the original uses InPO4 as its main example for the
second rule), so this script recomputes all 23,728 entries under `oxide_strict` and **does
not reuse `site/pair/struct.parquet`** (those three tables cover only `in_analysis_set`, and
`pair` has only the ChemEnv route and no cation-anion bond-strength table, so the second
rule's Sum s cannot be computed).

============================== operationalising the five rules (aligned to the original, rule by rule)
* **Rule 1 (radius ratio)** granularity=`site`/`orbit`. r_cation/r_anion use the **Pauling
  univalent radii** (`pauling_radii.py`; Shannon radii depend on CN and are circular, which
  G7 explicitly forbids).
  The predicted CN comes from the hard-sphere critical ratios, and a hit is predicted CN ==
  observed CN (strict equality).
  George: 66% of the tested local environments.
* **Rule 2 (electrostatic valence)** granularity=`site(anion)`/`orbit`. For each O site
  compute Sum s = Sum_cations (z_c / CN_c); the criterion is |Sum s - 2| <= 0.01.
  George: about 20% of all oxygen atoms.
* **Rule 3 (connection type)** granularity=`pair`. The corner/edge/face fractions over all
  connected polyhedron pairs.
  George: 62.5 / 27.2 / 10.3; **73.3 / 25.0 / 1.6 at CN <= 8**.
* **Rule 4 (adjacent polyhedra)** granularity=`structure`. The original defines a violation as
  "structures in which the polyhedra of cations with the highest valence and smallest
  coordination number are connected". That is: V = the highest oxidation state among the
  structure's cation sites, C = the smallest CN, A = {sites | ox==V and cn==C}; a connected
  pair within A is a violation.
  It applies only to structures **containing at least 2 distinct cation species** (the
  original: "In a crystal containing different cations"), and single-cation structures count
  as satisfied -- which is necessary to reproduce the positive example the original gives
  (rutile SnO2 is listed as satisfying all four).
  **Both an "oxidation-state version" (ox==V only) and a "CN version" (cn==C only) are
  given**: George shows only the latter holds.
  George: about 40% violate.
* **Rule 5 (parsimony)** granularity=`structure`. Each cation species (element, oxidation
  state) occupies only one local environment within a structure. The caption of the original
  Fig. 5b states "only coordination numbers are considered", so the main criterion uses
  **CN**; ChemEnv additionally provides a `ce_symbol` version as a sensitivity check.
  George: about 70.3%.
* **2-5 jointly** granularity=`structure`, conjunction. The structure-level boolean for rule 2
  is "every O site in the structure satisfies it", and for rule 3 it is "no face-sharing pair"
  (Pauling's own "particularly of shared faces").
  George: 13% (20% at CN <= 8).

============================================================================== usage
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    python src/reproduce_george.py --stage compute --limit 300 --workers 20 --force   # smoke test
    nohup python src/reproduce_george.py --stage compute --workers 20 --force \
          > $FEAT/reproduce_george.log 2>&1 &                                          # full run
    python src/reproduce_george.py --stage table                                       # emit Table S1
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
import build_features as BF          # noqa: E402  reuses read_cif / assign_oxi / symmetry_info
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

PER_STRUCT_TIMEOUT = 300   # seconds, as in build_features (section 6.5-3 measured ChemEnv
                           # routinely exceeding 60 s on cells above 200 atoms)
EPS_RULE2 = 0.01           # from George: an absolute deviation of 0.01 is allowed


# =================================== oxidation states (a patch on top of build_features)
def assign_oxi_fixed(struct, source):
    """A corrected version of `build_features.assign_oxi`.

    **A measured pitfall**: 3,658 of 38,307 ICSD entries (9.6%) have
    `_atom_type_oxidation_number` set to 0 everywhere (exp001 ZnO, for instance, is recorded
    as Zn0/O0). The original implementation judged `blob_ox_present = all(x is not None)`
    True and returned `ox_source='cif'`, whereupon `is_cat = [v>0]` left **no cation site at
    all** and the structure was silently emptied. A sanity check is added here: `cif` is
    trusted only when the anion has ox<0 and at least one site has ox>0, and otherwise it is
    downgraded to `guess`. This correction changes the coverage distribution of `ox_source`,
    and is reported separately.
    """
    v = [getattr(s.specie, "oxi_state", None) for s in struct]
    blob_ox = all(x is not None for x in v)
    meta = {"blob_ox_present": bool(blob_ox), "n_guess_sol": 0, "guess_unique": False,
            "cif_all_zero": False}
    if blob_ox and source == "icsd":
        vv = [float(x) for x in v]
        if any(x > 0 for x in vv) and any(x < 0 for x in vv):
            return vv, "cif", meta
        meta["cif_all_zero"] = True          # all zero / all one sign -> unusable, downgrade
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


# ============================ polyhedron connection enumeration (shared by all three algorithms)
def enumerate_connections(cat_idx, ligands):
    """Given the ligand set of each cation site, enumerate the polyhedron connection pairs
    within **each primitive cell**.

    `ligands[i]` = set of (anion site index j, periodic image (a,b,c)), the image being
    relative to i sitting at (0,0,0).
    The number of ligands shared by polyhedra (i @ 0) and (j @ T) is |L_i n (L_j + T)|.
    The de-duplication convention (matching ChemEnv `environment_subgraph`'s "one edge per
    primitive cell"):
      - i < j: enumerate all T;
      - i == j: T /= 0, and of T and -T keep only the lexicographically larger (they are two
        views of the same connection).
    Returns [(i, j, n_shared)] and not T (downstream only uses n_shared).
    **All three algorithms share this enumerator**, so G6 compares the single degree of
    freedom "neighbour definition" rather than also swapping the connectivity algorithm
    (agreement between ChemEnv's own ConnectivityFinder and this function is checked under
    --check-conn).
    """
    out = []
    # reverse index: anion site -> [(cation, image)], for finding candidate translations quickly
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
    """{1: corner, 2: edge, >=3: face} (the same convention as George 2020 and ChemEnv)"""
    return "corner" if ns == 1 else ("edge" if ns == 2 else "face")


# ================================================================ per-structure main flow
def process_one(rec):
    """rec = (source_id, source, blob_offset, blob_length). The anion is always O (the domain
    is oxide_strict)."""
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

        # deciding what is a cation: ox>0 where valences exist, otherwise fall back to
        # "element != O" (the domain already guarantees O is the only anion)
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

        # ---------------- each algorithm's ligand sets (cation-anion bonds only)
        ligands = {a: {i: set() for i in cat_idx} for a in ALGOS}
        cn_all = {a: {i: np.nan for i in cat_idx} for a in ALGOS}   # CN before the anion filter (for audit)
        ok = {a: False for a in ALGOS}
        ce_sym = [None] * n
        csm = [np.nan] * n
        err = {}

        # --- CrystalNN (section 6.3 pitfall B: x_diff_weight defaults to 3.0)
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

        # --- BrunnerNN_relative (default parameters)
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

        # --- ChemEnv (section 6.3 pitfall A: valences must be passed explicitly, or
        #     only_cations=True returns garbage)
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

        # ---------------- each algorithm's CN / anion Sum s / polyhedron pairs
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


# ================================================================ table schemas
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
        raise RuntimeError(f"no {name} shards; refusing to emit an empty table")
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
        print(f"[warn] oxide_strict = {len(sub)}, which does not match the reported 23,728; "
              f"check provenance")
    if limit:
        # smoke tests use a **random sample** (fixed seed) rather than the smallest cells:
        # picking small cells underestimates the per-structure cost by an order of magnitude
        # and ruins the extrapolation to a full run (section 6.6 requires "smoke test on a
        # small subset, measure the per-structure cost properly, then run in full")
        sub = sub.sample(n=min(limit, len(sub)), random_state=0)
    recs = list(zip(sub.source_id, sub.source, sub.blob_offset, sub.blob_length))
    shard_dir = SHARD_DIR + ("_smoke" if limit else "")
    if force and os.path.isdir(shard_dir):
        import shutil
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)
    tasks = [(k, recs[i:i + chunk], shard_dir) for k, i in enumerate(range(0, len(recs), chunk))]
    print(f"[compute] {len(recs)} structures / {len(tasks)} shards / {workers} processes", flush=True)
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
                print(f"  {c}/{len(tasks)} shards | {ndone} structures | ok {nok} | "
                      f"{el/60:.1f} min | eta {el/c*(len(tasks)-c)/60:.1f} min", flush=True)
    suf = "_smoke" if limit else ""
    for name, cols in (("site", SITE_COLS), ("anion", ANION_COLS),
                       ("pair", PAIR_COLS), ("struct", STRUCT_COLS), ("fail", FAIL_COLS)):
        p = OUT.get(name, f"{FEAT}/george_{name}.parquet").replace(".parquet", f"{suf}.parquet")
        nrow = merge_shards(name, shard_dir, p, cols)
        print(f"[merge] {name}: {nrow} rows -> {p}")
    print(f"[compute] done, {(time.time()-t0)/60:.1f} min wall clock")


# ================================================================ statistics helpers
def wilson(k, n, z=1.959963985):
    """Wilson 95% confidence interval (more reliable than Wald at extreme proportions; PREREG
    requires a CI for every rule)."""
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
    ap.add_argument("--smoke", action="store_true",
                    help="in the table stage, read the _smoke outputs")
    a = ap.parse_args()
    if a.stage == "compute":
        stage_compute(a.limit, a.workers, a.chunk, a.force)
    else:
        from george_table import stage_table
        stage_table(smoke=a.smoke)
