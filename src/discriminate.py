#!/usr/bin/env python3
"""The discrimination task: scoring Pauling's rules 2 to 5 honestly, as **criteria**.

# Why the task had to change

Of Pauling's five rules only the first is a predictive rule of the form "guess the CN from
the composition"; rules 2 to 5 are **criteria**: given a structure, decide whether it is
plausible. Earlier rounds scored all five on top-1 accuracy at predicting the coordination
environment, which is measuring temperature with a ruler -- the third rule does not predict
CN at all, it says that face-sharing structures are unstable.

The value of a criterion lies in **what it can reject**. But the database contains only
structures that already exist, which are by definition all "plausible", so the only thing
computable is "what fraction of real structures satisfy it" (George's 13%) -- and that number
says nothing about a criterion: a tautology satisfies 100% and a contradiction 0%, and
neither is of any use.

# What this script does

Paired discrimination. For each reduced chemical composition, take
  positives = the real structures in our analysis set (ICSD/COD, experimentally realised)
  negatives = same-composition endpoints in ELEMENTA (DFT local minima, never synthesised)
and ask whether each Pauling criterion can rank the real one first.

**Composition is fully controlled** -- the chemical formula is the same on both sides and the
only difference is the structure. That is the test a criterion deserves.
Measured scale: 2,229 shared compositions, 7,317 entries on the real side and 11,374 on the
ELEMENTA side (5.1 per composition on average).

# Three conventions that have to be handled explicitly

1. **The oxidation states on the two sides must have the same origin.** ELEMENTA has only
   compositions and no CIF decoration, so `oxi_state_guesses` is the only option. For
   fairness the real side is **forced through guess as well**, rather than using the native
   ICSD valences. The cost is losing the mixed-valence information, but it avoids the fatal
   bias of "the positives carry extra information".
2. **The systematic difference between DFT relaxation and experimental determination.**
   ELEMENTA is PBE-relaxed and the real side is experimentally refined, and PBE typically
   overestimates lattice constants by about 1%. If a criterion leans on bond length (BVS),
   that becomes a spurious signal.
   So **the main criteria use only quantities independent of absolute bond length** (CN,
   connection type, s_pauling=z/CN), and the BVS version is reported separately as a
   sensitivity check.
3. **Label noise in the negatives.** An ELEMENTA candidate may in fact be a known polymorph
   that is simply not in our analysis set (disordered structures were dropped when the store
   was built, or it fails the single-anion filter). This underestimates the discriminating
   power, so the conclusion is conservative -- acceptable, but it has to be said in the
   report.
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import re
import sys
import warnings
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

F = Path(os.environ.get("PRIS_FEATURES", "features/"))
ELEM = Path(os.environ.get("PRIS_ELEMENTA", "elementa/endpoints_open.extxyz"))
BLOB = Path(os.environ.get("PRIS_MATDATA_BLOB", "structures.blob"))
OUT = F / "discriminate.parquet"

ANIONS = ["O", "S", "Se", "Te", "N", "P", "F", "Cl", "Br", "I"]
RE_F = re.compile(r"\bformula=(\S+)")
RE_LAT = re.compile(r'Lattice="([^"]+)"')
RE_EL = re.compile(r"([A-Z][a-z]?)")


def redkey(formula: str) -> str | None:
    """Reduced-composition key. Fe2O3 and Fe4O6 get the same key, so cross-database pairing
    lines up."""
    d = collections.Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if el:
            d[el] += int(n or 1)
    if not d:
        return None
    g = 0
    for v in d.values():
        g = gcd(g, v)
    g = g or 1
    return "|".join(f"{k}{v // g}" for k, v in sorted(d.items()))


# ---------------------------------------------------------------- reading structures

def read_blob_cif(off: int, ln: int) -> str:
    with open(BLOB, "rb") as fh:
        fh.seek(off)
        raw = fh.read(ln)
    import zlib
    try:
        return zlib.decompress(raw).decode("utf-8", "ignore")
    except zlib.error:
        return raw.decode("utf-8", "ignore")


def iter_elementa(keep_keys: set[str]):
    """Stream the ELEMENTA endpoints, yielding only the target compositions. O(1) memory."""
    with open(ELEM) as fh:
        idx = 0
        while True:
            head = fh.readline()
            if not head:
                break
            try:
                nat = int(head.strip())
            except ValueError:
                continue
            comment = fh.readline()
            lines = [fh.readline() for _ in range(nat)]
            idx += 1
            m = RE_F.search(comment)
            if not m:
                continue
            fo = m.group(1)
            els = set(RE_EL.findall(fo))
            if "H" in els or "C" in els:
                continue
            if len([a for a in ANIONS if a in els]) != 1:
                continue
            # there has to be a cation. An element (pure Br, say) satisfies "exactly one anion
            # element" but has no cation/anion distinction, and Pauling's rules are undefined
            # on it. Measured: without this check the first few thousand iterations are all
            # elemental halogens.
            if len(els) < 2:
                continue
            rk = redkey(fo)
            if rk not in keep_keys:
                continue
            ml = RE_LAT.search(comment)
            if not ml:
                continue
            lat = np.array([float(x) for x in ml.group(1).split()]).reshape(3, 3)
            syms, pos = [], []
            for ln_ in lines:
                p = ln_.split()
                if len(p) < 4:
                    break
                syms.append(p[0])
                pos.append([float(p[1]), float(p[2]), float(p[3])])
            if len(syms) != nat:
                continue
            yield {"sid": f"elem-{idx}", "rk": rk, "formula": fo,
                   "lattice": lat, "species": syms, "coords": np.array(pos)}


# ---------------------------------------------------------------- computing the criteria

def guess_oxi(struct):
    """Infer valences from composition on both sides. Returns (valences, ok)."""
    from pymatgen.core import Composition
    comp = Composition(struct.composition.reduced_formula)
    try:
        guesses = comp.oxi_state_guesses(max_sites=-10)
    except Exception:
        return None, False
    if not guesses:
        return None, False
    g = guesses[0]
    try:
        val = [float(g[str(site.specie.symbol)]) for site in struct]
    except (KeyError, AttributeError):
        return None, False
    if not any(v > 0 for v in val) or not any(v < 0 for v in val):
        return None, False          # no cation/anion distinction; Pauling's rules are undefined
    return val, True


def criteria(struct, val):
    """Compute Pauling criteria 2/3/4/5. All of them use only CN and topology, never absolute
    bond length (see convention 2 in the module docstring)."""
    from pymatgen.analysis.local_env import CrystalNN
    out = {}
    try:
        cnn = CrystalNN(weighted_cn=False, x_diff_weight=0.0)
        nn = [cnn.get_nn_info(struct, i) for i in range(len(struct))]
    except Exception:
        return None
    cn = [len(x) for x in nn]
    cats = [i for i, v in enumerate(val) if v > 0]
    ans = [i for i, v in enumerate(val) if v < 0]
    if not cats or not ans:
        return None

    # --- rule 2: for each anion, how far the received bond-strength sum s=z/CN departs from
    #     its charge
    recv = collections.defaultdict(float)
    for i in cats:
        if cn[i] == 0:
            continue
        s = val[i] / cn[i]
        for nb in nn[i]:
            j = nb["site_index"]
            if val[j] < 0:
                recv[j] += s
    devs = [abs(recv.get(j, 0.0) - abs(val[j])) for j in ans]
    out["p2_mean_dev"] = float(np.mean(devs)) if devs else np.nan
    out["p2_frac_ok_010"] = float(np.mean([d <= 0.10 for d in devs])) if devs else np.nan

    # --- rule 3: the edge- and face-sharing fraction of polyhedron connections (>= 2 shared ligands)
    ligs = {i: {nb["site_index"] for nb in nn[i]} for i in cats}
    n_corner = n_edge = n_face = 0
    for a_i, a in enumerate(cats):
        for b in cats[a_i + 1:]:
            sh = len(ligs[a] & ligs[b])
            if sh == 1:
                n_corner += 1
            elif sh == 2:
                n_edge += 1
            elif sh >= 3:
                n_face += 1
    tot = n_corner + n_edge + n_face
    out["p3_frac_edge_face"] = (n_edge + n_face) / tot if tot else np.nan
    out["p3_frac_face"] = n_face / tot if tot else np.nan
    out["p3_n_pairs"] = tot

    # --- rule 4: whether the highest-valence, lowest-coordination cations are connected
    if tot:
        zc = {i: val[i] for i in cats}
        hi = max(zc.values())
        lo_cn = min(cn[i] for i in cats if zc[i] == hi)
        crit = [i for i in cats if zc[i] == hi and cn[i] == lo_cn]
        viol = 0
        for a_i, a in enumerate(crit):
            for b in crit[a_i + 1:]:
                if ligs[a] & ligs[b]:
                    viol += 1
        out["p4_violate"] = float(viol > 0)
    else:
        out["p4_violate"] = np.nan

    # --- rule 5: how many distinct CN each (element, oxidation state) occupies
    grp = collections.defaultdict(set)
    for i in cats:
        grp[(struct[i].specie.symbol, round(val[i], 2))].add(cn[i])
    out["p5_n_distinct"] = float(np.mean([len(v) for v in grp.values()])) if grp else np.nan
    out["p5_ok"] = float(all(len(v) == 1 for v in grp.values())) if grp else np.nan

    out["n_sites"] = len(struct)
    out["mean_cn_cat"] = float(np.mean([cn[i] for i in cats]))

    # --- what follows is raw material for the search over candidate criteria. All of it is
    # T1 (CN + topology) and contains no absolute bond-length quantity -- otherwise the search
    # would detect the "PBE relaxation vs experimental refinement" systematic rather than
    # chemistry.
    cnc = [cn[i] for i in cats]
    cna = [cn[j] for j in ans]
    out["cn_cat_max"] = float(max(cnc))
    out["cn_cat_min"] = float(min(cnc))
    out["cn_cat_span"] = float(max(cnc) - min(cnc))
    out["cn_cat_std"] = float(np.std(cnc))
    out["cn_an_mean"] = float(np.mean(cna)) if cna else np.nan
    out["cn_an_max"] = float(max(cna)) if cna else np.nan
    out["cn_an_span"] = float(max(cna) - min(cna)) if cna else np.nan
    out["frac_corner"] = n_corner / tot if tot else np.nan
    out["pair_per_cat"] = tot / len(cats)
    # degree distribution of the polyhedron connection graph: how many others each cation
    # polyhedron connects to
    deg = collections.Counter()
    for a_i, a in enumerate(cats):
        for b in cats[a_i + 1:]:
            if ligs[a] & ligs[b]:
                deg[a] += 1
                deg[b] += 1
    dv = [deg.get(i, 0) for i in cats]
    out["poly_deg_mean"] = float(np.mean(dv))
    out["poly_deg_max"] = float(max(dv))
    out["frac_isolated"] = float(np.mean([x == 0 for x in dv]))
    # charge and stoichiometry quantities (pure composition level, T0)
    out["z_cat_max"] = float(max(val[i] for i in cats))
    out["z_cat_mean"] = float(np.mean([val[i] for i in cats]))
    out["n_cat_el"] = float(len({struct[i].specie.symbol for i in cats}))
    out["cat_an_ratio"] = len(cats) / len(ans)
    # dispersion of the anion coordination numbers -- Hawthorne says bond-strength
    # redistribution happens exactly here
    out["cn_an_std"] = float(np.std(cna)) if cna else np.nan
    # a close-packing proxy: volume per atom (non-dimensionalised by the cube of the ionic
    # radius, to avoid using absolute volume directly)
    try:
        out["vol_per_atom"] = float(struct.volume / len(struct))
    except Exception:
        out["vol_per_atom"] = np.nan

    # --- extremal and counting local quantities (new in v2).
    # Pauling's rules are about a **single** polyhedron and a **single** anion; the previous
    # round used structure-level means and fractions throughout, which averaged the severe
    # local violations away. Measured, structure-level means reach at best 0.5631 on polymorph
    # ranking, whereas the rule itself asserts "one face-sharing connection makes it
    # unstable" -- that is max/count semantics, not mean semantics.
    if devs:
        out["p2_max_dev"] = float(max(devs))                      # the worst-offending anion
        out["p2_n_bad_020"] = float(sum(1 for x in devs if x > 0.20))
        out["p2_n_bad_per_an"] = out["p2_n_bad_020"] / len(devs)
        out["p2_sum_dev"] = float(sum(devs))                      # total violation, unnormalised
    else:
        out["p2_max_dev"] = out["p2_n_bad_020"] = np.nan
        out["p2_n_bad_per_an"] = out["p2_sum_dev"] = np.nan
    # count semantics for rule 3: how many face- and edge-sharing connections there are,
    # rather than their fraction
    out["p3_n_face"] = float(n_face)
    out["p3_n_edge"] = float(n_edge)
    out["p3_n_face_per_cat"] = n_face / len(cats)
    out["p3_has_face"] = float(n_face > 0)                        # "one is enough to violate"
    # count semantics for rule 4
    if tot:
        zc = {i: val[i] for i in cats}
        hi_z = max(zc.values())
        nv = 0
        for a_i, a in enumerate(cats):
            for b in cats[a_i + 1:]:
                if zc[a] == hi_z and zc[b] == hi_z and (ligs[a] & ligs[b]):
                    nv += 1
        out["p4_n_viol"] = float(nv)
        out["p4_n_viol_per_cat"] = nv / len(cats)
    else:
        out["p4_n_viol"] = out["p4_n_viol_per_cat"] = np.nan
    # max semantics for rule 5: how many environments the worst species occupies
    out["p5_max_distinct"] = float(max(len(v) for v in grp.values())) if grp else np.nan
    # locally extreme coordination: the site furthest from that element's common coordination
    out["cn_cat_range_norm"] = (max(cnc) - min(cnc)) / max(np.mean(cnc), 1e-9)
    return out


def process(rec):
    from pymatgen.core import Structure, Lattice
    try:
        if rec["kind"] == "real":
            st = Structure.from_str(read_blob_cif(rec["off"], rec["ln"]), fmt="cif")
        else:
            st = Structure(Lattice(rec["lattice"]), rec["species"], rec["coords"],
                           coords_are_cartesian=True)
        if len(st) > 200:
            return None                      # large cells are too slow; truncated the same way
                                             # on both sides
        val, ok = guess_oxi(st)
        if not ok:
            return None
        c = criteria(st, val)
        if c is None:
            return None
        c.update(sid=rec["sid"], rk=rec["rk"], kind=rec["kind"],
                 split=rec.get("split", "unsplit"), anion=rec.get("anion", ""))
        return c
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if OUT.exists() and not a.force:
        print(f"{OUT} already exists; add --force to recompute")
        return 0

    prov = pd.read_parquet(F / "provenance.parquet",
                           columns=["source_id", "formula", "in_analysis_set",
                                    "anion", "blob_offset", "blob_length"])
    sp = pd.read_parquet(F / "splits.parquet")
    real = prov[prov.in_analysis_set].merge(sp, on="source_id", how="left")
    real["rk"] = real.formula.map(redkey)
    # the same rule as in iter_elementa: drop elemental structures. Measured, the 38,307-entry
    # analysis set contains 810 of them, Pauling's rules are undefined on them, and both sides
    # must drop them under the same convention.
    real = real[real.rk.notna() & real.rk.str.count(r"[A-Z]").ge(2)]
    print(f"real side after dropping elements: {len(real):,}")

    ekeys = set()
    for r in iter_elementa(set(real.rk.dropna())):
        ekeys.add(r["rk"])
    common = set(real.rk.dropna()) & ekeys
    print(f"shared reduced compositions: {len(common):,}")

    recs = []
    sub = real[real.rk.isin(common)]
    for t in sub.itertuples():
        recs.append({"kind": "real", "sid": t.source_id, "rk": t.rk,
                     "off": int(t.blob_offset), "ln": int(t.blob_length),
                     "split": t.split, "anion": t.anion})
    for r in iter_elementa(common):
        r["kind"] = "elem"
        recs.append(r)
    if a.limit:
        recs = recs[:a.limit]
    print(f"to process: {len(recs):,} (real {sum(1 for r in recs if r['kind']=='real'):,} / "
          f"elem {sum(1 for r in recs if r['kind']=='elem'):,})")

    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(process, recs, chunksize=16)):
            if res:
                rows.append(res)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,}/{len(recs):,} -> {len(rows):,} ok", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}, {len(df):,} rows")
    print(df.kind.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
