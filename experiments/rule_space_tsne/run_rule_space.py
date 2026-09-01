#!/usr/bin/env python3
"""Map of the searched law space: ~2M documented candidate rules, embedded and projected.

Population: the candidate laws documented in SI Note S1.3 ("Itemisation of the
candidate count", 2,037,606 lower bound), regenerated per family from the archived
frozen search artefacts:

  crystal-plausibility   L4 chain: 8,466 threshold predicates (regenerated exactly via
                         src/l4_search.candidates(); per-candidate satisfaction and
                         detection rates evaluated on the discovery split).
  sorbent (ODAC23)       NEXT72 (869,855), NEXT78 (886,095), NEXT79 (435): finite
                         tail-correction searches, regenerated exactly from the archived
                         search records (term lists x 29 rejection fractions).
                         NEXT534 + NEXT540 (36,564 each): framework-guard grids,
                         regenerated exactly (label-free) from archived feature tables.
                         NEXT76 (876,757, archived but NOT in the SI itemisation) is
                         optional via --include-next76.
  generative screening   SCIGEN/WyFormer cross-source searches: per-candidate tables
                         archived as parquet (NEXT98b/103/106/107/108/111/114/117/121/
                         122/125/127/158), read back verbatim.

Honest coverage: the SI entries 144,237 ("mechanism-family grids"), 12,909, 402, 192,
21 and 3 could not be uniquely relocated in the archive and are NOT regenerated;
coverage.json records the reconciliation. Nothing is padded or synthesised.

Stages (each cached to disk; rerun independently with --stages):
  enumerate -> out/parts/*.parquet, out/candidates.parquet, out/stars.json, out/coverage.json
  textualize -> out/texts.parquet
  embed     -> out/emb_shards/*.npy, out/embeddings.npy (+ embeddings_meta.json)
  project   -> out/tsne.csv
  plot      -> out/rule_space_map.pdf/.png, out/rule_space_plaus.pdf

Smoke test (runs offline in minutes):
  python run_rule_space.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- defaults
DEF_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_ARCHIVE = os.environ.get("PRIS_ARCHIVE", "archive/")
SEED = 20260815
REJECTION_FRACTIONS = [round(0.02 + 0.01 * i, 2) for i in range(29)]  # next57 frozen grid
SINGLE_WEIGHTS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)              # next72 frozen grid
PAIR_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
PAIR_SHORTLIST = 24

PROBLEM_LABEL = {
    "plaus": "crystal plausibility (L4)",
    "sorbent": "sorbent frameworks (ODAC23)",
    "genscreen": "generative-model screening",
}
PROBLEM_COLOR = {"plaus": "#D55E00", "sorbent": "#0072B2", "genscreen": "#009E73"}

# ------------------------------------------------------------------- feature glosses
# Glosses for the 94 numeric features feeding the L4 candidate grid. Sources:
# src/phys_law.py, src/geom_feat.py, src/elec_feat.py, src/discriminate.py,
# src/f3_features.py docstrings and code. Features absent from this dict fall back
# to their raw name (none at the time of writing; see README).
FEATURE_GLOSS = {
    # contact / geometry
    "bl_min": "shortest cation-anion bond relative to the Shannon radius sum (reduced contact ratio)",
    "bl_mean": "mean cation-anion bond length relative to the Shannon radius sum",
    "bl_cat_max": "largest per-cation shortest reduced bond length (most under-bonded cation)",
    "bl_rsd_max": "largest relative spread of bond lengths within any coordination polyhedron",
    "aa_min": "closest ligand-ligand contact relative to the ligand radius sum",
    "aa_mean": "mean closest ligand-ligand contact ratio over coordination polyhedra",
    "angvar_max": "largest variance of ligand-cation-ligand bond angles over polyhedra",
    "angvar_mean": "mean variance of ligand-cation-ligand bond angles",
    "ecc_max": "largest off-centre displacement of a cation inside its coordination polyhedron",
    "ecc_mean": "mean off-centre displacement of cations inside their coordination polyhedra",
    "dist_rsd": "mean relative spread of neighbour distances per site",
    "dist_rsd_max": "largest relative spread of neighbour distances at any site",
    "phi_mean": "mean convex-hull filling fraction of the coordination polyhedra",
    "phi_min": "lowest convex-hull filling fraction of any coordination polyhedron",
    "mef_mean": "mean Hoppe MEFIR effective-ionic-radius mismatch",
    "mef_min": "most negative Hoppe MEFIR effective-ionic-radius mismatch",
    "mef_max": "largest Hoppe MEFIR effective-ionic-radius mismatch",
    "mef_absmax": "largest-magnitude Hoppe MEFIR effective-ionic-radius mismatch",
    "rep9_ca_pa": "ninth-power Born repulsion per atom over cation-anion close contacts",
    "rep9_aa_pa": "ninth-power Born repulsion per atom over anion-anion close contacts",
    "rep9_cc_pa": "ninth-power Born repulsion per atom over cation-cation close contacts",
    "repexp_ca_pa": "Born-Mayer exponential repulsion per atom over cation-anion close contacts",
    "repexp_aa_pa": "Born-Mayer exponential repulsion per atom over anion-anion close contacts",
    "repexp_cc_pa": "Born-Mayer exponential repulsion per atom over cation-cation close contacts",
    "strain2_ca_pa": "quadratic bond-strain energy per atom over cation-anion contacts",
    # bond valence
    "gii": "global instability index of the bond-valence sums",
    "bv_mean_abs": "mean absolute deviation of bond-valence sums from the nominal valence",
    "bv_max_abs": "largest absolute deviation of a bond-valence sum from the nominal valence",
    "bv_rel_mean": "mean relative bond-valence deviation",
    "bv_rel_max": "largest relative bond-valence deviation at any site",
    "bv_frac_bad": "fraction of sites whose bond-valence sum deviates beyond tolerance",
    "bv_param_cov": "fraction of bonds covered by tabulated bond-valence parameters",
    # Madelung / electrostatics
    "ewald_per_atom": "Ewald electrostatic energy per atom",
    "ewald_real": "real-space part of the Ewald energy per atom",
    "ewald_recip": "reciprocal-space part of the Ewald energy per atom",
    "ewald_point": "point self-energy part of the Ewald energy per atom",
    "mad_max": "highest site Madelung energy",
    "mad_min": "lowest site Madelung energy",
    "mad_range": "range of the site Madelung energies",
    "mad_std": "standard deviation of the site Madelung energies",
    "mad_an_mean": "mean site Madelung energy over anions",
    "mad_cat_mean": "mean site Madelung energy over cations",
    "madz_mean": "mean charge-normalised site Madelung energy",
    "madz_range": "range of the charge-normalised site Madelung energies",
    "madz_std": "standard deviation of the charge-normalised site Madelung energies",
    "mef_field": "unused",  # placeholder never emitted; kept out of coverage counts
    # ionicity / composition
    "fi": "Pauling ionicity fraction from the electronegativity difference",
    "dchi": "electronegativity difference between the anion and the cations",
    "dchi_min": "smallest cation-to-anion electronegativity difference",
    "frac_like_bonds": "fraction of bonded like-charge (cation-cation or anion-anion) contacts",
    "min_opp_frac": "lowest per-site fraction of opposite-charge neighbours",
    "n_cat_el": "number of distinct cation elements",
    "z_cat_max": "highest formal cation charge",
    "z_cat_mean": "mean formal cation charge",
    "cat_an_ratio": "cation-to-anion count ratio",
    # coordination
    "cn_an_mean": "mean anion coordination number",
    "cn_an_max": "highest anion coordination number",
    "cn_an_span": "span of the anion coordination numbers",
    "cn_an_std": "standard deviation of the anion coordination numbers",
    "cn_cat_max": "highest cation coordination number",
    "cn_cat_min": "lowest cation coordination number",
    "cn_cat_span": "span of the cation coordination numbers",
    "cn_cat_std": "standard deviation of the cation coordination numbers",
    "cn_cat_range_norm": "cation coordination-number range normalised by its mean",
    "mean_cn_cat": "mean cation coordination number",
    "econ_mean": "mean Hoppe effective coordination number",
    "econ_min": "lowest Hoppe effective coordination number",
    "econ_max": "highest Hoppe effective coordination number",
    "econ_std": "standard deviation of the Hoppe effective coordination numbers",
    "poly_deg_mean": "mean number of polyhedron-polyhedron linkages per cation polyhedron",
    "poly_deg_max": "largest number of polyhedron-polyhedron linkages at any cation polyhedron",
    "frac_isolated": "fraction of coordination polyhedra sharing no anion with another polyhedron",
    "frac_corner": "fraction of polyhedron pairs sharing only a corner",
    "pair_per_cat": "number of anion-sharing polyhedron pairs per cation",
    "p2_mean_dev": "mean deviation of the Pauling bond-strength sum at the anions (rule 2)",
    "p2_max_dev": "largest deviation of a Pauling bond-strength sum at any anion (rule 2)",
    "p2_frac_ok_010": "fraction of anions whose bond-strength sum is within 0.10 of the charge (rule 2)",
    "p2_n_bad_per_an": "rule-2 violations beyond 0.20 per anion",
    "p3_frac_edge_face": "fraction of polyhedron pairs sharing an edge or a face (rule 3)",
    "p3_frac_face": "fraction of polyhedron pairs sharing a face (rule 3)",
    "p3_has_face": "whether any two coordination polyhedra share a face (rule 3)",
    "p3_n_face_per_cat": "face-sharing polyhedron pairs per cation (rule 3)",
    "p4_violate": "whether high-valence low-coordination cations share polyhedron elements (rule 4)",
    "p4_n_viol_per_cat": "rule-4 violations per cation",
    "p5_n_distinct": "mean number of distinct coordination environments per cation species (rule 5)",
    "p5_max_distinct": "largest number of distinct coordination environments of any cation species (rule 5)",
    "p5_ok": "whether every cation species keeps a single coordination environment (rule 5)",
    # symmetry / Wyckoff
    "sg_num_001": "space-group number at symmetry tolerance 0.01",
    "sg_num_01": "space-group number at symmetry tolerance 0.1",
    "wyckoff_econ_001": "fraction of symmetry-inequivalent sites at tolerance 0.01 (Wyckoff economy)",
    "wyckoff_econ_01": "fraction of symmetry-inequivalent sites at tolerance 0.1 (Wyckoff economy)",
    "csys_rank_001": "crystal-system rank from triclinic (1) to cubic (7)",
    # packing / volume
    "sh_pack": "Shannon hard-sphere packing fraction of the cell",
    "density": "mass density",
    "vol_per_atom": "cell volume per atom",
}

FAMILY = {}  # feature -> physical family for the plausibility panel
for _f in ("bl_min bl_mean bl_cat_max bl_rsd_max aa_min aa_mean angvar_max angvar_mean "
           "ecc_max ecc_mean dist_rsd dist_rsd_max phi_mean phi_min mef_mean mef_min "
           "mef_max mef_absmax rep9_ca_pa rep9_aa_pa rep9_cc_pa repexp_ca_pa "
           "repexp_aa_pa repexp_cc_pa strain2_ca_pa").split():
    FAMILY[_f] = "contact/geometry"
for _f in "gii bv_mean_abs bv_max_abs bv_rel_mean bv_rel_max bv_frac_bad bv_param_cov".split():
    FAMILY[_f] = "bond valence"
for _f in ("ewald_per_atom ewald_real ewald_recip ewald_point mad_max mad_min mad_range "
           "mad_std mad_an_mean mad_cat_mean madz_mean madz_range madz_std").split():
    FAMILY[_f] = "Madelung/electrostatic"
for _f in "sg_num_001 sg_num_01 wyckoff_econ_001 wyckoff_econ_01 csys_rank_001".split():
    FAMILY[_f] = "symmetry/Wyckoff"
for _f in ("cn_an_mean cn_an_max cn_an_span cn_an_std cn_cat_max cn_cat_min cn_cat_span "
           "cn_cat_std cn_cat_range_norm mean_cn_cat econ_mean econ_min econ_max econ_std "
           "poly_deg_mean poly_deg_max frac_isolated frac_corner pair_per_cat "
           "p2_mean_dev p2_max_dev p2_frac_ok_010 p2_n_bad_per_an p3_frac_edge_face "
           "p3_frac_face p3_has_face p3_n_face_per_cat p4_violate p4_n_viol_per_cat "
           "p5_n_distinct p5_max_distinct p5_ok").split():
    FAMILY[_f] = "coordination"
for _f in "sh_pack density vol_per_atom".split():
    FAMILY[_f] = "packing/volume"
for _f in ("fi dchi dchi_min frac_like_bonds min_opp_frac n_cat_el z_cat_max z_cat_mean "
           "cat_an_ratio").split():
    FAMILY[_f] = "ionicity/composition"
FAM_COLORS = {
    "contact/geometry": "#4477AA", "bond valence": "#EE6677",
    "Madelung/electrostatic": "#228833", "symmetry/Wyckoff": "#CCBB44",
    "coordination": "#66CCEE", "packing/volume": "#AA3377",
    "ionicity/composition": "#BBBBBB", "other": "#000000",
}

# The eight surviving rules (tex/body.tex). Star = nearest L4 candidate by
# (feature, direction, guard), then closest threshold.
D_RULES = [
    ("D1", "bl_min", "ge", 0.804, None),
    ("D2", "bl_min", "le", 1.05, 0.50),
    ("D3", "bl_mean", "le", 1.081, None),        # native guard is CN-based, outside grid
    ("D4", "madz_range", "le", 31.45, None),
    ("D5", "mad_max", "le", 15.17, None),
    ("D6", "frac_like_bonds", "le", 1e-4, 0.55),
    ("D7", "wyckoff_econ_001", "le", 2.0 / 3.0, None),
    ("D8", "bv_rel_mean", "le", 0.7143040821865658, None),
]

# Archived per-candidate parquet families (generative-model screening programme).
# (tag, reldir, parquet, expected_rows, in_SI_itemisation)
CATALOGUE_FAMILIES = [
    ("next98b", "next98b_cross_source_exhaustive_search_v1",
     "next98b_cross_source_exhaustive_candidate_search.parquet", 12111, False),
    ("next103", "next103_dobvr_optional_guard_search_v1",
     "next103_optional_guard_candidate_search.parquet", 4757, True),
    ("next106", "next106_cmvf_optional_guard_search_v1",
     "next106_optional_guard_candidate_search.parquet", 2077, True),
    ("next107", "next107_two_axis_cmvf_guard_search_v1",
     "next107_two_axis_candidate_search.parquet", 12127, True),
    ("next108", "next108_near_miss_cmvf_rescue_v1",
     "next108_near_miss_rescue_search.parquet", 9178, True),
    ("next111", "next111_cmvo_optional_search_v1",
     "next111_cmvo_optional_candidate_search.parquet", 22592, True),
    ("next114", "next114_cmvom_frontier_rescue_v1",
     "next114_cmvom_frontier_candidate_search.parquet", 2688, False),
    ("next117", "next117_hcid_frontier_rescue_v1",
     "next117_hcid_frontier_candidate_search.parquet", 11349, False),
    ("next121", "next121_bvtbd_frontier_rescue_v1",
     "next121_bvtbd_frontier_candidate_search.parquet", 59319, False),
    ("next122", "next122_safe12_bvtc_prlr_rescue_v1",
     "next122_safe12_bvtc_prlr_candidate_search.parquet", 14292, False),
    ("next125", "next125_mhcr_frontier_rescue_v1",
     "next125_mhcr_frontier_candidate_search.parquet", 57178, False),
    ("next127", "next127_hall_profile_persistence_rescue_v1",
     "next127_hpp_rescue_candidate_search.parquet", 1300, False),
    ("next158", "next158_mechanism_family_consensus_search_v1",
     "next158_mechanism_family_consensus_candidate_search.parquet", 176, False),
]
# ODAC23 finite tail searches: (tag, reldir, search_json, expected, itemised, physics label)
TAIL_FAMILIES = [
    ("next72", "next72_odac23_anchored_tail_correction_search_v1",
     "NEXT72_ODAC23_ANCHORED_TAIL_SEARCH.json", 869855, True,
     "metal-donor bond-valence and motif"),
    ("next76", "next76_odac23_rigidity_tail_search_v1",
     "NEXT76_ODAC23_RIGIDITY_TAIL_SEARCH.json", 876757, False,
     "metal-ligand rigidity"),
    ("next78", "next78_odac23_electrostatic_tail_search_v1",
     "NEXT78_ODAC23_ELECTROSTATIC_TAIL_SEARCH.json", 886095, True,
     "analytic electrostatic"),
    ("next79", "next79_odac23_electrostatic_residual_guard_v1",
     "NEXT79_ODAC23_ELECTROSTATIC_RESIDUAL_SEARCH.json", 435, True,
     "electrostatic residual"),
]
# ODAC23 framework guard grids (label-free recomputation): (tag, reldir, deficit label)
GUARD_FAMILIES = [
    ("next534", "next534_odac23_sssp_framework_guard_v1",
     "same-sign shell-purity deficit", 36564, True),
    ("next540", "next540_odac23_pbaaa_framework_guard_v1",
     "bond-angle affine-accommodation deficit", 36564, True),
]
# SI itemisation entries that could NOT be uniquely relocated in the archive.
UNMAPPED_ITEMISED = {
    "mechanism-family grids (explicit totals)": 144237,
    "unidentified grid": 12909,
    "unidentified grid (402)": 402,
    "NEXT562 HEA direction combinations (not regenerated)": 192,
    "margin-local eligible candidates (not regenerated)": 21,
    "broad-residual diagnostic (not regenerated)": 3,
}
SI_ITEMISED_TOTAL = 2037606


def pretty(name: str) -> str:
    """Human-readable phrase from an archived term id or feature name."""
    s = str(name)
    hi = s.endswith("__high")
    lo = s.endswith("__low")
    s = s.removesuffix("__high").removesuffix("__low").replace("_", " ").strip()
    if hi:
        return f"high {s}"
    if lo:
        return f"low {s}"
    return s


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================= stage 1: enumerate
def enum_l4(repo: str, eval_metrics: bool = True, sample_n: int | None = None,
            rng: np.random.Generator | None = None) -> pd.DataFrame:
    sys.path.insert(0, os.path.join(repo, "src"))
    import l4_search  # noqa: E402

    lr, lb = l4_search.load_tables()
    lrd = lr[lr.split == "discovery"].reset_index(drop=True)
    lbd = lb[lb.psplit == "discovery"].reset_index(drop=True)
    cands = l4_search.candidates(lrd)
    log(f"l4: {len(cands)} candidate predicates "
        f"(discovery real {len(lrd)}, perturbed {len(lbd)})")
    rows = pd.DataFrame(cands, columns=["feature", "direction", "threshold", "guard"])
    rows.insert(0, "family", "l4")
    rows.insert(0, "problem", "plaus")
    rows["kind"] = "threshold"
    idx = np.arange(len(rows))
    if sample_n is not None and sample_n < len(rows):
        idx = (rng or np.random.default_rng(SEED)).choice(len(rows), sample_n, replace=False)
        idx.sort()
        rows = rows.iloc[idx].reset_index(drop=True)
    sat = np.full(len(rows), np.nan)
    det = np.full(len(rows), np.nan)
    if eval_metrics:
        t0 = time.time()
        for i, t in enumerate(rows.itertuples(index=False)):
            c = (t.feature, t.direction, t.threshold, t.guard)
            sat[i] = float(l4_search.pred_mask(lrd, *c).mean())
            det[i] = float(1.0 - l4_search.pred_mask(lbd, *c).mean())
        log(f"l4: metrics for {len(rows)} candidates in {time.time()-t0:.1f}s")
    rows["sat_disc"] = sat
    rows["det_disc"] = det
    return rows


def find_stars(l4: pd.DataFrame) -> dict:
    stars = {}
    for name, feat, drc, th, guard in D_RULES:
        m = (l4.feature == feat) & (l4.direction == drc)
        mg = m & (l4.guard == guard) if guard is not None else m & l4.guard.isna()
        pick = l4[mg] if mg.any() else l4[m]
        if pick.empty:
            continue
        j = (pick.threshold - th).abs().idxmin()
        stars[name] = {"rule_id": int(l4.loc[j, "rule_id"]),
                       "feature": feat, "target_threshold": th,
                       "matched_threshold": float(l4.loc[j, "threshold"]),
                       "guard_matched": bool(mg.any())}
    return stars


def enum_tail(tag: str, reldir: str, search_json: str, expected: int, archive: str,
              physics: str) -> tuple[pd.DataFrame, bool]:
    """Regenerate a NEXT72-style finite tail-correction candidate set.

    Enumeration = [anchor] + singles(ranked features x 2 directions x 7 weights)
    + pairs(C(shortlist,2) x 4 direction pairs x 25 weight pairs), each crossed with
    the 29 frozen rejection fractions. Term lists and shortlist are read from the
    archived frozen search record; nothing is re-searched.
    """
    d = Path(archive) / reldir
    rec = json.loads((d / search_json).read_text())
    ranking = rec["guard_feature_ranking"]
    shortlist = [str(x) for x in rec.get("pair_shortlist", [])]
    feats = [str(r["feature"]) for r in ranking]
    nthr = len(REJECTION_FRACTIONS)

    kinds, f1s, d1s, w1s, f2s, d2s, w2s, rfs = [], [], [], [], [], [], [], []

    def emit(kind, f1=None, d1=0, w1=np.nan, f2=None, d2=0, w2=np.nan):
        for rf in REJECTION_FRACTIONS:
            kinds.append(kind); f1s.append(f1); d1s.append(d1); w1s.append(w1)
            f2s.append(f2); d2s.append(d2); w2s.append(w2); rfs.append(rf)

    emit("anchor")
    for f in feats:
        for drc in (-1, 1):
            for w in SINGLE_WEIGHTS:
                emit("single", f, drc, w)
    short = shortlist[: min(PAIR_SHORTLIST, len(shortlist))]
    from itertools import combinations, product
    for fa, fb in combinations(short, 2):
        a, b = sorted((fa, fb))
        for da, db in product((-1, 1), repeat=2):
            for wa, wb in product(PAIR_WEIGHTS, repeat=2):
                emit("pair", a, da, wa, b, db, wb)
    df = pd.DataFrame({"kind": kinds, "feature": f1s, "dir1": d1s, "weight": w1s,
                       "feature2": f2s, "dir2": d2s, "weight2": w2s,
                       "reject_frac": rfs})
    df.insert(0, "family", tag)
    df.insert(0, "problem", "sorbent")
    df["physics"] = physics
    exact = len(df) == expected == int(rec.get("candidate_count", -1))
    log(f"{tag}: regenerated {len(df)} candidates "
        f"(archived {rec.get('candidate_count')}, itemised {expected}, exact={exact})")
    return df, exact


def enum_guard(tag: str, reldir: str, deficit_label: str, expected: int, repo: str,
               archive: str) -> tuple[pd.DataFrame, bool]:
    """Regenerate a NEXT534/540 framework-guard grid label-free.

    Candidates = 6 frozen weights x the unique score values over the supported
    development rows; the score is anchor(NEXT79) + weight x bounded deficit,
    recomputed from the archived label-free feature tables. No endpoint labels
    are read.
    """
    if repo not in sys.path:
        sys.path.insert(0, repo)
    man = json.loads((Path(archive) / reldir / "MANIFEST.json").read_text())
    ins = {k: v["path"] for k, v in man["inputs_sha256"].items()}
    if tag == "next534":
        from src.next534_odac23_sssp_framework_guard import (
            apply_sssp_framework_guard as apply_guard, WEIGHTS, DEVELOPMENT_ROLES)
        aux = pd.read_parquet(ins["sssp_features"])
    else:
        from src.next540_odac23_pbaaa_framework_guard import (
            apply_pbaaa_framework_guard as apply_guard, WEIGHTS, DEVELOPMENT_ROLES)
        aux = pd.read_parquet(ins["pbaaa_features"])
    frame = pd.read_parquet(ins["framework_features"])
    anchor = json.loads(Path(ins["next79_formula"]).read_text())
    # restrict to the archived development identity: rows present in the two label
    # files (only the material_id column is read; no endpoint values are opened)
    ids = pd.concat([
        pd.read_parquet(ins["discovery_labels"], columns=["material_id"]),
        pd.read_parquet(ins["validation_labels"], columns=["material_id"]),
    ])["material_id"].astype(str)
    dev = frame[frame["partition_role"].isin(DEVELOPMENT_ROLES)
                & frame["material_id"].astype(str).isin(set(ids))].merge(
        aux.drop(columns=[c for c in ("partition_role",) if c in aux.columns]),
        on="material_id", how="left", validate="one_to_one")
    ws, ths = [], []
    for w in WEIGHTS:
        score, supported, _r, _d = apply_guard(
            dev, anchor, weight=float(w), threshold=float(anchor["threshold"]))
        for th in np.unique(score[supported]).tolist():
            ws.append(float(w)); ths.append(float(th))
    df = pd.DataFrame({"weight": ws, "threshold": ths})
    df.insert(0, "family", tag)
    df.insert(0, "problem", "sorbent")
    df["kind"] = "guard_grid"
    df["physics"] = deficit_label
    exact = len(df) == expected
    log(f"{tag}: regenerated {len(df)} candidates (itemised {expected}, exact={exact})")
    return df, exact


def enum_catalogue(tag: str, reldir: str, parquet: str, expected: int,
                   archive: str) -> tuple[pd.DataFrame, bool]:
    t = pd.read_parquet(Path(archive) / reldir / parquet)
    out = pd.DataFrame(index=t.index)
    out["problem"] = "genscreen"
    out["family"] = tag
    out["kind"] = "score_formula"
    if "base_term_ids_json" in t.columns:
        out["terms_json"] = t["base_term_ids_json"]
        out["weights_json"] = t["base_weights_json"]
        out["opt_term"] = t.get("optional_term_id")
        out["opt_weight"] = t.get("optional_weight")
    else:  # next98b schema
        out["terms_json"] = t["term_ids_json"]
        out["weights_json"] = t["weights_json"]
        out["opt_term"] = None
        out["opt_weight"] = np.nan
    out["threshold"] = pd.to_numeric(t.get("safe_threshold"), errors="coerce")
    exact = len(out) == expected
    log(f"{tag}: read {len(out)} archived candidates (expected {expected}, exact={exact})")
    return out.reset_index(drop=True), exact


def stage_enumerate(a) -> None:
    out = Path(a.out)
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    coverage = {"si_itemised_total": SI_ITEMISED_TOTAL, "families": {},
                "unmapped_itemised": UNMAPPED_ITEMISED}
    smoke_caps = {"l4": 800, "next72": 400, "next103": 300, "next106": 200,
                  "next111": 300}
    parts = []

    def cache(tag, fn):
        p = parts_dir / f"{tag}.parquet"
        if p.exists() and not a.force:
            df = pd.read_parquet(p)
            log(f"{tag}: cached ({len(df)})")
            meta = json.loads((parts_dir / f"{tag}.meta.json").read_text())
        else:
            df, exact = fn()
            if a.smoke and tag in smoke_caps and len(df) > smoke_caps[tag]:
                df = df.sample(smoke_caps[tag], random_state=SEED).reset_index(drop=True)
                exact = False
            df.to_parquet(p, index=False)
            meta = {"count": len(df), "exact": bool(exact)}
            (parts_dir / f"{tag}.meta.json").write_text(json.dumps(meta))
        coverage["families"][tag] = meta
        parts.append(df)

    # crystal plausibility
    cache("l4", lambda: (enum_l4(a.repo, eval_metrics=True,
                                 sample_n=smoke_caps["l4"] if a.smoke else None,
                                 rng=rng), not a.smoke))
    if not a.smoke:
        for tag, reldir, sj, exp, itemised, physics in TAIL_FAMILIES:
            if tag == "next76" and not a.include_next76:
                continue
            cache(tag, lambda tag=tag, reldir=reldir, sj=sj, exp=exp, physics=physics:
                  enum_tail(tag, reldir, sj, exp, a.archive, physics))
        for tag, reldir, lbl, exp, itemised in GUARD_FAMILIES:
            def _guard(tag=tag, reldir=reldir, lbl=lbl, exp=exp):
                try:
                    return enum_guard(tag, reldir, lbl, exp, a.repo, a.archive)
                except Exception as e:  # keep going; record honestly
                    log(f"{tag}: FAILED to regenerate ({e}); family omitted")
                    return pd.DataFrame(), False
            cache(tag, _guard)
        cat_list = CATALOGUE_FAMILIES
    else:
        # smoke: one tail family (sampled) + a few archived catalogues
        cache("next72", lambda: enum_tail(*[f for f in TAIL_FAMILIES if f[0] == "next72"][0][:3],
                                          869855, a.archive, "metal-donor bond-valence and motif"))
        cat_list = [f for f in CATALOGUE_FAMILIES if f[0] in ("next103", "next106", "next111")]
    for tag, reldir, pq, exp, itemised in cat_list:
        cache(tag, lambda tag=tag, reldir=reldir, pq=pq, exp=exp:
              enum_catalogue(tag, reldir, pq, exp, a.archive))

    allc = pd.concat([p for p in parts if len(p)], ignore_index=True, sort=False)
    allc.insert(0, "rule_id", np.arange(len(allc), dtype=np.int64))
    allc.to_parquet(out / "candidates.parquet", index=False)
    l4 = allc[allc.family == "l4"]
    stars = find_stars(l4)
    (out / "stars.json").write_text(json.dumps(stars, indent=1))
    regen_itemised = sum(m["count"] for t, m in coverage["families"].items()
                         if m.get("exact") and t in
                         {"l4", "next72", "next78", "next79", "next534", "next540",
                          "next103", "next106", "next107", "next108", "next111"})
    coverage["total_plotted"] = int(len(allc))
    coverage["itemised_regenerated_exactly"] = int(regen_itemised)
    coverage["itemised_share"] = round(regen_itemised / SI_ITEMISED_TOTAL, 4)
    (out / "coverage.json").write_text(json.dumps(coverage, indent=1))
    log(f"enumerate: {len(allc)} candidates -> candidates.parquet; "
        f"{len(stars)}/8 survivor stars matched; "
        f"exact itemised coverage {coverage['itemised_share']:.1%}")


# ================================================================ stage 2: textualize
def texts_l4(df: pd.DataFrame) -> pd.Series:
    gloss = df.feature.map(lambda f: FEATURE_GLOSS.get(f, f))
    rel = np.where(df.direction == "le", "at most", "at least")
    th = df.threshold.map(lambda v: f"{v:.4g}")
    base = gloss + " is " + rel + " " + th
    g = df.guard
    guarded = ("if the Pauling ionicity exceeds " + g.map(lambda v: f"{v:.2f}")
               + ", the " + base)
    return pd.Series(np.where(g.notna(), guarded, base.str[0].str.upper() + base.str[1:]),
                     index=df.index)


def _tail_term(feat, drc, w):
    side = "high" if drc == 1 else "low"
    return f"a {w:g}-weighted risk increase for {side} {pretty(feat)}"


def texts_tail(df: pd.DataFrame) -> pd.Series:
    out = np.empty(len(df), dtype=object)
    for i, t in enumerate(df.itertuples(index=False)):
        pct = f"{t.reject_frac * 100:.0f}%"
        head = (f"ODAC23 sorbent screening: score frameworks with the sealed "
                f"{t.physics} instability axis")
        if t.kind == "anchor":
            body = ""
        elif t.kind == "single":
            body = " plus " + _tail_term(t.feature, t.dir1, t.weight)
        else:
            body = (" plus " + _tail_term(t.feature, t.dir1, t.weight)
                    + " and " + _tail_term(t.feature2, t.dir2, t.weight2))
        out[i] = f"{head}{body}, rejecting the riskiest {pct} of frameworks"
    return pd.Series(out, index=df.index)


def texts_guard(df: pd.DataFrame) -> pd.Series:
    return ("ODAC23 sorbent screening: NEXT79 electrostatic risk score plus a "
            + df.weight.map(lambda v: f"{v:g}") + "-weighted " + df.physics
            + ", rejecting frameworks scoring above "
            + df.threshold.map(lambda v: f"{v:.4g}"))


def texts_catalogue(df: pd.DataFrame) -> pd.Series:
    out = np.empty(len(df), dtype=object)
    for i, t in enumerate(df.itertuples(index=False)):
        try:
            terms = json.loads(t.terms_json)
            weights = json.loads(t.weights_json)
        except Exception:
            terms, weights = [], []
        phr = ", ".join(f"{pretty(x)} (w {w:g})" for x, w in zip(terms, weights))
        opt = ""
        if isinstance(t.opt_term, str) and t.opt_term:
            ow = t.opt_weight if np.isfinite(t.opt_weight) else 1.0
            opt = f", plus an optional guard on {pretty(t.opt_term)} at weight {ow:g}"
        th = f" above threshold {t.threshold:.4g}" if np.isfinite(t.threshold) else ""
        out[i] = (f"Generative-structure screening ({t.family.upper()}): flag likely "
                  f"failures by a risk score over {phr}{opt}{th}")
    return pd.Series(out, index=df.index)


def stage_textualize(a) -> None:
    out = Path(a.out)
    df = pd.read_parquet(out / "candidates.parquet")
    text = pd.Series(index=df.index, dtype=object)
    m = df.kind == "threshold"
    if m.any():
        text[m] = texts_l4(df[m])
    m = df.kind.isin(["anchor", "single", "pair"])
    if m.any():
        text[m] = texts_tail(df[m])
    m = df.kind == "guard_grid"
    if m.any():
        text[m] = texts_guard(df[m])
    m = df.kind == "score_formula"
    if m.any():
        text[m] = texts_catalogue(df[m])
    t = df[["rule_id", "problem", "family"]].copy()
    t["text"] = text
    assert t.text.notna().all(), "untextualized candidates remain"
    t.to_parquet(out / "texts.parquet", index=False)
    log(f"textualize: {len(t)} statements -> texts.parquet")
    for fam in t.family.unique():
        log(f"  e.g. [{fam}] {t[t.family == fam].text.iloc[0][:120]}")


# ===================================================================== stage 3: embed
def stage_embed(a) -> None:
    out = Path(a.out)
    t = pd.read_parquet(out / "texts.parquet")
    texts = t.text.tolist()
    n = len(texts)
    meta_p = out / "embeddings_meta.json"
    if a.backend == "st":
        from sentence_transformers import SentenceTransformer
        shard_dir = out / "emb_shards"
        shard_dir.mkdir(exist_ok=True)
        model = SentenceTransformer(a.model, device=a.device)
        if a.device == "cuda":
            model.half()
        shard = 50000
        nshard = math.ceil(n / shard)
        for i in range(nshard):
            p = shard_dir / f"shard_{i:05d}.npy"
            if p.exists() and not a.force:
                continue
            t0 = time.time()
            emb = model.encode(texts[i * shard:(i + 1) * shard],
                               batch_size=a.batch_size, show_progress_bar=False,
                               normalize_embeddings=True)
            np.save(p, emb.astype(np.float16))
            log(f"embed: shard {i + 1}/{nshard} ({time.time()-t0:.0f}s)")
        dim = np.load(shard_dir / "shard_00000.npy").shape[1]
        X = np.lib.format.open_memmap(out / "embeddings.npy", mode="w+",
                                      dtype=np.float16, shape=(n, dim))
        pos = 0
        for i in range(nshard):
            e = np.load(shard_dir / f"shard_{i:05d}.npy")
            X[pos:pos + len(e)] = e
            pos += len(e)
        X.flush()
        meta = {"backend": "st", "model": a.model, "dim": int(dim), "n": n}
    else:  # tfidf fallback (offline)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        from scipy.sparse import hstack
        vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=30000,
                             sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
                             sublinear_tf=True)
        M = hstack([vw.fit_transform(texts), vc.fit_transform(texts)]).tocsr()
        k = min(a.svd_dim, M.shape[1] - 1, max(2, n - 1))
        emb = TruncatedSVD(n_components=k, random_state=SEED).fit_transform(M)
        emb = normalize(emb).astype(np.float32)
        np.save(out / "embeddings.npy", emb)
        meta = {"backend": "tfidf", "dim": int(emb.shape[1]), "n": n}
    meta_p.write_text(json.dumps(meta))
    log(f"embed: {n} x {meta['dim']} ({meta['backend']}) -> embeddings.npy")


# =================================================================== stage 4: project
def stage_project(a) -> None:
    out = Path(a.out)
    t = pd.read_parquet(out / "texts.parquet")
    X = np.load(out / "embeddings.npy", mmap_mode="r")
    n = len(t)
    idx = np.arange(n)
    if a.subsample and a.subsample < n:  # stratified by family
        rng = np.random.default_rng(SEED)
        picks = []
        share = a.subsample / n
        for fam, g in t.groupby("family").groups.items():
            g = np.asarray(g)
            k = max(min(len(g), 200), int(round(len(g) * share)))
            picks.append(rng.choice(g, size=min(k, len(g)), replace=False))
        idx = np.concatenate(picks)
        stars_p = out / "stars.json"
        if stars_p.exists():  # always keep the eight survivors in the projection
            star_ids = {s["rule_id"] for s in json.loads(stars_p.read_text()).values()}
            extra = t.index[t.rule_id.isin(star_ids)].to_numpy()
            idx = np.concatenate([idx, extra])
        idx = np.unique(idx)
        log(f"project: stratified subsample {len(idx)}/{n}")
    Xs = np.asarray(X[idx], dtype=np.float32)
    if Xs.shape[1] > 50:
        from sklearn.decomposition import PCA
        Xs = PCA(n_components=50, random_state=SEED).fit_transform(Xs)
    coords = None
    try:
        from openTSNE import TSNE
        log(f"project: openTSNE on {len(idx)} points (perplexity {a.perplexity})")
        coords = np.asarray(TSNE(perplexity=a.perplexity, initialization="pca",
                                 random_state=SEED, n_jobs=-1,
                                 verbose=False).fit(Xs))
    except ImportError:
        pass
    if coords is None and len(idx) <= 100000:
        from sklearn.manifold import TSNE as SkTSNE
        log(f"project: sklearn TSNE on {len(idx)} points")
        coords = SkTSNE(perplexity=min(a.perplexity, (len(idx) - 1) // 3),
                        init="pca", random_state=SEED).fit_transform(Xs)
    if coords is None:
        try:
            import umap
            log(f"project: UMAP fallback on {len(idx)} points")
            coords = umap.UMAP(n_neighbors=30, min_dist=0.1,
                               random_state=SEED).fit_transform(Xs)
        except ImportError:
            raise SystemExit("N too large for sklearn t-SNE: pip install openTSNE "
                             "(preferred) or umap-learn, or pass --subsample N")
    d = t.iloc[idx][["rule_id", "problem", "family"]].copy()
    d["x"], d["y"] = coords[:, 0], coords[:, 1]
    d.to_csv(out / "tsne.csv", index=False)
    log(f"project: wrote {len(d)} coordinates -> tsne.csv")


# ====================================================================== stage 5: plot
def stage_plot(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    out = Path(a.out)
    d = pd.read_csv(out / "tsne.csv")
    cand = pd.read_parquet(out / "candidates.parquet",
                           columns=["rule_id", "feature", "sat_disc", "det_disc"])
    d = d.merge(cand, on="rule_id", how="left")
    stars = json.loads((out / "stars.json").read_text())

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sizes = {"sorbent": 1.0, "genscreen": 1.6, "plaus": 3.0}
    counts = d.problem.value_counts()
    for prob in ("sorbent", "genscreen", "plaus"):
        g = d[d.problem == prob]
        if g.empty:
            continue
        # alpha adapts to the territory's population so both smoke and full runs read
        kw = dict(s=sizes[prob],
                  alpha=float(np.clip(60000.0 / max(len(g), 1), 0.06, 0.6)))
        if prob == "plaus":  # size by detection rate where sat >= 0.90
            det = g.det_disc.fillna(0.0).to_numpy()
            sat = g.sat_disc.fillna(0.0).to_numpy()
            kw["s"] = np.where(sat >= 0.90, 2.0 + 14.0 * det, 1.6)
        ax.scatter(g.x, g.y, c=PROBLEM_COLOR[prob], linewidths=0, rasterized=True,
                   label=f"{PROBLEM_LABEL[prob]}  (n = {counts.get(prob, 0):,})", **kw)
    sd = d.set_index("rule_id")
    for name, s in stars.items():
        rid = s["rule_id"]
        if rid not in sd.index:
            continue
        x, y = sd.loc[rid, "x"], sd.loc[rid, "y"]
        ax.scatter([x], [y], marker="*", s=230, c="#FFD700", edgecolors="black",
                   linewidths=0.7, zorder=6)
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, fontweight="bold", zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=7.5, markerscale=6,
                    handletextpad=0.4, borderaxespad=0.2)
    for h in getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", []):
        h.set_alpha(1.0)
    ax.text(0.99, 0.01, f"t-SNE of {len(d):,} candidate rule statements; "
            "stars mark the eight surviving rules D1-D8",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.35")
    fig.tight_layout()
    fig.savefig(out / "rule_space_map.pdf", dpi=400)
    fig.savefig(out / "rule_space_map.png", dpi=400)
    plt.close(fig)
    log(f"plot: rule_space_map.pdf/.png ({len(d)} points, {len(stars)} stars)")

    # plausibility-only panel coloured by physical feature family
    g = d[d.problem == "plaus"].copy()
    if not g.empty:
        g["fam"] = g.feature.map(lambda f: FAMILY.get(f, "other"))
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        for fam, gg in g.groupby("fam"):
            det = gg.det_disc.fillna(0.0).to_numpy()
            sat = gg.sat_disc.fillna(0.0).to_numpy()
            s = np.where(sat >= 0.90, 3.0 + 22.0 * det, 2.2)
            ax.scatter(gg.x, gg.y, s=s, c=FAM_COLORS.get(fam, "#000000"),
                       alpha=0.65, linewidths=0, rasterized=True,
                       label=f"{fam} ({len(gg)})")
        for name, srec in stars.items():
            rid = srec["rule_id"]
            if rid not in sd.index:
                continue
            x, y = sd.loc[rid, "x"], sd.loc[rid, "y"]
            ax.scatter([x], [y], marker="*", s=200, c="#FFD700", edgecolors="black",
                       linewidths=0.7, zorder=6)
            ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, fontweight="bold", zorder=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=6.5, markerscale=3)
        fig.tight_layout()
        fig.savefig(out / "rule_space_plaus.pdf", dpi=400)
        plt.close(fig)
        log("plot: rule_space_plaus.pdf (plausibility territory by feature family)")


# ============================================================================= main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all",
                    help="comma list of enumerate,textualize,embed,project,plot (or all)")
    ap.add_argument("--out", default=None, help="output dir (default ./out, ./out_smoke with --smoke)")
    ap.add_argument("--repo", default=DEF_REPO)
    ap.add_argument("--archive", default=DEF_ARCHIVE)
    ap.add_argument("--backend", choices=["st", "tfidf"], default="st")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--svd-dim", type=int, default=256, help="tfidf SVD dimensions")
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--subsample", type=int, default=0,
                    help="stratified subsample for projection (0 = all points)")
    ap.add_argument("--include-next76", action="store_true",
                    help="also plot the archived NEXT76 tail search (876,757 candidates, "
                         "not part of the SI itemisation)")
    ap.add_argument("--force", action="store_true", help="recompute cached stages")
    ap.add_argument("--smoke", action="store_true",
                    help="~2,000-candidate offline end-to-end test (tfidf backend)")
    a = ap.parse_args()
    if a.out is None:
        a.out = "out_smoke" if a.smoke else "out"
    if a.smoke:
        a.backend = "tfidf"
        a.svd_dim = min(a.svd_dim, 64)
        a.perplexity = min(a.perplexity, 30.0)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    stages = (["enumerate", "textualize", "embed", "project", "plot"]
              if a.stages == "all" else a.stages.split(","))
    t0 = time.time()
    for s in stages:
        {"enumerate": stage_enumerate, "textualize": stage_textualize,
         "embed": stage_embed, "project": stage_project, "plot": stage_plot}[s](a)
    log(f"done ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
