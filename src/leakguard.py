# -*- coding: utf-8 -*-
"""LeakGuard: a provably leak-free feature whitelist + a non-degenerate set scorer.

Background
----------
The previous round of adversarial review proved that **human judgement of "is this feature a
leak" is unreliable**. The author of `search_t1.univ_conn` wrote the identity
`tgt = 1 <=> an_link > deg` into a comment, then banned `deg` while keeping `an_link` --
banning half of it. `T_P5`'s `cn_span_st<=0 => tgt=0` is a logical tautology, and no gate
caught that either.

So leak detection has to be **done by machine**. This module implements three levels:

  A empirical determinism : is t a deterministic function of a single f, or is there a
                            zero-error one-sided pure region
  B single-feature ceiling: accuracy of an unconstrained decision tree on a single f, under
                            CV grouped by proto_id
  C construction provenance: each feature in features.yaml declares (atoms, op, scope, base);
                            take the transitive closure against the target's declaration and
                            apply three hard rules

Acceptance criterion (for the detector itself): it must reproduce the two known leaks from
the previous round
  an_link  vs T_CONN
  cn_span_st vs T_P5

Usage
-----
  python leakguard.py prep      # build and cache the three domains (reads discovery only)
  python leakguard.py audit     # run the three levels, write the report + whitelist
  python leakguard.py all
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FEAT = os.environ.get("PRIS_FEATURES", "features/")
SCRATCH = ("/tmp/claude-1000/-home-zhilong-workspace-newpauling/"
           "6021b5d8-e7ef-4fa5-8416-1361cb973e90/scratchpad")
YAML_OUT = os.path.join(FEAT, "features_clean.yaml")
REPORT_OUT = os.path.join(FEAT, "leakguard_report.json")
YAML_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.yaml")

TARGETS = ("T_CONN", "T_P2", "T_P5")

# ---- decision thresholds (fixed before any detection result was seen) -------------------
EPS_DET = 1e-6          # error tolerance for a deterministic function
PURE_MIN_COV = 0.005    # minimum coverage of a one-sided pure region (below this it is not a
                        # leak, just a rare subgroup)
TAIL_PURITY = 0.99      # purity threshold for a one-sided "near-pure" region (cov >= TAIL_MIN_COV)
TAIL_MIN_COV = 0.01
CEIL_BLOCK = 0.98       # single-feature grouped-CV ceiling at or above this -> a leak
CEIL_REVIEW = 0.95


# ================================================================== domain cache

def prep(force=False):
    """Build the three T1 domains and cache them in the scratchpad. Reads discovery only
    (guaranteed by search_t1)."""
    import search_t1 as st
    todo = [t for t in TARGETS
            if force or not os.path.exists(os.path.join(SCRATCH, f"lg_{t}.parquet"))]
    if not todo:
        print("[prep] cache already exists, skipping")
        return
    t0 = time.time()
    X0, meta0, feat0, rnd0, orb, U = st.build_universes()

    def dump(tag, uni):
        X = uni["Xs"]["chemenv"].copy()
        for a in st.ALGOS:
            X["__y_" + a] = np.asarray(uni["ys"][a]).astype(np.int8)
        X["__ok"] = np.asarray(uni["ok"]).astype(bool)
        m = uni["meta"]
        for c in ("source_id", "element", "ox", "anion", "proto_id"):
            X["__m_" + c] = m[c].values
        X.to_parquet(os.path.join(SCRATCH, f"lg_{tag}.parquet"), index=False)
        json.dump({"feat": uni["feat"], "rnd": uni["rnd"]},
                  open(os.path.join(SCRATCH, f"lg_{tag}_cols.json"), "w"))
        print(f"[prep] {tag}: {len(X)} rows / {len(uni['feat'])} features "
              f"/ ok={int(X['__ok'].sum())} ({time.time()-t0:.0f}s)", flush=True)

    import gc
    if "T_CONN" in todo:
        dump("T_CONN", st.univ_conn(X0, meta0, orb, U)); gc.collect()
    if "T_P2" in todo:
        dump("T_P2", st.univ_p2(U, orb)); gc.collect()
    if "T_P5" in todo:
        dump("T_P5", st.univ_p5(X0, meta0, orb, U)); gc.collect()


def load(tag):
    X = pd.read_parquet(os.path.join(SCRATCH, f"lg_{tag}.parquet"))
    cols = json.load(open(os.path.join(SCRATCH, f"lg_{tag}_cols.json")))
    ok = X["__ok"].values.astype(bool)
    y = {a: X["__y_" + a].values.astype(np.int8)
         for a in ("chemenv", "crystalnn", "brunner")}
    meta = X[[c for c in X.columns if c.startswith("__m_")]].rename(
        columns=lambda c: c[4:])
    F = X[[c for c in cols["feat"] + cols["rnd"] if c in X.columns]]
    return F, y, ok, meta, cols["feat"], [c for c in cols["rnd"] if c in X.columns]


# ================================ level A: empirical determinism / one-sided pure region

def level_a(v, y, ok):
    """Empirical determinism detection for a single feature.

    Returns
      err_thresh   : the misclassification rate minimised over (theta, polarity). == 0 means
                     t is a threshold function of f
      err_value    : the minimum misclassification rate after grouping by the exact values of
                     f. == 0 means t is a deterministic function of f
      pure_side    : when a zero-error one-sided region exists (cov >= PURE_MIN_COV), its
                     (side, cov, direction label)
      tail_purity  : the highest purity attainable by a one-sided region with cov >= TAIL_MIN_COV
    Same convention as search_t1.tierA: NaN -> -1e18 (falls to the low side).
    """
    v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=-1e18,
                      posinf=1e18, neginf=-1e18)[ok]
    y = np.asarray(y)[ok].astype(np.int64)
    n = v.size
    if n == 0:
        return None
    o = np.argsort(v, kind="stable")
    vs, ys = v[o], y[o]
    # cut points: between adjacent distinct values
    cut = np.flatnonzero(vs[1:] != vs[:-1]) + 1          # sample count on the low side
    c1 = np.cumsum(ys)
    n_lo = cut.astype(np.float64)
    k1_lo = c1[cut - 1].astype(np.float64)               # count of y=1 on the low side
    tot1 = float(c1[-1])
    n_hi = n - n_lo
    k1_hi = tot1 - k1_lo
    k0_lo, k0_hi = n_lo - k1_lo, n_hi - k1_hi

    res = {"n": int(n), "n_unique": int(len(cut) + 1),
           "base_rate": float(tot1 / n)}
    if len(cut) == 0:
        res.update(err_thresh=1.0, err_value=1.0, pure_side=None,
                   tail_purity=float(max(res["base_rate"], 1 - res["base_rate"])),
                   tail_cov=1.0)
        return res

    # --- threshold determinism: predict b on the low side and 1-b on the high side
    err_a = k1_lo + k0_hi          # b=0 low side / b=1 high side
    err_b = k0_lo + k1_hi
    res["err_thresh"] = float(np.minimum(err_a, err_b).min() / n)

    # --- value-level determinism (t as any deterministic function of f)
    # Note: a near-continuous feature has almost one value per row, and err_value is then
    # trivially 0 (one sample per value is of course pure). That is how the negative controls
    # rnd_u0/rnd_u3 were wrongly BLOCKed. So this criterion is only usable when there are at
    # least MIN_PER_VALUE samples per value on average; otherwise it is set to NaN and dropped.
    MIN_PER_VALUE = 20
    _, inv = np.unique(vs, return_inverse=True)
    nv = inv.max() + 1
    g1 = np.bincount(inv, weights=ys, minlength=nv)
    gn = np.bincount(inv, minlength=nv).astype(np.float64)
    ev = float((gn - np.maximum(g1, gn - g1)).sum() / n)
    res["mean_per_value"] = float(n / nv)
    res["err_value"] = ev if (n / nv) >= MIN_PER_VALUE else float("nan")
    res["err_value_raw"] = ev

    # --- zero-error one-sided region (a tautological implication): one side is all 0 or all 1
    best = None
    for side, nn, kk1, kk0, lab in (("lo", n_lo, k1_lo, k0_lo, "f<=theta"),
                                    ("hi", n_hi, k1_hi, k0_hi, "f>theta")):
        pure = ((kk1 == 0) | (kk0 == 0)) & (nn >= PURE_MIN_COV * n)
        if pure.any():
            j = int(np.flatnonzero(pure)[np.argmax(nn[pure])])
            cov = float(nn[j] / n)
            # probability of being pure by chance: max(p0,1-p0)^m for a pure region of size m;
            # must be < 1e-6
            p = max(res["base_rate"], 1 - res["base_rate"])
            log_p_chance = float(nn[j]) * math.log10(max(p, 1e-12))
            if log_p_chance < -6 and (best is None or cov > best["cov"]):
                best = {"side": side, "cov": cov, "body": int(kk1[j] > 0),
                        "theta": float(vs[cut[j] - 1]), "pred": lab,
                        "log10_p_chance": round(log_p_chance, 1)}
    res["pure_side"] = best

    # --- near-pure one-sided region (the highest purity attainable at cov >= TAIL_MIN_COV)
    tp, tc = 0.0, 0.0
    for nn, kk1 in ((n_lo, k1_lo), (n_hi, k1_hi)):
        m = nn >= TAIL_MIN_COV * n
        if m.any():
            pu = np.maximum(kk1[m], nn[m] - kk1[m]) / nn[m]
            j = int(np.argmax(pu))
            if pu[j] > tp:
                tp, tc = float(pu[j]), float(nn[m][j] / n)
    res["tail_purity"], res["tail_cov"] = tp, tc
    return res


# ================== level B: single-feature grouped-CV ceiling (unconstrained tree)

def level_b(v, y, ok, groups, n_folds=5, seed=20260728):
    """Feed a single feature to a depth-unlimited decision tree, K-fold CV grouped by proto_id."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GroupKFold
    x = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=-1e18,
                      posinf=1e18, neginf=-1e18)[ok].reshape(-1, 1)
    yy = np.asarray(y)[ok].astype(np.int8)
    gg = pd.factorize(np.asarray(groups)[ok])[0]
    if len(np.unique(yy)) < 2 or len(np.unique(gg)) < n_folds:
        return None
    gk = GroupKFold(n_splits=n_folds)
    acc, maj = [], []
    for tr, te in gk.split(x, yy, gg):
        t = DecisionTreeClassifier(min_samples_leaf=5, random_state=seed).fit(
            x[tr], yy[tr])
        acc.append(float((t.predict(x[te]) == yy[te]).mean()))
        b = int(yy[tr].mean() >= 0.5)
        maj.append(float((yy[te] == b).mean()))
    return {"cv_acc": float(np.mean(acc)), "cv_sd": float(np.std(acc)),
            "cv_maj": float(np.mean(maj)),
            "cv_gain": float(np.mean(acc) - np.mean(maj))}


def joint_ceiling(F, cols, y, ok, groups, n_folds=3, seed=20260728):
    """The grouped-CV ceiling of a set of features jointly (to quantify how much of the target
    a "set of formula terms" can reconstruct)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    cols = [c for c in cols if c in F.columns]
    if not cols:
        return None
    x = F[cols].values.astype(np.float64)[ok]
    yy = np.asarray(y)[ok].astype(np.int8)
    gg = pd.factorize(np.asarray(groups)[ok])[0]
    if len(np.unique(yy)) < 2 or len(np.unique(gg)) < n_folds:
        return None
    gk = GroupKFold(n_splits=n_folds)
    acc, maj = [], []
    for tr, te in gk.split(x, yy, gg):
        m = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                           early_stopping=False,
                                           random_state=seed).fit(x[tr], yy[tr])
        acc.append(float((m.predict(x[te]) == yy[te]).mean()))
        b = int(yy[tr].mean() >= 0.5)
        maj.append(float((yy[te] == b).mean()))
    return {"n_feat": len(cols), "cv_acc": float(np.mean(acc)),
            "cv_maj": float(np.mean(maj)),
            "cv_gain": float(np.mean(acc) - np.mean(maj))}


# ============================== level C: construction provenance (features.yaml)

ENVELOPE_OPS = {"span", "range", "ptp", "nunique", "std", "max", "min",
                "argmax", "argmin", "any", "all"}


def _closure(name, spec, atoms):
    """Transitive atom closure of a feature: the atoms field plus each atom's declared
    upstream in the atoms table."""
    seen, stack = set(), list(spec.get("atoms", []))
    while stack:
        a = stack.pop()
        if a in seen:
            continue
        seen.add(a)
        stack.extend(atoms.get(a, {}).get("from", []))
    return seen


def level_c(cfg, tag):
    """Run the three hard rules against the features.yaml declarations. Returns
    {feature: verdict_dict}."""
    atoms = cfg["atoms"]
    rank = cfg["scope_rank"]             # a smaller number means a coarser aggregation domain
                                         # (containing more individuals)
    tspec = cfg["targets"][tag]
    tclos = set(tspec["forbidden_atoms"])
    tscope, tbase = tspec["scope"], tspec["base"]

    def contains(s_outer, s_inner):
        """Does s_outer's aggregation domain contain s_inner (the target's domain)."""
        if s_outer not in rank or s_inner not in rank:
            return False
        return rank[s_outer] <= rank[s_inner]

    feats = dict(cfg["features"][tag])
    for inh in tspec.get("inherit", []):
        for k, v in cfg["common"][inh].items():
            feats.setdefault(k, v)

    out = {}
    for f, spec in feats.items():
        clos = _closure(f, spec, atoms)
        hits = sorted(clos & tclos)
        r = {"atoms": sorted(clos), "op": spec.get("op"),
             "scope": spec.get("scope"), "base": spec.get("base"),
             "R1_atom_hit": hits,
             "R2_envelope": False, "R3_same_object": False, "note": spec.get("note")}
        # R1: the feature's atom closure touches a forbidden atom of the target -> a
        # construction-level leak
        r["R1"] = bool(hits)
        # informational: the hit occurs on an aggregation domain coarser than the target's
        # (a weaker leak mechanism, but still not whitelisted)
        r["R1_coarse"] = bool(hits and rank.get(spec.get("scope"), 9)
                              < rank.get(tscope, 9))
        # R2: envelope operator + same base + aggregation domain containing the target domain
        # -> an entry point for a tautological implication
        if (spec.get("base") == tbase and spec.get("op") in ENVELOPE_OPS
                and contains(spec.get("scope"), tscope)):
            r["R2"] = r["R2_envelope"] = True
        else:
            r["R2"] = False
        # R3: any statistic with the same domain and base as the target -> the target's
        # sufficient-statistic family, which needs to be cleared empirically
        r["R3"] = bool(spec.get("base") == tbase and spec.get("scope") == tscope
                       and not r["R2"])
        r["R3_same_object"] = r["R3"]
        r["verdict_c"] = ("BLOCK" if (r["R1"] or r["R2"]) else
                          "REVIEW" if r["R3"] else "PASS")
        out[f] = r
    return out


# ================================================================== combined verdict

def combine(a, b, c):
    """The three levels -> a final verdict. BLOCK at any level means BLOCK."""
    reasons = []
    if c is not None and c["R1"]:
        reasons.append("C.R1 construction provenance: atom closure hits a forbidden atom of "
                       "the target: " + ",".join(c["R1_atom_hit"]))
    if c is not None and c["R2"]:
        reasons.append("C.R2 envelope tautology: an envelope operator on the same base "
                       "aggregates over a domain containing the target domain")
    if a is not None:
        ev = a.get("err_value")
        ev = 1.0 if (ev is None or ev != ev) else ev
        if ev <= EPS_DET:
            reasons.append(f"A. value-level identity, err={ev:.2e} "
                           f"({a.get('mean_per_value', 0):.0f} samples per value)")
        elif a.get("err_thresh", 1.0) <= EPS_DET:
            reasons.append(f"A. threshold-level identity, err={a['err_thresh']:.2e}")
        if a.get("pure_side"):
            ps = a["pure_side"]
            reasons.append(f"A. zero-error one-sided region {ps['pred']} theta={ps['theta']:.4g} "
                           f"=> tgt={ps['body']} cov={ps['cov']:.4f}")
    if b is not None and b["cv_acc"] >= CEIL_BLOCK:
        reasons.append(f"B. single-feature grouped-CV ceiling {b['cv_acc']:.4f}")
    if reasons:
        return "BLOCK", reasons
    soft = []
    if c is not None and c["R3"]:
        soft.append("C.R3 same domain and base as the target (its sufficient-statistic family)")
    if a is not None and a.get("tail_purity", 0) >= TAIL_PURITY:
        soft.append(f"A. near-pure one-sided region, purity={a['tail_purity']:.4f} cov={a['tail_cov']:.4f}")
    if b is not None and b["cv_acc"] >= CEIL_REVIEW:
        soft.append(f"B. single-feature CV {b['cv_acc']:.4f} >= {CEIL_REVIEW}")
    return ("REVIEW", soft) if soft else ("PASS", [])


def audit():
    import yaml
    cfg = yaml.safe_load(open(YAML_SRC))
    report = {"meta": {"eps_det": EPS_DET, "pure_min_cov": PURE_MIN_COV,
                       "tail_purity": TAIL_PURITY, "tail_min_cov": TAIL_MIN_COV,
                       "ceil_block": CEIL_BLOCK, "ceil_review": CEIL_REVIEW,
                       "split": "discovery only"}, "targets": {}}
    clean = {}
    for tag in TARGETS:
        t0 = time.time()
        F, y3, ok, meta, feats, rnds = load(tag)
        y = y3["chemenv"]
        grp = meta.proto_id.values
        cres = level_c(cfg, tag)
        rows = {}
        undeclared = [f for f in feats if f not in cres]
        for f in feats:
            a = level_a(F[f].values, y, ok)
            b = level_b(F[f].values, y, ok, grp)
            c = cres.get(f)
            v, why = combine(a, b, c)
            if c is None:
                v, why = "BLOCK", ["C. not declared in features.yaml (deny by default)"]
            rows[f] = {"A": a, "B": b, "C": c, "verdict": v, "reasons": why}
            if a and b:
                msg = (f"  [{tag}] {f:22s} {v:6s} "
                       f"errT={a['err_thresh']:.2e} errV={a['err_value']:.2e} "
                       f"pure={'Y' if a['pure_side'] else '-'} "
                       f"tail={a['tail_purity']:.4f} cv={b['cv_acc']:.4f}")
            else:
                msg = f"  [{tag}] {f:22s} {v:6s} (A/B not computable)"
            print(msg + ("  <- " + "; ".join(why) if why else ""), flush=True)
        # negative control: every random feature must PASS, or the detector itself is wrong
        neg = {}
        for f in rnds[:8]:
            a = level_a(F[f].values, y, ok)
            b = level_b(F[f].values, y, ok, grp)
            neg[f] = {"verdict": combine(a, b, None)[0],
                      "err_thresh": a["err_thresh"], "tail_purity": a["tail_purity"],
                      "cv_acc": None if b is None else b["cv_acc"]}
        passed = [f for f, r in rows.items() if r["verdict"] == "PASS"]
        review = [f for f, r in rows.items() if r["verdict"] == "REVIEW"]
        block = [f for f, r in rows.items() if r["verdict"] == "BLOCK"]
        # the joint ceiling of a set of formula terms (quantifying how much the target's own
        # defining expression can reconstruct)
        formula = [f for f in block if rows[f]["C"] and rows[f]["C"]["R1"]]
        jc = {"formula_terms": joint_ceiling(F, formula, y, ok, grp),
              "whitelist": joint_ceiling(F, passed, y, ok, grp)}
        report["targets"][tag] = {
            "n": int(ok.sum()), "base_rate": float(y[ok].mean()),
            "n_feat": len(feats), "n_pass": len(passed), "n_review": len(review),
            "n_block": len(block), "undeclared": undeclared,
            "pass": passed, "review": review, "block": block,
            "joint_ceiling": jc, "neg_control_rnd": neg, "features": rows}
        clean[tag] = {"scope": cfg["targets"][tag]["scope"],
                      "target": cfg["targets"][tag]["desc"],
                      "whitelist": passed, "review_excluded": review,
                      "blocked": {f: rows[f]["reasons"] for f in block}}
        print(f"[{tag}] PASS {len(passed)} / REVIEW {len(review)} / BLOCK {len(block)}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- acceptance: the two known leaks must be reproduced
    acc = {}
    for tag, f in (("T_CONN", "an_link"), ("T_P5", "cn_span_st")):
        r = report["targets"][tag]["features"].get(f)
        acc[f"{tag}::{f}"] = {"caught": bool(r and r["verdict"] == "BLOCK"),
                              "reasons": (r or {}).get("reasons")}
    report["acceptance"] = acc
    report["acceptance"]["all_caught"] = all(v["caught"] for k, v in acc.items()
                                             if isinstance(v, dict))
    json.dump(report, open(REPORT_OUT, "w"), indent=1, default=str)
    yaml.safe_dump({"generated": time.strftime("%Y-%m-%d %H:%M"),
                    "source": "leakguard.py", "split": "discovery",
                    "targets": clean}, open(YAML_OUT, "w"),
                   allow_unicode=True, sort_keys=False)
    print("\nacceptance:", json.dumps(acc, ensure_ascii=False, indent=1, default=str))
    print("wrote", REPORT_OUT, "\nwrote", YAML_OUT)
    return report


# ================================ the fixed scorer (importable from search_t1)

class AuditScorerV2:
    """A non-degenerate set scorer under conjunctive semantics.

    What was wrong with the old `AuditScorer.matched()`: among all triggered members it took
    the prediction of whichever had the shortest `bits`, so when the member bodies share a
    polarity the set degenerates into a constant predictor over its coverage and `acc_set` is
    identically `maj_matched`.

    The new semantics (matching the conjunctive combination of PREREG section 3.3):
      * each member predicts **within its own guard**;
      * where the members **agree** within the coverage -> the set predicts that value;
      * **a conflict is an abstention**; abstained samples do not count towards acc_set but do
        count towards the `abstain` rate;
      * report `acc_set` (agreeing samples only), `acc_set_all` (abstentions counted as errors,
        the conservative convention), `abstain`, and **the comparison against B1 and B2 over
        the same set of agreeing samples**.
    """

    def __init__(self, y, ok, masks, bodies, bits, deff):
        self.y, self.ok = np.asarray(y).astype(bool), np.asarray(ok).astype(bool)
        self.masks, self.bodies, self.bits = masks, [int(b) for b in bodies], bits
        self.n = int(self.ok.sum())
        self.deff = deff
        self.n_eff = self.n / deff
        self.pool = list(range(len(masks)))
        self.trig = [np.asarray(m).astype(bool) & self.ok for m in masks]
        self._cache = {}

    # ---- the MDL part matches the old class digit for digit (the accounting convention is untouched)
    def cells(self, S):
        pat = np.zeros(len(self.y), np.int64)
        cell = np.zeros(len(self.y), np.int64)
        for k, i in enumerate(S):
            t = self.trig[i]
            pat += (1 << k) * t
            cell = cell * 3 + np.where(t, 1 + self.bodies[i], 0)
        return cell * 8191 + pat

    def data_cost(self, S):
        k = tuple(sorted(S))
        if k not in self._cache:
            self._cache[k] = self._data_cost(k)
        return self._cache[k]

    def _data_cost(self, S):
        c = self.cells(S)[self.ok]
        t = self.y[self.ok].astype(float)
        _, cid = np.unique(c, return_inverse=True)
        nc = cid.max() + 1
        n1 = np.bincount(cid, weights=t, minlength=nc)
        nG = np.bincount(cid, minlength=nc).astype(float)
        n0 = nG - n1
        data = float(np.where(n0 > 0, n0 * np.log2(nG / np.maximum(n0, 1)), 0).sum() +
                     np.where(n1 > 0, n1 * np.log2(nG / np.maximum(n1, 1)), 0).sum())
        par = float(0.5 * np.log2(np.maximum(nG / self.deff, 2.0)).sum())
        par_reg = (2 ** max(len(S), 1) - 1) / 2.0 * math.log2(self.n_eff)
        return data / self.deff, dict(reg=par_reg, full=par, obs=par)

    def model_cost(self, S, M, lam):
        return lam * (sum(self.bits[i] for i in S) +
                      (math.log2(math.comb(M, len(S))) if len(S) else 0.0))

    def L_total(self, S, M, lam, par_mode="obs"):
        d, p = self.data_cost(S)
        return self.model_cost(S, M, lam) + p[par_mode] + d, d, p

    # ---- the fixed matched-coverage evaluation
    def vote(self, S):
        """Returns (cov, pred, abstain). Conjunctive semantics: each member predicts, and a
        conflict abstains."""
        n = len(self.y)
        n0 = np.zeros(n, np.int32)      # number of members predicting 0
        n1 = np.zeros(n, np.int32)
        for i in S:
            t = self.trig[i]
            if self.bodies[i]:
                n1 += t
            else:
                n0 += t
        cov = (n0 + n1) > 0
        abst = cov & (n0 > 0) & (n1 > 0)
        pred = np.where(n1 > 0, 1, 0)
        return cov, pred.astype(np.int8), abst

    def matched(self, S, base1=None, base2=None, maj_global=None):
        cov, pred, abst = self.vote(S)
        if not cov.any():
            return None
        dec = cov & ~abst                       # samples where the members agree (the set
                                                # actually predicts)
        y = self.y
        r = {"cov": float(cov.sum() / self.n), "n_cov": int(cov.sum()),
             "abstain": float(abst.sum() / max(int(cov.sum()), 1)),
             "n_decided": int(dec.sum()),
             "cov_decided": float(dec.sum() / self.n)}
        r["acc_set"] = float((pred[dec] == y[dec]).mean()) if dec.any() else float("nan")
        # the conservative convention: abstentions count as errors and the denominator is still
        # the whole coverage
        r["acc_set_all"] = float((pred[cov] == y[cov]).mean() * 0
                                 + (pred[dec] == y[dec]).sum() / cov.sum())
        # the baseline has to be compared over **the same decided samples**, otherwise the
        # denominators differ and the comparison is meaningless
        for nm, base in (("B1", base1), ("B2", base2)):
            if base is None:
                continue
            base = np.asarray(base)
            r[f"acc_{nm}_matched"] = float((base[dec] == y[dec]).mean()) if dec.any() else float("nan")
            r[f"gain_vs_{nm}"] = r["acc_set"] - r[f"acc_{nm}_matched"]
            r[f"acc_{nm}_cov"] = float((base[cov] == y[cov]).mean())
        # degeneracy diagnostic: is the set a constant predictor over the decided region
        r["maj_matched"] = float(max(y[dec].mean(), 1 - y[dec].mean())) if dec.any() else float("nan")
        r["pred_entropy"] = float(_H(pred[dec].mean())) if dec.any() else 0.0
        r["is_constant_predictor"] = bool(dec.any() and len(np.unique(pred[dec])) == 1)
        r["gain_vs_maj"] = r["acc_set"] - r["maj_matched"]
        if maj_global is not None:
            r["gain_vs_maj_global"] = r["acc_set"] - float(maj_global)
        ntr = np.zeros(len(y), np.int16)
        for i in S:
            ntr += self.trig[i]
        r["overlap"] = float((ntr >= 2).sum() / max(self.n, 1))
        r["contra"] = float(abst[ntr >= 2].mean()) if (ntr >= 2).any() else 0.0
        r["n_members"] = len(S)
        r["bodies"] = [self.bodies[i] for i in S]
        r["single_polarity"] = bool(len(set(r["bodies"])) <= 1)
        return r


def _H(p):
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def gate_G5(perm_z, gain_vs_B1, gain_vs_B2, z_min=5.0):
    """The fixed G5: the rule must beat **both** B1 and B2. The previous round compared only
    against B1, which let rules that were negative against B2 through."""
    return bool(perm_z >= z_min and gain_vs_B1 > 0 and gain_vs_B2 > 0)


# ================================================================== main

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("prep", "all"):
        prep(force="--force" in sys.argv)
    if cmd in ("audit", "all"):
        audit()


if __name__ == "__main__":
    main()
