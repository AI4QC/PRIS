#!/usr/bin/env python3
"""Source data for Supplementary Figs. S18-S19: how firmly the six PSS terms are
determined, and how they correlate with each other.

The six terms are the ones frozen in outputs/20260814_f3_synth/F3_frozen.json; they
are held fixed throughout.  Nothing here re-runs the feature search and nothing here
touches the sealed held-out split (PREREG-F3 Section 2): every number is computed on
the 919 development composition groups that the score was fitted on.

Two questions, two outputs:
  s8_pss_coefficient_stability.csv  refit the same six terms on random subsets of the
                                    development structures (50-90%, 200 draws each)
  s8_pss_bootstrap.csv              cluster bootstrap over composition groups on all
                                    of the development data (B=2000, the interval
                                    convention of Note S13)
  s9_pss_term_correlation.json      correlation of the six terms, both across
                                    structures and on the within-composition
                                    differences that the pairwise fit actually sees

Run: python src/pss_stability_analysis.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_rulesets import load  # noqa: E402  (published conventions)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SI_DATA = ROOT / "paper" / "si_data"
FROZEN = json.loads((ROOT / "outputs" / "20260814_f3_synth" / "F3_frozen.json").read_text())
F = os.environ.get("PRIS_FEATURES",
                   "features/")

SEED_SUB, SEED_BOOT = 20260826, 20260728
N_DRAWS, N_BOOT = 200, 2000
FRACTIONS = (0.5, 0.6, 0.7, 0.8, 0.9)

# display order of equation (1); the archived feature keys stay as they are
TERMS = ["vol_per_atom", "madz_mean", "bv_rel_mean",
         "wyckoff_econ_001", "poly_deg_max", "frac_isolated"]
LABEL = {"vol_per_atom": "v_atom", "madz_mean": "M_z", "bv_rel_mean": "D_BV",
         "wyckoff_econ_001": "eta_site", "poly_deg_max": "k_max",
         "frac_isolated": "f_iso"}


# ─────────────────────────────────────────────────────────────────────────────
# development set, frozen preprocessing, and every within-composition pair
# ─────────────────────────────────────────────────────────────────────────────
def is_dev(rk: str) -> bool:
    return zlib.crc32(f"{rk}|synthsplit20260814".encode()) % 10 < 6


def development_set():
    d = load()  # keep-filtered synth_rank; merges synth_rank_aug when it is present
    aug = pd.read_parquet(F + "synth_rank_aug.parquet")
    if not set(aug.columns) - {"mp_id"} <= set(d.columns):
        d = d.merge(aug, on="mp_id", how="left")
    return d[d.rk.map(is_dev)].reset_index(drop=True)


def fit_logistic(X, w, C=1e6, tol=1e-11, iters=100):
    """Weighted, intercept-free, antisymmetric logistic — the model class of Note S15.

    Newton solve of the objective sklearn minimises for the mirrored (X,1)/(-X,0)
    encoding used by src/f3_fit.py; agrees with that path to better than 0.3%.
    """
    b = np.zeros(X.shape[1])
    lam = 1.0 / (2.0 * C)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        g = -(X * (w * (1 - p))[:, None]).sum(0) + lam * b
        s = w * p * (1 - p)
        H = (X * s[:, None]).T @ X + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b -= step
        if np.max(np.abs(step)) < tol:
            break
    return b


def main() -> int:
    dev = development_set()
    med = pd.Series(FROZEN["impute_median"])
    mu, sd = pd.Series(FROZEN["mu"]), pd.Series(FROZEN["sd"])
    beta0 = np.array([dict(zip(FROZEN["features"], FROZEN["beta"]))[c] for c in TERMS])

    Z = ((dev[TERMS].fillna(med) - mu[TERMS]) / sd[TERMS]).values
    rk, y = dev.rk.values, dev.synth.values.astype(bool)

    pos, neg, gname = [], [], []
    for g in pd.unique(rk):
        ii = np.where(rk == g)[0]
        a, b = ii[y[ii]], ii[~y[ii]]
        if not len(a) or not len(b):
            continue
        pos.append(np.repeat(a, len(b)))
        neg.append(np.tile(b, len(a)))
        gname.append(np.full(len(a) * len(b), g))
    p_idx, q_idx = np.concatenate(pos), np.concatenate(neg)
    gcode = pd.factorize(np.concatenate(gname))[0]
    n_groups = int(gcode.max() + 1)
    print(f"development set: {len(dev)} structures, {n_groups} groups, "
          f"{len(p_idx)} pairs")

    def gweight(keep):
        gc = gcode[keep]
        return 1.0 / np.bincount(gc, minlength=n_groups)[gc]

    def group_equal_acc(score, keep):
        dv = score[p_idx[keep]] - score[q_idx[keep]]
        gc = gcode[keep]
        m = np.isfinite(dv) & (dv != 0)
        won = np.bincount(gc[m], weights=(dv[m] > 0).astype(float), minlength=n_groups)
        seen = np.bincount(gc[m], minlength=n_groups)
        k = seen > 0
        return float((won[k] / seen[k]).mean())

    everything = np.ones(len(p_idx), bool)
    s0 = Z @ beta0
    acc0 = group_equal_acc(s0, everything)
    assert abs(acc0 - FROZEN["dev_acc"]) < 1e-9, (acc0, FROZEN["dev_acc"])
    print(f"reproduced the frozen development accuracy: {acc0:.6f}")

    # ── S18a,c: refit the six fixed terms on random subsets of the structures ──
    rng = np.random.default_rng(SEED_SUB)
    rows = []
    for frac in FRACTIONS:
        for draw in range(N_DRAWS):
            take = rng.choice(len(dev), int(round(frac * len(dev))), replace=False)
            mask = np.zeros(len(dev), bool)
            mask[take] = True
            keep = mask[p_idx] & mask[q_idx]
            b = fit_logistic(Z[p_idx[keep]] - Z[q_idx[keep]], gweight(keep))
            s = Z @ b
            rows.append(dict(
                fraction=frac, draw=draw, n_structures=int(mask.sum()),
                n_groups=int(len(np.unique(gcode[keep]))), n_pairs=int(keep.sum()),
                cosine=float(b @ beta0 / np.linalg.norm(b) / np.linalg.norm(beta0)),
                spearman_vs_published=float(spearmanr(s, s0).statistic),
                dev_accuracy=group_equal_acc(s, everything),
                **{LABEL[c]: float(v) for c, v in zip(TERMS, b)}))
        print(f"  subsample {int(frac * 100)}%: {N_DRAWS} refits")
    sub = pd.DataFrame(rows)
    sub.to_csv(SI_DATA / "s8_pss_coefficient_stability.csv", index=False)

    # ── S18b: cluster bootstrap over composition groups, all development data ──
    order = np.argsort(gcode, kind="stable")
    starts = np.searchsorted(gcode[order], np.arange(n_groups))
    ends = np.searchsorted(gcode[order], np.arange(n_groups), side="right")
    per_group = [order[a:b] for a, b in zip(starts, ends)]
    DIFF = Z[p_idx] - Z[q_idx]

    def fit_groups(sel):
        idx = np.concatenate([per_group[k] for k in sel])
        w = np.concatenate([np.full(len(per_group[k]), 1.0 / len(per_group[k]))
                            for k in sel])
        return fit_logistic(DIFF[idx], w)

    rngb = np.random.default_rng(SEED_BOOT)
    boot = np.array([fit_groups(rngb.integers(0, n_groups, n_groups))
                     for _ in range(N_BOOT)])
    print(f"  cluster bootstrap: {N_BOOT} resamples of {n_groups} groups")
    bt = pd.DataFrame({
        "term": [LABEL[c] for c in TERMS],
        "published": beta0,
        "boot_mean": boot.mean(0),
        "cluster_se": boot.std(0, ddof=1),
        "ci_lo": np.percentile(boot, 2.5, axis=0),
        "ci_hi": np.percentile(boot, 97.5, axis=0),
        "abs_z": np.abs(beta0) / boot.std(0, ddof=1),
        "sign_agreement": [(np.sign(boot[:, k]) == np.sign(beta0[k])).mean()
                           for k in range(len(TERMS))],
    })
    bt.to_csv(SI_DATA / "s8_pss_bootstrap.csv", index=False)
    print(bt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ── S19: correlation of the six terms ──
    labels = [LABEL[c] for c in TERMS]
    weight_all = gweight(everything)
    wn = weight_all / weight_all.sum()
    centred = DIFF - wn @ DIFF
    cov = (centred * wn[:, None]).T @ centred
    s = np.sqrt(np.diag(cov))
    R_pair = cov / np.outer(s, s)
    R_struct = spearmanr(Z).statistic
    R_struct_p = np.corrcoef(Z.T)
    vif = np.diag(np.linalg.inv(R_pair))
    ev = np.linalg.eigvalsh(R_pair)
    corr = dict(
        terms=labels,
        spearman_structures=R_struct.tolist(),
        pearson_structures=R_struct_p.tolist(),
        pearson_pair_design=R_pair.tolist(),
        vif_pair_design=vif.tolist(),
        condition_number=float(ev[-1] / ev[0]),
        eigenvalues=ev.tolist(),
        bootstrap_coefficient_correlation=np.corrcoef(boot.T).tolist(),
        n_structures=int(len(dev)), n_groups=n_groups, n_pairs=int(len(p_idx)),
        complete_case_fraction=float(dev[TERMS].notna().all(axis=1).mean()),
    )
    (SI_DATA / "s9_pss_term_correlation.json").write_text(json.dumps(corr, indent=1))
    off = R_pair[np.triu_indices(len(TERMS), 1)]
    print(f"\npair design: max |r| {np.abs(off).max():.3f}, max VIF {vif.max():.2f}, "
          f"condition number {ev[-1] / ev[0]:.2f}")
    print("wrote s8_pss_coefficient_stability.csv, s8_pss_bootstrap.csv, "
          "s9_pss_term_correlation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
