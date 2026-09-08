#!/usr/bin/env python3
"""Apply a law set to an arbitrary structure -- the entry point for users.

Usage:
    python src/apply_rules.py POSCAR CONTCAR foo.cif ...        # the five by default
    python src/apply_rules.py --set core4 *.cif                 # the trusted core of four
    python src/apply_rules.py --set single *.cif                # the single core law only
    python src/apply_rules.py --verbose foo.cif                 # verdict and measured value, law by law

Law sets (all searched on discovery, verified on calibration, lockbox untouched):

Performance on the held-out calibration set (5,297 real / 3,612 perturbed), from
paper/FACTS.md section 16:

  single  the single core law   satisfaction 0.9919 / exclusion 0.2890
                                (false-positive gap on LeMat DFT-relaxed: -0.2 pt)
  core4   the trusted core      satisfaction 0.9579 / exclusion 0.6121
  five    the recommended five  satisfaction 0.9171 / exclusion 0.7004
          (default)

For reference: Pauling's rules 2-5 are jointly satisfied at 0.0651.

The thresholds are the published values 0.735 / 0.804 laid down in FACTS section 16.
An earlier version had 0.7353 / 0.8044 here, which disagreed with the numbers printed
in the rule descriptions and, against FACTS, shifted the calibration satisfaction rate
by 0.0019.

Which set to use:
  - rechecking existing experimental or computed structures -> core4
  - screening generative-model output (failure mode unknown) -> five (the fifth law
                                    targets charge placed on the wrong site, a class
                                    the bond-length criteria cannot see at all)
  - just the single most robust law                          -> single

**Valences are inferred from the composition's formal charges (guess_oxi), not from
BVAnalyzer** -- the latter back-solves valence from bond lengths and then tests laws
about bond valence, which is circular.

# Scope

When integer valences cannot be determined, it falls back automatically to
**non-integer mean valences** (e.g. Fe^2.67+ in Fe3O4): anions take their common
valence and the remaining charge is distributed over the variable-valence cations by
stoichiometry.

Measured (calibration, 737 entries):
  integer valences only     coverage 64.5%
  with the non-integer      coverage **80.9%**, and the four-law conjunction is
  fallback                  satisfied at 0.9256 on the new population
                            (0.9577 on the integer population; with 121 samples the
                            95% CI is about +/-4.7 pt, so the difference is not
                            significant)

**Not one threshold had to change to carry them over** -- because they are quantiles of
physical quantities, not fitted parameters.

About 19% still cannot be judged (several anions, complex molecular groups, or no
variable-valence element and no integer combination that balances). Those are not
"implausible", they are structures this method **cannot judge**. **Do not read
"skipped" as "passed".**
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discriminate import guess_oxi          # noqa: E402
from phys_law import phys_feats             # noqa: E402
from elec_feat import elec_feats            # noqa: E402
from geom_feat import geom_feats            # noqa: E402
from t0_guard import parse                  # noqa: E402

# (description, feature, direction, threshold, premise or None)
R_SINGLE = [("shortest cation-anion bond >= 0.735 x (sum of Shannon radii)",
             "bl_min", "lo", 0.735, None)]

R_CORE4 = [
    ("shortest cation-anion bond >= 0.804 x (sum of Shannon radii)",
     "bl_min", "lo", 0.804, None),
    ("if mean anion CN <= 3.333, then mean bond-length ratio <= 1.081",
     "bl_mean", "hi", 1.081, ("cn_an_mean", "lo", 3.333)),
    ("range of site Madelung energy per unit charge <= 31.45 eV",
     "madz_range", "hi", 31.45, None),
    ("highest site Madelung energy <= 15.17 eV", "mad_max", "hi", 15.17, None),
]

R_FIVE = R_CORE4 + [
    ("if ionic-character fraction fi > 0.55, then no like-charge bonds",
     "frac_like_bonds", "hi", 1e-4, ("fi", "hi", 0.55)),
]

SETS = {"single": R_SINGLE, "core4": R_CORE4, "five": R_FIVE}
ANI = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As", "H", "C"}


def ionicity(st, X):
    """Pauling ionic-character fraction fi = 1 - exp(-0.25 dchi^2). Composition only."""
    c = parse(st.composition.reduced_formula.replace(" ", ""))
    cand = [e for e in c if e in ANI and e in X and np.isfinite(X[e])]
    if not cand:
        return np.nan
    an = max(cand, key=lambda e: X[e])
    cats = {e: n for e, n in c.items()
            if e != an and e in X and np.isfinite(X[e])}
    if not cats:
        return np.nan
    d = sum(n * (X[an] - X[e]) for e, n in cats.items()) / sum(cats.values())
    return float(1 - np.exp(-0.25 * d * d))


COMMON = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Ag": 1, "Mg": 2, "Ca": 2,
          "Sr": 2, "Ba": 2, "Zn": 2, "Cd": 2, "Al": 3, "Ga": 3, "In": 3, "Sc": 3,
          "Y": 3, "La": 3, "Si": 4, "Ge": 4, "Zr": 4, "Hf": 4, "B": 3, "P": 5}
ANI_Q = {"O": -2, "S": -2, "Se": -2, "Te": -2, "F": -1, "Cl": -1,
         "Br": -1, "I": -1, "N": -3}


def frac_oxi(st):
    """Non-integer mean-valence fallback: anions take their common valence and the
    remaining charge goes to the variable-valence cations.

    About seven in ten of the structures with no integer valence solution are mixed
    valence (Fe^2.67+ in Fe3O4 and the like). Ewald supports fractional charges
    natively, and the Shannon radius lookup only needs the rounded value.
    Measured: coverage 64.5% -> 80.9%, and **not one threshold has to change**
    (satisfaction 0.9256 on the new population).
    """
    from collections import Counter
    syms = [s.specie.symbol for s in st]
    ans = [i for i, e in enumerate(syms) if e in ANI_Q]
    cats = [i for i, e in enumerate(syms) if e not in ANI_Q]
    if not ans or not cats:
        return None
    qa = sum(ANI_Q[syms[i]] for i in ans)
    cc = Counter(syms[i] for i in cats)
    val = [0.0] * len(st)
    for i in ans:
        val[i] = ANI_Q[syms[i]]
    if len(cc) == 1:
        q = -qa / len(cats)
        for i in cats:
            val[i] = q
        return val
    var = [e for e in cc if e not in COMMON]
    if len(var) != 1:          # with two or more variable-valence elements the
                               # assignment is not unique, so do not guess
        return None
    qf = sum(COMMON[e] * cc[e] for e in cc if e in COMMON)
    qv = (-qa - qf) / cc[var[0]]
    if not (0 < qv <= 8):
        return None
    for i in cats:
        val[i] = COMMON.get(syms[i], qv)
    return val


def features(st):
    """Compute every quantity the law sets need. Returns (dict, reason for failure)."""
    val, ok = guess_oxi(st)
    if not ok:
        val = frac_oxi(st)     # fall back to non-integer mean valences
        if val is None:
            return None, ("no charge-balanced valences could be assigned "
                          "(both the integer and the non-integer fallback failed)")
    f, errors = {}, []
    for fn in (phys_feats, elec_feats, geom_feats):
        try:
            g = fn(st, val)
            if g:
                f.update(g)
        except Exception as exc:
            errors.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
    # mean anion coordination number (used as a premise), same convention as criteria()
    try:
        from discriminate import criteria
        c = criteria(st, val)
        if c:
            f.update(c)
    except Exception as exc:
        errors.append(f"criteria: {type(exc).__name__}: {exc}")

    # fi depends only on composition, so take electronegativities straight from the
    # pymatgen element table. The public entry point should not depend on
    # _elem_props.parquet from the multi-GB external feature store for this one value.
    try:
        from pymatgen.core.periodic_table import Element
        comp = parse(st.composition.reduced_formula.replace(" ", ""))
        X = {}
        for symbol in comp:
            value = Element(symbol).X
            if value is not None and np.isfinite(value):
                X[symbol] = float(value)
        f["fi"] = ionicity(st, X)
    except Exception as exc:
        errors.append(f"ionicity: {type(exc).__name__}: {exc}")
    return f, "; ".join(errors) if errors else None


def judge(f, rules, verbose=False):
    """Judge law by law; returns True / False / None (not enough information).

    One computed violation is enough to return False. If there is no violation but
    some applicability or target quantity cannot be computed, this must return None
    -- "unknown" must never be reported as "plausible".
    """
    lines, failed, unknown = [], False, False
    for desc, col, side, th, g in rules:
        v = f.get(col, np.nan)
        applies = True
        if g:
            gc, gs, gt = g
            gv = f.get(gc, np.nan)
            if not np.isfinite(gv):
                unknown = True
                lines.append(("?", desc,
                              f"premise {gc} cannot be computed, so applicability is undecided"))
                continue
            else:
                applies = (gv > gt) if gs == "hi" else (gv <= gt)
        if not applies:
            lines.append(("-", desc,
                          f"premise not met ({g[0]}={f.get(g[0], float('nan')):.3f})"))
            continue
        if not np.isfinite(v):
            unknown = True
            lines.append(("?", desc, f"{col} cannot be computed, so this law is undecided"))
            continue
        ok = bool((v <= th) if side == "hi" else (v >= th))
        failed |= not ok
        lines.append(("✓" if ok else "✗", desc, f"{col} = {v:.4f}"))
    if verbose:
        for mark, desc, detail in lines:
            print(f"    {mark} {desc}")
            print(f"        {detail}")
    if failed:
        return False
    if unknown:
        return None
    return True


# The compact F2 formula over the experimental domain (7 terms). The z-scores use the
# whole-sample mean and standard deviation from real_all plus each feature table, which
# differs slightly from the within-fold standardisation used when fitting -- **S is only
# comparable between candidates of the same composition; its absolute value is meaningless**.
F2 = [("min_opp_frac", -0.3222), ("econ_mean", +0.1491), ("mef_mean", +0.1369),
      ("angvar_mean", +0.0599), ("econ_max", +0.0390), ("dist_rsd", -0.0300),
      ("p5_n_distinct", +0.0016)]
_STAT = {}


def f2_score(f):
    """The F2 score. Smaller S is more stable; returns nan if any term is missing."""
    import os
    import pandas as pd
    if not _STAT:
        base = os.environ.get("PRIS_FEATURES",
                              "features/")
        frames = []
        for fn in ("real_all.parquet", "phys_real.parquet",
                   "elec_real.parquet", "geom_real.parquet"):
            if os.path.exists(base + fn):
                frames.append(pd.read_parquet(base + fn))
        for fr in frames:
            for c, _ in F2:
                if c in fr.columns and c not in _STAT:
                    v = fr[c].dropna()
                    _STAT[c] = (float(v.mean()), float(v.std()) or 1.0)
    s = 0.0
    for c, w in F2:
        v = f.get(c, np.nan)
        if c not in _STAT or not np.isfinite(v):
            return np.nan
        mu, sd = _STAT[c]
        s += w * (v - mu) / sd
    return float(s)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Judge whether an ionic crystal structure is plausible, using the PRIS laws")
    ap.add_argument("files", nargs="+",
                    help="structure files (CIF / POSCAR / anything pymatgen can read)")
    ap.add_argument("--set", choices=list(SETS), default="five",
                    help="law set: single / core4 / five (default)")
    ap.add_argument("--verbose", action="store_true",
                    help="give the verdict and the measured value law by law")
    ap.add_argument("--formula", action="store_true",
                    help="also report the F2 score S (**only comparable between candidates "
                         "of the same composition**; smaller S is more stable)")
    a = ap.parse_args()
    rules = SETS[a.set]
    print(f"Law set [{a.set}], {len(rules)} laws\n")

    from pymatgen.core import Structure
    npass = nfail = nskip = 0
    for path in a.files:
        try:
            st = Structure.from_file(path)
        except Exception as e:
            print(f"[skip] {path}: could not read the structure ({e})")
            nskip += 1
            continue
        f, err = features(st)
        if f is None:
            print(f"[skip] {path}: {err}")
            nskip += 1
            continue
        verdict = judge(f, rules, a.verbose)
        if err and a.verbose:
            print(f"    ! some features failed to compute: {err}")
        if verdict is None:
            detail = f" ({err})" if err else ""
            print(f"[skip] {path}: the features the laws need are incomplete, "
                  f"so no verdict{detail}")
            nskip += 1
            continue
        extra = ""
        if a.formula:
            sv = f2_score(f)
            extra = f"   S = {sv:+.3f}" if np.isfinite(sv) else "   S = not computable"
        print(f"[{'plausible' if verdict else 'implausible'}] {path}  "
              f"({st.composition.reduced_formula}, {len(st)} atoms){extra}")
        npass += verdict
        nfail += (not verdict)
    print(f"\nplausible {npass} / implausible {nfail} / skipped {nskip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
