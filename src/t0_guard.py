#!/usr/bin/env python3
"""Pure-composition (T0) quantities -- not used as laws, but as **premises**.

# Why

An earlier conclusion of this work: T0 quantities carry **identically zero weight** on these
two targets, because a law's exclusion power is computed on damaged samples of the same
composition and the formula pairs within same-composition groups, so compositional terms
cancel exactly on both sides. That is an inevitable consequence of how the targets are
designed.

But "useless as a law" is not the same as "useless". The direct evidence is
`frac_like_bonds` (the fraction of bonds between like-charge ions):

  - on its own it excludes **96%** of S5 (cation-anion swap), the only quantity that works
    on that class
  - but it is satisfied by only **79.3%** of real structures, which does not clear the 0.95
    satisfaction floor
  - and those 20.7% of violations are **not randomly distributed**: phosphides 66.8% /
    tellurides 53.0% / selenides 33.8% / sulfides 28.1% / nitrides 25.9%, against only 16.0%
    for fluorides -- **concentrated in the least ionic chemistries**, where cation-cation
    bonding is genuinely real

So "there must be no like-charge bonds" is a law about **ionic crystals**, and what it lacks
is a premise: **"if the compound is ionic enough"**. And ionicity is exactly a
pure-composition quantity.

# What is computed

  dchi  = the stoichiometry-weighted mean of (electronegativity(anion) - electronegativity(cation))
  fi    = 1 - exp(-0.25 * dchi^2)      the Pauling ionic-character fraction

Both need only the chemical formula and no structure -- so they take identical values on
damaged samples of the same composition. As a law the exclusion power is identically 0; as a
premise they confine the law to the chemistry where it actually holds.
"""
from __future__ import annotations
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
F = os.environ.get("PRIS_FEATURES", "features/")


def parse(f):
    c = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(f)):
        if el:
            c[el] += float(n) if n else 1.0
    return c


def main() -> int:
    prov = pd.read_parquet(F + "provenance.parquet",
                           columns=["source_id", "formula", "anion", "n_elements"])
    ep = pd.read_parquet(F + "_elem_props.parquet")[["element", "X"]]
    X = dict(zip(ep.element, ep.X))

    rows = []
    for t in prov.itertuples():
        an = t.anion
        if not isinstance(an, str) or an not in X or not np.isfinite(X[an]):
            continue
        c = parse(t.formula)
        cats = {e: n for e, n in c.items()
                if e != an and e in X and np.isfinite(X[e])}
        if not cats:
            continue
        tot = sum(cats.values())
        d = sum(n * (X[an] - X[e]) for e, n in cats.items()) / tot
        rows.append({"source_id": t.source_id, "dchi": float(d),
                     "fi": float(1 - np.exp(-0.25 * d * d)),
                     "dchi_min": float(min(X[an] - X[e] for e in cats))})
    d = pd.DataFrame(rows)
    d.to_parquet(F + "t0_guard.parquet", index=False)
    print(f"wrote {len(d):,} rows")
    print(d[["dchi", "fi", "dchi_min"]].describe().round(3).to_string())

    # relation to frac_like_bonds -- checking "the more ionic, the fewer like-charge bonds"
    import os
    if os.path.exists(F + "phys_real.parquet"):
        pr = pd.read_parquet(F + "phys_real.parquet")[["source_id", "frac_like_bonds"]]
        m = d.merge(pr, on="source_id", how="inner").dropna()
        print(f"\n{len(m):,} rows comparable. Binned by ionic-character fraction:")
        m["bin"] = pd.qcut(m.fi, 5, duplicates="drop")
        g = m.groupby("bin").agg(n=("fi", "size"),
                                 fi_median=("fi", "median"),
                                 frac_no_like_bonds=("frac_like_bonds",
                                                     lambda x: float((x == 0).mean())))
        print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
