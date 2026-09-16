# -*- coding: utf-8 -*-
"""Table of Pauling univalent radii -- used only for Pauling's first rule.

Why not Shannon: Shannon radii are tabulated by coordination number, so using them to predict
coordination number is circular (this is exactly what hard gate G7 of the research plan,
section 4.3, is written to block). George 2020 likewise uses "Pauling's univalent radii"
(ref [7] = Pauling, *The Nature of the Chemical Bond*).

The radii come at two levels, and `radius_tier` records which:

* `tier='published'` -- the univalent radii Pauling himself gave (closed-shell ions: noble-gas
  configurations plus 18-electron configurations). This is the main statistical convention.
* `tier='extended'` -- our extrapolation to open-shell d-block ions using Pauling's own
  screening formula R1 = C_n / (Z - S). **This is our extrapolation, not Pauling's table**; it
  is reported only as a sensitivity layer and is never the headline number.

How the screening formula is calibrated (every constant can be rechecked):
    Pauling's 1927 original definition R1 = C_n /(Z - S), with one (C_n, S) per isoelectronic
    series. For each closed shell we solve (C_n, S) from **two published anchors** in that
    series and then check the rest of the series against them.
    For example, the Ne shell from Na+ 0.95 and Mg2+ 0.82 gives S=4.69, C=5.995;
        substituting back gives Al3+ 0.721 (published 0.72), Si4+ 0.644 (0.65),
        P5+ 0.582 (0.59) and O2- 1.811 (published 1.76, off by 2.9%).
        So published values take precedence and the formula is only a fallback.

How the open-shell extrapolation (tier='extended') is constructed:
    Within the d shell of one principal quantum number, the screening constant S is
    **linearly interpolated in the d-electron count** between n_d=0 (the noble-gas anchor) and
    n_d=10 (the 18-electron anchor), and C_n is interpolated the same way. Both ends reproduce
    Pauling's published values exactly.
    3d: S runs from 11.13 (Ar shell) to 18.00 (Ni shell) = 0.687 per electron; C from 10.47 to 10.56.
    4d: S from 28.75 (Kr) to 37.50 (Pd); C from 12.21 to 11.97.
    5d/4f: S from 45.40 (Xe) to 68.58 (Pt), interpolated over all 24 electrons of (n_f + n_d);
    C from 16.22 to 14.28.
"""
from __future__ import annotations

# ------------------------------------------------- published Pauling univalent radii (A)
# Key: (element symbol, integer oxidation state). Values from Pauling, *The Nature of the
# Chemical Bond*, 3rd edition, Table 13-3 (univalent radii) -- the ref [7] George 2020 cites.
PAULING_UNIVALENT = {
    # ---- He shell (2 e)
    ("H", -1): 2.08, ("Li", 1): 0.60, ("Be", 2): 0.44, ("B", 3): 0.35,
    ("C", 4): 0.29, ("N", 5): 0.25,
    # ---- Ne shell (10 e)
    ("N", -3): 2.47, ("O", -2): 1.76, ("F", -1): 1.36,
    ("Na", 1): 0.95, ("Mg", 2): 0.82, ("Al", 3): 0.72, ("Si", 4): 0.65,
    ("P", 5): 0.59, ("S", 6): 0.53, ("Cl", 7): 0.49,
    # ---- Ar shell (18 e)
    ("S", -2): 2.19, ("Cl", -1): 1.81,
    ("K", 1): 1.33, ("Ca", 2): 1.18, ("Sc", 3): 1.06, ("Ti", 4): 0.96,
    ("V", 5): 0.88, ("Cr", 6): 0.81, ("Mn", 7): 0.75,
    # ---- Ni (3d10) 18-electron shell
    ("Cu", 1): 0.96, ("Zn", 2): 0.88, ("Ga", 3): 0.81, ("Ge", 4): 0.76,
    ("As", 5): 0.71, ("Se", 6): 0.66, ("Br", 7): 0.62,
    # ---- Kr shell (36 e)
    ("Se", -2): 2.32, ("Br", -1): 1.95,
    ("Rb", 1): 1.48, ("Sr", 2): 1.32, ("Y", 3): 1.20, ("Zr", 4): 1.09,
    ("Nb", 5): 1.00, ("Mo", 6): 0.93, ("Tc", 7): 0.87,
    # ---- Pd (4d10) 18-electron shell
    ("Ag", 1): 1.26, ("Cd", 2): 1.14, ("In", 3): 1.04, ("Sn", 4): 0.96,
    ("Sb", 5): 0.89, ("Te", 6): 0.82, ("I", 7): 0.77,
    # ---- Xe shell (54 e)
    ("Te", -2): 2.50, ("I", -1): 2.16,
    ("Cs", 1): 1.69, ("Ba", 2): 1.53, ("La", 3): 1.39, ("Ce", 4): 1.27,
    # ---- Pt (5d10) 18-electron shell
    ("Au", 1): 1.37, ("Hg", 2): 1.25, ("Tl", 3): 1.15, ("Pb", 4): 1.06,
    ("Bi", 5): 0.98,
}

# --------------------------------------------- shell calibration constants for extrapolation
# the (C_n, S) anchors at both ends; see the calibration procedure in the module docstring
_SHELL_ANCHOR = {
    "3d": dict(Z0=18, C0=10.47, S0=11.13, C1=10.56, S1=18.00, nmax=10),   # Ar → Ni
    "4d": dict(Z0=36, C0=12.21, S0=28.75, C1=11.97, S1=37.50, nmax=10),   # Kr → Pd
    "5d": dict(Z0=54, C0=16.22, S0=45.40, C1=14.28, S1=68.58, nmax=24),   # Xe -> Pt (including 4f14)
}

_Z = {}  # element -> atomic number (pymatgen is loaded lazily, to avoid the cost when workers
         # import this module repeatedly)


def _atomic_number(el: str) -> int:
    if not _Z:
        from pymatgen.core.periodic_table import Element
        for e in Element:
            _Z[e.symbol] = e.Z
    return _Z[el]


def _extended_radius(el: str, ox: int):
    """Extrapolated univalent radius for an open-shell d-block ion. None means no
    extrapolation is possible."""
    z = _atomic_number(el)
    ne = z - ox                      # the ion's electron count
    if ne < 2:
        return None
    for shell, a in _SHELL_ANCHOR.items():
        k = ne - a["Z0"]             # electrons beyond the noble-gas core
        if 0 < k < a["nmax"]:        # the endpoints (k=0 / k=nmax) are in the published table
            f = k / a["nmax"]
            C = a["C0"] + (a["C1"] - a["C0"]) * f
            S = a["S0"] + (a["S1"] - a["S0"]) * f
            den = z - S
            return C / den if den > 0.5 else None
    return None


def univalent_radius(el: str, ox, allow_extended: bool = False):
    """Returns (radius in A, tier); (None, None) when there is no entry.

    ox is rounded to an integer -- Pauling's table itself has only integer valences.
    Fractional valences (Fe2.5+, say) take the nearest integer, and can be removed afterwards
    through the `frac_ox` flag in the feature store.
    """
    if ox is None or ox != ox:
        return None, None
    o = int(round(float(ox)))
    r = PAULING_UNIVALENT.get((el, o))
    if r is not None:
        return r, "published"
    if allow_extended:
        r = _extended_radius(el, o)
        if r is not None:
            return r, "extended"
    return None, None


# ------------------------------------------------------- radius ratio -> coordination number
# The critical radius ratios of Pauling's first rule (hard-sphere geometry; see George 2020
# Fig. 1a):
#   <0.155 -> CN 2 (linear) / 0.155-0.225 -> CN 3 (trigonal planar) / 0.225-0.414 -> CN 4 (tetrahedral)
#   0.414-0.732 -> CN 6 (octahedral) / 0.732-1.000 -> CN 8 (cubic) / >=1.000 -> CN 12 (cuboctahedral)
RATIO_BOUNDS = [(0.0, 0.155, 2), (0.155, 0.225, 3), (0.225, 0.414, 4),
                (0.414, 0.732, 6), (0.732, 1.000, 8), (1.000, 1e9, 12)]


def predict_cn(ratio):
    if ratio is None or ratio != ratio:
        return None
    for lo, hi, cn in RATIO_BOUNDS:
        if lo <= ratio < hi:
            return cn
    return None


if __name__ == "__main__":
    # self-check: how far the formula misses the published values (for the README / reports)
    import statistics
    errs = []
    for (el, o), r in sorted(PAULING_UNIVALENT.items()):
        rx = _extended_radius(el, o)
        if rx:
            errs.append(abs(rx - r) / r)
    print(f"{len(PAULING_UNIVALENT)} published ions; the formula reproduces {len(errs)}, "
          f"median relative error {statistics.median(errs):.3%}" if errs else "no overlap")
    for t in [(("Fe", 3)), (("Fe", 2)), (("Mn", 2)), (("Cu", 2)), (("W", 6)), (("Ta", 5)), (("Nd", 3))]:
        print(t, univalent_radius(*t, allow_extended=True))
