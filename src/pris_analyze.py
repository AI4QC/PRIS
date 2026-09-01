#!/usr/bin/env python3
"""Run the full PRIS analysis on one or more crystal structures.

    python pris_analyze.py mystructure.cif
    python pris_analyze.py --json POSCAR CONTCAR *.cif
    python pris_analyze.py --quiet *.cif          # one verdict line per file

This is the entry point described in the manuscript: give it a structure file
and it reports, for every one of the eight laws, the measured quantity, the
threshold, the verdict, and the physicochemical mechanism the law tests.  It
then reports the nested law sets (Set 1, Set 1', Set 2, Set 3, Set 4) and the
PRIS-derived synthesis score (PSS).

Nothing is fitted here.  Every threshold and every PSS coefficient is read from
the frozen artefacts committed with this repository, so a verdict produced today
is the verdict the manuscript reports.

Formal charges come from the composition, never from bond lengths: inferring
valence from bond lengths and then testing a bond-valence law on the result
would use the conclusion as the premise.  Integer charge balancing is tried
first; structures with no integer solution fall back to a non-integer mean
valence (Fe3O4's Fe(2.67+), for instance).

About one structure in five cannot be judged at all -- several anions, complex
molecular groups, or no charge assignment.  Those return NO VERDICT.
**"No verdict" does not mean "plausible."**
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from apply_rules import frac_oxi, ionicity            # noqa: E402
from t0_guard import parse                            # noqa: E402
from discriminate import criteria, guess_oxi           # noqa: E402
from elec_feat import elec_feats                       # noqa: E402
from f3_features import _feats as sym_born_feats       # noqa: E402
from phys_law import phys_feats                        # noqa: E402

# --------------------------------------------------------------------------
# The eight laws, frozen.  Thresholds are percentiles of the discovery split,
# not fitted parameters.  Every value here is the one the manuscript reports.
#
#   key       the archived feature name the law is measured on
#   trigger   (key, direction, value) or None -- a law whose trigger is not met
#             is satisfied by definition, which is what confines each law to
#             its domain
# --------------------------------------------------------------------------
TAU_PERMISSIVE, TAU_STRICT = 0.735, 0.804

LAWS = {
    1: dict(name="Law 1", quantity="reduced contact rho",
            key="bl_min", direction="min", threshold=TAU_STRICT,
            mechanism="short-range repulsion",
            statement="rho >= tau"),
    2: dict(name="Law 2", quantity="reduced contact rho",
            key="bl_min", direction="max", threshold=1.05,
            trigger=("fi", "above", 0.50),
            mechanism="ionic contact",
            statement="f_i > 0.50  =>  rho <= 1.05"),
    3: dict(name="Law 3", quantity="mean reduced cation-anion contact",
            key="bl_mean", direction="max", threshold=1.081,
            trigger=("cn_an_mean", "below", 3.333),
            mechanism="packing",
            statement="mean anion CN <= 3.333  =>  mean d/(r+ + r-) <= 1.081"),
    4: dict(name="Law 4", quantity="range of site Madelung energy / valence",
            key="madz_range", direction="max", threshold=31.45,
            mechanism="electrostatic balance",
            statement="range_i of V_M(i)/v_i <= 31.45 eV"),
    5: dict(name="Law 5", quantity="largest site Madelung energy",
            key="mad_max", direction="max", threshold=15.17,
            mechanism="electrostatic balance",
            statement="max_i V_M(i) <= 15.17 eV"),
    6: dict(name="Law 6", quantity="fraction of like-charge bonds",
            key="frac_like_bonds", direction="max", threshold=1e-4,
            trigger=("fi", "above", 0.55),
            mechanism="electrostatic balance",
            statement="f_i > 0.55  =>  no like-charge bonds"),
    7: dict(name="Law 7", quantity="inequivalent sites / sites",
            key="wyckoff_econ_001", direction="max", threshold=2.0 / 3.0,
            mechanism="crystallographic site complexity",
            statement="inequivalent sites / sites <= 2/3"),
    8: dict(name="Law 8", quantity="mean |BV sum - v_i| / v_i",
            key="bv_rel_mean", direction="max", threshold=0.7143,
            mechanism="bond-valence conservation",
            statement="mean of |BV sum - v_i|/v_i <= 0.7143"),
}

# Each set demands a fuller model of an ionic crystal.  Set 1' is the guarded
# two-sided window that stands beside the chain rather than inside it, which is
# why the catalogue holds eight laws while Set 4 applies seven.
SETS = {
    "Set 1":  dict(laws=(1,), tau=TAU_PERMISSIVE, model="hard-sphere contact floor"),
    "Set 1'": dict(laws=(1, 2), tau=TAU_PERMISSIVE, model="two-sided contact window"),
    "Set 2":  dict(laws=(1, 3, 4, 5), tau=TAU_STRICT, model="rigid-ion lattice"),
    "Set 3":  dict(laws=(1, 3, 4, 5, 6), tau=TAU_STRICT, model="ionic network"),
    "Set 4":  dict(laws=(1, 3, 4, 5, 6, 7, 8), tau=TAU_STRICT, model="crystal chemistry"),
}
SET_ORDER = ("Set 1", "Set 1'", "Set 2", "Set 3", "Set 4")

# which sets apply each law -- Law 2 belongs to Set 1' alone, so a Law 2
# failure is not a Set 4 failure and must not read as one
LAW_SETS = {i: [n for n in SET_ORDER if i in SETS[n]["laws"]] for i in LAWS}

# the frozen synthesis score travels with the agent-loop record
PSS_FROZEN = os.path.join(ROOT, "agent_loop", "frozen",
                          "20260814_f3_synth", "F3_frozen.json")


def load_pss():
    """Frozen PSS: the six standardised terms, their weights and the
    development-set mean, spread and impute median.  Returns None when the
    frozen artefact is not shipped alongside this script."""
    try:
        with open(PSS_FROZEN) as fh:
            d = json.load(fh)
    except OSError:
        return None
    return dict(features=d["features"], beta=d["beta"],
                mu=d["mu"], sd=d["sd"], impute=d["impute_median"])


def pauling_ionicity(struct):
    """Pauling's composition-based ionic character, f_i = 1 - exp(-0.25 dchi^2).

    Electronegativities come from pymatgen's element table, so the public entry
    point never needs the multi-gigabyte feature store for this one quantity.
    """
    from pymatgen.core.periodic_table import Element
    try:
        comp = parse(struct.composition.reduced_formula.replace(" ", ""))
        X = {}
        for symbol in comp:
            value = Element(symbol).X
            if value is not None and np.isfinite(value):
                X[symbol] = float(value)
        return ionicity(struct, X)
    except Exception:                           # noqa: BLE001
        return np.nan


def measure(struct):
    """Every quantity the eight laws and PSS are measured on, for one structure.

    Returns (values, charge_note) or (None, reason) when the structure cannot
    be judged.
    """
    val, integral = guess_oxi(struct)
    note = "integer charge balancing"
    if not integral:
        val = frac_oxi(struct)
        note = "non-integer mean valence"
    if val is None:
        return None, "no charge assignment (several anions, or no valid combination)"

    v = {}
    try:
        v.update(phys_feats(struct, val))       # rho, mean contact, like-charge bonds
    except Exception as exc:                    # noqa: BLE001
        return None, f"contact features failed ({exc})"
    try:
        v.update(elec_feats(struct, val))       # Madelung, bond valence, volume
    except Exception as exc:                    # noqa: BLE001
        return None, f"electrostatic features failed ({exc})"
    try:
        v.update(criteria(struct, val))         # coordination, polyhedron topology
    except Exception:                           # noqa: BLE001
        pass                                    # only PSS needs these
    try:
        v.update(sym_born_feats(struct, val))   # symmetry: inequivalent-site fraction
    except Exception:                           # noqa: BLE001
        pass

    v["fi"] = pauling_ionicity(struct)
    return v, note


def judge_law(idx, v, tau=None):
    """Verdict for one law.  Returns (state, value, threshold) where state is
    'pass', 'fail', 'not triggered' or 'no verdict'."""
    law = LAWS[idx]
    thr = tau if (idx == 1 and tau is not None) else law["threshold"]

    trig = law.get("trigger")
    if trig is not None:
        tkey, tdir, tval = trig
        t = v.get(tkey, np.nan)
        if t is None or not np.isfinite(t):
            return "no verdict", np.nan, thr
        if tdir == "above" and not t > tval:
            return "not triggered", t, thr
        if tdir == "below" and not t <= tval:
            return "not triggered", t, thr

    x = v.get(law["key"], np.nan)
    if x is None or not np.isfinite(x):
        return "no verdict", np.nan, thr
    ok = x >= thr if law["direction"] == "min" else x <= thr
    return ("pass" if ok else "fail"), float(x), thr


def pss_score(v, frozen):
    """PSS = sum of standardised term x weight.  Missing terms take the frozen
    development-set median, exactly as the fit did."""
    if frozen is None:
        return None, []
    total, terms = 0.0, []
    for key, beta in zip(frozen["features"], frozen["beta"]):
        raw = v.get(key, np.nan)
        imputed = raw is None or not np.isfinite(raw)
        if imputed:
            raw = frozen["impute"][key]
        z = (raw - frozen["mu"][key]) / frozen["sd"][key]
        total += beta * z
        terms.append(dict(term=key, raw=float(raw), standardised=float(z),
                          weight=float(beta), contribution=float(beta * z),
                          imputed=bool(imputed)))
    return float(total), terms


def analyse(path):
    """Full PRIS analysis of one structure file."""
    from pymatgen.core import Structure

    out = dict(file=path)
    try:
        struct = Structure.from_file(path)
    except Exception as exc:                    # noqa: BLE001
        out.update(verdict="no verdict", reason=f"cannot read structure ({exc})")
        return out

    out["formula"] = struct.composition.reduced_formula
    out["n_sites"] = len(struct)

    v, note = measure(struct)
    if v is None:
        out.update(verdict="no verdict", reason=note)
        return out
    out["charges"] = note
    out["ionic_character"] = (float(v["fi"]) if np.isfinite(v.get("fi", np.nan))
                              else None)

    laws = {}
    for idx in sorted(LAWS):
        state, value, thr = judge_law(idx, v)
        laws[idx] = dict(name=LAWS[idx]["name"], quantity=LAWS[idx]["quantity"],
                         statement=LAWS[idx]["statement"],
                         mechanism=LAWS[idx]["mechanism"],
                         state=state, value=None if not np.isfinite(value) else value,
                         threshold=thr, sets=LAW_SETS[idx])
    out["laws"] = laws

    sets = {}
    for name in SET_ORDER:
        spec = SETS[name]
        states = [judge_law(i, v, tau=spec["tau"])[0] for i in spec["laws"]]
        if "fail" in states:
            verdict = "implausible"
        elif "no verdict" in states:
            verdict = "no verdict"
        else:
            verdict = "plausible"
        broken = [LAWS[i]["name"] for i, s in zip(spec["laws"], states) if s == "fail"]
        missing = [LAWS[i]["name"] for i, s in zip(spec["laws"], states) if s == "no verdict"]
        sets[name] = dict(model=spec["model"], verdict=verdict,
                          unsatisfied=broken, unavailable=missing)
    out["sets"] = sets

    score, terms = pss_score(v, load_pss())
    if score is not None:
        out["pss"] = score
        out["pss_terms"] = terms

    s4 = sets["Set 4"]
    out["verdict"] = s4["verdict"]
    if s4["verdict"] == "implausible":
        out["mechanisms_to_review"] = sorted({
            LAWS[i]["mechanism"] for i in SETS["Set 4"]["laws"]
            if laws[i]["state"] == "fail"})
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
MARK = {"pass": "  ok  ", " fail": " FAIL ", "fail": " FAIL ",
        "not triggered": "  --  ", "no verdict": "  ?   "}


def report(a):
    print(f"\n{a['file']}")
    if "formula" not in a:
        print(f"  NO VERDICT -- {a['reason']}")
        return
    if "laws" not in a:
        print(f"  {a['formula']}, {a['n_sites']} sites")
        print(f"  NO VERDICT -- {a['reason']}")
        return
    print(f"  {a['formula']}, {a['n_sites']} sites, charges from {a['charges']}", end="")
    if a.get("ionic_character") is not None:
        print(f", ionic character f_i = {a['ionic_character']:.3f}")
    else:
        print()

    print()
    print(f"    {'law':<7}{'quantity':<38}{'measured':>10}{'threshold':>11}"
          f"  {'verdict':^8} {'applied in':<16} mechanism")
    for idx in sorted(a["laws"], key=int):
        L = a["laws"][idx]
        val = "     -" if L["value"] is None else f"{L['value']:10.4f}"
        thr = f"{L['threshold']:11.4f}"
        applied = ", ".join(s.replace("Set ", "") for s in L["sets"])
        print(f"    {L['name']:<7}{L['quantity']:<38}{val}{thr}"
              f"  {MARK[L['state']]:^8} {applied:<16} {L['mechanism']}")

    print()
    for name in SET_ORDER:
        S = a["sets"][name]
        line = f"    {name:<7}{S['model']:<28}{S['verdict']}"
        if S["unsatisfied"]:
            line += "  (unsatisfied: " + ", ".join(S["unsatisfied"]) + ")"
        elif S["unavailable"]:
            line += "  (unavailable: " + ", ".join(S["unavailable"]) + ")"
        print(line)

    if "pss" in a:
        print(f"\n    PSS  {a['pss']:+.3f}   "
              "(higher scores sit closer to what has been made)")
        if any(t["imputed"] for t in a["pss_terms"]):
            miss = [t["term"] for t in a["pss_terms"] if t["imputed"]]
            print(f"         imputed from development medians: {', '.join(miss)}")

    print(f"\n    VERDICT  {a['verdict'].upper()}")
    if a.get("mechanisms_to_review"):
        print("    review   " + "; ".join(a["mechanisms_to_review"]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the full PRIS analysis on one or more structures.",
        epilog="Verdicts use Set 4, the full crystal-chemical set. "
               "A structure that cannot be charge-assigned returns NO VERDICT, "
               "which does not mean plausible.")
    ap.add_argument("files", nargs="+",
                    help="structure files (CIF, POSCAR, or anything pymatgen reads)")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON record per structure instead of a table")
    ap.add_argument("--quiet", action="store_true",
                    help="one verdict line per file")
    a = ap.parse_args()

    results = [analyse(p) for p in a.files]

    if a.json:
        json.dump(results, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0

    if a.quiet:
        for r in results:
            broken = ", ".join(r.get("mechanisms_to_review", []))
            print(f"{r['verdict'].upper():<12}{r['file']}"
                  + (f"   {broken}" if broken else ""))
    else:
        for r in results:
            report(r)

    n = {k: sum(r["verdict"] == k for r in results)
         for k in ("plausible", "implausible", "no verdict")}
    print(f"\nplausible {n['plausible']} / implausible {n['implausible']} "
          f"/ no verdict {n['no verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
