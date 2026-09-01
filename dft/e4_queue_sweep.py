#!/usr/bin/env python3
"""Reproduce Fig. 4f's queue sweep, first from the UMA proxy and then from DFT.

Figure 4f plots how much of the DFT validation queue a PSS threshold removes against how
much of the high-property subset it keeps. The subset is defined in the manuscript by the
UMA bulk-modulus proxy at 400 GPa. E4 computes the same quantity from first principles, so
the same sweep can be run against a DFT-defined subset and drawn as a second curve.

The reimplementation is checked before it is trusted: run with --check and it reproduces the
published UMA numbers (1,081 candidates, 61 screened, 140 priority, all retained) from the
frozen PSS cutoff. Only if those come back exactly does the DFT curve mean anything.

    python dft/e4_queue_sweep.py --check          # reproduce the published UMA numbers
    python dft/e4_queue_sweep.py                  # both curves, once E4 has bulk moduli
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INVERSE = ROOT / "outputs/20260822_property_design_synthesis_score/inverse_scores.parquet"
PSS_CUTOFF = -0.6368790173149083     # frozen in the support-matched calibration
TARGET_GPA = 400.0


def load_candidates() -> pd.DataFrame:
    d = pd.read_parquet(INVERSE)
    return d[["candidate_id", "synthesis_score", "clamped_bulk_modulus_proxy_gpa",
              "rung_L4_verdict"]].copy()


def sweep(score: np.ndarray, priority: np.ndarray, cutoffs: np.ndarray) -> pd.DataFrame:
    """Queue reduction and high-property retention as the PSS threshold moves."""
    rows = []
    for c in cutoffs:
        screened = score < c
        rows.append({
            "cutoff": float(c),
            "queue_reduction": float(screened.mean()),
            "priority_retained": float((~screened[priority]).mean()) if priority.any() else np.nan,
            "n_screened": int(screened.sum()),
            "n_priority_screened": int((screened & priority).sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="only reproduce the published UMA numbers and stop")
    a = ap.parse_args()

    d = load_candidates()
    score = d["synthesis_score"].to_numpy(float)
    uma = d["clamped_bulk_modulus_proxy_gpa"].to_numpy(float)
    uma_priority = uma >= TARGET_GPA

    screened = score < PSS_CUTOFF
    print("reproducing the published UMA figures at the frozen cutoff")
    print(f"  candidates          {len(d):5d}   expected  1081")
    print(f"  screened by PSS     {int(screened.sum()):5d}   expected    61")
    print(f"  priority (UMA>=400) {int(uma_priority.sum()):5d}   expected   140")
    print(f"  priority screened   {int((screened & uma_priority).sum()):5d}   expected     0")
    ok = (len(d) == 1081 and int(screened.sum()) == 61
          and int(uma_priority.sum()) == 140
          and int((screened & uma_priority).sum()) == 0)
    print(f"  -> {'reproduced' if ok else 'DOES NOT MATCH, do not trust the DFT curve'}")
    if a.check or not ok:
        return 0 if ok else 1

    # --- the same sweep against a DFT-defined subset ---------------------------
    bm_path = HERE / "E4_design" / "bulk_moduli.json"
    if not bm_path.exists():
        print("\nno bulk_moduli.json yet; run analyze.py once stage B is complete")
        return 0
    bm = {r["candidate_id"]: r["dft_bulk_modulus_gpa"] for r in json.loads(bm_path.read_text())
          if r.get("candidate_id") and r.get("dft_bulk_modulus_gpa")}
    d["dft_b"] = d["candidate_id"].map(bm)
    have = d["dft_b"].notna().to_numpy()
    print(f"\nDFT bulk moduli available for {int(have.sum())} of {len(d)} candidates")
    print("The DFT curve is drawn on that subset only; the UMA curve uses all 1,081, so the "
          "two are not read off the same denominator and the caption must say so.")

    # The 400 GPa target was calibrated on the proxy, and the proxy runs high, so carrying
    # the same absolute number onto the DFT scale leaves almost nobody above the line: the
    # "retention" it reports is then one or two candidates and means nothing. The subset is
    # therefore also defined at the target rescaled by the measured proxy bias, which is the
    # same line drawn on the axis the DFT numbers actually live on.
    bias = float(np.median(d.loc[have, "dft_b"].to_numpy(float)
                           / d.loc[have, "clamped_bulk_modulus_proxy_gpa"].to_numpy(float)))
    rescaled = TARGET_GPA * bias
    n_abs = int((d.loc[have, "dft_b"] >= TARGET_GPA).sum())
    n_res = int((d.loc[have, "dft_b"] >= rescaled).sum())
    print(f"proxy bias: median DFT/UMA = {bias:.3f}, so {TARGET_GPA:.0f} GPa on the proxy "
          f"scale is {rescaled:.0f} GPa on the DFT scale")
    print(f"  candidates above {TARGET_GPA:.0f} GPa under DFT: {n_abs}"
          f"{'  <- too few to read a retention off' if n_abs < 20 else ''}")
    print(f"  candidates above {rescaled:.0f} GPa under DFT: {n_res}")

    cutoffs = np.quantile(score, np.linspace(0.0, 0.5, 60))
    uma_curve = sweep(score, uma_priority, cutoffs)
    dft_priority = np.zeros(len(d), dtype=bool)
    dft_priority[have] = d.loc[have, "dft_b"].to_numpy(float) >= rescaled
    dft_curve = sweep(score[have], dft_priority[have], cutoffs)

    out = HERE / "E4_design" / "queue_sweep.json"
    out.write_text(json.dumps({
        "pss_cutoff": PSS_CUTOFF, "target_gpa": TARGET_GPA,
        "n_candidates": int(len(d)), "n_with_dft": int(have.sum()),
        "uma_priority": int(uma_priority.sum()),
        "dft_priority": int(dft_priority[have].sum()),
        "proxy_bias_dft_over_uma": bias,
        "dft_threshold_gpa": rescaled,
        "n_above_absolute_target": n_abs,
        "n_above_rescaled_target": n_res,
        "uma_curve": uma_curve.to_dict("records"),
        "dft_curve": dft_curve.to_dict("records"),
    }, indent=1) + "\n")
    print(f"\nat the frozen cutoff:")
    print(f"  UMA-defined priority retained: "
          f"{100 * (~(score < PSS_CUTOFF))[uma_priority].mean():.1f}%")
    sub = score[have] < PSS_CUTOFF
    if dft_priority[have].any():
        n = int(dft_priority[have].sum())
        print(f"  DFT-defined priority retained: "
              f"{100 * (~sub)[dft_priority[have]].mean():.1f}%  (over {n} candidates, "
              f"the subset above {rescaled:.0f} GPa)")
        abs_mask = np.zeros(len(d), dtype=bool)
        abs_mask[have] = d.loc[have, "dft_b"].to_numpy(float) >= TARGET_GPA
        if abs_mask[have].any():
            print(f"  at the unrescaled {TARGET_GPA:.0f} GPa the subset is only "
                  f"{int(abs_mask[have].sum())} candidates; a retention read off that is "
                  f"not a measurement and is not reported")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
