# NEXT131 protected BROAD residual diagnostic

## Purpose

Diagnose, without searching or changing a formula, the exact strict BROAD
constraint residual of every NEXT130 candidate that passed SAFE12.

## Frozen inputs

- The 1,560-candidate NEXT130 search output and its manifest.
- The exact NEXT130 source and inherited NEXT125/NEXT129 artifacts.
- Discovery endpoints only, used as offline diagnostic labels.
- No validation or replication endpoint may be opened.

## Procedure

1. Reconstruct the exact NEXT130 virtual scores from the frozen physical
   candidate keys.
2. Select only records with `passes_safe_all_cells == true` from the published
   NEXT130 result; do not alter their safe thresholds.
3. Enumerate every distinct score threshold strictly below the published safe
   threshold.
4. Evaluate the same 12 source/fold cells and the same 62 strict BROAD
   inequalities used by NEXT128.
5. Rank diagnostic thresholds only by failed-constraint count, normalized
   shortfall, threshold, and candidate identity.
6. Publish the per-candidate residual and aggregate failure frequencies.

## Boundaries

- This stage searches no new formula and authorizes no validation opening.
- It performs no DFT calculation, learned energy/force/stress inference, or
  structural relaxation.
- DFT-derived discovery outcomes remain offline labels only and cannot enter an
  executable law.
