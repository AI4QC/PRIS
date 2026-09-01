# NEXT181 Strong-Closure Repair-Width Search

## Purpose and frozen universe

Search the six NEXT180-eligible strong-neighborhood directional-closure
features with the unchanged bounded repair-width operator. This is additive
and does not replace any prior script or formula.

- Exact NEXT163 global-closest base score/support.
- Active only when the original base score satisfies frozen
  `BROAD <= base < SAFE`.
- Eligible features exactly as published by NEXT180:
  CrystalNN closure min/q10/mean, CrystalNN volume q10/mean, and Voronoi
  closure min.
- Attenuations: `0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00`.
- Score in the active interval:
  `max(0, base - alpha * (SAFE - BROAD) * strong_closure)`.
- Missing feature, unsupported base, or score outside the interval keeps the
  exact base score and support.
- Exactly one base plus six by eight candidates: 49 total.

The universe is frozen before evaluator execution. No feature, attenuation,
gate, cutoff, score, or threshold may be added after seeing results.

## Evaluation and boundaries

Use the unchanged cross-source source-AUC, SAFE, and BROAD discovery gates,
fixed formula-group folds, and Pauling baselines. Freeze authorization requires
one candidate to pass every discovery gate. Otherwise terminate the branch and
keep validation/replication sealed.

The executable inputs are structure/composition derived. Discovery outcomes
are offline search labels only. No DFT calculation/value, learned
energy/force/stress proxy, or physical relaxation is used.

## Outputs

Publish atomically under
`$PRIS_ARCHIVE/next181_strong_closure_repair_width_search_v1`:

- `MANIFEST.json`;
- `NEXT181_STRONG_CLOSURE_REPAIR_WIDTH_CATALOGUE.json`;
- `NEXT181_DISCOVERY_EVALUATION.json`;
- `NEXT181_FROZEN_CANDIDATE.json`;
- `next181_strong_closure_repair_width_search.parquet`.

Append results only to the standalone investigation report; do not edit
canonical paper, notes, TeX, README, or preregistration files.
