# NEXT177 Weighted-Rigidity Amplitude-Extension Search

## Purpose

Test the only boundary signal exposed by NEXT176: weighted CrystalNN minimum
and q10 tightness achieved their smallest BROAD residual at the maximum frozen
NEXT175 attenuation. This is a finite additive discovery search; it does not
modify or replace any prior formula or artifact.

## Frozen candidate universe

- Base score and support: exact hash-pinned NEXT163 global-closest candidate.
- Active region: the original base score satisfies frozen
  `BROAD <= base < SAFE`.
- Eligible label-free features only:
  `pwldr_crystalnn_tightness_min` and
  `pwldr_crystalnn_tightness_q10`.
- Attenuations: `1.0, 1.25, 1.5, 2.0, 3.0, 4.0`.
- Candidate count: one unchanged base plus two features by six attenuations,
  for exactly 13 candidates.
- Score inside the active region:
  `max(0, base - alpha * (SAFE - BROAD) * weighted_rigidity)`.
- Outside the active region, under missing rigidity, or without base support,
  return the exact unchanged base score/support.

The grid is frozen before running the evaluator. No further attenuation,
feature, cutoff, threshold, or gate may be added after seeing NEXT177 results.

## Gates

Use the unchanged cross-source discovery evaluator, source AUC gates, SAFE
gates, BROAD gates, formula-group folds, and Pauling baselines. A candidate is
eligible for freezing only if it passes every frozen discovery gate.

## Boundaries

- Structure/composition-derived executable inputs only.
- Discovery outcomes are offline search labels only.
- No DFT calculation or DFT value enters the executable law.
- No learned energy/force/stress proxy and no physical relaxation.
- No validation or replication endpoint path exists in the runner.
- Validation and replication remain sealed unless every discovery gate passes.

## Outputs

Publish atomically under
`$PRIS_ARCHIVE/next177_weighted_rigidity_amplitude_extension_search_v1`:

- `MANIFEST.json`;
- `NEXT177_WEIGHTED_RIGIDITY_AMPLITUDE_EXTENSION_CATALOGUE.json`;
- `NEXT177_DISCOVERY_EVALUATION.json`;
- `NEXT177_FROZEN_CANDIDATE.json`;
- `next177_weighted_rigidity_amplitude_extension_search.parquet`.

Append the result only to the standalone investigation report. Do not edit
canonical paper, notes, TeX, README, or preregistration files.
