# NEXT218--NEXT219 lower-anchored repair-band relief freeze

Date: 2026-08-08

Status: frozen before any NEXT218 candidate is joined to discovery outcomes.

## Purpose

NEXT216 tested multiplicative protection relief inside the fixed NEXT214 repair
band. NEXT217 showed that every new candidate worsened the frozen BROAD
residual. The failure has a specific score-geometry explanation: multiplying
the complete score can move an in-band record below the frozen lower boundary,
where unchanged lower-score records retain their original values. The observed
best thresholds moved below the repair boundary and the ordering around that
boundary was no longer coherent.

NEXT218 changes only this geometry. It contracts the excess above the lower
boundary and cannot move an in-band score across that boundary. NEXT219 is a
diagnostic-only BROAD residual analysis if NEXT218 has AUC+SAFE candidates but
no all-gate candidate.

## Immutable scientific boundary

The executable law may use only composition and initial, unrelaxed geometry.
It must not use a DFT calculation or DFT value, a learned energy/force/stress
proxy, a model or proxy potential, a relaxed structure, a relaxation
trajectory, or physical relaxation. Discovery outcomes are offline labels only.
Validation and replication endpoints remain sealed unless every frozen
discovery gate passes.

All code, tests, reports, and formal artifacts are additive. Existing scripts
and content and all canonical paper/report files remain unchanged.

## Frozen provenance

- Base score and support: exact final NEXT214 three-term record.
- Repair band: lower-inclusive, upper-exclusive
  `[0.17470215862148156, 0.570892727856757)`.
- Protection hypotheses: the exact 22 NEXT215-eligible identities.
- Eligible-identity SHA-256:
  `2e5000a319188a6191922a499b8151e28bb603ba06e70cff8750ec582e887b41`.
- Feature cutoffs: the already-defined endpoint-blind `1/16` and `15/16`
  inverted-CDF quantiles over all supported rows in the fixed repair band.
- Protection certificate: the exact NEXT216 bounded directional certificate
  `P in [0,1]`.

No feature, direction, cutoff, cohort, threshold, or amplitude is chosen from
NEXT218 outcomes.

## NEXT218 executable grammar

The amplitude grid is exactly `{1/16, 1/8, 1/4, 1/2}`. For an eligible
certificate and an initial score `s`, define

```text
active = support AND lower <= s < upper AND feature is finite

if active:
    s' = lower + (s - lower) * (1 - amplitude * P)
else:
    s' = s

support' = support
```

This is a lower-anchored contraction, not a multiplication of the whole score.
It guarantees `lower <= s' <= s < upper` for every active record. Missing
features deactivate the optional correction and retain the exact NEXT214
score. The unchanged base plus `22 * 4` anchored contractions gives exactly
`89` candidates.

## Frozen evaluation and selection

Use the unchanged discovery evaluator and its exact source-AUC, SAFE, BROAD,
fold, threshold, confidence-bound, and deterministic-selection rules. Require
exact reproduction of the NEXT214 base and the NEXT215 eligible universe before
evaluation.

- If at least one candidate passes every discovery gate, publish the unchanged
  evaluator selection and authorize only the already-defined next validation
  step; do not claim replacement before sealed validation.
- If no candidate passes every gate but at least one passes AUC+SAFE and fails
  BROAD, freeze the exact sorted candidate-key population for NEXT219.
- Otherwise terminate the branch without a diagnostic.

## NEXT219 diagnostic rule

NEXT219 searches and selects no formula. It must reproduce the exact frozen
NEXT218 AUC+SAFE/non-BROAD population, rerun the unchanged BROAD residual
diagnostic, and rank records lexicographically by:

1. failed BROAD constraint count;
2. normalized shortfall sum;
3. candidate key.

It must reproduce the NEXT214 reference of six failed constraints and
normalized shortfall `0.26893426117441227`. Any claimed improvement requires a
strictly smaller diagnostic tuple. This diagnostic cannot authorize validation.

## Stop rule

If the globally closest NEXT219 record is the unchanged base, or if no anchored
candidate strictly improves the NEXT214 diagnostic tuple, close the
lower-anchored relief branch. Do not densify amplitudes, retune band boundaries,
add conjunctions, or reuse NEXT218 outcomes to define another candidate in this
branch.
