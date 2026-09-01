# NEXT220--NEXT221 signed repair-band redistribution freeze

Date: 2026-08-08

Status: frozen before any NEXT220 candidate is joined to discovery outcomes.

## Purpose

NEXT216 showed that one-sided multiplication can cross the repair boundary and
worsen threshold geometry. NEXT218 prevented that crossing and preserved AUC,
but NEXT219 showed that its best threshold remained exactly at the anchor, so
no protected record changed decision. A useful correction must change local
protected-versus-severe ordering rather than move both groups in one direction.

NEXT220 therefore applies one bounded, zero-centered signed redistribution:
protected-like records move down and severe-like records move up under the same
audited initial-geometry certificate. NEXT221 is diagnostic-only if the search
has AUC+SAFE candidates but no all-gate candidate.

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
- Feature cutoffs: the endpoint-blind `1/16` and `15/16` inverted-CDF
  quantiles already frozen in NEXT216.
- Protection certificate: the exact NEXT216 bounded directional value
  `P in [0,1]`, where larger `P` means more protected-like.
- Band width: `W = upper - lower`.

No feature, direction, cutoff, cohort, threshold, or amplitude is chosen from
NEXT220 outcomes.

## NEXT220 executable grammar

The redistribution grid is exactly
`beta in {1/64, 1/32, 1/16, 1/8, 1/4}`. For an initial score `s`, define

```text
active = support AND lower <= s < upper AND feature is finite

if active:
    s' = s + beta * W * (1 - 2 * P)
else:
    s' = s

support' = support
```

Thus `P=1` lowers risk by `beta*W`, `P=0` raises it by the same amount, and
`P=1/2` leaves it unchanged. With the frozen band and maximum `beta=1/4`, all
active scores remain nonnegative without clipping. Missing features deactivate
the optional correction and retain the exact NEXT214 score.

The unchanged base plus `22 * 5` signed redistributions gives exactly `111`
candidates. No pair, conjunction, learned weight, intercept, calibration,
source-specific term, or result-dependent continuation is allowed.

## Frozen evaluation and selection

Use the unchanged discovery evaluator and its exact source-AUC, SAFE, BROAD,
fold, threshold, confidence-bound, and deterministic-selection rules. Require
exact reproduction of NEXT214, the NEXT215 eligible universe, and the closed
NEXT216--NEXT219 branches before evaluation.

- If at least one candidate passes every discovery gate, publish the unchanged
  evaluator selection and authorize only the already-defined next validation
  step; do not claim replacement before sealed validation.
- If no candidate passes every gate but at least one passes AUC+SAFE and fails
  BROAD, freeze the exact sorted candidate-key population for NEXT221.
- Otherwise terminate the branch without a diagnostic.

## NEXT221 diagnostic rule

NEXT221 searches and selects no formula. It must reproduce the exact frozen
NEXT220 AUC+SAFE/non-BROAD population, rerun the unchanged BROAD residual
diagnostic, and rank records lexicographically by failed constraint count,
normalized shortfall sum, and candidate key. It must reproduce the NEXT214
reference of six failures and normalized shortfall `0.26893426117441227`.

## Stop rule

If no new candidate strictly improves that diagnostic tuple, close signed
single-certificate redistribution. Do not densify beta, retune the band, or add
pairs. A strict diagnostic improvement without an all-gate pass may justify
only a separately frozen forward step; it does not authorize validation.
