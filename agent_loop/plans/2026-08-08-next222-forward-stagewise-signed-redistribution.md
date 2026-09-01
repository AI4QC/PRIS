# NEXT222 forward-stagewise signed redistribution freeze

Date: 2026-08-08

Status: frozen before any NEXT222 proposal is joined to discovery outcomes.

## Purpose and starting point

NEXT221 established the first strict improvement from repair-band protection:
`sivr_cell_hydro_abs__protected_low` at `beta=1/4` reduced the frozen BROAD
shortfall from `0.26893426117441227` to `0.2572028547126239` while preserving
both-source AUC and all `12` SAFE cells. It still failed the same six SCIGEN
`protected_kept` constraints.

NEXT222 tests whether the same signed, zero-centered mechanism has independent
multi-signal information. This is one predeclared forward-stagewise loop, not a
sequence of manually chosen searches.

## Immutable scientific boundary

Every executable quantity uses only composition and initial, unrelaxed
geometry. The law must not use a DFT calculation or value, a learned
energy/force/stress proxy, a model or proxy potential, a relaxed structure, a
trajectory, or physical relaxation. Discovery outcomes are offline labels
only. Validation and replication remain sealed unless every discovery gate
passes. All additions preserve existing scripts and canonical content.

## Frozen candidate universe

- Base score/support: exact NEXT214 final record.
- Activation band: exact original NEXT214 lower-inclusive,
  upper-exclusive band `[0.17470215862148156, 0.570892727856757)`.
- Eligible hypotheses: exact `22` NEXT215 identities, digest
  `2e5000a319188a6191922a499b8151e28bb603ba06e70cff8750ec582e887b41`.
- Cutoffs and certificates: exact endpoint-blind NEXT216 definitions.
- Beta grid: exact NEXT220 grid `{1/64, 1/32, 1/16, 1/8, 1/4}`.
- Frozen first signed term:
  `sivr_cell_hydro_abs__protected_low`, `beta=1/4`,
  `q_lo=0.0004926449347397526`, `q_hi=0.030114620288954386`.
- Maximum signed terms: `6`, including the frozen first term.

At every depth, evaluate the unchanged current path plus every unused
hypothesis at every beta. No feature reuse, pair search, beam, pruning,
refitting, new cutoff, new feature, source-specific term, or manual override is
allowed.

## Executable cumulative formula

For the original NEXT214 score `s0`, band width `W`, and frozen certificates
`P_j`, define each term only when `s0` lies in the activation band and its raw
feature is finite. Missing terms contribute zero. The cumulative score is

```text
delta = W * sum_j beta_j * (1 - 2 * P_j) over active signed terms
score = max(0, s0 + delta)
support = support_NEXT214
```

The nonnegative floor is fixed before evaluation and is the only clipping.

## Frozen loop and stop rules

At each depth use the unchanged evaluator. If any proposal passes every
discovery gate, publish the evaluator-selected passing path and stop. Otherwise
restrict the residual calculation to candidates passing both-source AUC and all
`12` SAFE cells but not BROAD. Rank them by failed BROAD constraint count,
normalized shortfall sum, and candidate key.

Accept a non-null proposal only if it strictly improves the current tuple, with
shortfall tolerance `1e-12`. Stop on the first depth with no strict improvement,
with no AUC+SAFE/non-BROAD candidate, on an all-gate pass, or at six signed
terms. Do not continue or retune after seeing the result.
