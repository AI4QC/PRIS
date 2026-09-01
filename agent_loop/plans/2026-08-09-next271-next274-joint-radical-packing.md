# NEXT271--NEXT274 Joint Radical-Packing Development Plan

**Status:** frozen before opening any NEXT271--NEXT274 feature outcomes or
endpoint-conditioned results.

**Sequential-development disclosure:** NEXT267--NEXT270 discovery outcomes are
already known. In particular, separate radical-cell volume and Chebyshev-radius
heterogeneity passed the raw gates, and the best NEXT270 residual was
`(5, 0.0955435292756307)`. This plan is a new prospective continuation after
that result. It is not independent confirmation.

## Scientific boundary

The executable feature and any candidate score may use only composition and
the initial, raw, unrelaxed periodic geometry. They must not use a DFT
calculation or per-structure DFT value, a learned energy/force/stress proxy, a
model or proxy potential, a relaxed structure, a trajectory, or physical
relaxation. Discovery outcomes are offline labels only. Validation and
replication outputs remain sealed throughout NEXT271--NEXT274.

All work is additive. Existing scripts, outputs, reports, and canonical paper,
note, preregistration, and README files remain unchanged. The independent
exploration report may receive an appended section only after the branch is
complete.

## Frozen inputs

- NEXT267 plan SHA-256:
  `258aade1020fa7911b293a4201dc4f72f428f5df6f1870fe584d64aa3b7b154a`.
- NEXT267 SCIGEN feature table SHA-256:
  `2e400676b94110fa9e64715840f26873855416a12df199181c8150df6d4fe7c0`.
- NEXT267 WyFormer feature table SHA-256:
  `ae6a9b76e39e603541a1065aea46fdac6ad3e3ab633e9e55a210dfa35977b827`.
- NEXT268 eligible-set digest:
  `d0804006dd42d8cdc3fabc4483a0d12bae98ddc11096b99e196b521288ead82e`.
- NEXT269 authorized diagnostic-set digest:
  `3af318aaa0f14a4849483e0d9e616ccc14954469b13d7817af563f1fbee22263`.
- NEXT270 diagnostic SHA-256:
  `5b413d16ef703b74ff5988b96cc66369ad5ef4260d4061e69411554d6633e5a6`.

## Fixed mechanism

For each supported structure, let

```text
X = prv_volume_ratio_cv
Y = prv_chebyshev_ratio_cv.
```

Use the label-free inverse-CDF anchors already published by NEXT268:

```text
x = max(0, (X - 0.021517581455692707)
           / (0.6977318301246591 - 0.021517581455692707))
y = max(0, (Y - 0.011985598809152042)
           / (0.28180046821941024 - 0.011985598809152042)).
```

Do not upper-clip `x` or `y`. Materialize exactly these twelve finite,
dimensionless structure features, in this order:

1. `prvj_joint_min = min(x, y)`
2. `prvj_joint_harmonic = 0 if x + y == 0 else 2*x*y/(x+y)`
3. `prvj_joint_geometric = sqrt(x*y)`
4. `prvj_joint_product = x*y`
5. `prvj_joint_mean = (x+y)/2`
6. `prvj_joint_max = max(x, y)`
7. `prvj_joint_l1_gap = abs(x-y)`
8. `prvj_volume_minus_chebyshev = x-y`
9. `prvj_chebyshev_minus_volume = y-x`
10. `prvj_volume_excess = max(0, x-y)`
11. `prvj_chebyshev_excess = max(0, y-x)`
12. `prvj_balance_weighted_joint = 0 if max(x,y) == 0 else
    min(x,y)^2/max(x,y)`

Rows unsupported by NEXT267 remain unsupported and abstain. No imputation is
allowed. Arithmetic outputs are quantized to the existing `1e12` grid before
publication. The executable transform accepts no endpoint or label input.

## NEXT271: label-free materialization

Implement a pure transform and materialize both discovery sources using only
the frozen NEXT267 feature tables. Tests must first fail, then cover exact
schema/order, analytic pairs, non-finite refusal, row-order invariance, support
preservation, input hash refusal, boundary flags, and atomic publication.

The formal output must include a manifest, catalogue, and one Parquet table per
source. NEXT272 is authorized only if all twelve values are finite on every
NEXT267-supported row and the supported-row identities are unchanged.

## NEXT272: prospective feature audit

Reconstruct the exact NEXT224 rejected-extreme discovery cohort and audit both
`protected_low` and `protected_high` directions for all twelve features: 24
fixed hypotheses. Use the unchanged source aggregate, source-fold macro,
worst-fold, coverage, and raw eligibility gates from NEXT268. Quantiles use the
same inverse-CDF method and the same combined finite discovery population.

Rank and publish every direction. NEXT273 is authorized only for directions
that pass every frozen raw gate in both sources. Publish the exact eligible-set
digest before any coefficient search.

## NEXT273: bounded margin-local search

Reproduce the exact NEXT224 base candidate. For each NEXT272-eligible
direction, evaluate exactly the existing seven local-width fractions and three
nonnegative amplitude fractions used by NEXT269, plus one exact reproduction
control. The support policy, normalization population, missing policy,
triangular term, evaluator, folds, AUC gates, twelve SAFE cells, and BROAD gate
are unchanged from NEXT269.

No adaptive feature, direction, width, amplitude, threshold, or coefficient may
be added after results are visible. Freeze a candidate only if it passes every
cross-source discovery gate. Otherwise, NEXT274 is authorized only for the
exact new-candidate identities that pass both source AUC gates and all SAFE
cells but fail BROAD.

## NEXT274: unchanged BROAD diagnostic

Exactly reproduce the authorized NEXT273 candidate records and recompute their
unchanged BROAD threshold tables. Rank residuals lexicographically by

```text
(failed_constraint_count, normalized_shortfall_sum, candidate_key).
```

Compare the global closest residual with the frozen NEXT270 reference
`(5, 0.0955435292756307)`. Strict improvement means a lower failed-constraint
count, or the same count and a lower normalized shortfall sum. NEXT274 performs
no new formula search and opens no validation or replication output.

If a candidate passes all discovery gates, or if NEXT274 strictly improves the
reference, continued work requires another new preoutcome freeze. Otherwise
this joint-radical branch closes.

## Verification and reporting

- Run all focused NEXT271--NEXT274 tests and the complete repository suite.
- Verify every frozen input, executed-source, and published-output SHA-256.
- Confirm all no-DFT/no-proxy/no-relaxation flags and sealed validation and
  replication flags.
- Check CodeGraph status after edits.
- Append an evidence section to the independent exploration report; do not edit
  earlier report text or any canonical document.
