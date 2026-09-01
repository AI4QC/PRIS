# NEXT172 Repair-Interval BROAD Residual Diagnostic

## Purpose

Diagnose why the frozen NEXT171 repair-interval local-rigidity operator cannot
produce a common BROAD operating point. This is an additive discovery-label
diagnostic, not a formula search, threshold search, validation run, or claim.

## Frozen inputs and population

- Reconstruct the exact NEXT171 label-free features, 41 virtual scores, source
  folds, discovery outcomes, and Pauling cell baselines from hash-pinned formal
  inputs.
- Reproduce the published NEXT171 candidate records before diagnosis.
- Retain exactly candidates satisfying published `passes_source_auc_gates`,
  `passes_safe_all_cells`, and not `passes_broad_all_cells`.
- Expected retained count: 39.
- SHA-256 of newline-joined, lexicographically sorted retained candidate keys:
  `223affb83d00d773fd8b04b4e875fd676dd595a8e9b7910c4abd7dfb457129a9`.
- Expected feature counts: base 1; CrystalNN tightness minimum 8;
  tightness q10 8; volume q10 8; tightness mean 7; volume mean 7.

## Diagnostic

For every retained candidate, enumerate evaluator thresholds strictly below
its already-published SAFE threshold. At each threshold and in every fixed
source aggregate/fold cell, compare against the unchanged Pauling baseline:

- coverage lower bound: strict improvement;
- protected structures kept: no decrease;
- severe structures rejected: strict improvement;
- severe-rejection precision lower bound: strict improvement;
- savings lower bound: strict improvement;
- source-aggregate precision lower bound: at least the frozen BROAD minimum.

Choose the closest threshold solely by minimum failed-constraint count, then
minimum normalized shortfall sum, then smaller threshold. Publish every
failure component, per-candidate records, component frequencies, the unchanged
base residual, and the global closest residual.

## Boundaries

- Discovery outcomes are offline labels only.
- No DFT calculation or DFT value enters an executable formula.
- No learned energy/force/stress proxy and no physical relaxation.
- No validation or replication path exists in the runner interface.
- No candidate, feature, attenuation, score, cutoff, or gate changes.
- No new formula is selected or authorized by this diagnostic.
- Internal validation and replication remain sealed regardless of outcome.

## Outputs

Publish atomically under
`$PRIS_ARCHIVE/next172_repair_interval_broad_residual_diagnostic_v1`:

- `MANIFEST.json`;
- `NEXT172_REPAIR_INTERVAL_BROAD_RESIDUAL_DIAGNOSTIC.json`;
- `next172_repair_interval_broad_residual_per_candidate.parquet`.

Append the result only to the standalone investigation report. Do not edit
canonical paper, notes, TeX, README, or preregistration files.
