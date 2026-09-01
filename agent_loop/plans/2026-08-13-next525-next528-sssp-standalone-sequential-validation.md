# NEXT525–NEXT528: SSSP standalone sequential validation

Date frozen: 2026-08-13 (America/Chicago)

## Purpose

Test whether the already-frozen same-sign shell purity (SSSP) law is a useful
standalone, zero-DFT prescreen rather than only a residual feature inside the
failed NEXT414/NEXT415 combination search.  This is an additive branch.  It
does not revise NEXT411–NEXT415, any existing report, or any paper artifact.

## Immutable executable boundary

The executable law receives only composition and one raw, initial, fully
periodic geometry.  It may not receive a DFT value, relaxed geometry, later
trajectory frame, learned energy/force/stress value, MLIP value, proxy
potential, or relaxation result.  DFT-derived endpoints are offline labels
opened only after the feature and evaluation artifacts for that role are
frozen.

The NEXT411 formula and direction are unchanged:

1. assign formal charge signs with the frozen NEXT19 policy;
2. construct the periodic opposite-sign radical-Voronoi graph;
3. for each site `i`, let `R_i` be the largest opposite-sign incident edge
   distance;
4. let `D_i` be the nearest same-sign periodic distance within `R_i`, or
   `R_i` if none exists;
5. `u_i = min(1, D_i / R_i)`;
6. `SSSP = q10_inverted_cdf(u_i)` on the frozen `1e-10` output grid.

Higher SSSP is more protected; standalone risk is `-SSSP`.  Unsupported
structures abstain.

## Evidence already available before this freeze

NEXT413 passed its discovery residual-feature gates on both sources.  A
separate discovery-only feasibility calculation, performed before this
document was written, found standalone protected-versus-severe AUCs of
`0.8558947767973231` on SCIGEN and `0.6546082319787324` on WyFormer.  The five
reduced-formula fold AUC minima were `0.8292102462951909` and
`0.632093449381585`, respectively.  These values authorize a standalone
holdout test; they are not holdout evidence.

WyFormer's internal-validation endpoint was used previously by NEXT97, so it
is not globally pristine.  It has not been used to formulate, calibrate, or
evaluate SSSP.  The protocol therefore calls it an SSSP-specific holdout, not
a globally untouched holdout.  Internal replication remains the decisive
sequential test and must not be opened unless internal validation passes.

## NEXT525: discovery calibration and immutable formula artifact

Join the frozen NEXT412 discovery feature tables to the already-open
discovery endpoints and the corresponding NEXT85/NEXT94 Pauling decisions.
No validation or replication endpoint may be read.

Search only observed, supported SSSP values as a single shared reject
threshold `t`, with `reject := supported and SSSP <= t`.  A threshold is
feasible only if, on each discovery source:

- Wilson lower protected recall is at least `0.95`;
- Wilson lower savings is at least `0.02`.

Rank feasible thresholds lexicographically by:

1. the minimum Wilson lower severe-rejection precision across the two
   sources;
2. the minimum severe recall;
3. the minimum Wilson lower savings;
4. the larger threshold.

The expected deterministic threshold from the pre-freeze feasibility pass is
`0.5231805323`.  NEXT525 must reproduce it from the frozen discovery inputs;
it may not hard-code a different threshold.  Publish one formula JSON,
discovery evaluation, discovery predictions, and a manifest with hashes.

## NEXT526: all-holdout label-free feature freeze

Before opening either validation endpoint, compute unchanged NEXT411 SSSP for
both `internal_validation` and `internal_replication` raw geometries for both
sources.  Publish one table per source and role plus a manifest and catalogue.
The manifest must state that no endpoint was opened, no DFT value entered the
feature, and no relaxation/model/proxy potential was executed.  Minimum
feature coverage is `0.95` in every source-role cell and at least 20 rounded
unique values are required.

The NEXT525 formula hash, NEXT411 implementation hash, raw geometry hashes,
metadata hashes, and all produced feature hashes are frozen before NEXT527.

## Fixed evaluation statistics

For either holdout role, join only by `material_id`.  SCIGEN uses
`distortion_ratio`; WyFormer maps endpoint strata as protected `0`, middle
`1.5`, and severe `2`.  Protected means endpoint `<= 1`; severe means endpoint
`>= 2`.

For each source report:

- support coverage and Wilson lower coverage;
- protected recall and Wilson lower protected recall at the frozen threshold;
- severe-rejection precision and its Wilson lower bound;
- savings and its Wilson lower bound;
- severe recall;
- pooled protected-versus-severe AUC of `-SSSP`;
- five deterministic reduced-formula-fold AUCs, their macro mean and minimum;
- a deterministic 1,000-draw reduced-formula cluster-bootstrap 95% interval
  for pooled AUC (seed `20260813`);
- the frozen Pauling P2–P5 binary decision baseline on the same rows.

No threshold, direction, transform, endpoint definition, gate, bootstrap
seed, fold function, or missing-value policy may change after an endpoint is
opened.

## Per-source acceptance gates

Every source must satisfy all of the following:

- Wilson lower coverage `>= 0.90`;
- Wilson lower protected recall `>= 0.95`;
- Wilson lower severe-rejection precision `>= 0.60`;
- Wilson lower savings `>= 0.02`;
- pooled AUC `>= 0.60`;
- macro reduced-formula-fold AUC `>= 0.60`;
- worst reduced-formula-fold AUC `>= 0.55`;
- all five folds evaluable with at least 10 supported protected and 10
  supported severe rows;
- cluster-bootstrap AUC lower bound `> 0.50`;
- SSSP binary rejection AUC exceeds the Pauling P2–P5 binary rejection AUC;
- SSSP Wilson lower coverage, protected recall, and severe-rejection precision
  each exceed the corresponding Pauling value.

This is a safe-screen gate; SSSP need not exceed Pauling severe recall because
the frozen operating point intentionally protects stable structures.  Both
sources must pass for the role to pass.

## NEXT527 and NEXT528 sequence

NEXT527 may open only the two `internal_validation` endpoint tables.  If
either source fails, publish the failure and stop: NEXT528 is unauthorized and
replication endpoint values remain unread.

Only if NEXT527 passes both sources may NEXT528 open the two
`internal_replication` endpoint tables and repeat the identical evaluation.
No recalibration or rescue search is allowed between roles.

If both roles pass, write a new independent report describing SSSP as a
replicated zero-DFT prescreen candidate, with exact limitations and without
editing canonical reports or the paper.  Canonical integration remains gated
on user confirmation.  If a role fails, preserve every artifact, report the
negative result, and return to discovery with a non-duplicate mechanism.
