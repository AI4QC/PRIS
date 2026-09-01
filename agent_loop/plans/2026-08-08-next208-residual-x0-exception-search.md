# NEXT208 Residual X0 Exception Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Test whether one of the 44 NEXT207-stable raw initial-structure
features can grant a fixed, interpretable exception to the protected
structures still rejected by the exact NEXT206 candidate while preserving all
cross-source AUC, SAFE, and BROAD gates.

**Architecture:** Reconstruct the exact NEXT206 score and support arrays from
the already verified discovery-only inputs.  For every NEXT207-eligible
feature direction, derive 15 cutoffs from the pooled, endpoint-blind empirical
distribution among rows rejected by the frozen NEXT206 residual threshold.
Each candidate changes the current score to zero only when its one raw x0
condition is met; support and every other score are unchanged.  Encode each
corrected score as one exactly invertible virtual term and evaluate all 661
candidates with the unchanged cross-source evaluator.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and the existing
NEXT125/NEXT205 discovery evaluator and provenance reconstruction.

## Scientific and provenance boundary

- Executable inputs remain composition plus initial unrelaxed geometry only.
- Discovery outcomes are offline labels used only by the unchanged evaluator;
  they are not accepted by the cutoff builder or executable score function.
- Forbidden: DFT calculations or DFT values in the executable rule; learned
  energy/force/stress proxies; model or proxy potentials; relaxed structures,
  trajectories, or physical relaxation.
- Validation and replication artifacts remain physically unopened even if a
  discovery candidate passes. User review is required before opening them or
  modifying a canonical report/paper.
- Additive files only. Preserve all existing scripts and artifacts. Do not
  modify `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.
- Continue in the user-authorized existing dirty checkout. Do not commit or
  create a worktree.

## Evidence authorizing the search

NEXT207 tested exactly 484 raw-feature direction hypotheses in the fixed
NEXT206 rejected-extreme cohort. Exactly 44 passed both-source aggregate and
fold-stability gates, with sorted newline-joined hypothesis SHA-256:

```text
9d5ccc3ca8dd31c2b4b230330d141f9a05202900cd1f0e243f4140efb60ec24a
```

The top-ranked hypothesis is
`scbv_mismatch_max__protected_low`: aggregate AUC is `0.715165452399365`
for SCIGEN and `0.6505500113075599` for WyFormer, and the minimum worst-fold
AUC across sources is `0.6044456845238095`. NEXT208 retains all 44 eligible
hypotheses rather than choosing a family after seeing threshold-search
outcomes.

## Alternatives considered and frozen choice

1. **All eligible one-dimensional certificates (chosen).** This preserves
   every independently qualified physical mechanism while keeping the search
   finite and auditable.
2. **Only the NEXT207 rank-one feature.** Smaller, but it prematurely assumes
   mismatch is the only transferable mechanism and can miss a better BROAD
   tradeoff.
3. **Pairwise conjunctions or threshold/depth co-tuning.** Deferred because it
   multiplies label-driven degrees of freedom before the simpler family is
   falsified.

No feature subset, conjunction, score multiplier, residual threshold, or
operating gate may be tuned in NEXT208.

## Frozen candidate universe

Let `s206`, `u206`, and `r` be the exact NEXT206 global-closest score, support,
and residual threshold:

```text
candidate key SHA-256 =
26ee85c8dbb8f810eb5baf8c8be07f61d390f2c02c9e45d147c786212b7acc38
r = 0.16344427817025572
```

The endpoint-blind cutoff-fit mask is exactly:

```text
u206 AND finite(s206) AND s206 >= r
```

It contains no endpoint, source, fold, identifier, lattice label, Pauling
value, or support flag beyond `u206`. For each eligible feature, discard only
non-finite feature values inside this mask and use NumPy's empirical
`method="inverted_cdf"` quantile.

For exception fractions `p = k/16`, `k = 1,...,15`:

```text
protected_low:  cutoff = quantile(x, p);     safe(x) := finite(x) AND x <= cutoff
protected_high: cutoff = quantile(x, 1 - p); safe(x) := finite(x) AND x >= cutoff
```

For each feature/direction/fraction candidate:

```text
active := u206 AND finite(s206) AND s206 >= r AND safe(x)
s208  := 0 if active else s206
u208  := u206
```

Missing feature values fail open by keeping `s206`. Zero is the only pardon
depth: NEXT208 does not search a multiplier. The unchanged base candidate is
included once. The exact candidate count is therefore:

```text
1 + 44 * 15 = 661
```

Candidate keys record the complete base key, feature, direction, fraction
numerator/denominator, realized cutoff, residual threshold, inclusive
comparison, missing policy, and score composition. Distinct fractions remain
distinct hypotheses even when a discrete feature yields an equal cutoff.

## Frozen evaluation and selection

Encode each `s208` by the existing exact `asinh(sinh(score/divisor))` virtual
term mechanism. Reuse
`search_optional_guard_laws_parallel` without modifying it. This preserves:

- separate SCIGEN and WyFormer pooled/macro/worst AUC gates;
- SAFE threshold selection across both source aggregates and ten source-fold
  cells;
- BROAD comparison against Pauling in every one of the same 12 cells;
- the existing deterministic rank and candidate-key tie break.

NEXT208 succeeds only if the selected candidate has
`passes_all_discovery_gates=true`. Passing AUC and SAFE while failing BROAD is
not success; those exact candidates authorize diagnostic-only NEXT209. If no
candidate passes all gates, do not loosen a threshold or open validation.

## Task 1: Write NEXT208 contract tests

**Files:**

- Create: `tests/test_next208_residual_x0_exception_search.py`

**Steps:**

1. Test endpoint-free inverted-CDF cutoffs for both directions and all
   inclusive-boundary/missing-value cases.
2. Test the exact exception score: activation only above `r`, fixed zero
   pardon, unchanged support, and fail-open missing values.
3. Test deterministic base-plus-feature candidate construction, key contents,
   fraction grid, and exact candidate accounting on a toy universe.
4. Test exact virtual-term round trip and evaluator-facing runtime specs.
5. Test sealed formal interface, atomic fail-closed output behavior, and
   explicit no-DFT/no-proxy/no-relaxation flags.
6. Run the module and observe the expected missing-module RED failure.

## Task 2: Implement the minimal NEXT208 search

**Files:**

- Create: `src/next208_residual_x0_exception_search.py`

**Steps:**

1. Implement only the tested cutoff, score, candidate, and virtual-term
   helpers.
2. Extend NEXT207 path reconstruction and verify the exact NEXT207 manifest,
   source hash, output hashes, 44 eligible identities, and hypothesis digest.
3. Reconstruct and verify the exact NEXT206 score, support, candidate key, and
   residual threshold.
4. Build exactly 661 candidates without accepting endpoint values in cutoff
   construction, then call the unchanged parallel evaluator once.
5. Publish atomically: catalogue JSON, evaluation JSON, frozen-candidate JSON,
   all-candidate Parquet, and manifest with complete hashes and boundary flags.

## Task 3: Run and verify NEXT208

1. Run the test module after implementation and require GREEN.
2. Compile the new source.
3. Compute the design SHA-256 and freeze it in the source before the formal
   run; rerun tests after the identity edit.
4. Run the formal search once with the established `newpauling` environment
   and output root
   `$PRIS_ARCHIVE/next208_residual_x0_exception_search_v1`.
5. Verify candidate counts, output hashes, source hash, safety flags, selected
   record, and all input hashes from the published manifest.

## Task 4: Conditional NEXT209 handoff

If NEXT208 has no all-gate candidate but has one or more candidates satisfying
source AUC plus SAFE and failing only BROAD, freeze their sorted candidate-key
SHA-256 and write a separate diagnostic-only NEXT209 plan before implementing
it. NEXT209 may reproduce and localize the residual but may not search a new
formula. Otherwise close this exact one-dimensional exception branch.

## Task 5: Standalone report and repository verification

Append the verified NEXT207/NEXT208 and conditional NEXT209 results only to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Run targeted tests,
source compilation, full pytest, `git diff --check`, manifest output-hash
verification, report-fence checks, canonical-path status checks, and
CodeGraph status. Do not edit a canonical manuscript or report.
