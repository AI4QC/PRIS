# NEXT215--NEXT217 Continuous Repair-Band Relief Certificate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Determine whether a raw, interpretable initial-structure descriptor
can identify protected structures inside the remaining NEXT214 rejection band,
then, only if the audit succeeds, test a bounded continuous relief term without
using DFT or weakening the frozen high-risk SAFE region.

**Architecture:** NEXT215 reconstructs the exact three-term NEXT214 score and
audits the same frozen 242-column raw x0 feature universe only within the fixed
interval between NEXT214's closest BROAD threshold and its exact SAFE threshold.
It searches no formula. If and only if a feature is stable in the same
protection direction across both source aggregates and every reduced-formula
fold, NEXT216 receives a separately hashed design that attenuates the current
score only inside this interval. NEXT217 is diagnostic-only and is authorized
only for AUC+SAFE/non-BROAD candidates.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and the existing
NEXT207/NEXT210/NEXT212/NEXT214 provenance and dual-source evaluation helpers.

## Non-negotiable scientific and repository boundary

- Executable inputs are composition plus initial unrelaxed geometry only.
- Forbidden in an executable law: DFT calculation or value; learned
  energy/force/stress proxy; model or proxy potential; relaxed structure;
  trajectory; physical relaxation.
- Discovery endpoints are offline audit/evaluation labels only and never a
  runtime input.
- Validation and replication endpoints remain physically unopened throughout
  NEXT215--NEXT217. Passing discovery gates does not authorize opening them.
- Additive files only. Preserve every existing script, output, and document.
  Do not modify `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.
- Work in the user-authorized dirty checkout without a commit or worktree.
- Publish formal results outside the repository and append only verified facts
  to the existing standalone report.

## Frozen NEXT214 starting point

Formal NEXT214 identities are:

- design: `2f8eee54c6af050d00ec09f76ca39d2b5751390393e23c5d347a52fe129ad630`
- source: `fb1c4b7d2ca0db9af12b6ebf9b88e538f4694ac6f96a83bb01bfc84d9d71f149`
- manifest: `35e1da8433880ea5dd9436f68b6d18b8fed9e135aaf2bb457c528063cc8eed18`
- catalogue: `f11d9549feb54d4acdb4aa740a0fdd76bbb8b2bcfe9b0cc5bbf0b4e4f9015ac6`
- evaluation: `cd05d9f7a684eed330603c45ac84b4098363d904b7b66533ba03aa6129a97c29`
- formula: `1ba0de2b8f77e393c11849e5a013722a4b453a7f744894fb4d8cd61d24a3e412`
- candidate table: `a22f3f350345c94400f673aba4ff308ab0ba579c5be88f6b377aaa1a930b4afc`

Reconstruct the exact score by applying these immutable terms in order:

```text
1/16 * scbv_mismatch_max__protected_low
1/16 * nm_site_max__protected_low
1/16 * steric_overlap2_vector_q95__protected_low
```

The final path key has SHA-256
`bf9c47811a270ba46f0acb5a982f42c45cacf54c8801733333289b03bf99810e`.
The unchanged depth-4 identity has SHA-256
`3bac3825e0a4ac36caf41e8170f6f266876ab830ea9676610ed4c019db3b461f`.
It passes the frozen source-AUC and SAFE gates but misses all six SCIGEN
`protected_kept` BROAD constraints. Its normalized shortfall is
`0.26893426117441227`.

The repair interval is immutable:

```text
lower = NEXT214 closest BROAD threshold = 0.17470215862148156
upper = NEXT214 exact SAFE threshold    = 0.570892727856757
```

NEXT215's audit cohort is exactly:

```text
NEXT214 support
AND finite NEXT214 final score
AND lower <= final score < upper
AND (discovery endpoint <= 1 OR discovery endpoint >= 2)
```

The expected accounting is frozen and any mismatch fails closed:

| Cell | Protected in repair band | Severe in repair band |
|---|---:|---:|
| SCIGEN aggregate | 221 | 1,432 |
| SCIGEN folds 0--4 | 42, 49, 40, 42, 48 | 302, 269, 301, 282, 278 |
| WyFormer aggregate | 317 | 369 |
| WyFormer folds 0--4 | 72, 53, 66, 70, 56 | 68, 72, 84, 81, 64 |

Rows at or above `upper` are deliberately outside the repair cohort. The
corresponding frozen counts are SCIGEN 84 protected / 1,674 severe and
WyFormer 9 protected / 150 severe; they must be left unchanged by any later
relief formula.

## Frozen NEXT215 audit

Use the exact NEXT207 column-selection policy. It yields 242 sorted raw numeric
x0 features with name-list SHA-256
`87a20f191ca47b6fb3e9f0255ae8d1e98bcf41e21991af3d290ff222c446f07c`.
Audit both directions for all 242 features, giving exactly 484 audit rows:

```text
protected_high -> protection score = +feature
protected_low  -> protection score = -feature
```

The target is `endpoint <= 1`. Within every aggregate/fold cell, use only
finite feature values. Preserve the NEXT207 gates exactly:

- finite coverage at least `0.90` in every cell;
- at least 20 protected and 20 severe finite rows in every cell;
- each source aggregate protection AUC at least `0.55`;
- each source macro-fold protection AUC at least `0.53`;
- each source worst-fold protection AUC at least `0.50`;
- the opposite direction for the same feature must not also pass;
- identities and metrics must reproduce deterministically.

Audit the three NEXT214-used features for evidence, but mark both directions
ineligible for NEXT216 with reason `already_in_next214_path`. Rank remaining
eligible hypotheses by descending minimum worst-fold AUC, then minimum
aggregate AUC, then mean aggregate AUC, then hypothesis name. Publish the
complete 484-row table; do not select a formula, attenuation, cutoff, or
validation candidate in NEXT215.

Formal output directory:

```text
$PRIS_ARCHIVE/next215_repair_band_relief_audit_v1/
```

Publish atomically:

- `MANIFEST.json`
- `NEXT215_REPAIR_BAND_RELIEF_CATALOGUE.json`
- `NEXT215_REPAIR_BAND_RELIEF_AUDIT.json`
- `next215_repair_band_relief_audit.parquet`

## Conditional NEXT216 search contract

Create a separate dated NEXT216 plan only after the formal NEXT215 hashes and
eligible-hypothesis digest exist. If NEXT215 yields zero eligible hypotheses,
stop this branch without loosening gates.

If eligible hypotheses exist, freeze the exact eligible identities and use
only endpoint-blind feature values from all rows in the fixed repair band to
compute 1/16 and 15/16 inverted-CDF cutoffs. Define a bounded protection
certificate `P`:

```text
protected_high: P = clip((x - q_lo) / (q_hi - q_lo), 0, 1)
protected_low:  P = clip((q_hi - x) / (q_hi - q_lo), 0, 1)
```

For an eligible certificate, use the existing multiplicative repair-loop
semantics:

```text
if support AND lower <= s < upper AND finite(P):
    s' = s * (1 - alpha * P)
else:
    s' = s
```

Support never changes; missing values keep the NEXT214 score. Rows below
`lower` and rows at or above `upper` remain bitwise unchanged. The NEXT216
plan must freeze the amplitude grid before evaluation; the default proposed
grid is `{1/16, 1/8, 1/4, 1/2}`. It may not be expanded after outcomes are
seen. Include the unchanged NEXT214 path exactly once and run the unchanged
dual-source AUC/SAFE/BROAD evaluator. No beam search, conjunction, refit,
manual candidate selection, or validation access is allowed.

## Conditional NEXT217 diagnostic

Only if NEXT216 produces AUC+SAFE/non-BROAD candidates and no all-gate result,
freeze the exact candidate count and sorted identity digest in a new dated
NEXT217 plan. NEXT217 may reproduce their threshold tables and rank BROAD
residuals by `(failed_constraint_count, normalized_shortfall_sum,
candidate_key)`. It searches no new formula and opens no validation or
replication output.

## TDD and execution tasks

### Task 1: Write NEXT215 contract tests and observe RED

**Files:**

- Create: `tests/test_next215_repair_band_relief_audit.py`

**Steps:**

1. Test protection direction mapping, input validation, and protected-positive
   AUC semantics.
2. Test strict lower-inclusive/upper-exclusive repair-band membership.
3. Test coverage/count/AUC gates, opposite-direction veto, used-feature veto,
   deterministic ranking, and empty eligibility.
4. Test that the formal interface exposes discovery endpoints but no validation
   or replication endpoint.
5. Test fail-closed missing inputs.
6. Run the module with the repository interpreter and observe the expected
   missing-module failure.

### Task 2: Implement minimal NEXT215 and reach GREEN

**Files:**

- Create: `src/next215_repair_band_relief_audit.py`

**Steps:**

1. Implement only the tested direction, cohort, per-source audit, veto, and
   ranking helpers.
2. Verify NEXT214's design/source/manifest/output hashes and explicit no-DFT
   boundary flags.
3. Reconstruct the exact NEXT214 score and verify term identities, path-key
   hashes, thresholds, support, feature universe, and all frozen cohort counts.
4. Write the four formal artifacts atomically with complete input, executed
   source, and output hashes.
5. Run the targeted test module and source compilation.

### Task 3: Run formal NEXT215 and make the branch decision

1. Run once into the frozen external output directory.
2. Independently recompute every output hash and boundary flag.
3. If zero eligible hypotheses, record the predeclared stop and skip NEXT216.
4. Otherwise write a separate frozen NEXT216 design containing the exact
   NEXT215 hashes, eligible count/digest, candidate count, amplitude grid, and
   formula semantics before writing NEXT216 code.

### Task 4: Conditionally implement NEXT216 and NEXT217 by TDD

For each authorized stage, write tests first, observe the expected RED failure,
implement minimally, run the targeted module, compile, execute formally into a
new external directory, and verify hashes. Never overwrite a prior directory.

### Task 5: Report and full verification

Append only verified NEXT215--NEXT217 evidence to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Run all targeted
tests, source compilation, the full pytest suite, manifest/output-hash checks,
boundary-flag checks, report-fence and canonical-path checks,
`git diff --check`, trailing-whitespace checks, and CodeGraph status. Keep the
overall goal active unless an all-gate discovery candidate exists and the
standalone report is ready for user review.
