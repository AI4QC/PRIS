# NEXT289--NEXT290 PRV Confidence-Deadzone Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the two cross-source PRV protection certificates improve
the unchanged BROAD gate when only high-confidence certificate tails may alter
the frozen NEXT224 risk score.

**Architecture:** NEXT289 reuses the exact NEXT268 PRV hypotheses, endpoint-
blind cutoffs, NEXT224 frontier, threshold, support, local-width and amplitude
grids, evaluator, folds, and SAFE/BROAD gates. It replaces the full linear
signed certificate used by NEXT269 with a symmetric central deadzone. NEXT290
is an exact diagnostic reproducer for AUC+SAFE/non-BROAD NEXT289 candidates.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, the existing
NEXT223/NEXT268/NEXT269/NEXT270 evaluator and provenance helpers, SHA-256
manifests, and atomic directory publication.

## Frozen scientific design

### Competing approaches considered

1. **Confidence deadzone (selected).** Preserve two-sided PRV evidence but
   leave the score unchanged for ambiguous certificates. This directly tests
   whether the NEXT269 protected-retention loss came from treating intermediate
   PRV values as decisive.
2. **Asymmetric risk/relief amplitudes (rejected).** Interpolating between the
   NEXT269 symmetric score and NEXT287 one-sided score would add a tunable trade-
   off chosen after both outcomes were visible. It is less interpretable and
   more vulnerable to discovery overfit.
3. **New weighted-Delaunay invariant (deferred).** This is the next mechanism
   family if confidence gating fails, but it requires a separate feature
   materialization and cross-source audit rather than another score transform.

### Exact formula

Let `P(x) in [0,1]` be a published NEXT268 bounded protection certificate and
define signed confidence `z = 2 P - 1`. For frozen deadzone `d`, define

```text
G_d(z) = sign(z) * max((abs(z) - d) / (1 - d), 0).
```

For frozen NEXT224 score `s`, threshold `t`, repair width `R`, local-width
fraction `f`, and amplitude fraction `a`, define

```text
h = f R
w(s) = max(0, 1 - abs(s - t) / h)
s_deadzone = max(0, s - a h w(s) G_d(2 P(x) - 1)).
```

The exact margin edges have `w=0`. If `P` lies in the confidence deadzone,
the correction is exactly zero. High protection (`P` near one) lowers risk;
high severe-like confidence (`P` near zero) raises risk. Unsupported rows,
nonfinite certificates, and rows outside the local interval keep the NEXT224
score and support. At analytic `d=0`, the implementation must reproduce the
NEXT269 signed score exactly; `d=0` is a test identity, not a NEXT289 candidate.

### Candidate universe

Use exactly the two published NEXT268 eligible hypotheses:

- `prv_chebyshev_ratio_cv__protected_low`
- `prv_volume_ratio_cv__protected_low`

Freeze only:

```text
deadzone_fraction in {1/2, 3/4}
local_width_fraction in {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}
amplitude_fraction in {1/4, 1/2, 1}
```

`d=1/2` acts only for `P<1/4` or `P>3/4`; `d=3/4` acts only for
`P<1/8` or `P>7/8`. The formal universe is one unchanged NEXT224 reproduction
control plus `2 * 2 * 7 * 3 = 84` new candidates, total 85. No intermediate
deadzone, asymmetric amplitude, extra cutoff, direction reversal, feature
interaction, source-specific coefficient, or post-outcome candidate is
allowed.

### Gates and branch decision

Use the unchanged source-AUC, twelve SAFE-cell, and BROAD gates. A reportable
candidate must pass both source-AUC gates and all SAFE cells. If any new
candidate passes BROAD, freeze the selected discovery formula but keep
validation and replication sealed and stop for the independent report. If no
candidate passes BROAD, authorize NEXT290 only for the exact sorted
AUC+SAFE/non-BROAD identities and their SHA-256 digest.

NEXT290 adds no search dimension. It must reproduce every authorized evaluator
record and compare the closest tuple `(failed_constraint_count,
normalized_shortfall_sum)` with NEXT270 `(5, 0.0955435292756307)`. A tied
failure count requires strictly smaller shortfall. If it does not improve,
terminate PRV score-combination development; any continuation must use a newly
frozen physical invariant rather than another deadzone/amplitude refinement.

### Hard no-DFT boundary

Every executable quantity may use only composition and the initial, raw,
unrelaxed periodic geometry. Discovery outcomes are offline labels only. All
formal manifests must record false for DFT calculation, per-structure DFT
values, learned energy/force/stress proxies, model/proxy potentials, physical
relaxation, opened validation outputs, and replication endpoints. Validation
and replication geometry remains unopened. Canonical paper, note, README, and
preregistration files must not change.

## Task 1: NEXT289 score and grammar with TDD

**Files:**

- Create: `tests/test_next289_prv_confidence_deadzone_search.py`
- Create: `src/next289_prv_confidence_deadzone_search.py`

**Step 1: Write failing analytic tests**

Test exact `G_d` values at `P={0,1/8,1/4,1/2,3/4,7/8,1}` for both frozen
deadzones. Test exact margin-center, interior, edge, outside, missing, and
unsupported behavior. At analytic `d=0`, compare the full score vector with
`next269.prv_margin_local_score` exactly.

**Step 2: Run focused tests and verify red**

```bash
python -m pytest \
  tests/test_next289_prv_confidence_deadzone_search.py -q
```

Expected: collection fails because the NEXT289 module does not exist.

**Step 3: Implement minimal transform and score**

Validate array shapes, certificate range, exact inherited grids, deadzone
identity, finite base values on support, exact edge zeroing, nonnegative output,
unchanged support, and missing-certificate fallback.

**Step 4: Add failing grammar/materialization tests**

Assert one control, 84 new candidates, unique stable JSON keys, exact two
deadzones, inherited grids, no hidden degrees of freedom, exact virtual-score
encoding, and deterministic confidence-active counts.

**Step 5: Implement specs and materializer; rerun green**

Use the existing endpoint-blind bounded-protection and exact asinh/sinh score
encoding patterns. The focused file must pass.

## Task 2: NEXT289 formal discovery runner

**Files:**

- Modify: `tests/test_next289_prv_confidence_deadzone_search.py`
- Modify: `src/next289_prv_confidence_deadzone_search.py`

**Step 1: Test fail-closed runner boundaries**

Cover missing/forked inputs, output overwrite refusal, exact candidate counts,
eligible hypothesis digest, false forbidden-mechanism flags, sealed endpoints,
and atomic publication.

**Step 2: Implement the frozen runner**

Reconstruct the exact NEXT224 frontier via existing helpers, verify NEXT268,
materialize 85 candidates, run the unchanged evaluator, reproduce the base,
and publish:

- `MANIFEST.json`
- `NEXT289_PRV_CONFIDENCE_DEADZONE_CATALOGUE.json`
- `NEXT289_DISCOVERY_EVALUATION.json`
- `NEXT289_FROZEN_CANDIDATE.json`
- `next289_prv_confidence_deadzone_search.parquet`

Record every input, executed-source, and output hash and publish atomically.

**Step 3: Run formal NEXT289 once**

Publish only to:

```text
$PRIS_ARCHIVE/next289_prv_confidence_deadzone_search_v1
```

Then enforce the frozen branch decision before creating NEXT290.

## Task 3: Conditional NEXT290 exact diagnostic

**Files:**

- Create: `tests/test_next290_prv_confidence_deadzone_broad_diagnostic.py`
- Create: `src/next290_prv_confidence_deadzone_broad_diagnostic.py`

**Step 1: Write and run failing tests**

Test exact AUC+SAFE/non-BROAD selection, sorted identity digest, exclusion of
any added/missing candidate, evaluator reproduction, failure-table definition,
lexicographic closest selection, and comparison with NEXT270.

**Step 2: Implement exact reproduction only and verify green**

Reuse the unchanged NEXT270 BROAD diagnostic definition. Do not add a candidate,
feature, deadzone, amplitude, width, threshold, or direction.

**Step 3: Run formal NEXT290 only if authorized**

Publish only to:

```text
$PRIS_ARCHIVE/next290_prv_confidence_deadzone_broad_diagnostic_v1
```

## Task 4: Evidence and independent report

**Files:**

- Modify: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

Run the focused tests and the complete repository suite. Independently verify
published hashes, candidate digests, forbidden-mechanism flags, and sealed
endpoints. Append the prospective formula, full results, closest residual or
BROAD pass, hashes, and conclusion to the independent report. Check CodeGraph
has no pending files and confirm no canonical protected path changed.

There are intentionally no commit, branch, merge, or cleanup steps because the
user requires additive work in the existing dirty checkout.
