# NEXT287--NEXT288 PRV One-Sided Relief Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the two cross-source periodic radical-Voronoi (PRV)
protection certificates can repair the current SCIGEN protected-retention
bottleneck when they are allowed only to reduce, never increase, the frozen
NEXT224 risk score.

**Architecture:** NEXT287 reuses the exact frozen NEXT268 eligible hypotheses,
endpoint-blind cutoffs, NEXT224 base score, threshold, support, repair width,
width grid, amplitude grid, evaluator, folds, and SAFE/BROAD gates. It changes
only the score geometry from the NEXT269 symmetric signed correction to a
one-sided triangular relief. NEXT288 is an exact diagnostic reproducer for the
NEXT287 candidates that pass both-source AUC and all SAFE cells but do not pass
BROAD. Validation and replication remain sealed in either outcome.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, the existing
NEXT223/NEXT227/NEXT257/NEXT268/NEXT269/NEXT270 evaluator and artifact helpers,
SHA-256 manifests, and atomic directory publication.

## Frozen scientific design

### Why this is a distinct test

The NEXT269 correction is symmetric:

```text
s_signed = max(0, s + a h w(s) (1 - 2 P(x)))
```

where `s` is the frozen NEXT224 risk score, `P in [0,1]` is an endpoint-blind
bounded protection certificate, `h` is the local width, and `w` is the frozen
triangular margin kernel. Low protection therefore increases risk as strongly
as high protection reduces it. NEXT270's best record failed only the pooled
SCIGEN and folds 0--3 `protected_kept` constraints. This plan tests the
conservative alternative

```text
h = local_width_fraction * repair_width
w(s) = max(0, 1 - abs(s - base_threshold) / h)
s_relief = max(0, s - amplitude_fraction * h * w(s) * P(x)).
```

The exact edge `abs(s - base_threshold) >= h`, including numerical equality,
has `w=0`. Unsupported rows, nonfinite certificates, and rows outside the
local interval keep the NEXT224 score and support. Thus `0 <= s_relief <= s`
for every supported row. A good PRV packing certificate may pardon a
near-boundary structure; absence of that certificate is not itself evidence of
invalidity.

Earlier NEXT216/NEXT218 branches tested one-sided relief using a different
base frontier and 22 older raw-x0 certificates. They do not answer this
question because the two PRV certificates were not yet defined and the PRV
signed branch is the only recent mechanism that strictly improved the frozen
BROAD residual. This plan does not reopen or retune those earlier branches.

### Candidate universe

Use exactly the two NEXT268 eligible directions and their published cutoffs:

- `prv_chebyshev_ratio_cv__protected_low`
- `prv_volume_ratio_cv__protected_low`

Use exactly the inherited grids:

```text
local_width_fraction in {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}
amplitude_fraction in {1/4, 1/2, 1}
```

The universe is one unchanged NEXT224 reproduction control plus
`2 * 7 * 3 = 42` new candidates. There is no extra feature, cutoff, exponent,
kernel, interaction, source-specific coefficient, fold-specific coefficient,
or post-outcome candidate addition. The feature normalization population and
cutoffs remain endpoint-blind and frozen by NEXT268.

### Gates and continuation rule

NEXT287 must use the unchanged NEXT269 evaluator and fixed selector. A new
candidate is reportable only if it passes the two frozen source-AUC gates and
all twelve SAFE cells. A candidate is a discovery BROAD pass only if the
unchanged evaluator says so; no threshold or requirement may be weakened.

If any new candidate passes BROAD, publish the selected discovery formula but
do not open validation or replication. Stop for an independent report and user
review. If no candidate passes BROAD, authorize NEXT288 only for the exact
sorted AUC+SAFE/non-BROAD identities and freeze their SHA-256 digest in the
NEXT287 manifest. NEXT288 must reproduce those records and compare the closest
tuple `(failed_constraint_count, normalized_shortfall_sum)` with the published
NEXT270 tuple `(5, 0.0955435292756307)`. A tie in failure count requires a
strictly smaller shortfall. No new search follows NEXT288 without another
pre-outcome design.

### Hard boundary

Every executable quantity may use only composition and initial, raw,
unrelaxed periodic geometry. Discovery outcomes are offline evaluation labels
only. The scripts must record false for DFT calculation, per-structure DFT
value use, learned energy/force/stress proxies, model/proxy potentials,
relaxation, trajectories, internal validation geometry, internal replication
geometry, and both replication endpoints. No canonical paper, note, README,
or preregistration file may change.

## Task 1: NEXT287 score and candidate grammar

**Files:**

- Create: `tests/test_next287_prv_one_sided_relief_search.py`
- Create: `src/next287_prv_one_sided_relief_search.py`

**Step 1: Write failing analytic score tests**

Test an exact hand-calculated vector containing `P=0`, `P=1/2`, `P=1`, a
nonfinite certificate, an unsupported row, the center, an interior point, and
both exact interval edges. Assert unchanged support, exact edge inactivity,
missing-term fallback, nonnegativity, and elementwise `score <= base_score`.

**Step 2: Run the focused test and verify red**

Run:

```bash
python -m pytest \
  tests/test_next287_prv_one_sided_relief_search.py -q
```

Expected: collection fails because the NEXT287 module does not exist.

**Step 3: Implement the minimal score function**

Implement `prv_one_sided_relief_score` with exact-type/grid validation,
finite-on-support validation, explicit edge zeroing, nonfinite-certificate
fallback, unchanged support, and postcondition checks for finite,
nonnegative, nonincreasing scores.

**Step 4: Add failing grammar tests**

Construct a two-row synthetic eligible table. Assert one reproduction control,
42 new specs, stable unique JSON keys, exact inherited fractions/amplitudes,
the one-sided score-composition marker, and rejection of malformed hypotheses,
directions, cutoffs, grids, and duplicate identities.

**Step 5: Implement candidate specs and exact virtual materialization**

Reuse only the existing bounded-protection and exact asinh/sinh virtual-score
encoding pattern. Assert the decoded evaluator value equals the physical
one-sided score and that support never changes.

**Step 6: Run NEXT287 unit tests and verify green**

Run the focused command from Step 2. Expected: all NEXT287 tests pass.

## Task 2: NEXT287 formal discovery runner

**Files:**

- Modify: `tests/test_next287_prv_one_sided_relief_search.py`
- Modify: `src/next287_prv_one_sided_relief_search.py`

**Step 1: Write failing runner-boundary tests**

Cover missing/forked input hashes, pre-existing output refusal, formal
candidate counts, eligible digest identity, false forbidden-mechanism flags,
sealed validation/replication flags, and atomic output publication.

**Step 2: Implement the runner**

Reconstruct the exact NEXT224 frontier through the existing NEXT269 path and
verification helpers, attach exact NEXT267 PRV values, verify the NEXT268
eligible digest, materialize 43 candidates, run the unchanged evaluator, and
publish:

- `MANIFEST.json`
- `NEXT287_PRV_ONE_SIDED_RELIEF_CATALOGUE.json`
- `NEXT287_DISCOVERY_EVALUATION.json`
- `NEXT287_FROZEN_CANDIDATE.json`
- `next287_prv_one_sided_relief_search.parquet`

Record every input, executed-source, and output SHA-256. Publish through a
temporary sibling directory and `os.replace` only after all checks pass.

**Step 3: Run focused tests**

Expected: all NEXT287 tests pass.

**Step 4: Run formal NEXT287 exactly once**

Use `python` and publish to:

```text
$PRIS_ARCHIVE/next287_prv_one_sided_relief_search_v1
```

Do not overwrite an existing directory.

**Step 5: Enforce the branch decision**

If a BROAD pass exists, stop the computational branch, keep validation and
replication sealed, and write the independent report. Otherwise freeze the
exact AUC+SAFE/non-BROAD candidate digest before creating NEXT288.

## Task 3: NEXT288 exact BROAD diagnostic

**Files:**

- Create: `tests/test_next288_prv_one_sided_broad_diagnostic.py`
- Create: `src/next288_prv_one_sided_broad_diagnostic.py`

**Step 1: Write failing diagnostic tests**

Test exact identity filtering, rejection of any extra/missing candidate,
reproduction of evaluator records, deterministic failure-table construction,
lexicographic closest selection, and strict comparison with the NEXT270
reference.

**Step 2: Run and verify red**

Run:

```bash
python -m pytest \
  tests/test_next288_prv_one_sided_broad_diagnostic.py -q
```

Expected: collection fails because the NEXT288 module does not exist.

**Step 3: Implement exact reproduction only**

Reuse NEXT270's unchanged BROAD table builder and normalized-shortfall
definition. NEXT288 must not add a candidate, feature, threshold, direction,
or formula. Publish only if all NEXT287 authorized evaluator records reproduce
exactly.

**Step 4: Run focused tests and formal diagnostic**

Publish to:

```text
$PRIS_ARCHIVE/next288_prv_one_sided_broad_diagnostic_v1
```

Only run this task if NEXT287 has no BROAD pass and formally authorizes it.

## Task 4: Verification and independent report

**Files:**

- Modify: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

**Step 1: Run focused tests**

Run the NEXT287 test and, if created, NEXT288 test together.

**Step 2: Run the complete suite**

```bash
python -m pytest -q
```

**Step 3: Verify formal artifacts**

Independently recompute all published-output and executed-source hashes, assert
all forbidden-mechanism and sealed-endpoint flags, and check the eligible and
diagnostic identity digests.

**Step 4: Append the independent report**

Report the prospective rationale, exact formula, candidate universe, all
cross-source/gate results, closest residual or BROAD pass, hashes, tests, and
the no-DFT/sealed-endpoint boundary. Do not modify canonical content.

**Step 5: Check CodeGraph and protected paths**

Require no pending CodeGraph sync and an empty status for `paper/`, `tex/`,
`notes/`, `README.md`, and `PREREG.md` attributable to this branch.

There are intentionally no commit steps: the user requires additive work in
the existing dirty checkout and has not authorized commits, branches, merges,
or cleanup.
