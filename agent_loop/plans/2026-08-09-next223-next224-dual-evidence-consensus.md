# NEXT223--NEXT224 Dual-Evidence Consensus Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether distinct pre-DFT protection and severity certificates can
resolve the remaining NEXT222 SCIGEN-protection versus WyFormer-savings conflict
without weakening any frozen discovery gate.

**Architecture:** Reconstruct the exact NEXT222 two-term score, then add one
ordered, budget-allocated dual-evidence correction built from the exact 22
endpoint-blind NEXT215/NEXT216 certificates. NEXT223 evaluates the complete
frozen candidate universe; NEXT224 only reproduces its eligible
AUC+SAFE/non-BROAD population and computes the unchanged BROAD residual.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, existing NeoPauling
cross-source evaluator, pytest, SHA-256 manifests.

Date: 2026-08-09

Status: frozen before any NEXT223 candidate score is joined to discovery
outcomes.

## Scientific rationale and alternatives

NEXT222 stopped because no additional univariate signed certificate strictly
improved its two-term path. That path reduced normalized BROAD shortfall to
`0.1564570050830728` but still failed five SCIGEN `protected_kept`
requirements and WyFormer fold 4 `savings_lower`.

Three mechanisms were considered before opening this branch:

1. another univariate term or denser amplitude grid, rejected by the exact
   NEXT222 stopping rule;
2. product/minimum protection conjunctions, rejected as the primary direction
   because earlier frozen branches showed substantial information loss;
3. distinct protection and severity evidence, selected because it strictly
   generalizes the successful signed correction while explicitly allocating a
   fixed correction budget between protection rescue and severe rejection.

The third mechanism tests a concrete diagnosis: one feature need not be an
equally good detector of both protected and severe structures. An ordered pair
can lower risk when its first certificate says protected, raise risk when its
second certificate says severe, and cancel when the evidence conflicts.

## Immutable no-DFT and data boundary

- Executable quantities may use composition and initial, unrelaxed geometry
  only.
- The executable law must not use a DFT calculation or value; a learned
  energy, force, or stress proxy; a model or proxy potential; a relaxed
  structure; a trajectory; or physical relaxation.
- Discovery outcomes are offline labels only.
- Use only the already opened SCIGEN and WyFormer discovery endpoints.
- Internal validation and replication endpoints remain physically sealed
  unless an eligible new candidate passes every discovery gate.
- Preserve every existing script, result, report, and canonical artifact. Add
  new scripts, tests, formal directories, and an independent-report section
  only.

## Frozen inputs and certificate universe

- Base score/support: exact NEXT222 final two-term path and exact NEXT214
  support.
- Activation: original NEXT214 score in the exact lower-inclusive,
  upper-exclusive interval
  `[0.17470215862148156, 0.570892727856757)`.
- Certificate identities: exact 22 NEXT215 eligible hypotheses, digest
  `2e5000a319188a6191922a499b8151e28bb603ba06e70cff8750ec582e887b41`.
- Certificate cutoffs and directions: exact endpoint-blind NEXT216
  definitions; no refit or new cutoff.
- Amplitude grid: exact NEXT220 grid
  `{1/64, 1/32, 1/16, 1/8, 1/4}`.
- Protection-budget fractions: exact grid `{1/4, 1/2, 3/4}`.
- Support remains identical to NEXT214. If either pair operand is missing, the
  proposed correction is zero for that row.

## Frozen executable grammar

For original NEXT214 score `s0`, exact NEXT222 cumulative delta `d222`, repair
width `W`, protection-role certificate `P_protect`, risk-role certificate
`P_risk`, amplitude `beta`, and protection-budget fraction `lambda`, define on
active rows

```text
risk_evidence = 1 - P_risk
pair_delta = 2 * beta * W * (
    (1 - lambda) * risk_evidence - lambda * P_protect
)
score = max(0, s0 + d222 + pair_delta)
```

The pair is ordered. Interchanging its operands generally changes the score.
When both roles use the same certificate and `lambda=1/2`, the correction
reduces exactly to the already tested univariate form
`beta * W * (1 - 2P)`; only these equal-budget diagonal records are grammar
controls and are not eligible new laws. Of the 110 controls, the 100 whose
feature is unused by the final path must exactly reproduce the corresponding
NEXT222 depth-3 records; the 10 involving either already-used feature are
closed-form identity controls. The factor two keeps their amplitude identical
to NEXT220/NEXT222.

The complete NEXT223 catalogue contains:

- one unchanged NEXT222 control;
- `22 * 22 * 5 * 3 = 7260` ordered-pair/amplitude/allocation records;
- `22 * 5 = 110` equal-budget diagonal grammar controls, including 100 exact
  NEXT222 depth-3 reproduction controls;
- `7260 - 110 = 7150` eligible new candidates;
- `7261` total records.

No pair pruning, family filtering, outcome-dependent cutoff, coefficient
refit, source-specific rule, fold-specific rule, beam search, or manual
override is allowed.

## Frozen evaluation and selection

Run the unchanged cross-source evaluator. Report source pooled/macro/worst AUC,
all 12 SAFE cells, BROAD, support, and the complete gate counts separately for
all records, equal-budget diagonal controls, and eligible new candidates.

Only eligible new candidates may be selected. If one or more pass all
discovery gates, select deterministically using the unchanged evaluator order
and stop the discovery search. Otherwise select the unchanged evaluator's
best eligible AUC+SAFE record for reporting, without using BROAD residual to
choose it.

NEXT224 must reconstruct the exact eligible
AUC+SAFE/non-BROAD population, verify its sorted-key digest, and compute the
unchanged BROAD threshold diagnostic for every member. Rank by failed
constraint count, normalized shortfall sum, and candidate key. Compare the
global closest eligible record with the exact NEXT222 reference
`(6, 0.1564570050830728)`.

The branch is closed if no eligible candidate strictly improves that tuple.
If it strictly improves but does not pass all gates, any continuation requires
a new pre-outcome freeze. Validation/replication remains sealed in either
case. If an eligible candidate passes every discovery gate, freeze a separate
validation protocol before opening any validation endpoint.

## Task 1: NEXT223 score and catalogue helpers

**Files:**

- Create: `tests/test_next223_dual_evidence_consensus_search.py`
- Create: `src/next223_dual_evidence_consensus_search.py`

**Step 1: Write failing score tests**

Test exact ordered-pair algebra, operand-order sensitivity, budget-allocation
sensitivity, diagonal equal-budget reduction
to the univariate signed form, original-band activation, nonnegative flooring,
unchanged support, and missing-pair term-off behavior.

**Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_next223_dual_evidence_consensus_search.py
```

Expected: collection/import failure because the NEXT223 module does not exist.

**Step 3: Implement minimal score and catalogue helpers**

Implement `dual_evidence_consensus_score()` and
`build_dual_evidence_candidate_specs()` with exact validation and deterministic
JSON candidate keys. Expose constants for the fixed beta grid and candidate
counts.

**Step 4: Run focused tests and verify GREEN**

Expected: all NEXT223 helper tests pass.

## Task 2: NEXT223 formal discovery search

**Files:**

- Modify: `tests/test_next223_dual_evidence_consensus_search.py`
- Modify: `src/next223_dual_evidence_consensus_search.py`
- Create formal directory:
  `$PRIS_ARCHIVE/next223_dual_evidence_consensus_search_v1`

**Step 1: Add failing interface and fail-closed tests**

Require discovery-only endpoint parameters, exact input identity, output
non-overwrite, the 7261/110/7150 catalogue partition, and explicit false flags
for every forbidden mechanism.

**Step 2: Verify RED, implement the minimal formal runner, then verify GREEN**

Reconstruct NEXT222 from formal inputs, materialize all virtual score columns,
run the unchanged evaluator, and write catalogue, evaluation, selected-formula,
candidate Parquet, and manifest files atomically into a new directory.

**Step 3: Run the formal search once**

Use `python`, all exact formal
NEXT98--NEXT222 directories and design files, and no validation or replication
path.

## Task 3: NEXT224 exact BROAD residual diagnostic

**Files:**

- Create: `tests/test_next224_dual_evidence_broad_diagnostic.py`
- Create: `src/next224_dual_evidence_broad_diagnostic.py`
- Create formal directory:
  `$PRIS_ARCHIVE/next224_dual_evidence_broad_diagnostic_v1`

**Step 1: Write and run failing tests**

Test exact eligible AUC+SAFE/non-BROAD filtering, sorted-key digest,
deterministic residual ordering, discovery-only interface, and missing-input
failure.

**Step 2: Implement and verify GREEN**

Reproduce NEXT223 candidates and evaluator records exactly; do not search or
select a new formula. Compute the unchanged threshold tables and residuals,
write diagnostic JSON, per-candidate Parquet, and manifest, then apply the
frozen close/continue rule.

## Task 4: Report and verification

**Files:**

- Modify only the additive independent report:
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

**Step 1: Append NEXT223--NEXT224 methods, exact results, limitations, formal
paths, and hashes**

Do not modify `paper/`, `tex/`, `notes/`, `README.md`, `PREREG.md`, or any
existing script/result.

**Step 2: Run verification**

Run focused tests, `py_compile` for both scripts, the full pytest suite,
independent SHA-256 checks against both manifests, `git diff --check`, exact
trailing-whitespace checks for new files, canonical-path status checks, and
CodeGraph status after indexing catches up.

**Step 3: Keep the overall goal active unless every discovery, validation, and
replication requirement is actually satisfied**

Do not claim a replacement law from discovery-only evidence.
