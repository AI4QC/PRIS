# NEXT23 Analytic Relaxation-Change Screening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze and blind-test a DFT-free analytic law for screening WBM unrelaxed structures that undergo large structural reorganization during DFT relaxation.

**Architecture:** Add four no-replace stages: a disjoint geometry-only WBM freezer, a development-only formula freezer, a label-free frozen-rule applier, and a post-freeze evaluator.  Reuse the already tested NEXT20--NEXT22 analytic feature builders.  Keep identifier-bearing blind artifacts outside the repository and publish only aggregate evidence in a new standalone report.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, ASE, pymatgen, pytest, Parquet, deterministic ZIP/JSON manifests.

### Task 1: Disjoint geometry-only WBM cohort

**Files:**
- Create: `tests/test_next23_wbm_holdout.py`
- Create: `src/next23_wbm_holdout.py`

**Step 1: Write failing tests**

Test deterministic salted selection after exclusions, exact ID disjointness,
atom-count bounds, rejection of label-like exclusion columns, no-replace
publication, manifest hashes, canonical ZIP order, and unchanged upstream
inputs.

**Step 2: Verify RED**

Run: `pytest -q tests/test_next23_wbm_holdout.py`

Expected: collection fails because `src.next23_wbm_holdout` does not exist.

**Step 3: Implement the minimal freezer**

Reuse the validated input readers and canonical frame writer from NEXT14, but
use protocol `2026-08-02-next23-wbm-relaxation-change-holdout-v1`, salt
`next23-wbm-relaxation-change-blind-v1`, formal sample size 8,192, and a required
metadata-only exclusion file whose material IDs are hashed into the manifest.
Never import or accept a label table.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_next23_wbm_holdout.py tests/test_next14_wbm_holdout.py`

Expected: all pass.

### Task 2: Development-only finite formula freeze

**Files:**
- Create: `tests/test_next23_relaxation_rule.py`
- Create: `src/next23_relaxation_rule.py`

**Step 1: Write failing tests**

Test the fixed 17-candidate catalogue, robust median/IQR transform, low-risk
directions, fail-open missing values, one-sided Wilson bounds, deterministic
threshold/tie selection, all-gate eligibility, immutable frozen JSON, and
rejection of forbidden feature names or non-finite parameters.

**Step 2: Verify RED**

Run: `pytest -q tests/test_next23_relaxation_rule.py`

Expected: collection fails because the module does not exist.

**Step 3: Implement the minimal search/freezer**

Read only the exposed 2,048-row NEXT20/NEXT21/NEXT22 features and the declared
development endpoint.  Freeze exactly one candidate if it meets all development
gates; otherwise publish a no-candidate result and stop before blind labels.
Record all source/input hashes and the complete finite scan.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_next23_relaxation_rule.py`

Expected: all pass.

### Task 3: Label-free frozen-rule application

**Files:**
- Create: `tests/test_next23_apply_rule.py`
- Create: `src/next23_apply_rule.py`

**Step 1: Write failing tests**

Test exact ID joins across the three feature tables, source-manifest hash
validation, score reproduction from the frozen formula, fail-open unsupported
rows, prohibition of label/DFT/relaxation columns, no-replace prediction
publication, and frozen-law/source hash checks.

**Step 2: Verify RED**

Run: `pytest -q tests/test_next23_apply_rule.py`

Expected: collection fails because the module does not exist.

**Step 3: Implement the minimal applier**

Produce only identifiers, analytic support, risk score, reject decision, and
the frozen provenance needed for later evaluation.  Do not accept a label path.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_next23_apply_rule.py`

Expected: all pass.

### Task 4: Post-freeze evaluation and Pauling comparison

**Files:**
- Create: `tests/test_next23_evaluate.py`
- Create: `src/next23_evaluate.py`

**Step 1: Write failing tests**

Test that prediction publication predates label opening, hashes are unchanged,
IDs join one-to-one, primary Wilson gates are exact, continuous/tier diagnostics
are label-only, Pauling comparison uses the identical cohort, outputs are
aggregate-only in the repository, and no-refit fields are explicit.

**Step 2: Verify RED**

Run: `pytest -q tests/test_next23_evaluate.py`

Expected: collection fails because the module does not exist.

**Step 3: Implement the minimal evaluator**

Open the blind endpoint once after predictions exist, save identifier-bearing
joins only to the external private directory, and publish immutable aggregate
JSON/manifest to a new repository output directory.

**Step 4: Verify GREEN**

Run: `pytest -q tests/test_next23_evaluate.py`

Expected: all pass.

### Task 5: Execute the protocol

**Files:**
- Create externally: `next23_wbm_relaxation_change_holdout/`
- Create externally: `next23_wbm_{sivr,madelung,scbve}/`
- Create externally: `next23_relaxation_rule_freeze/`
- Create externally: `next23_relaxation_predictions/`
- Create externally: `next23_relaxation_evaluation_private/`
- Create: `outputs/20260802_next23_relaxation_change_evaluation/`

Run the geometry freezer, then the three existing no-DFT feature builders, then
the frozen applier.  Confirm prediction manifests and hashes before invoking the
evaluator.  Do not open blind labels if the development freeze has no candidate.

### Task 6: Verification and standalone report

**Files:**
- Create: `reports/2026-08-02-next23-analytic-relaxation-change-screening.md`

Run focused tests, the full repository test suite, CodeGraph status, manifest
hash verification, forbidden-token scans, ZIP integrity checks, and independent
metric recomputation.  Write the standalone report only after results exist.
State separately what is implemented, what passed development, what passed blind
validation, and what remains unproven.  Do not modify any existing report,
paper, README, or canonical scientific document.

