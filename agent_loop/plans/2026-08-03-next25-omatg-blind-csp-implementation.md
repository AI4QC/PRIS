# NEXT25 OMatG Composition-Only Blind CSP Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task with test-driven development.

**Goal:** Generate a composition-only OMatG CSP cohort, seal frozen NEXT23
analytic and Pauling decisions without reference geometry, then perform one
blind evaluation against physically separated MP-20 DFT reference structures.

**Architecture:** Add a field-restricted MP-20 composition selector, an OMatG
output sanitizer, a source-specific frozen-rule applier, a post-freeze
reference extractor/evaluator, and an aggregate reporter.  Reuse the tested
NEXT20/NEXT22 feature builders and NEXT24 Pauling controls without modifying
their scientific definitions.  Keep identifier-bearing and reference
artifacts outside the repository.

**Tech Stack:** Python 3.11 for repository code; isolated OMatG runtime;
PyTorch, PyTorch Geometric, ASE, pymatgen, LMDB, pandas, NumPy, pytest,
Parquet, deterministic ZIP/JSON manifests.

### Task 1: Freeze composition-only cohort

**Files:**
- Create: `tests/test_next25_omatg_compositions.py`
- Create: `src/next25_omatg_compositions.py`

Write failing tests for allowed-field-only LMDB access, exact source hashes,
reduced-formula uniqueness and split exclusion, deterministic salted 512-row
selection, exact full-composition preservation, projected metadata, dummy-LMDB
schema, no-replace publication, and input/source rehash.  Implement the
minimum selector and publish the external composition-only cohort.

### Task 2: Build and verify isolated OMatG runtime

Create an external no-replace environment and official source/model snapshot.
Verify source and Hugging Face revisions, model/config hashes, checkpoint load,
CUDA/software versions, deterministic seed propagation, and a non-cohort smoke
generation.  The runtime configuration must point train, validation, and
prediction inputs to composition-only dummy LMDB files and must not reference
the MP-20 test LMDB.

### Task 3: Run OMatG and sanitize generated x0

**Files:**
- Create: `tests/test_next25_omatg_holdout.py`
- Create: `src/next25_omatg_holdout.py`

Write failing tests for raw-output provenance, exact frame/order/composition
coverage, full periodicity, finite coordinates/cells, canonical deterministic
archive generation, no label-like metadata, numeric geometry preservation,
no-replace publication, and source rehash.  Run the frozen OMatG command once
for all 512 inputs, retain all outputs, and sanitize them.

### Task 4: Seal analytic and Pauling predictions

**Files:**
- Create: `tests/test_next25_apply_rule.py`
- Create: `src/next25_apply_rule.py`

Write failing tests for immutable NEXT23 law validation, exact cohort/feature
joins, fail-open unsupported rows, endpoint-free schema, no-refit manifest,
no-replace publication, and source rehash.  Run SIVR and SCBVE, apply the
frozen law, and run unchanged Pauling 2--5 controls on the identical archive.
Record a freeze timestamp and hashes before any reference extraction.

### Task 5: Open DFT references once and evaluate

**Files:**
- Create: `tests/test_next25_omatg_evaluate.py`
- Create: `src/next25_omatg_evaluate.py`

Write failing tests for prediction-before-label chronology, exact source row
and composition pairing, reference extraction only after freeze, official
StructureMatcher tolerances, corrected RMSD, Wilson gates, identical-cohort
Pauling comparisons, subgroup accounting, and no post-label refit.  Extract
only selected reference rows, evaluate once, and seal all outputs.

### Task 6: Verify and report

Independently recompute source/output hashes, selection ranks, frozen scores,
decisions, matches, corrected RMSDs, Wilson bounds, and Pauling metrics.  Scan
prediction artifacts for forbidden endpoint fields.  Run focused tests, the
full repository suite, and CodeGraph status.  Write a new standalone NEXT25
report with the CSP-versus-stability limitation explicit.  Do not modify any
existing report or paper before user confirmation.
