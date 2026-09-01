# NEXT48 QMOF External Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run a sealed-order, zero-refit external QMOF validation of the frozen NEXT31 DFT-free packing law and compare it with frozen Pauling 2--5 controls.

**Architecture:** One additive module owns three irreversible stages: metadata-only protocol/cohort freeze, unrelaxed-x0 prediction freeze, and post-prediction relaxed-structure evaluation.  It reuses the deterministic geometry archive, NEXT27 features, NEXT31 score, Pauling controls, CrystalNN fingerprint, Wilson intervals, and atomic no-replace publishers already in the repository.

**Tech Stack:** Python 3.12, ASE 3.28, pymatgen 2026.5.4, matminer 0.10.1, NumPy, pandas, SciPy, scikit-learn, pytest, ZIP/CSV/JSON/Parquet.

### Task 1: Freeze mapping and protocol

**Files:**
- Create: `tests/test_next48_qmof_external_validation.py`
- Create: `src/next48_qmof_external_validation.py`

1. Write a fixture archive with malformed initial/final CIF payloads plus a
   metadata CSV, proving mapping can be frozen from central directories and
   whitelisted columns without parsing coordinates.
2. Write tests for exact archive/rule hashes, exact-name mapping, unmatched
   accounting, relaxed-ID presence, endpoint/gate constants, and no-replace.
3. Run `conda run -n newpauling pytest tests/test_next48_qmof_external_validation.py -q`
   and confirm failure because the module does not exist.
4. Implement `freeze_qmof_protocol()` and minimal strict ZIP/CSV helpers.
5. Re-run the focused tests and confirm the protocol tests pass.

### Task 2: Freeze x0 predictions

**Files:**
- Modify: `tests/test_next48_qmof_external_validation.py`
- Modify: `src/next48_qmof_external_validation.py`

1. Add tests with valid tiny unrelaxed CIFs and an intentionally invalid relaxed
   payload, injected analytic/Pauling calculators, one supported row, and one
   fail-open row.
2. Confirm the prediction tests fail because `freeze_qmof_predictions()` is
   absent.
3. Implement strict protocol validation, CIF-to-geometry projection,
   deterministic geometry archive publication, existing NEXT27 feature
   calculation, exact NEXT31 scoring, Pauling decisions, error capture, source
   accounting, and prediction/source hashes.
4. Confirm prediction tests pass and that the invalid relaxed payload was never
   opened.

### Task 3: Evaluate relaxed QMOF structures

**Files:**
- Modify: `tests/test_next48_qmof_external_validation.py`
- Modify: `src/next48_qmof_external_validation.py`

1. Add tests that evaluation refuses unsealed or hash-mismatched predictions,
   opens only mapped relaxed members, accepts atom reordering with equal element
   counts, fails unsupported pairs open, and computes frozen NEXT31 and Pauling
   decision metrics.
2. Confirm the evaluator tests fail because `evaluate_qmof_relaxation()` is
   absent.
3. Implement independent CrystalNN fingerprints, absolute log-volume change,
   endpoint accounting, overall/source-slice metrics, Pauling head-to-head,
   deterministic joined/result artifacts, and no-refit/no-replace manifests.
4. Re-run focused tests and confirm all pass.

### Task 4: Freeze and run the real external cohort

**Files:**
- Create externally under: `$PRIS_ARCHIVE/next48_*`

1. Freeze the real QMOF protocol before opening any CIF coordinate payload.
2. Validate the protocol artifact, archive/rule hashes, 4,119 included rows, and
   28 unmatched rows.
3. Run x0 prediction freeze on all eligible initial CIFs; validate geometry,
   feature, Pauling, and prediction hashes and fail-open accounting.
4. Only after step 3 succeeds, run the relaxed-structure evaluator.
5. Record total and per-source metrics without changing any threshold or cohort.

### Task 5: Report and regression verification

**Files:**
- Create: `reports/2026-08-03-next48-qmof-external-validation.md`

1. Write a standalone report with protocol chronology, formulas, source/citation
   provenance, support/failure accounting, overall and per-source results,
   Pauling comparison, limitations, and exact artifact hashes.
2. Run focused tests for NEXT48 plus reused NEXT27/NEXT31/Pauling/fingerprint
   components.
3. Load `superpowers:verification-before-completion` and run the full repository
   test suite in the `newpauling` Conda environment.
4. Recheck CodeGraph health and artifact/report SHA-256 values.
5. Confirm `paper/`, `README.md`, prior scripts, prior reports, and existing
   outputs were not modified or replaced.  Do not commit the shared dirty
   checkout.
