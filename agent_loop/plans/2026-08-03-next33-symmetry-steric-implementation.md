# NEXT33 Symmetry-Steric Law Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and prospectively gate a single-x0, fully DFT-free law from approximate symmetry-recovery residuals and directional steric imbalance.

**Architecture:** One additive feature module computes representation-invariant symmetry and steric quantities, then an immutable batch builder joins them to the already sealed NEXT32 analytic table without endpoints. A separate bounded rule module scans only the frozen candidate catalogue on exposed development labels and either publishes one rule or stops before confirmation.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, ASE, spglib, pymatgen periodic distances, pytest, existing NEXT32 cohort/rule/evaluation infrastructure.

**Workspace note:** The checkout is a shared, largely untracked research tree. A separate worktree would omit NEXT19--NEXT32 dependencies. Execute additively in place; do not commit, overwrite, reset, clean, or change canonical documents.

### Task 1: Multi-tolerance symmetry recovery

**Files:**

- Create: `src/next33_symmetry_steric_features.py`
- Create: `tests/test_next33_symmetry_steric_features.py`

1. Write failing tests for translation, atom permutation, rigid rotation and integer-supercell invariance.
2. Add a near-symmetric toy structure where larger displacement increases recovery onset and residual while an exact P1/no-recovery structure returns zeros.
3. Test that the API reads only `Atoms` geometry and does not expose `sid`, space-group metadata, refined structures or forbidden endpoint fields.
4. Run the focused test and confirm the import failure.
5. Implement relative symprec grids, normalized point-operation counts and orbit fractions.
6. Implement element-wise Hungarian periodic matching for operation residuals without constructing a symmetrized structure.
7. Re-run until green.

### Task 2: Directional steric imbalance

**Files:**

- Modify: `src/next33_symmetry_steric_features.py`
- Modify: `tests/test_next33_symmetry_steric_features.py`

1. Add failing tests for unique self-image handling, supercell invariance, vector cancellation in a symmetric cell and monotonic compression response.
2. Freeze `w12` and overlap-square kernels and exact feature names from the design.
3. Implement unique periodic pairs with oriented displacement vectors, site scalar/vector loads and normalized tensor deviators.
4. Test finite fail-open behavior for invalid cells/radii and absence of DFT/energy/force/stress names.
5. Re-run the focused tests.

### Task 3: Immutable geometry-only batch

**Files:**

- Modify: `src/next33_symmetry_steric_features.py`
- Modify: `tests/test_next33_symmetry_steric_features.py`

1. Add a failing integration test using a deterministic geometry ZIP, NEXT32 feature parquet and both manifests.
2. Require exact cohort/feature hashes, `labels_opened=false`, `endpoint_fields_read=false`, identity equality and no-overwrite publication.
3. Publish `next33_symmetry_steric_features.parquet`, explicit family support/failure columns and a manifest with all executed source hashes.
4. Run a 16-structure smoke batch and record coverage/runtime.

### Task 4: Bounded NEXT33 rule search

**Files:**

- Create: `src/next33_symmetry_steric_rule.py`
- Create: `tests/test_next33_symmetry_steric_rule.py`

1. Add failing tests for the frozen term catalogue, risk directions, candidate pairs and symmetry-only promotion prohibition.
2. Reuse the NEXT32 endpoint classifier, one-sided Wilson bounds and six unchanged gates.
3. Test that zero-IQR terms disable only their formulas and no continuous weight is fitted.
4. Implement robust-z scan over five fixed rejection fractions and deterministic unique selection.
5. Always publish a scan; publish `NEXT33_FROZEN_SYMMETRY_STERIC_RULE.json` only when every gate passes.
6. Add label-free application tests with rule/feature manifest hash binding and no-overwrite output.

### Task 5: Execute exposed development

1. Build NEXT33 features on the sealed 4,096-row NEXT32 development cohort before reading endpoints in the new process.
2. Verify exact identity join with the existing immutable NEXT32 feature table.
3. Open only the already-exposed development endpoint artifact and run the bounded scan once.
4. If no candidate passes, preserve the scan and stop before all confirmation downloads.
5. If a candidate passes, freeze the formula, normalization constants, threshold and existing NEXT32 confirmation protocol.

### Task 6: Conditional confirmation

1. Only after promotion, download the three corrected official validation archives as opaque files and record size/ETag/last-modified/SHA-256.
2. Sanitize and parent-disjoint 2,048 structures per source without endpoint conversion.
3. Compute NEXT32+NEXT33 features, Pauling controls and fixed predictions for all 6,144 rows.
4. Freeze the confirmation protocol and identity hashes, then decode endpoints exactly once.
5. Evaluate aggregate and each source using the existing unchanged gates; never refit.

### Task 7: Standalone report and verification

**Files:**

- Create: `reports/2026-08-03-next33-symmetry-steric.md`

1. Report the exact formula or no-promotion outcome, symmetry-artifact safeguard, full candidate failures and Pauling comparison.
2. Keep all old scripts, reports, paper, README and PREREG unchanged.
3. Run focused NEXT33 tests and `conda run -n newpauling python -m pytest -q`.
4. Verify every manifest output/source hash and CodeGraph synchronization.
5. Stop at the standalone-report boundary for user confirmation; keep the long-running law-discovery goal active unless its full objective is actually proven.
