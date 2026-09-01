# NEXT112--NEXT114 Obstruction Morphology Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and discovery-test a canonical, strictly no-DFT obstruction-morphology certificate without changing NEXT109--NEXT111 or any canonical manuscript content.

**Architecture:** NEXT112 independently solves the frozen NEXT109 primary LP, then derives component aggregates and secondary-LP optimum values rather than reading a non-unique optimizer. NEXT113 reuses the locked discovery geometry routers to materialize core and expanded features. NEXT114 freezes eligible tails and a finite near-miss search before opening discovery endpoints.

**Tech Stack:** Python 3, NumPy, SciPy HiGHS linear programming, pandas/Parquet, pymatgen periodic Voronoi geometry, pytest, SHA-256 manifests.

**Repository constraint:** Work in the current checkout, make additive edits only, and do not commit or create a worktree. Do not modify `paper/`, `tex/`, `notes/`, `README.md`, `PREREG.md`, NEXT109--NEXT111, or prior reports.

### Task 1: Prove the graph-level NEXT112 contract with failing tests

**Files:**
- Create: `tests/test_next112_obstruction_morphology.py`
- Create after RED: `src/next112_obstruction_morphology.py`

**Steps:**
1. Add a feasible-network test expecting all six morphology values to be zero.
2. Run `python -m pytest tests/test_next112_obstruction_morphology.py -q` and confirm import failure.
3. Add tests for global imbalance, disconnected imbalance, the connected Hall cut, an isolated site, and an empty graph.
4. Add exact tests for site/edge permutation, integer replication, common charge scaling, and a symmetric multi-optimum graph.
5. Implement the minimum code for the primary LP, component aggregates, minimax LP, and positive-side slack-range LPs.
6. Re-run the focused test after each behavior and keep every feature finite and bounded.

### Task 2: Prove the structure-level no-DFT contract with failing tests

**Files:**
- Modify: `tests/test_next112_obstruction_morphology.py`
- Modify after RED: `src/next112_obstruction_morphology.py`

**Steps:**
1. Add tests for the exact protocol/schema, deterministic pure evaluation, catalogue abstention, and sign-pattern cap.
2. Add an actual primitive/supercell invariance test.
3. Add a test that fails if Brown bond-valence parameters are consulted.
4. Implement the additive structure wrapper by reusing only NEXT109 catalogue and opposite-sign graph helpers.
5. Select the sign pattern with the exact NEXT109 four-scalar rank before exposing the six new values.
6. Run the NEXT109 and NEXT112 tests together to prove no regression.

### Task 3: Build NEXT113 cross-source discovery features with TDD

**Files:**
- Create: `tests/test_next113_cross_source_cmvom_features.py`
- Create after RED: `src/next113_cross_source_cmvom_features.py`
- Create: `docs/plans/2026-08-08-next113-cmvom-feature-freeze.json`

**Steps:**
1. Freeze graph mode, catalogue modes, schema, discovery-only paths, process count, thread environment, and expected input/design SHA-256 values before the formal run.
2. Add failing tests for row schema, independent core/expanded abstention, payload identity, output refusal, endpoint firewall, and manifest fields.
3. Implement a builder patterned on NEXT110 without modifying or importing endpoint values.
4. Run focused tests, then materialize into a new external directory under `$PRIS_ARCHIVE/` using only `python` and one BLAS thread per worker.
5. Verify row counts, support counts, finite counts, deterministic ordering, catalogue digest, and every output SHA-256.

### Task 4: Freeze NEXT114 without labels

**Files:**
- Create: `tests/test_next114_cmvom_optional_search.py`
- Create after RED: `src/next114_cmvom_optional_search.py`
- Create: `docs/plans/2026-08-08-next114-cmvom-search-freeze.json`

**Steps:**
1. Inspect pooled discovery feature coverage and dispersion without opening endpoints.
2. Exclude coverage failures and exact zero-IQR terms without epsilon.
3. Freeze median, IQR, p99.5 clipping, direction set, weight grid, grouping, missing policy, near-miss base identity, candidate count, and all AUC/SAFE/BROAD gates.
4. Add failing tests for exact freeze identity, deterministic candidate enumeration, reversible tail encoding, no duplicate keys, provenance refusal, and unopened validation/replication flags.
5. Implement only the frozen finite search.

### Task 5: Run discovery once and obey the gates

**Files:**
- External create: `$PRIS_ARCHIVE/next114_cross_source_cmvom_search_v1/`

**Steps:**
1. Verify every formal input hash and the freeze hash before endpoints are read.
2. Run the finite candidate universe on SCIGEN and WyFormer discovery endpoints only.
3. Report six AUC cells, twelve SAFE cells, BROAD precision, pass counts, and the deterministic selected candidate.
4. If any discovery gate fails, keep validation and replication unopened and record `freeze_authorized=false`.
5. If and only if all gates pass, stop before opening new payloads and record that a separate authorization step is required.

### Task 6: Report and verify

**Files:**
- Create: `reports/2026-08-08-next112-next114-cmvom-no-dft-search.md`

**Steps:**
1. Write the mathematical certificate, no-DFT boundary, feature coverage, frozen universe, discovery results, failures, and next decision in a new standalone report.
2. State clearly whether the branch is a confirmed replacement, a useful orthogonal mechanism, or a stopped negative result.
3. Run focused NEXT109--NEXT114 tests and the full pytest suite.
4. Check CodeGraph pending sync, repository status, protected paths, artifact manifests, and report SHA-256.
5. Do not modify canonical reports or manuscript files before user confirmation.
