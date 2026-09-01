# NEXT109--NEXT111 Convex Mixed-Valence Obstruction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and discovery-evaluate a Brown-free, representation-invariant mixed-valence graph obstruction certificate under the strict pre-DFT boundary.

**Architecture:** NEXT109 provides a pure normalized slack LP and a raw-structure sign-pattern wrapper. NEXT110 materializes only physically isolated discovery features. NEXT111 performs one preregistered finite additive search while keeping validation and replication unopened.

**Tech Stack:** Python 3.11, NumPy, SciPy `linprog`/HiGHS, pymatgen, pandas/pyarrow, pytest, CodeGraph.

**Repository constraints:** Additive files only. Do not commit, create a worktree, edit protected canonical documents or old reports, or open validation/replication artifacts. These explicit task constraints override the generic skill handoff suggestions.

### Task 1: Pure obstruction solver

**Files:**
- Create: `tests/test_next109_convex_mixed_valence_obstruction.py`
- Create after RED: `src/next109_convex_mixed_valence_obstruction.py`

1. Write tests for a zero-slack feasible network and a fixed `+1/-2` global imbalance with positive slack.
2. Run the focused tests with `python -m pytest -q` and verify RED because NEXT109 is absent.
3. Implement strict interval/endpoint validation, unsigned incidence, the L1 slack LP, and global balance gap.
4. Verify the two tests GREEN.
5. Add RED tests for disconnected-component imbalance, a connected Hall/cut obstruction, an isolated site, and an empty graph.
6. Implement component discovery, component gaps, and unserved fraction; verify GREEN.
7. Add and observe RED as needed for site/edge permutation, integer replication, and common charge scaling invariance; make only minimal corrections and verify GREEN.

### Task 2: Raw-structure wrapper

**Files:**
- Modify after RED: `tests/test_next109_convex_mixed_valence_obstruction.py`
- Modify after RED: `src/next109_convex_mixed_valence_obstruction.py`

1. Write RED tests for the exact four-feature schema, deterministic CsCl output, raw-structure immutability, expanded-only chemistry, integer supercell invariance, and sign-pattern overflow.
2. Implement catalogue/sign helpers by reusing immutable NEXT104 definitions and a Brown-free all-site Voronoi endpoint builder that records rather than aborts on unserved sites.
3. Select a coherent best sign pattern lexicographically and emit support/failure provenance.
4. Verify NEXT109 plus NEXT104 dependency tests GREEN.
5. Add a test that monkeypatches NEXT104 Brown strength construction to fail and proves NEXT109 remains supported.

### Task 3: Discovery-only cross-source feature freeze

**Files:**
- Create: `tests/test_next110_cross_source_cmvo_features.py`
- Create after RED: `src/next110_cross_source_cmvo_features.py`

1. Write a tiny builder test with corrupt, inaccessible validation/replication decoys and verify RED.
2. Implement only discovery geometry/metadata inputs, core and expanded independent abstention, atomic output, row accounting, environment/catalogue provenance, and immutable hashes.
3. Verify the builder cannot accept or discover validation/replication paths.
4. Run formal NEXT110 with one solver thread per worker to a new immutable directory under `$PRIS_ARCHIVE/`.
5. Rehash every formal input and output and record label-free support/failure counts.

### Task 4: Frozen finite search

**Files:**
- Create before endpoint access: `docs/plans/2026-08-08-next111-cmvo-search-freeze.md`
- Create: `tests/test_next111_cmvo_optional_search.py`
- Create after RED: `src/next111_cmvo_optional_search.py`

1. Record exact NEXT108/NEXT110 hashes, eligible-term rules, calibration equations, term directions, weights, base count, candidate order, and unchanged gates in the freeze file.
2. Write RED tests proving calibration precedes endpoint join, unsupported terms contribute zero, base scores/support remain unchanged, and finite catalogue enumeration is exact.
3. Implement the label-free calibration and catalogue builder; hash the frozen catalogue before endpoint loading.
4. Reuse the existing 12-cell evaluator without weakening any threshold or AUC gate.
5. Run the formal discovery-only search once to a new immutable external directory.
6. Record maximum all-gate count, SAFE and AUC failure cells, selected formula if any, and explicit validation/replication unopened state.

### Task 5: Outcome-dependent stop or replication freeze

**Files:**
- Create on discovery failure: `reports/2026-08-08-next109-next111-cmvo-no-dft-search.md`
- Create a new replication freeze only if every discovery gate passes.

1. If no candidate passes all frozen gates, stop this branch, keep external endpoints closed, and write a standalone negative/partial report.
2. If a candidate passes all gates, freeze formula, calibration, thresholds, candidate hash, and row predictions before opening any isolated replication endpoint.
3. Never revise the formula or gate after an endpoint is observed.

### Task 6: Final verification

1. Load `superpowers:verification-before-completion`.
2. Rehash all NEXT110/NEXT111 artifacts and compare them with manifests.
3. Run focused NEXT104--NEXT111 tests.
4. Run `python -m pytest -q`.
5. Check CodeGraph pending sync and inspect only listed stale files if any.
6. Check scoped Git status proves protected files and old reports were not edited.
7. Keep the overall scientific goal active unless a frozen external replication justifies the requested scientific end state.
