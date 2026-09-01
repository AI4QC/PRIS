# NEXT104 Convex Mixed-Valence Flow Certificate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and cross-source evaluate a representation-invariant, strictly no-DFT mixed-valence periodic bond-flow certificate.

**Architecture:** A pure LP solver jointly chooses normalized periodic edge flow and site charges inside frozen oxidation-state convex hulls. A raw-structure wrapper enumerates electronegativity-oriented sign patterns and charge-blind Brown priors. Separate discovery-only feature and optional-guard search stages preserve the existing physical lockboxes.

**Tech Stack:** Python 3.11, NumPy, SciPy `linprog`/HiGHS, pymatgen, pandas/pyarrow, pytest, CodeGraph.

**Repository constraints:** Additive files only; no commits, worktree changes, canonical report edits, or validation/replication access because the user explicitly requested preservation and autonomous in-place research.

### Task 1: Pure convex mixed-valence flow solver

**Files:**
- Create: `tests/test_next104_convex_mixed_valence_flow.py`
- Create after RED: `src/next104_convex_mixed_valence_flow.py`

1. Write failing tests for compatible, interval-infeasible, and isolated-site networks.
2. Run the three tests and verify failure because the module is absent.
3. Implement input validation, normalized priors, the first total-variation LP, the overload LP, and canonical-scale LP.
4. Run the three tests and verify GREEN.
5. Add permutation, prior-scale, and supercell-replication tests.
6. Run them RED before any corresponding implementation adjustment.
7. Make the minimal invariant implementation changes and verify GREEN.

### Task 2: Raw periodic structure wrapper

**Files:**
- Modify after RED: `tests/test_next104_convex_mixed_valence_flow.py`
- Modify after RED: `src/next104_convex_mixed_valence_flow.py`

1. Write failing tests for catalogue schema, NaCl, expanded-only chemistry, raw-structure immutability, determinism, and sign-pattern overflow.
2. Verify RED.
3. Implement frozen catalogue construction, sign-pattern enumeration, electronegativity orientation, dummy-neutral Voronoi graph construction, Brown generic priors, and best-certificate selection.
4. Verify all NEXT104 tests GREEN.
5. Run NEXT19/22/101/101b dependency tests.

### Task 3: Discovery-only cross-source feature freeze

**Files:**
- Create: `tests/test_next105_cross_source_cmvf_features.py`
- Create after RED: `src/next105_cross_source_cmvf_features.py`

1. Write a failing builder test using tiny isolated discovery fixtures and deliberately corrupt validation/replication paths.
2. Verify RED because NEXT105 does not exist.
3. Implement a builder whose CLI accepts only discovery geometry/metadata, NEXT104 design, output directory, and worker count.
4. Emit core/expanded features, support/failure provenance, catalogue and environment hashes, atomic outputs, and a manifest stating all forbidden inputs were unopened.
5. Verify the builder test and NEXT104 tests GREEN.
6. Run the formal two-source discovery feature freeze to a new immutable external output directory.

### Task 4: Frozen cross-source optional-guard search

**Files:**
- Create: `tests/test_next106_cmvf_optional_guard_search.py`
- Create after RED: `src/next106_cmvf_optional_guard_search.py`
- Create: `docs/plans/2026-08-04-next106-cmvf-search.md`

1. Freeze the exact feature directions, eligibility, weights, missing policy, 67-base input hash, and unchanged gates in the NEXT106 design file before reading endpoint rows.
2. Write failing tests proving label-free calibration precedes endpoint join, missing guards preserve base scores/support, and gate calculations match NEXT103.
3. Verify RED.
4. Implement the finite candidate catalogue and reuse the single-pass 12-cell evaluator.
5. Verify GREEN.
6. Run the formal discovery-only search to a new immutable external output directory.
7. Assert all hashes, candidate counts, gate outcomes, and replication-unopened flags.

### Task 5: Outcome-dependent freeze or stop

**Files:**
- Create only if every discovery gate passes: new NEXT107 freeze/prediction files and tests.
- Otherwise create: `reports/2026-08-04-next104-next106-cmvf-no-dft-search.md`

1. If any discovery gate fails, keep all replication endpoints closed and write a standalone negative/partial report.
2. If every gate passes, freeze formula, calibration, thresholds, and row predictions before opening any replication endpoint.
3. Never weaken gates or change the formula after observing results.

### Task 6: Final verification

1. Load `superpowers:verification-before-completion`.
2. Run all NEXT104--NEXT106 and dependency tests.
3. Run `python -m pytest -q`.
4. Recompute formal source/input/output hashes and assert manifests.
5. Check CodeGraph for pending sync.
6. Check scoped Git status proves only additive files changed and protected paths are absent.
7. Keep the overall scientific goal active unless every requested end-state claim is supported by frozen replication evidence.
