# NEXT30 Inorganic Hull Law Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and audit a sparse, single-x0, fully analytic WBM high-hull-energy screening law without using DFT or energy proxies at execution time.

**Architecture:** Add one self-contained NEXT30 module with four fail-closed phases: split, development/freeze, label-free confirmation prediction, and confirmation evaluation. Reuse immutable NEXT23 feature tables and the existing Wilson metric semantics, but never modify NEXT23 or canonical documents.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, scikit-learn metrics, pytest, JSON/Parquet manifests, SHA-256.

### Task 1: Freeze the split and candidate catalogue

**Files:**
- Create: `src/next30_inorganic_hull_law.py`
- Create: `tests/test_next30_inorganic_hull_law.py`

1. Write failing tests for deterministic 4,096/4,096 ID separation, zero overlap, forbidden label-like feature columns, exact feature-table joins, and no-overwrite publication.
2. Run `pytest -q tests/test_next30_inorganic_hull_law.py` and verify failure because NEXT30 does not exist.
3. Implement constants, SHA helpers, immutable JSON publication, split creation, term catalogue, and formula catalogue.
4. Re-run the focused test until it passes.

### Task 2: Implement development-only discovery and rule freeze

**Files:**
- Modify: `src/next30_inorganic_hull_law.py`
- Modify: `tests/test_next30_inorganic_hull_law.py`

1. Add failing synthetic tests proving that confirmation IDs cannot enter discovery, robust-z constants come only from development rows, missing terms fail open, and the deterministic tie-break chooses one formula.
2. Implement development label filtering, finite catalogue scoring, Wilson gates, and immutable rule/scan manifests.
3. Test both promotion and no-promotion paths. A no-promotion run must not create a confirmation prediction.

### Task 3: Seal confirmation predictions before label evaluation

**Files:**
- Modify: `src/next30_inorganic_hull_law.py`
- Modify: `tests/test_next30_inorganic_hull_law.py`

1. Add failing tests for exact rule-hash validation, confirmation-only IDs, fail-open support, and absence of endpoint fields in predictions.
2. Implement fixed-score application and immutable prediction publication.
3. Verify predictions are byte-bound to the frozen rule and input manifests.

### Task 4: Evaluate once and compare with Pauling

**Files:**
- Modify: `src/next30_inorganic_hull_law.py`
- Modify: `tests/test_next30_inorganic_hull_law.py`

1. Add failing tests for endpoint thresholds, Wilson metrics, same-cohort Pauling comparison, and the rule that `beyond_pauling` is false unless NEXT30 passes while every Pauling control fails.
2. Implement aggregate-only public results plus identifier-bearing private join.
3. Run the real phases in order. If development does not promote, stop without reading confirmation labels; otherwise seal confirmation predictions first and then evaluate once.

### Task 5: Report and verify

**Files:**
- Create: `reports/2026-08-03-next29-next30-cross-domain-energy-boundary.md`

1. Record NEXT29 negative transfer, NEXT30 protocol, exact formula or no-promotion result, hashes, Pauling comparison, and evidence limitations.
2. Run focused NEXT29/NEXT30 tests, then the full repository suite.
3. Check CodeGraph pending sync and record exact artifact hashes.

No git commit is part of this run because the shared checkout contains a large pre-existing untracked research tree; committing would exceed the user-authorized additive experiment scope.
