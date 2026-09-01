# NEXT115--NEXT117 Hall-Closure Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development
> task-by-task.  The user's autonomy instruction overrides the generic skill's
> question, commit, worktree, and subagent handoffs: work in this checkout,
> create additive files only, and do not commit.

**Goal:** Build and evaluate a DFT-free generalized-Hall local obstruction
certificate without changing any existing script, result, report, or canonical
document.

**Architecture:** NEXT115 provides a pure graph LP with canonical secondary
optimum values plus a wrapper that reuses NEXT109's frozen structure inputs and
sign-pattern choice.  NEXT116 materializes unlabeled cross-source features.
After a label-free finite freeze, NEXT117 adds the retained terms only to the
existing frozen frontier and evaluates the unchanged discovery gates.

**Tech stack:** Python 3 in
`<env>`, NumPy, SciPy HiGHS, pandas,
pymatgen, pytest, JSON/SHA-256 manifests.

### Task 1: Graph-level RED tests

**Files:**

- Create: `tests/test_next115_hall_cut_interval_deficit.py`
- Create later: `src/next115_hall_cut_interval_deficit.py`

1. Write an import/API test for `solve_hall_cut_interval_deficit` and run it to
   observe a missing-module failure.
2. Add separate failing tests for a feasible edge, a global one-sided mismatch,
   a connected local Hall violation, an empty graph, invalid intervals, bad
   edge orientation, and duplicate-edge determinism.
3. Add a brute-force subset oracle over seeded small graphs and assert both LP
   primary optima equal enumeration.
4. Add a direct interval-flow feasibility oracle and assert feasibility iff
   both directional primary deficits are zero.
5. Add permutation, common-charge-scale, exact replication, and symmetric
   multi-optimum invariance tests.

Run each new test before implementation with:

```
python -m pytest \
  tests/test_next115_hall_cut_interval_deficit.py -q
```

### Task 2: Minimal NEXT115 graph implementation

**Files:**

- Create: `src/next115_hall_cut_interval_deficit.py`

1. Define immutable result records and the ten frozen feature names.
2. Validate finite same-sign intervals, positive tolerance, endpoint shape,
   range, orientation, and sign coverage; deduplicate endpoints.
3. Implement one directional closure LP with variables in `[0, 1]` and
   constraints `x_origin <= y_neighbor`.
4. Clamp only numerical noise at zero and reject invalid optima.
5. On a positive primary face, run secondary LPs for minimum charge support,
   minimum/maximum origin count, and minimum neighbor count.
6. Normalize, check `[0, 1]`, and return exact zeros when the primary deficit is
   nonpositive within tolerance.
7. Run all graph tests to GREEN, then refactor only while they remain green.

### Task 3: NEXT115 structure wrapper

**Files:**

- Extend additively within the new NEXT115 source.
- Extend the new NEXT115 test file.

1. Add failing tests proving the wrapper imports only frozen analytic helpers,
   exposes ten terms, and abstains on the same unsupported cases as NEXT109.
2. For every frozen sign pattern, call NEXT109's obstruction solver and rank by
   exactly `(min_interval_slack, global_balance_gap,
   component_balance_gap, unserved_site_fraction)`.
3. Select the identical best pattern and calculate HCID on its oriented graph.
4. Test deterministic repeated calls and a real pymatgen supercell.
5. Scan the new source for forbidden executable dependencies and run focused
   NEXT109--115 regression tests.

### Task 4: NEXT116 builder through TDD

**Files:**

- Create: `tests/test_next116_cross_source_hcid_features.py`
- Create: `src/next116_cross_source_hcid_features.py`
- Create: `docs/plans/2026-08-08-next116-hcid-feature-freeze.json`
- Write formal artifacts only below
  `$PRIS_ARCHIVE/next116_cross_source_hcid_features_v1`.

1. RED-test row schema, stable ordering, catalogue hash checks, skip accounting,
   smoke isolation, manifest hashes, and absence of labels/forbidden inputs.
2. Reuse NEXT113's frozen structure catalogue and multiprocessing conventions;
   compute only NEXT115 terms.
3. Run a small smoke build, independently recompute sampled rows, then run the
   formal SCIGEN and WyFormer build with fixed thread limits and 12 workers.
4. Verify row counts, source identity/order, technical support equality, finite
   ranges, source hashes, and the manifest.

### Task 5: Label-free NEXT117 freeze

**Files:**

- Create: `tests/test_next117_hcid_frontier_rescue.py`
- Create: `src/next117_hcid_frontier_rescue.py`
- Create: `docs/plans/2026-08-08-next117-hcid-search-freeze.json`

1. Without opening labels, audit per source/support stratum: missingness, median,
   IQR, p99.5, cap, exact-zero rate, and pairwise Spearman correlation.
2. Retain only nondegenerate high-tail terms and remove one of each
   `|rho| >= 0.98` pair by a deterministic, label-free rule.
3. Freeze centers, scales, caps, polarities, coefficient grid, allowed term
   count, exact NEXT111/NEXT114 base frontier identities, candidate count, and
   catalogue hashes.
4. RED-test that changed freezes, labels opened before freeze, duplicate
   candidates, or changed base reproduction fail closed.

### Task 6: Frozen discovery evaluation

**Formal output:**

`$PRIS_ARCHIVE/next117_hcid_frontier_rescue_v1`

1. Reproduce every imported base candidate's six AUCs and twelve SAFE cells
   within the previously frozen tolerance before adding HCID.
2. Enumerate only the frozen finite catalogue; evaluate unchanged SCIGEN and
   WyFormer pooled/macro/worst AUC, all SAFE cells, and BROAD gates.
3. Select deterministically by the frozen hierarchy.
4. If no candidate passes every discovery gate, record the negative result and
   keep validation/replication unopened.  If and only if all discovery gates
   pass, follow the already frozen routing policy; do not invent a new endpoint.
5. Write a hashed manifest and independently verify every artifact hash.

### Task 7: Report and verification

**Files:**

- Create: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

1. Report the mathematical certificate, invariants, coverage, label-free
   freeze, finite search, all gate outcomes, limitations, and whether any
   endpoint was opened.
2. Explicitly distinguish analytic feasibility from empirical stability
   prediction and from DFT-quality screening.
3. Run focused tests, then the entire suite with the mandated interpreter.
4. Verify CodeGraph has no pending files, all formal hashes match, protected
   paths are unchanged, and only additive NEXT115--117/report/plan files were
   created.
5. Load `superpowers:verification-before-completion` before making any final
   completion claim.  The overall goal remains active unless a genuinely
   all-gate law and its required validation are completed.
