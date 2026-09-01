# Better Laws and Formulas Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fully reproducible, lockbox-safe search loop for stronger interpretable
structure laws and sparse ranking formulas without modifying any existing analysis,
report, or manuscript content.

**Architecture:** A new descriptor module computes literature-frozen local descriptors
for the exact deterministic S1-S5 lineage. A separate search module owns split assertions,
candidate generation, group-equal metrics, sparse formula fitting, and aggregate output.
Raw identifier-bearing caches stay outside the repository; only aggregate results and a
new report are retained.

**Tech Stack:** Python 3.11, NumPy, pandas, pyarrow, pymatgen, SciPy, scikit-learn,
pytest/unittest.

### Task 1: Freeze invariants in tests

**Files:**
- Create: `tests/test_better_search.py`
- Create: `src/better_search.py`

**Step 1: Write failing tests**

Test that:

- null or `lockbox` split rows raise before fitting;
- group-equal accuracy differs from pair-weighted accuracy on an imbalanced toy example;
- pair construction never crosses groups and gives both orientations equal weight;
- a rule violation wins over unknown values, while an unknown-only case is indeterminate;
- a guarded rule does not require its body when the guard is false;
- deterministic group splitting never separates one group.

**Step 2: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_better_search.py -q
```

Expected: collection fails because `better_search` does not exist.

**Step 3: Implement the minimal utilities**

Implement only the APIs exercised by the tests.

**Step 4: Run tests and verify GREEN**

Expected: all new utility tests pass.

### Task 2: Implement and test the P1 descriptor

**Files:**
- Create: `src/advanced_local_features.py`
- Modify: `tests/test_better_search.py`

**Step 1: Write failing tests**

Use small synthetic ionic structures to test:

- bond-valence weights are positive and parameter coverage is bounded in [0, 1];
- a symmetric environment has lower vector asymmetry than an intentionally displaced
  centre;
- entropy effective coordination is invariant to uniform scaling of all bond-valence
  weights;
- missing bond-valence parameters yield explicit low coverage, not a fabricated pass.

**Step 2: Verify RED**

Expected: import or missing-function failure.

**Step 3: Implement the descriptor**

Reuse `elec_feat.bv_table`, `discriminate.guess_oxi`,
`make_negatives.perturb`, `make_negatives.swapped_val`, and
`phys_law.seed_of`. Do not modify those files.

**Step 4: Verify GREEN and all regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Expected: all tests pass.

### Task 3: Add descriptor-cache CLI

**Files:**
- Modify: `src/advanced_local_features.py`
- Modify: `tests/test_better_search.py`

**Steps:**

1. Write a failing smoke test for explicit `--features-dir`, `--out`, `--mode`,
   `--limit`, and deterministic output.
2. Implement real and S1-S5 modes with multiprocessing.
3. Refuse repository output paths unless `--allow-private-output` is explicit.
4. Run a 20-structure smoke pass twice and compare aggregate hashes.

Commands:

```bash
python src/advanced_local_features.py real --limit 20 \
  --out /tmp/np-next-20260731/p1_real_smoke.parquet
python src/advanced_local_features.py bad --limit 20 \
  --out /tmp/np-next-20260731/p1_bad_smoke.parquet
```

### Task 4: Implement law candidate and Pareto search

**Files:**
- Modify: `src/better_search.py`
- Modify: `tests/test_better_search.py`

**Steps:**

1. Write failing toy tests for one-sided, band, and guarded candidates.
2. Write a failing toy beam test where balanced retention is required to find the best
   two-rule set.
3. Implement vectorised candidate masks and a diversity-preserving Pareto beam.
4. Add exact L1/L1'/L2/L3 definitions and a regression test against the committed
   calibration CSV.
5. Add LOKO and anion-stratified aggregate evaluation.

### Task 5: Implement nested, group-equal sparse formula fitting

**Files:**
- Modify: `src/better_search.py`
- Modify: `tests/test_better_search.py`

**Steps:**

1. Write failing tests proving scaler/imputer statistics come only from training rows.
2. Write a failing test proving each formula group has total sample weight one.
3. Write a failing test proving abstention thresholds are learned from development scores
   and applied numerically to evaluation scores.
4. Implement grouped inner selection, at-most-seven-term refit, outer evaluation, and
   saved fold statistics.
5. Add group bootstrap and energy-gap strata.

### Task 6: Add the experiment driver

**Files:**
- Create: `src/run_better_search.py`
- Modify: `tests/test_better_search.py`

**Steps:**

1. Write a failing CLI smoke test using tiny synthetic parquet inputs.
2. Implement explicit input/output paths and `--dry-run`.
3. Refuse any row whose split is null or `lockbox`.
4. Emit `manifest.json`, `law_results.json`, `formula_results.json`, and compact CSVs.
5. Record Git SHA, input file hashes, package versions, seed, candidate vocabulary, and
   exact commands.

### Task 7: Run descriptors and reproduce baselines

**Files:**
- Runtime cache: `/tmp/np-next-20260731/`
- Aggregate output: `outputs/20260731_better_laws_formulas/`

**Steps:**

1. Run P1 real and bad descriptor passes.
2. Verify identical S1-S5 parent lineage against `phys_bad.parquet`.
3. Reproduce L1/L1'/L2/L3 exactly before accepting new law results.
4. Refit a documented sparse formula baseline and store every fold statistic.
5. Stop if any baseline or lineage check fails.

### Task 8: Run the frozen search and falsification suite

**Steps:**

1. Run law search on development data only.
2. Freeze selected identities and thresholds.
3. Evaluate deterministic outer splits and historical calibration.
4. Run S1-S5, LOKO, anion, coverage, and DFT-relaxed false-positive diagnostics.
5. Run the grouped nested formula search and fixed-threshold abstention evaluation.
6. Compare against the success gates in the design document.

### Task 9: Write a new report only

**Files:**
- Create: `reports/2026-07-31-better-laws-formulas.md`

The report must contain:

- repository progress and maturity audit;
- exact reproducibility failures found in the old loop;
- frozen protocol and data lineage;
- baseline reproduction;
- all new candidates, including negative results;
- selected laws/formulas only if they pass the frozen gates;
- LOKO, coverage, chemistry, group-weighting, and false-positive caveats;
- direct links to primary literature;
- an explicit statement that existing reports and manuscript files were not modified.

### Task 10: Final verification

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
git diff --check
git status --short
```

Confirm that no file under `paper/`, `notes/`, or `tex/`, and none of
`README.md`, `src/README.md`, or `PREREG.md`, changed.
