# PRIS Full-Pool Continuous Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fail-closed analyzer relating all three continuous CSAgent CLscores to three-state PRIS outcomes on their complete 8,108,676-row common support.

**Architecture:** Validate the compact score table and every immutable PRIS shard before analysis, then map shared `orig_index` values into preallocated NumPy arrays while checking the complete 8,125,976-row pool. Compute deterministic equal-frequency score deciles, global and chemistry-size-stratified rank associations, cross-model score agreement, and disagreement examples without copying CIF text into any output.

**Tech Stack:** Python 3.11+, PyArrow Parquet metadata/batch reads, pandas, NumPy, SciPy, pytest.

### Task 1: Freeze validation and merge contracts

**Files:**
- Create: `tests/test_pu_synth_full_pool_analysis.py`
- Create: `experiments/pu_synthesizability_20260821/analyze_full_pool.py`

1. Write synthetic Parquet fixtures with input and shard manifests.
2. Add failing tests for exact score support, null/duplicate scores, complete `pool_row` tiling, duplicate `orig_index`, manifest mismatch, and output SHA mismatch.
3. Run `pytest tests/test_pu_synth_full_pool_analysis.py -q` and confirm the missing module/API causes RED.
4. Implement score-table and shard-stream validation with configurable expected row counts for tests and frozen production defaults.
5. Re-run the focused tests to GREEN.

### Task 2: Implement deterministic three-state summaries

**Files:**
- Modify: `tests/test_pu_synth_full_pool_analysis.py`
- Modify: `experiments/pu_synthesizability_20260821/analyze_full_pool.py`

1. Add failing tests showing equal-frequency bins differ by at most one row, tied boundaries are reported, and pass/explicit-violation/no-verdict rates sum to one for L2, L4, and D1--D8.
2. Implement `score + orig_index` deterministic tie breaking, boundary annotations, state-rate aggregation, and size summaries.
3. Run focused tests to GREEN.

### Task 3: Implement rank associations and score agreement

**Files:**
- Modify: `tests/test_pu_synth_full_pool_analysis.py`
- Modify: `experiments/pu_synthesizability_20260821/analyze_full_pool.py`

1. Add failing tests for global Spearman, constant-stratum exclusion, all-constant strata, and pairwise A/B/Jang agreement.
2. Cache score ranks, calculate separate state-indicator associations, and pool within-stratum centered ranks only over informative `n_elements + site_bin` strata.
3. Emit per-stratum diagnostics and pairwise agreement metrics.
4. Run focused tests to GREEN.

### Task 4: Emit compact, auditable outputs

**Files:**
- Modify: `tests/test_pu_synth_full_pool_analysis.py`
- Modify: `experiments/pu_synthesizability_20260821/analyze_full_pool.py`

1. Add a failing end-to-end test for expected CSV/JSON files and examples containing indices/metadata but no CIF column.
2. Implement high-score-violation and low-score-pass example selection for each score, integrity JSON, result summary, and atomic CSV/JSON writes.
3. Run focused and adjacent PU tests, `git diff --check`, and CLI help.

