# PRIS + S_syn Combined Screening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the frozen PRIS L4 gate and frozen synthesis formula identify complementary subsets of CSAgent PU-low-score structures, including comparisons at matched experimental-structure retention.

**Architecture:** Reuse the local binary dual-route shards, recompute the frozen formula scores from retained raw terms, and equalize exact CIFs before analysis. Treat L4 explicit violations as a fixed categorical gate and combine it with a low-`S_syn` threshold using an OR cascade. Calibrate thresholds only on the experimental cohort, then evaluate the frozen decisions on the PU-low-score cohort; CLscore defines the candidate cohort but is never an outcome label.

**Tech Stack:** Python, NumPy, pandas, SciPy, PyArrow, Matplotlib, pytest.

### Task 1: Specify and test matched-retention decisions

**Files:**
- Create: `tests/test_pu_synth_combined_screening.py`
- Create: `experiments/pu_synthesizability_20260821/combined_screening.py`

**Steps:**
1. Write failing tests for exact-CIF validation, low-score threshold calibration, fixed L4 decisions, formula-only decisions, and the combined `L4 OR low formula` cascade.
2. Run the targeted tests and confirm failure because the module does not exist.
3. Implement deterministic matched-retention calibration, including explicit tie accounting.
4. Run the targeted tests and confirm they pass.

### Task 2: Add overlap, gains, and bootstrap uncertainty

**Files:**
- Modify: `tests/test_pu_synth_combined_screening.py`
- Modify: `experiments/pu_synthesizability_20260821/combined_screening.py`

**Steps:**
1. Write failing tests for the four-way overlap table and paired detection gain.
2. Implement exact counts, absolute and relative gains, and independent-cohort structure-level bootstrap with threshold recalibration.
3. Verify deterministic output under a fixed seed and passing tests.

### Task 3: Add the additive full-data CLI and outputs

**Files:**
- Modify: `experiments/pu_synthesizability_20260821/combined_screening.py`
- Create: `outputs/20260822_pu_formula_scores/combined_screening_v1/`

**Steps:**
1. Write a CLI integration test against synthetic Parquet shards and frozen-formula fixtures.
2. Implement fail-closed input checks, frozen formula loading, exact-CIF equalization, CSV/JSON output, and SHA-256 manifest generation.
3. Run the full local binary cohort with the complete `S_syn` and the no-D7/D8 shared-term sensitivity.
4. Verify all row counts, output hashes, and rerun determinism.

### Task 4: Visualize and summarize without editing the current report

**Files:**
- Create: `outputs/20260822_pu_formula_scores/combined_screening_v1/combined_screening.png`
- Create: `outputs/20260822_pu_formula_scores/combined_screening_v1/combined_screening.pdf`
- Create: `outputs/20260822_pu_formula_scores/combined_screening_v1/RESULTS.zh-CN.md`

**Steps:**
1. Plot the matched-retention detection frontier and the natural-threshold overlap.
2. Write a concise independent interpretation that calls PU low score a model proxy, not ground truth.
3. Visually inspect the PNG and run targeted plus complete PU experiment tests.

