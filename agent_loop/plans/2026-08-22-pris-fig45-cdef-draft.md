# PRIS Fig. 4–5 c–f Draft Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible, non-canonical c–f figure draft that compares independent pre-DFT screening choices and leaves explicit interfaces for corrected eHull and PU-model performance data.

**Architecture:** Read frozen aggregate CSV/NPZ outputs only. Panel c uses retention on the experimental cohort as its x-axis and PU-queue screening as its y-axis; L4, S_syn, and distance cutoffs remain separate operating choices, while corrected eHull is an optional curve loaded by path. Panel d is a nested two-row plot with one shared percentile x-axis: L4 violation rate above and S_syn below. If row-level common-ID scores are unavailable, the script derives a labelled decile-level provisional view from the frozen A/B aggregates and records that status in JSON. Panels e/f are callback slots with transparent pending placeholders.

**Tech Stack:** Python, pandas, NumPy, Matplotlib; SVG/PDF/PNG exports and Nature-figure source/PDF QA.

### Task 1: Audit and freeze the data contract

**Files:**
- Read: `outputs/20260821_pu_synthesizability/analysis_v1/{rung_summary,distance_cutoff_summary}.csv`
- Read: `outputs/20260822_pu_formula_scores/independent_choices_v1/independent_frontier.csv`
- Read: `outputs/20260821_pu_synthesizability/full_pool_analysis_v1/score_deciles.csv`
- Read: `outputs/20260822_pu_formula_scores/full_pool_dual_v2/direct_formula_plots_v3/clscore_formula_density_data.npz`
- Optional: corrected eHull threshold CSV and future row-level consensus CSV

**Step 1:** Validate required columns and record row counts.

**Step 2:** Write a manifest that distinguishes exact, provisional, and pending evidence; never silently substitute D7 or a raw A/B arithmetic mean.

### Task 2: Add the c–f plotting draft

**Files:**
- Create: `experiments/pu_synthesizability_20260821/plot_fig45_cdef_draft.py`
- Create: `outputs/20260822_pu_formula_scores/fig45_cdef_draft_v1/` (generated only)

**Step 1:** Implement independent retention–screening curves/points for L4, S_syn, and distance cutoffs; omit D7 from panel c entirely.

**Step 2:** Implement optional corrected-eHull loader with an explicit proxy/coverage annotation and a pending placeholder when no file is supplied.

**Step 3:** Implement d's shared-x nested axes and a strict optional schema for exact normalized-consensus deciles; use a labelled aggregate fallback only when exact rows are absent.

**Step 4:** Implement `panel_e_callback` and `panel_f_callback` hooks plus pending placeholders; do not import or mutate canonical figure scripts.

**Step 5:** Export SVG, PDF, and 600-dpi PNG, source-data snapshots, status JSON, and SHA256 manifest.

### Task 3: Validate and hand off

**Files:**
- Generated: `outputs/20260822_pu_formula_scores/fig45_cdef_draft_v1/*`

**Step 1:** Run the Nature figure source validator and PDF glyph audit.

**Step 2:** Inspect the rendered figure at final size and check that c contains no D7 label, d shares x, and placeholders are visibly marked as pending.

**Step 3:** Report paths, exact/provisional/pending status, and the command for replacing the optional data to `/root`; do not edit `tex/`, `paper/`, or existing figure files.
