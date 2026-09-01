# NEXT32 OMat24 Inorganic Response Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Build and prospectively evaluate a single-`x0`, fully DFT-free-at-execution inorganic law for severe OMat24 DFT force/stress response.

**Architecture:** Four additive modules separate raw-record geometry projection, analytic feature construction, bounded development/freeze/application, and post-freeze endpoint evaluation. `rattled-relax` is exposed development only; three unopened OMat24 perturbation sources are sanitized and predicted together before any confirmation DFT value is decoded.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, ASE, pymatgen, LMDB, pytest, existing NEXT19–NEXT23 analytic kernels and Pauling controls.

**Workspace note:** The repository contains a shared untracked NEXT19–NEXT31 research tree. A new worktree would not contain those inputs, so this plan is executed additively in the current checkout. Do not commit, overwrite, reset, or clean shared files.

### Task 1: Raw OMat24 projection and identity-locked endpoints

**Files:**

- Create: `src/next32_omat24_cohort.py`
- Create: `tests/test_next32_omat24_cohort.py`

1. Write failing tests using a tiny synthetic ASE-LMDB payload. Prove that metadata projection extracts only `sid`, `parent_id`, geometry and record key while a sentinel endpoint object is never numerically decoded.
2. Test deterministic one-row-per-parent selection, salt ordering, cumulative parent exclusions, duplicate identity rejection, and no-overwrite publication.
3. Test endpoint decoding separately: exact prediction ID lock, `force_max`, `force_rms`, Voigt stress norm, finite-value checks, and refusal before a frozen prediction manifest.
4. Run `conda run -n newpauling python -m pytest -q tests/test_next32_omat24_cohort.py` and confirm the import/function failures.
5. Implement a top-level/nested JSON projector using the existing NEXT26 skip parser, geometry ZIP publication, metadata parquet, parent list, manifests and SHA-256.
6. Implement endpoint publication in a separate function that validates prediction hashes before decoding `forces` and `stress`.
7. Re-run the focused tests until green.

### Task 2: Periodic contact strain and reused analytic features

**Files:**

- Create: `src/next32_inorganic_response_features.py`
- Create: `tests/test_next32_inorganic_response_features.py`

1. Add failing unit tests for translation, atom permutation and supercell invariance of the dimensionless contact quantities; compression must monotonically lower `q` and raise overlap loads.
2. Add tests for exact periodic self-image handling, missing-radius fail-open behavior, and absence of forbidden endpoint/DFT columns.
3. Add a failing integration test that loads a geometry-only ZIP and rejects a manifest with `labels_opened=true` or a mismatched geometry hash.
4. Implement unique periodic pair enumeration and publish `cov_q01`, `cov_q05`, `cov_contact085_pa`, `cov_overlap2_pa`, `cov_site_overlap_q95`, and `cov_site_overlap_max`.
5. Reuse one Voronoi/valence graph per structure to compute the frozen SIVR, normalized Madelung and SCBVE subset from NEXT20–NEXT22; emit NaN plus explicit failure reasons when unsupported.
6. Implement immutable batch feature and unchanged Pauling 2–5 control publication from geometry-only inputs.
7. Run the focused tests and a 16-structure smoke batch.

### Task 3: Bounded development search, freeze and label-free application

**Files:**

- Create: `src/next32_inorganic_response_rule.py`
- Create: `tests/test_next32_inorganic_response_rule.py`

1. Write failing tests for the frozen endpoint thresholds, Wilson lower/upper intervals, prevalence-lift clause and exact six promotion gates.
2. Write tests proving that confirmation rows cannot affect development medians, IQRs, candidate ranking or threshold selection.
3. Freeze the term catalogue, risk directions, selected mechanism-linked pairs and rejection fractions from the design. Disable only formulas containing a zero-IQR term.
4. Implement robust-z scoring, deterministic thresholding, fail-open support, candidate scan and unique tie-breaking.
5. Publish a scan always; publish `NEXT32_FROZEN_INORGANIC_RESPONSE_RULE.json` only when a candidate passes all development gates.
6. Implement rule application that validates rule and feature manifests, writes predictions without label fields, records source names, and refuses overwrite.
7. Run `conda run -n newpauling python -m pytest -q tests/test_next32_inorganic_response_rule.py`.

### Task 4: Frozen confirmation evaluation and Pauling comparison

**Files:**

- Create: `src/next32_inorganic_response_evaluate.py`
- Create: `tests/test_next32_inorganic_response_evaluate.py`

1. Add failing tests for aggregate and per-source gates, exact ID/source joins, prediction/rule/protocol hash binding, and fail-open semantics.
2. Add tests that `beyond_pauling_on_this_endpoint` is true only when NEXT32 passes aggregate plus every source gate and every fixed Pauling control fails the aggregate gate set.
3. Implement protocol freeze before confirmation labels, endpoint opening after all hashes validate, ROC AUC/Spearman diagnostics, Wilson metrics and source tables.
4. Publish immutable evaluation JSON, joined parquet and manifest; refuse threshold refit or input change during publication.
5. Run all NEXT32 unit tests.

### Task 5: Execute exposed development

**External artifacts:** `$PRIS_ARCHIVE/next32_*`

1. Record the official URL, archive size, ETag, last-modified value and SHA-256 for the already exposed `rattled-relax` archive.
2. Use metadata-only deterministic selection to publish 4,096 parent-unique development geometries.
3. Build analytic features and Pauling controls before writing the selected development endpoint table.
4. Open only development DFT force/stress labels and run the bounded scan.
5. If `promotion=false`, preserve results, write a negative report and stop before downloading confirmation sources.
6. If promoted, hash-freeze the formula and confirmation evaluation protocol.

### Task 6: Execute three-source confirmation

1. Download `rattled-300`, `rattled-500`, and `rattled-1000` as opaque archives; record hashes before extraction.
2. Sanitize 2,048 parent-unique, cumulatively parent-disjoint geometries per source without decoding DFT numeric fields.
3. Build all features, fixed Pauling controls and NEXT32 predictions for all 6,144 rows.
4. Verify prediction/rule/protocol hashes and only then decode identity-locked force/stress endpoints.
5. Evaluate once. Do not modify the cohort, formula, constants, threshold, endpoint cutoffs or gates after label opening.

### Task 7: Report and verification

**Files:**

- Create: `reports/2026-08-03-next32-omat24-inorganic-response.md`

1. Report official data provenance, no-DFT execution contract, exact formula or no-promotion result, all failed candidates/source strata, Pauling comparison and claim boundary.
2. Keep NEXT29–NEXT31 and every canonical document unchanged.
3. Run focused NEXT32 tests, then `conda run -n newpauling python -m pytest -q`.
4. Check CodeGraph synchronization and verify every published hash against its manifest.
5. Stop at the standalone-report boundary for user confirmation; do not edit the manuscript or canonical reports.

