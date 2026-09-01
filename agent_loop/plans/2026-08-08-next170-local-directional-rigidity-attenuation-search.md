# NEXT170 Local Directional Rigidity Attenuation Search Implementation Plan

> **For Codex:** REQUIRED SUB-SKILLS: Use `superpowers:test-driven-development` before implementation and `superpowers:verification-before-completion` before reporting.

**Goal:** Determine whether the five NEXT169-eligible local directional-rigidity certificates can repair the frozen NEXT163/NEXT164 no-DFT score while preserving every existing discovery gate.

**Architecture:** Reconstruct exactly one frozen NEXT163/NEXT164 base score, then apply one bounded multiplicative attenuation at a time. The only searched quantities are the already eligible feature identity and a short preregistered attenuation grid. The existing cross-source evaluator remains authoritative for SOURCE_AUC, SAFE12, and BROAD gates.

**Tech Stack:** Python 3.11 from `<env>`, NumPy, pandas, pytest, the existing NEXT125/NEXT130 cross-source evaluator, Parquet, JSON, SHA-256 manifests.

## Frozen inputs and grammar

- NEXT169 manifest SHA-256: `13249710d94e1950ffed4b84a40eca7f754bd292bb794275095d81cf1bbab643`.
- NEXT169 audit SHA-256: `4135933b256e6820e27093446f4afc6e83a7cb341670add8ef6b01fbaf74920a`.
- NEXT169 table SHA-256: `af58c9c5d59d209baa2727bb3f39ec3f36a50c1498c8410ef07172354bee53a0`.
- The eligible feature set is frozen to:
  - `pldr_crystalnn_tightness_min`
  - `pldr_crystalnn_tightness_q10`
  - `pldr_crystalnn_tightness_mean`
  - `pldr_crystalnn_volume_q10`
  - `pldr_crystalnn_volume_mean`
- The attenuation grid is exactly `(0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40)`.
- Include the unmodified base exactly once. Total candidate count is therefore `1 + 5 * 8 = 41`.

For base risk score \(s\), bounded feature \(f\), and attenuation \(\alpha\), define

\[
s' = \max(0, s(1-\alpha f)).
\]

The feature direction cannot change. There is no threshold, quantile calibration, transform, pair, conjunction, source-specific parameter, or family-specific exception. If the NEXT168 feature is unsupported or nonfinite, use the original base score for that row (`TERM_OFF_KEEP_BASE`); base support is unchanged.

## Frozen evaluation and success rule

- Reproduce the exact NEXT163/NEXT164 base candidate identity before evaluating any correction.
- Use the existing evaluator's source AUC gates exactly: five evaluable folds, pooled extreme AUC at least 0.75, macro fold AUC at least 0.60, and worst-fold AUC at least 0.55.
- Preserve all SAFE12 gates exactly, including coverage lower bound 0.90, protected recall lower bound 0.90, severe precision lower bound 0.80, and savings lower bound 0.02.
- Pass the existing BROAD gates in every discovery cell, including severe precision lower bound 0.45.
- A candidate is successful only when `passes_all_discovery_gates` is true. Validation and replication outputs remain unopened even if discovery succeeds.
- Candidate ranking and tie breaking are delegated unchanged to the existing evaluator.

## TDD tasks

### Task 1: Pure attenuation and candidate grammar

**Files:**

- Create: `tests/test_next170_local_directional_rigidity_attenuation_search.py`
- Create after red: `src/next170_local_directional_rigidity_attenuation_search.py`

**Steps:**

1. Test the exact five-feature allowlist, eight-value attenuation grid, and 41-candidate identity.
2. Test monotonicity, `[0, base]` bounds, `alpha=0` identity, and missing-feature keep-base behavior.
3. Test that support remains exactly the base support.
4. Test deterministic candidate keys and no duplicate base.
5. Run the test red, implement minimally, and run green.

### Task 2: Formal search and publication

**Files:**

- Extend: `src/next170_local_directional_rigidity_attenuation_search.py`
- Extend tests before implementation.

**Steps:**

1. Add a formal runner whose interface accepts discovery endpoints but no validation or replication endpoint.
2. Verify all NEXT168/NEXT169 boundary flags and hashes, then reconstruct the exact NEXT163 candidate and score.
3. Materialize the 41 evaluator-compatible virtual terms and run the existing cross-source evaluator.
4. Publish catalogue, evaluation, complete candidate table, formula, and `MANIFEST.json` atomically under `$PRIS_ARCHIVE/next170_local_directional_rigidity_attenuation_search_v1`.
5. Record whether all discovery gates pass and whether a frozen candidate is authorized. Do not open validation or replication.

### Task 3: Standalone report and verification

**Files:**

- Modify only: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

**Steps:**

1. Append NEXT168, NEXT169, and NEXT170 methods, provenance, hashes, complete gate outcome, and limitations.
2. Run the full NEXT142--NEXT170 tests.
3. Independently recompute artifact hashes and boundary flags, balance Markdown fences, check CodeGraph sync, and inspect scoped Git status.
4. Do not edit canonical paper/report content before user confirmation.

