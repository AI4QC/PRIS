# NEXT171 Repair-Interval Local Rigidity Search Implementation Plan

> **For Codex:** REQUIRED SUB-SKILLS: Use `superpowers:executing-plans`, `superpowers:test-driven-development`, and `superpowers:verification-before-completion`.

**Goal:** Test whether the NEXT169-validated local directional-rigidity signal repairs BROAD when applied only inside the exact frozen NEXT164 repair interval, while leaving the SAFE region pointwise unchanged.

**Architecture:** Reconstruct the exact NEXT163/NEXT164 no-DFT base score and reuse the five NEXT169-eligible CrystalNN certificates. For one certificate and one fixed attenuation at a time, change the score only when the unmodified base lies in the frozen repair interval. Evaluate the resulting 41-member finite grammar with the unchanged cross-source SOURCE_AUC, SAFE12, and BROAD gates.

**Tech Stack:** Python 3.11 from `<env>`, NumPy, pandas, pytest, existing NEXT125/NEXT130 evaluator, Parquet, JSON, SHA-256 manifests.

## Frozen rationale and alternatives

NEXT170 established that all 41 global attenuations retain the SOURCE_AUC gates and that larger attenuation improves pooled AUC, but only four retain SAFE12 and none passes BROAD. NEXT169 established the local-rigidity direction specifically inside the fixed repair shell. Therefore the next falsifiable hypothesis is that the physical signal is valid locally but the global operator changes rows that were not part of the repair problem.

Rejected alternatives:

- a smooth window is not used because it adds a width/shape parameter;
- a rigidity cutoff is not used because it adds a result-sensitive threshold;
- source-, fold-, chemistry-, and family-specific intervals or coefficients are forbidden;
- feature pairs, conjunctions, direction reversal, and any new descriptor are forbidden.

No existing script applies a correction only when `BROAD_THRESHOLD <= base_score < SAFE_THRESHOLD`; NEXT146 gates on a physical residual, while NEXT153, NEXT163, and NEXT170 alter the score globally.

## Frozen inputs and formula

- NEXT170 manifest SHA-256: `f8ca4aab7ec58597dd2bd101f2cb0b18cc91b2ba227ecddbf7a50be200a918c4`.
- NEXT170 evaluation SHA-256: `b13305d04b2920be83db1227e88136aa341cb82f5917c29962b48b8d2bbbe668`.
- NEXT170 complete table SHA-256: `6dc294b9ce690364c2abf4ef1492f92ea21ffc138ed739c69c5e2b65ee8a2300`.
- Exact base candidate key SHA-256: `1d0ea8331f38aa69cfdedbe664d5ceb46c14e166e121bae92d9e14dd4fc6109e`.
- Exact interval: `0.21976295573076796 <= s < 0.5415470292150686`.
- Eligible features, unchanged from NEXT169:
  - `pldr_crystalnn_tightness_min`
  - `pldr_crystalnn_tightness_q10`
  - `pldr_crystalnn_tightness_mean`
  - `pldr_crystalnn_volume_q10`
  - `pldr_crystalnn_volume_mean`
- Attenuation grid: `(0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)`.
- Candidate count: `1 + 5 * 8 = 41`, including the unchanged base exactly once.

For base score `s`, feature `f`, and attenuation `alpha`, define

```text
active = feature is finite and BROAD_THRESHOLD <= s < SAFE_THRESHOLD
s' = max(0, s * (1 - alpha * f)) if active else s
```

The interval is evaluated from the unmodified base score, never recursively. Missing or unsupported rigidity keeps the base. Base support is unchanged. Every row with `s >= SAFE_THRESHOLD` and every row with `s < BROAD_THRESHOLD` must remain bitwise equal to the base score.

## Frozen evaluation and stopping rule

- Before searching, reproduce the unchanged NEXT163/NEXT164 base metrics and all four gate booleans to absolute tolerance `1e-12`.
- Source AUC gates remain five evaluable folds, pooled AUC at least `0.75`, macro AUC at least `0.60`, and worst-fold AUC at least `0.55`.
- SAFE12 remains the existing complete gate set, including coverage lower bound `0.90`, protected-recall lower bound `0.90`, severe-precision lower bound `0.80`, and savings lower bound `0.02`.
- BROAD remains the existing every-cell gate with severe-precision lower bound `0.45`.
- Success requires `passes_all_discovery_gates == true`. A successful discovery candidate authorizes a frozen formula artifact, but not validation opening in this stage.
- If no candidate passes all gates, terminate this operator family. Do not extend the attenuation grid, add a feature cutoff, or add a second feature after seeing the results.

## Task 1: Pure interval-local operator and grammar

**Files:**

- Create: `tests/test_next171_repair_interval_local_rigidity_search.py`
- Create after red: `src/next171_repair_interval_local_rigidity_search.py`

**Steps:**

1. Write tests for exact interval inclusivity/exclusivity and pointwise identity outside it.
2. Test monotonic boundedness inside the interval, missing keep-base behavior, support identity, and use of the original base to determine activity.
3. Test the exact five features, eight attenuations, 41 canonical candidate keys, and one unchanged base.
4. Run the test file and verify failure because the module is absent.
5. Implement the minimal pure kernel and grammar, then run green.

## Task 2: Evaluator materialization and formal runner

**Files:**

- Extend test first: `tests/test_next171_repair_interval_local_rigidity_search.py`
- Extend after red: `src/next171_repair_interval_local_rigidity_search.py`

**Steps:**

1. Test exact evaluator recovery through the existing `sinh/asinh` virtual-term encoding.
2. Test that the formal runner accepts discovery endpoints but no validation or replication path, and fails closed on missing input.
3. Verify NEXT168--NEXT170 manifests, hashes, eligible features, no-DFT flags, and sealed validation/replication flags.
4. Reconstruct and reproduce the base, materialize exactly 41 candidates, and call the unchanged evaluator.
5. Publish atomically under `$PRIS_ARCHIVE/next171_repair_interval_local_rigidity_search_v1`: catalogue, evaluation, complete candidate table, formula record, and `MANIFEST.json`.

## Task 3: Report and full verification

**Files:**

- Modify only: `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

**Steps:**

1. Append the frozen NEXT171 method, full gate counts, selected metrics, hashes, and explicit authorization or termination decision.
2. Run all directed NEXT142--NEXT171 tests.
3. Independently recompute manifest/output/source hashes and inspect all DFT/proxy/relaxation/validation/replication flags.
4. Verify balanced Markdown fences, CodeGraph sync, scoped Git status, and no changes to `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.

