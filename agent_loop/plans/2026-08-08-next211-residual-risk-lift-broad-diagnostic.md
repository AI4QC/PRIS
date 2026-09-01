# NEXT211 Continuous Residual-Risk Lift BROAD Diagnostic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Reproduce and diagnose the exact 91 NEXT210 candidates that pass
both-source AUC and every SAFE cell but fail at least one BROAD comparison,
without searching or selecting a new formula.

**Architecture:** Verify the complete NEXT210 provenance, reconstruct the exact
NEXT206 base score and frozen NEXT210 221-candidate universe, and require an
exact identity match to the published search table. Re-run only the frozen 91
candidate keys through the unchanged evaluator, then use the existing
threshold-table diagnostic to measure each candidate's minimum BROAD residual.
Publish an additive diagnostic table and summary; do not alter any law, report,
paper, validation, or replication artifact during this task.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and the existing
NEXT125/NEXT164/NEXT210 evaluators and provenance helpers.

## Frozen boundary and input identity

- Executable inputs remain composition plus initial unrelaxed geometry only.
- No DFT calculation/value, learned energy/force/stress proxy, model or proxy
  potential, relaxed structure, trajectory, or physical relaxation may enter
  the executable score.
- Discovery outcomes are offline evaluation labels only and are not score
  inputs.
- Validation and replication endpoints remain physically unopened.
- Additive files only. Do not edit `paper/`, `tex/`, `notes/`, `README.md`, or
  `PREREG.md`.
- Do not tune features, directions, quantiles, amplitudes, residual thresholds,
  score composition, operating gates, or diagnostic ordering in NEXT211.

The formal NEXT210 search emitted exactly 221 candidates and no all-gate pass.
The frozen diagnostic population is the 91 rows satisfying

```text
passes_source_auc_gates
AND passes_safe_all_cells
AND NOT passes_broad_all_cells
```

after stable sorting by `candidate_key`. Its newline-joined key digest is
`a23090645432ef0518bf273128047663debe8f4f227d8fdf62c1838a2ca05b19`.

Formal NEXT210 identities are:

- design: `6cdf054a87a4a07ca761d63af24fbd519b27bcb1957247aef67f3a3054cacd70`
- source: `143d71ce856a49e6afe90d60448e1e114981b83bdc38974917b28dc851dd0c1b`
- manifest: `9162d949a08e88a5f2635c2289d4cb402fa8a823ddbef7f16127a11ad5700326`
- catalogue: `64f7af7ce8826fdfebe1a46228764192578591b00321c464baf6d48acb9c57ad`
- evaluation: `3ad11fa8ec5449def5c42ef52488537b7eb8411c05e33fe96b9f9a4532944960`
- frozen candidate: `b5d7a84ec4a1600526181de76688daf3c0004d753d920dcc0e8c31f52d57a6d1`
- search table: `39f9365f51387aa3e9808ebf3b1c8a4f028b64d0552b82d223ce977bd33d39d3`

## Frozen diagnostic procedure

For each of the exact 91 keys:

1. Rebuild the score from the already-frozen NEXT210 feature, direction,
   quantile cutoffs, amplitude, risk scale, and residual threshold.
2. Re-run the unchanged dual-source evaluator and exactly reproduce the
   published candidate record.
3. Construct the unchanged source/fold cells and Pauling baselines.
4. Starting from the candidate's published SAFE threshold, enumerate its exact
   threshold tables and diagnose BROAD failures with
   `diagnose_broad_threshold_tables`.
5. Record the best threshold, failed-constraint count, normalized shortfall,
   and exact failure list.

Select the diagnostic closest candidate deterministically by

```text
(failed_constraint_count, normalized_shortfall_sum, candidate_key)
```

This ordering is descriptive only. It does not authorize a new formula or
validation opening. The continuous residual-risk lift branch closes after this
diagnostic because the frozen NEXT210 search has no all-gate candidate.

### Task 1: Write NEXT211 contracts first

**Files:**

- Create: `tests/test_next211_residual_risk_lift_broad_diagnostic.py`

**Steps:**

1. Test exact AUC+SAFE/non-BROAD filtering and newline-joined digest.
2. Test deterministic residual ranking and identity tie-break.
3. Test the formal interface exposes discovery inputs but no validation or
   replication path.
4. Test missing formal inputs fail closed.
5. Run the targeted test and observe the expected missing-module RED.

### Task 2: Implement the additive diagnostic

**Files:**

- Create: `src/next211_residual_risk_lift_broad_diagnostic.py`

**Steps:**

1. Verify NEXT210 manifest, output hashes, boundary flags, source identity,
   candidate count, frozen subset count, and subset digest.
2. Reconstruct all 221 specifications and require exact key equality with the
   published table.
3. Materialize and re-run only the frozen 91 candidates.
4. Require exact evaluator reproduction, calculate per-candidate BROAD
   residuals, and aggregate failure frequencies.
5. Publish the JSON summary, Parquet table, and hash-complete manifest
   atomically.
6. Run the targeted test and require GREEN.

### Task 3: Formal execution and evidence

**Files:**

- Create outside the repository:
  `$PRIS_ARCHIVE/next211_residual_risk_lift_broad_diagnostic_v1/`

**Steps:**

1. Run with the sealed formal inputs and four evaluator workers.
2. Verify the manifest and every published output hash independently.
3. Inspect the closest residual and failure-frequency evidence before defining
   any later search branch.

### Task 4: Standalone report and verification

**Files:**

- Modify only:
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`

**Steps:**

1. Append the verified NEXT210 and NEXT211 results, clearly separating search
   results from diagnostic interpretation.
2. Run targeted tests, Python compilation, the full pytest suite,
   `git diff --check`, formal hash checks, canonical-path checks, report-fence
   checks, and CodeGraph status.
3. Do not commit or modify canonical manuscript/report files in the existing
   dirty additive workspace.
