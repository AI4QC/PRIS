# NEXT217 Repair-Band BROAD Diagnostic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Reproduce and rank the exact BROAD residuals of the 88 frozen
NEXT216 AUC+SAFE/non-BROAD candidates without searching a new formula.

**Architecture:** Verify NEXT216 and its full candidate table, reconstruct all
89 exact physical scores, select the prepublished 88-candidate diagnostic
population, and run the existing BROAD threshold-table diagnostic at each
candidate's published SAFE threshold. Rank by failed constraint count,
normalized shortfall, and candidate key. Publish evidence only.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and existing
NEXT128/NEXT164/NEXT215/NEXT216 helpers.

## Hard boundary

- Diagnostic-only: no new feature, formula, cutoff, amplitude, interval,
  conjunction, refit, or evaluator is searched.
- Executable candidates remain composition plus initial unrelaxed geometry.
- No DFT calculation/value, learned energy/force/stress proxy, model or proxy
  potential, relaxed structure, trajectory, or physical relaxation may enter.
- Validation and replication artifacts remain physically unopened.
- Preserve all existing scripts/results; publish additively and do not edit
  canonical paper/report paths.

## Frozen NEXT216 identity

- design: `f01ffa96607ffb489aa9b01b00a47235c03d8efae5b7530553b5dfa5bf0c3a96`
- source: `9aa58d9807400ad0729ed5f589eb5a7d34c82b36e6867a29d8f0a7d5afeb2050`
- manifest: `c3a53bb248c612a00fe493b7bb3ae7fb695cf80253e11f0c296ffceaa9738838`
- catalogue: `dd180b13be58860511ef55bb0c4ea99ff213f75bf81fd5f7d73ce187ade989c5`
- evaluation: `d31ee14913f3a993d64a862f46bd8e7f9a6e17ed046e3af02fe9832921eea914`
- formula: `feda3a7baf4ff690bcad2468619b7c8a9d6ccc6c0f643d6ee8b36909c3c107dd`
- candidate table: `93ebafe13f1c66ea1560c0ddb76b2c01d8b04dc6cb1c4a208c4ae2a60213905c`

The table contains exactly 89 candidates: 89 SAFE, 88 source-AUC, zero BROAD,
and zero all-gate. The diagnostic population is exactly:

```text
passes_source_auc_gates
AND passes_safe_all_cells
AND NOT passes_broad_all_cells
```

It contains 88 unique candidate keys whose sorted newline-joined SHA-256 is
`550ba336091545f8040ac62cc9b4e7f426fb196757e2ad8d815b00cb1cc90c35`.
Any mismatch fails closed.

## Frozen diagnostic

Reconstruct the exact 89 candidate specs and scores using the published
NEXT216 eligible identities, endpoint-blind cutoffs, fixed interval, and fixed
amplitudes. For each of the 88 diagnostic candidates:

1. Reproduce its score and unchanged support.
2. Build the unchanged source/fold threshold tables.
3. Use its exact published `safe_threshold` as the upper search bound.
4. Run the existing `diagnose_broad_threshold_tables` helper.
5. Record best threshold, failed constraints, normalized shortfall, and exact
   failure list.

Rank candidates by:

```text
(failed_constraint_count, normalized_shortfall_sum, candidate_key)
```

Compare the global closest against the unchanged NEXT214 residual baseline:

```text
failed constraints = 6
normalized shortfall = 0.26893426117441227
```

This comparison is diagnostic evidence, not formula selection. NEXT217 never
authorizes validation and never searches a follow-up formula.

Formal output directory:

```text
$PRIS_ARCHIVE/next217_repair_band_broad_diagnostic_v1/
```

Publish atomically:

- `MANIFEST.json`
- `NEXT217_REPAIR_BAND_BROAD_DIAGNOSTIC.json`
- `next217_repair_band_broad_diagnostic.parquet`

## Tasks

1. Create `tests/test_next217_repair_band_broad_diagnostic.py`; observe RED for
   exact candidate selection/digest, deterministic closest ranking, sealed
   interface, and fail-closed missing inputs.
2. Create `src/next217_repair_band_broad_diagnostic.py`; verify NEXT216,
   reconstruct scores, diagnose the exact 88 candidates, and publish atomically.
3. Run targeted tests and compilation, then execute formally once.
4. Independently verify hashes, candidate identities, residual ranking,
   boundary flags, no-formula-search flags, and unopened validation/replication.
5. Append verified NEXT215--NEXT217 results to the standalone report and run
   the full verification matrix.
