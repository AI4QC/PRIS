# NEXT185 Conditional Closure BROAD Residual Diagnostic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diagnose why every frozen NEXT184 candidate fails BROAD without searching or selecting another formula.

**Architecture:** Reproduce the exact 121-candidate NEXT184 score universe, select only the 93 published candidates that pass source AUC and SAFE while failing BROAD, and reuse the frozen NEXT164 threshold-table diagnostic. Publish per-candidate failed constraints and normalized shortfalls plus aggregate failure frequencies and closest candidates.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, existing NEXT164 and NEXT184 evaluation helpers.

## Frozen diagnostic population and outputs

- Population rule: `passes_source_auc_gates AND passes_safe_all_cells AND NOT passes_broad_all_cells`.
- Expected rows: `93`.
- Sorted candidate-key digest: `a0ebde986998c5c2a3c400ca53dfbe4ad59fe0eb8eb5e4ae8a32a46505fa7626`.
- No formula search, candidate selection, threshold change, validation/replication read, DFT calculation, learned energy/force/stress proxy, or relaxation.
- Rank residual proximity by increasing failed-constraint count, then normalized shortfall sum, then candidate key.
- Report the unchanged base residual, global closest residual, closest residual per conditional hypothesis, and frequency of every `cell_id::component` failure.

### Task 1: Write and run failing tests

**Files:**

- Create: `tests/test_next185_conditional_closure_broad_residual_diagnostic.py`

Test exact population filtering/order, deterministic closest selection, discovery-only formal interface, and fail-closed missing inputs. Run the target test and confirm collection fails because the module is absent.

### Task 2: Implement the diagnostic

**Files:**

- Create: `src/next185_conditional_closure_broad_residual_diagnostic.py`

Verify exact NEXT184 provenance and hashes, reconstruct the frozen base/families/certificates, materialize all NEXT184 scores, reproduce the 93 published records, evaluate threshold tables with NEXT164, and publish JSON/Parquet/manifest atomically.

### Task 3: Execute and report

Create `$PRIS_ARCHIVE/next185_conditional_closure_broad_residual_diagnostic_v1/`, verify all hashes and boundary flags, then append the result additively to `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Do not change canonical manuscript/report files.
