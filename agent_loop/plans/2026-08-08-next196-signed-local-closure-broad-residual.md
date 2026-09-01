# NEXT196 Signed-Local Closure BROAD Residual Diagnostic Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diagnose every exact BROAD constraint failure for the frozen NEXT195 candidates that pass both source-AUC and SAFE gates, without searching or changing any formula.

**Architecture:** Reconstruct the exact 37 NEXT195 scores and reproduce their published discovery metrics. For every evaluator threshold below the candidate's published SAFE threshold, reuse the frozen NEXT164 threshold-table diagnostic and select the closest residual by failed-constraint count, normalized shortfall, then candidate key.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, and existing NEXT164/NEXT194/NEXT195 reconstruction and diagnostic helpers.

## Frozen diagnostic design

- Diagnostic population: exact NEXT195 rows satisfying `passes_source_auc_gates && passes_safe_all_cells && !passes_broad_all_cells`.
- Expected population: 37 candidates.
- Sorted candidate-key SHA-256: `c9019f11352eb4cbf44731ac6d4b182dfc8339a9aaf552e40a56631e340a91fb`.
- Threshold universe, source/fold cells, Pauling baselines, components, normalized shortfall, and candidate reproduction tolerances are unchanged from NEXT164/NEXT185.
- Closest ordering: minimum failed-constraint count, then minimum normalized-shortfall sum, then lexical candidate key.
- Report the unchanged base residual, closest residual by certificate, global closest residual, and exact failure-frequency map.
- Search no formula, select no formula, change no threshold, and open no validation or replication endpoint.
- The executable scores remain composition/raw-geometry only; no DFT value, learned energy/force/stress proxy, or physical relaxation enters them.

### Task 1: TDD and formal NEXT196 diagnostic

**Files:**

- Create: `tests/test_next196_signed_local_closure_broad_residual.py`
- Create: `src/next196_signed_local_closure_broad_residual.py`
- Create externally: `$PRIS_ARCHIVE/next196_signed_local_closure_broad_residual_v1/`

Test exact diagnostic-population filtering, closest ordering, discovery-only interface, and missing-input failure. Confirm RED before implementation. Require exact NEXT195 plan/source/input/output provenance, reproduce all 37 records, publish per-candidate residuals and summary atomically, and make no new formula decision.

### Task 2: Interpret and report

Compare the closest shortfall with the base `0.860419`, NEXT178 `0.834952`, and NEXT185 `0.836140`. Append the result to the standalone report while preserving every prior script/report and every canonical document.
