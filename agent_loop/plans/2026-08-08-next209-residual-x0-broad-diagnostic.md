# NEXT209 Residual X0 BROAD Diagnostic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Reproduce and diagnose the sole NEXT208 candidate that passes both
source-AUC and all SAFE cells but fails BROAD, without searching or selecting a
new formula.

**Architecture:** Verify the complete NEXT208 provenance and select candidates
by the already frozen predicate `AUC AND SAFE AND NOT BROAD`. Reconstruct the
candidate score from x0 inputs, reproduce its published evaluation record, and
run the unchanged threshold-residual diagnostic against Pauling in the same 12
source/fold cells. Publish only a diagnostic JSON, one-row Parquet table, and
manifest.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and the existing
NEXT164/NEXT206 threshold-residual helpers.

## Frozen population and expected identity

NEXT208 evaluated 661 candidates and produced:

```text
passes all discovery gates                 = 0
passes source AUC                          = 6
passes SAFE in all cells                   = 17
passes BROAD in all cells                  = 0
passes AUC + SAFE but not BROAD             = 1
sorted candidate-key SHA-256                =
e1f1ab49dfe24fa449275bf24ab8882f2850be53ccc6422aa3674516d9feb312
```

The sole diagnostic candidate is the unchanged NEXT206 base: `feature`,
`direction`, `comparison`, and `cutoff` are null; exception fraction numerator
and active-row count are zero. Thus zero newly activated raw-x0 exceptions
reach AUC+SAFE. NEXT209 must verify this rather than infer it.

Formal NEXT208 identity:

```text
MANIFEST.json                                      ba61979f3731bb0f26da817e71a313b15b6f42580c2d6edbc67ec91553bff9a0
NEXT208_RESIDUAL_X0_EXCEPTION_CATALOGUE.json       b8d35584f51d1e16a47fce17daa49bf9fdb891344a62a8052a2c0af27c480511
NEXT208_DISCOVERY_EVALUATION.json                  3e1f3c58af5e917972b13ac1de9f556c5520520211a53df10c7b63e161ee8313
NEXT208_FROZEN_CANDIDATE.json                      58d4bc7a86784b8c08d921bb54e9ff7f6146eb15566142efa97c39547feff649
next208_residual_x0_exception_search.parquet       b04fbef394eeedca385cf7d3609e2a088ae9cfe09b774c46fb30c4e632396860
executed NEXT208 source                              a8415f792050c14c72155f63231e8a348be3f96b2bcfe1a8659a18084073dd94
```

## Frozen diagnostic

1. Select exactly the published `AUC AND SAFE AND NOT BROAD` row and require
   the count and key digest above.
2. Require that it is the unchanged base with zero exception-active rows.
3. Rebuild all NEXT208 specs only to prove candidate-universe identity, then
   materialize and rerun only the selected one.
4. Require the rerun evaluation record to reproduce the published row.
5. Enumerate the unchanged candidate's finite threshold tables, restrict BROAD
   thresholds to the published SAFE threshold, and compute failures with
   `diagnose_broad_threshold_tables`.
6. Require exact equality with the NEXT206 global-closest residual (apart from
   the wrapper candidate key). This proves the raw-x0 exception search did not
   improve the best admissible BROAD residual.

NEXT209 searches no formula, opens no validation/replication artifact, and
uses no DFT value, learned energy/force/stress proxy, potential, or relaxation
in the executable rule. After successful reproduction it closes this exact
single-feature raw-x0 exception branch.

## Task 1: Write contract tests

**Files:**

- Create: `tests/test_next209_residual_x0_broad_diagnostic.py`

**Steps:**

1. Test exact AUC+SAFE/non-BROAD filtering and deterministic candidate digest.
2. Test fail-closed rejection when the selected row is not an inactive base.
3. Test deterministic closest-residual selection and tie break.
4. Test the sealed discovery-only formal interface and missing-input failure.
5. Run the module and observe the missing-module RED failure.

## Task 2: Implement and run the diagnostic

**Files:**

- Create: `src/next209_residual_x0_broad_diagnostic.py`

**Steps:**

1. Implement only the tested selection, inactive-base verification, ranking,
   provenance, reproduction, and diagnostic helpers.
2. Publish atomically:
   `NEXT209_RESIDUAL_X0_BROAD_DIAGNOSTIC.json`,
   `next209_residual_x0_broad_diagnostic.parquet`, and `MANIFEST.json`.
3. Record explicit branch-closed, no-new-formula, no-DFT/no-proxy/no-relaxation,
   and sealed-validation flags.
4. Run targeted tests, compile the source, then execute the formal diagnostic
   once at
   `$PRIS_ARCHIVE/next209_residual_x0_broad_diagnostic_v1`.

## Task 3: Report and verify

Append NEXT207--NEXT209 verified outcomes to the standalone report only. Run
targeted and full tests, compilation, hash checks, `git diff --check`, report
fence checks, canonical-path status checks, and CodeGraph status. Keep the
overall research goal active: closing this branch is not discovery success.
