# NEXT192 Signed Safe-Margin Audit and NEXT193 Certificate Search Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Recover interpretable information discarded by the frozen hinge-risk law by testing whether depth on the physically safe side of its exact term centers is a transferable protection certificate, then search a bounded repair only if every cross-source audit gate passes.

**Architecture:** NEXT192 reconstructs the exact selected NEXT163 base terms, weights, transformations, centers, scales, support, and four mechanism-family assignments. For each term it replaces the existing positive risk hinge by its complementary negative-side hinge, caps contributions in the same units, and audits ten frozen family/consensus summaries. NEXT193 is conditional on audit eligibility and can subtract only normalized eligible certificates inside the unchanged repair interval.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, and the existing NEXT87/NEXT151/NEXT163/NEXT164/NEXT169/NEXT183/NEXT190 provenance and evaluation helpers.

## Frozen scientific design

The selected base represents every physical term as

```text
z_j = direction_j * (transform_j(x_j) - center_j) / scale_j
risk_j = max(0, z_j)
```

and therefore maps every observation with `z_j <= 0` to the same zero contribution. NEXT183's cleanliness summaries operate on these already-truncated family risks, so they distinguish lower risk from higher risk but cannot distinguish a term barely below its center from one far into its frozen physical safe side.

NEXT192 introduces no new fitted value. It reconstructs the complementary margin

```text
safe_j = min(0.5, weight_j * max(0, -z_j))
```

where `0.5` is the unchanged NEXT163 contribution cap. Exact selected base terms are assigned to the same four frozen mechanism families and averaged within each family:

```text
S_local    = mean(safe_j in local_geometry)
S_charge   = mean(safe_j in charge_flow_feasibility)
S_valence  = mean(safe_j in valence_transport)
S_contact  = mean(safe_j in contact_robustness)
```

The exact ten protection-high hypotheses are:

```text
safe_local_geometry
safe_charge_flow_feasibility
safe_valence_transport
safe_contact_robustness
safe_family_mean
safe_family_min
safe_family_second_min
safe_nonlocal_mean
safe_nonlocal_min
safe_local_nonlocal_min
```

Definitions:

```text
safe_family_mean       = mean(S_local, S_charge, S_valence, S_contact)
safe_family_min        = min(S_local, S_charge, S_valence, S_contact)
safe_family_second_min = second-smallest(S_local, S_charge, S_valence, S_contact)
safe_nonlocal_mean     = mean(S_charge, S_valence, S_contact)
safe_nonlocal_min      = min(S_charge, S_valence, S_contact)
safe_local_nonlocal_min = min(S_local, safe_nonlocal_mean)
```

All values remain in `[0, 0.5]`; higher predicts protected (`direction=+1`). The single-family hypotheses test attribution, while the minimum/second-minimum hypotheses test whether genuinely transferable protection requires concordance rather than one unusually clean subsystem. No term cutoff, feature subset, alternative cap, sign flip, source-specific rule, or extra aggregation may be added after outcomes are opened.

Support and audit gates remain exactly:

```text
SCIGEN full support >= 0.90
WyFormer full support >= 0.90
SCIGEN repair-shell worst-fold AUC >= 0.55 with all 5 folds evaluable
WyFormer repair-shell pooled AUC >= 0.55
SCIGEN full-extreme pooled AUC >= 0.50
WyFormer full-extreme pooled AUC >= 0.50
```

Eligible hypotheses are ranked by decreasing minimum key AUC, decreasing mean key AUC, then lexical name. NEXT192 searches no formula.

If and only if NEXT192 has an eligible certificate, NEXT193 searches:

```text
W = SAFE_THRESHOLD - BROAD_THRESHOLD
certificate = clip(safe_margin / 0.5, 0, 1)
active iff BROAD_THRESHOLD <= base_score < SAFE_THRESHOLD
score = max(0, base_score - alpha * W * certificate)
alpha in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
```

Rows outside the interval, unsupported rows, and missing certificate rows retain the exact base score and support. Candidate count is `1 + eligible_count * 6`. Binary cutoffs and learned thresholds are excluded.

All transforms, centers, scales, signs, term weights, and family assignments were frozen before discovery outcomes. The executable value uses only composition and the initial raw geometry. No DFT calculation/value, learned energy/force/stress proxy, relaxed structure, trajectory, or physical relaxation enters the formula. Discovery outcomes remain offline audit/search labels only.

### Task 1: TDD and formal NEXT192 audit

**Files:**

- Create: `tests/test_next192_signed_safe_margin_audit.py`
- Create: `src/next192_signed_safe_margin_audit.py`
- Create externally: `$PRIS_ARCHIVE/next192_signed_safe_margin_audit_v1/`

Test exact complementary hinge reconstruction, cap, support, four family assignments, ten summaries, unchanged eligibility, deterministic selection, discovery-only interface, and missing-input failure. Confirm RED before implementation. Require exact frozen feature, endpoint, base, NEXT190, plan, and source provenance. Reconstruct the same base score, fixed folds, repair shell, and full-extreme populations used by NEXT190. Publish atomically.

### Task 2: Conditionally TDD and run NEXT193

If NEXT192 has eligible certificates, create:

- `tests/test_next193_signed_safe_margin_search.py`
- `src/next193_signed_safe_margin_search.py`
- `$PRIS_ARCHIVE/next193_signed_safe_margin_search_v1/`

Test exact normalization, interval-only subtraction, missing/base fallback, six amplitudes, exact candidate count, materialized-score recovery, discovery-only interface, and missing-input failure. The formal runner may use only exact NEXT192-eligible certificates.

### Task 3: Diagnose and report

If NEXT193 runs but fails, publish a frozen BROAD residual diagnostic before changing the mechanism. Append NEXT186, NEXT188, NEXT190, and NEXT192/NEXT193 results to the standalone report. Keep all validation/replication endpoints sealed and do not modify canonical manuscript/report files.
