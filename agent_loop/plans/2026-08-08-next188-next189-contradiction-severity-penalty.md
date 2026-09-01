# NEXT188 Contradiction Severity Audit and NEXT189 Penalty Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the local/nonlocal contradiction variable rejected as a protection relief is instead a transferable pre-DFT severity signal, then add it as a bounded risk penalty only if the reversed direction passes frozen cross-source gates.

**Architecture:** NEXT188 reuses the exact 24 NEXT186 values but preregisters the opposite direction: larger contradiction means more severe, represented by audit direction `-1` against the protected endpoint. NEXT189 is conditional on eligibility and searches one unchanged base plus the eligible severity penalties over the original repair interval with a frozen six-value amplitude grid.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, existing NEXT151/NEXT163/NEXT179/NEXT183/NEXT186 provenance and evaluation helpers.

## Frozen scientific design

NEXT186 showed that strong-closure-weighted positive local-risk surplus is not a protection certificate. The new hypothesis is not a post hoc sign flip inside the same audit: it is a separate, prospectively frozen mechanism with a different physical meaning. A structure whose local-geometry risk remains anomalously larger than independent charge-flow, contact, and valence-transport risks despite strong directional closure may contain a localized incompatibility that the additive base underweights.

The exact value universe remains:

```text
surplus_max  = max(0, L - max(N_charge, N_contact, N_valence))
surplus_mean = max(0, L - mean(N_charge, N_contact, N_valence))
product      = closure * surplus
minimum      = min(0.5 * closure, surplus)
```

Crossing six frozen closure features, two surplus references, and two conjunctions gives exactly 24 hypotheses. Every NEXT188 hypothesis uses direction `-1`: lower value predicts protected, equivalently higher value predicts severe. Names are `<closure>__<surplus>__<conjunction>__severe_high`.

NEXT188 reuses the unchanged support and cross-source AUC gates from NEXT186. Eligible rows are ranked by decreasing minimum key AUC, decreasing mean, then name. No formula is searched in NEXT188.

If NEXT188 has at least one eligible penalty, NEXT189 searches:

```text
alpha in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
active iff BROAD_THRESHOLD <= base_score < SAFE_THRESHOLD
score = base_score + alpha * contradiction_penalty
```

Rows outside the interval, unsupported rows, and missing penalties retain the exact base score and support. The addition is interval-local rather than global so already clear safe and severe regions are not perturbed. Binary cutoffs and learned thresholds are excluded. Candidate count is `1 + eligible_count * 6`; amplitudes, thresholds, gates, and candidate identities may not change after NEXT188 outcomes are visible.

### Task 1: TDD and formal NEXT188 audit

**Files:**

- Create: `tests/test_next188_contradiction_severity_audit.py`
- Create: `src/next188_contradiction_severity_audit.py`
- Create externally: `$PRIS_ARCHIVE/next188_contradiction_severity_audit_v1/`

Test the exact 24 severe-high hypotheses, direction `-1`, unchanged gates, deterministic selection, discovery-only interface, and missing-input failure. Confirm RED before implementation. Require exact NEXT186 provenance, reuse its pure value functions, rerun the four fixed populations, and publish atomically.

### Task 2: Conditionally TDD and run NEXT189

If NEXT188 has eligible penalties, create:

- `tests/test_next189_contradiction_severity_penalty_search.py`
- `src/next189_contradiction_severity_penalty_search.py`
- `$PRIS_ARCHIVE/next189_contradiction_severity_penalty_search_v1/`

Test exact addition inside the interval, missing/base fallback, support preservation, six amplitudes, exact candidate count, virtual-term recovery, discovery-only interface, and missing-input failure. The formal runner may use only exact NEXT188-eligible penalties.

### Task 3: Diagnose and report

If NEXT189 fails, publish a frozen BROAD residual diagnostic before changing mechanism. Append NEXT186 and NEXT188/189 results additively to the standalone report. Keep validation/replication sealed and do not modify canonical manuscript/report files.
