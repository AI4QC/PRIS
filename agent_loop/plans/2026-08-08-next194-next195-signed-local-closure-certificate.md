# NEXT194 Signed-Local Closure Audit and NEXT195 Search Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize the promising but fold-unstable signed local-geometry safety margin by requiring independent strong-neighborhood directional closure, then search a bounded pre-DFT correction only if the conjunction passes every frozen cross-source gate.

**Architecture:** NEXT194 reconstructs the exact NEXT192 `safe_local_geometry` value and crosses its fixed `[0,1]` normalization with the six NEXT180-eligible strong-neighborhood closure features through product and minimum conjunctions. NEXT195 is conditional on eligibility and subtracts only eligible certificates inside the unchanged repair interval.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pytest, and existing NEXT151/NEXT163/NEXT164/NEXT179/NEXT180/NEXT190/NEXT192 provenance and evaluation helpers.

## Frozen scientific design

NEXT192 showed that the discarded signed safe side is not a four-family consensus. Charge-flow and contact safe margins are identically zero for the selected base, valence safety reverses direction in the WyFormer shell, and the local-geometry family is the only transferable component. Its key AUCs were:

```text
SCIGEN shell pooled/worst = 0.587391 / 0.520498
WyFormer shell pooled     = 0.644059
SCIGEN full pooled        = 0.832189
WyFormer full pooled      = 0.752703
```

Thus local signed safety is strong in pooled and full populations but not stable in the worst SCIGEN formula fold. Strong-neighborhood directional closure independently passed the unchanged five-fold gate in NEXT180. The new mechanism requires both properties: a structure is protectable only when its exact selected local-risk terms lie materially below their frozen centers and the local neighbor graph still closes in three strong directions.

Define

```text
S = clip(safe_local_geometry / 0.5, 0, 1)
C = one of the six exact NEXT180-eligible strong-closure features
product = C * S
minimum = min(C, S)
```

The exact six closure features are the NEXT180 eligible set, copied from its frozen audit artifact and verified against source provenance. Crossing six closures and two conjunctions gives exactly 12 protection-high hypotheses named:

```text
<closure_feature>__signed_local_safe__product__high
<closure_feature>__signed_local_safe__minimum__high
```

Every hypothesis has `direction=+1`. No weighted sum, alternative safety cap, feature cutoff, extra closure, opposite direction, source-specific choice, or binary rule may be added after outcomes are opened.

Support and audit gates remain exactly:

```text
SCIGEN full support >= 0.90
WyFormer full support >= 0.90
SCIGEN repair-shell worst-fold AUC >= 0.55 with all 5 folds evaluable
WyFormer repair-shell pooled AUC >= 0.55
SCIGEN full-extreme pooled AUC >= 0.50
WyFormer full-extreme pooled AUC >= 0.50
```

Eligible hypotheses are ranked by decreasing minimum key AUC, decreasing mean key AUC, then lexical name. NEXT194 searches no formula.

If and only if NEXT194 has at least one eligible certificate, NEXT195 searches:

```text
W = SAFE_THRESHOLD - BROAD_THRESHOLD
active iff BROAD_THRESHOLD <= base_score < SAFE_THRESHOLD
score = max(0, base_score - alpha * W * certificate)
alpha in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
```

Rows outside the interval, unsupported rows, and missing certificate rows retain the exact base score and support. Candidate count is `1 + eligible_count * 6`. Binary cutoffs and learned thresholds are excluded.

All term transforms/centers/scales/weights, the `0.5` cap, closure features, graph construction, and conjunctions are deterministic and use only composition plus raw initial geometry. No DFT calculation/value, learned energy/force/stress proxy, relaxed structure, trajectory, or physical relaxation enters the executable formula. Discovery outcomes remain offline audit/search labels only.

### Task 1: TDD and formal NEXT194 audit

**Files:**

- Create: `tests/test_next194_signed_local_closure_audit.py`
- Create: `src/next194_signed_local_closure_audit.py`
- Create externally: `$PRIS_ARCHIVE/next194_signed_local_closure_audit_v1/`

Test the exact 12 hypothesis identities, signed-local normalization, product/minimum conjunctions, support/missingness, unchanged eligibility, deterministic selection, discovery-only interface, and missing-input failure. Confirm RED before implementation. Require exact frozen feature, endpoint, base, NEXT180, NEXT192, plan, and source provenance. Reconstruct the same base score, fixed folds, repair shell, and full-extreme populations used by NEXT192. Publish atomically.

### Task 2: Conditionally TDD and run NEXT195

If NEXT194 has eligible certificates, create:

- `tests/test_next195_signed_local_closure_search.py`
- `src/next195_signed_local_closure_search.py`
- `$PRIS_ARCHIVE/next195_signed_local_closure_search_v1/`

Test exact interval-only subtraction, missing/base fallback, six amplitudes, exact candidate count, materialized-score recovery, discovery-only interface, and missing-input failure. The formal runner may use only exact NEXT194-eligible certificates.

### Task 3: Diagnose and report

If NEXT195 runs but fails, publish a frozen BROAD residual diagnostic before changing the mechanism. Append NEXT186 through NEXT195 results to the standalone report. Keep all validation/replication endpoints sealed and do not modify canonical manuscript/report files.
