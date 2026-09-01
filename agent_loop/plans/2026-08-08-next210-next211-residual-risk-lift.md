# NEXT210--NEXT211 Continuous Residual-Risk Lift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Test whether the 44 NEXT207-stable raw x0 signals can improve the
exact NEXT206 rule through a continuous high-risk lift, preserving ranking and
SAFE behavior that hard zero-score exceptions destroyed.

**Architecture:** Verify NEXT209 branch closure, reconstruct the exact NEXT206
score, and robustly normalize each eligible feature using endpoint-blind
`1/16` and `15/16` empirical quantiles in the current rejected population.
Add one bounded nonnegative risk term only above the frozen residual threshold.
Evaluate the unchanged base plus 220 feature/amplitude candidates with the
unchanged dual-source AUC/SAFE/BROAD evaluator. If no candidate passes all
gates, NEXT211 may diagnose only the exact AUC+SAFE/non-BROAD population.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and the existing
NEXT125/NEXT164 evaluator and residual-diagnostic helpers.

## Boundary and prior evidence

- Executable inputs remain composition plus initial unrelaxed geometry only.
- Discovery outcomes are offline evaluation labels only and are absent from
  normalization and score-construction interfaces.
- No DFT calculation/value, learned energy/force/stress proxy, model or proxy
  potential, relaxed structure, trajectory, or physical relaxation may enter
  the executable score.
- Validation and replication remain physically unopened. Even a complete
  discovery pass requires user review before they are opened.
- Additive files only; preserve all old scripts/content and do not edit
  `paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.

NEXT207 supplied 44 dual-source/fold-stable directional hypotheses with digest
`9d5ccc3ca8dd31c2b4b230330d141f9a05202900cd1f0e243f4140efb60ec24a`.
NEXT208 showed that five new `1/16` hard exceptions preserve source AUC but
fail one SAFE cell, while 16 new exceptions pass SAFE but fail source AUC. The
sole AUC+SAFE survivor was the inactive NEXT206 base. NEXT209 exactly
reproduced the NEXT206 residual and closed the hard-exception branch.

Formal NEXT209 hashes are: design
`6a2b6debb1974231a9b962d8d2d46d7dc7dd6d82a688c31d6b5bc15966076493`,
source `303f2162b820732d470d5d0b708f2cc45d53938115631724940959d26f5e1a7d`,
manifest `0956cbf629092e25f63769e46775ea172ebcd41992338baa6b69dbc28616758d`,
diagnostic `d35960c9c08241a846fd0bde3267173f48ff77bfcaada24c8868d6268ce55b18`,
and table `702346e58a04afcba08c7be73c25aca14f80b17f52338b5523bb8de2cbdd6507`.

## Alternatives considered

1. **Bounded continuous risk lift (chosen).** It uses the audited risk
   direction, preserves fine ordering, never lowers a structure's base risk,
   and has one amplitude degree of freedom.
2. **Empirical-CDF rank score.** Deferred because deployment would require the
   entire empirical distribution rather than two frozen physical cutoffs.
3. **Pairwise conjunctions, denser hard cutoffs, or another pardon depth.**
   Rejected because NEXT208 falsified hard exceptions and these alternatives
   multiply label-driven degrees of freedom without new information.

## Frozen normalization and formula

Let `s`, `u`, and `r=0.16344427817025572` be the exact NEXT206 score, support,
and residual threshold. For each eligible feature `x`, fit cutoffs only on

```text
u AND finite(s) AND s >= r AND finite(x)
```

using NumPy `method="inverted_cdf"`:

```text
q_lo = quantile(x, 1/16)
q_hi = quantile(x, 15/16)
```

Require `q_hi > q_lo`; otherwise fail closed for that hypothesis. Define a
bounded severe-positive risk:

```text
protected_low:  R(x) = clip((x - q_lo) / (q_hi - q_lo), 0, 1)
protected_high: R(x) = clip((q_hi - x) / (q_hi - q_lo), 0, 1)
```

Let the fixed scale be the NEXT206 base candidate's original operating gap:

```text
G = SAFE - BROAD = 0.5415470292150686 - 0.21976295573076796
```

For `a in {1/16, 1/8, 1/4, 1/2, 1}`:

```text
active := u AND finite(s) AND s >= r AND finite(x)
s' := s + a * G * R(x) if active else s
u' := u
```

Missing values keep the base score. The term never reduces risk. No cutoff,
amplitude, residual threshold, feature subset, or operating gate may be tuned
outside this grid. Include the unchanged base once:

```text
1 + 44 * 5 = 221 candidates
```

## Frozen evaluation

Encode each score by the existing exact virtual-term round trip and reuse
`search_optional_guard_laws_parallel` unchanged. A candidate succeeds only if
it passes both-source pooled/macro/worst AUC, every SAFE cell, and every BROAD
comparison with Pauling. Selection uses the existing deterministic evaluator
rank and candidate-key tie break. No validation is opened automatically.

## Task 1: NEXT210 TDD contracts

**Files:**

- Create: `tests/test_next210_residual_risk_lift_search.py`

**Steps:**

1. Test endpoint-free robust cutoffs and exact high/low bounded-risk maps.
2. Test nonnegative lift, threshold activation, missing fail-open, unchanged
   support, and exact amplitude/scale behavior.
3. Test deterministic 221-candidate construction and complete key metadata.
4. Test exact virtual-term round trip, formal interface sealing, and atomic
   missing-input failure.
5. Observe missing-module RED, implement minimally, and require GREEN.

## Task 2: Run NEXT210 formally

**Files:**

- Create: `src/next210_residual_risk_lift_search.py`

**Steps:**

1. Verify complete NEXT209 provenance and branch-closed flags.
2. Reconstruct the exact NEXT206 base and 44 eligible hypotheses.
3. Build exactly 221 endpoint-blind candidates and run the unchanged evaluator
   once.
4. Publish atomically: catalogue, evaluation, frozen-candidate JSON,
   all-candidate Parquet, and hash-complete manifest.

## Task 3: Conditional NEXT211 diagnostic

If NEXT210 has no all-gate pass but has AUC+SAFE/non-BROAD candidates, freeze
their sorted key digest and reproduce their BROAD residuals without searching
a new formula. If none exist, close the continuous raw-x0 lift branch. If an
all-gate candidate exists, stop before validation and report it for user review.

## Task 4: Report and verification

Append verified results to the standalone report only. Run targeted tests,
compilation, full pytest, `git diff --check`, hash checks, canonical-path
checks, report-fence checks, and CodeGraph status. Closing this branch alone
does not complete the overall discovery goal.
