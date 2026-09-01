# NEXT212--NEXT213 Anchored Two-Signal Risk-Lift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Test whether one frozen secondary continuous x0 risk signal can
complement the exact closest NEXT211 `scbv_mismatch_max` anchor while retaining
both-source AUC and SAFE behavior and reducing or eliminating its BROAD
protected-retention residual.

**Architecture:** Verify NEXT211 and reconstruct the exact NEXT210 universe.
Select the NEXT211 closest candidate only by its already-frozen diagnostic
ordering, reproduce its score, and add at most one second bounded nonnegative
risk lift using the remaining 43 NEXT207-stable hypotheses and the unchanged
five-amplitude grid. Evaluate all 216 candidates once with the unchanged
dual-source evaluator. If no all-gate candidate exists, NEXT213 may diagnose
only the exact AUC+SAFE/non-BROAD population.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and the existing
NEXT125/NEXT164/NEXT210/NEXT211 evaluators and provenance helpers.

## Boundary and prior evidence

- Executable inputs remain composition plus initial unrelaxed geometry only.
- No DFT calculation/value, learned energy/force/stress proxy, model or proxy
  potential, relaxed structure, trajectory, or physical relaxation may enter
  the executable score.
- Discovery outcomes are offline evaluation labels only and are absent from
  score-construction interfaces.
- Validation and replication remain physically unopened.
- Additive files only. Do not edit `paper/`, `tex/`, `notes/`, `README.md`, or
  `PREREG.md`.

NEXT211 reproduced all 91 frozen candidates. Every one failed only the six
SCIGEN `protected_kept` constraints. Its closest candidate reduced the NEXT206
normalized BROAD shortfall from `0.2905066371709464` to
`0.2766359997905851`, but did not pass BROAD.

Formal NEXT211 identities are:

- design: `d00ede3926d3a525327e9f2fbb9728fa57b47089695efe4d43fd115663b49409`
- source: `a3e8e84b7646c4376d5bce4673faf1a22dbfee488ce1ad698a1b3cbfb36962cb`
- manifest: `fa5287b8099d7b302caf360ddcab617b3ec46b58227b5d09352cdf5f80406de2`
- diagnostic: `c48bc36d7a00f3a48e3e6c91248e69b30525acd888adc3890daa1f90d5926f1a`
- table: `a860da0c31382aa582b3de158d87bce9e0790707e631ddbf0b979b8fa395809a`

## Frozen anchor and candidate universe

The anchor is the exact NEXT211 closest key with SHA-256
`c66e9a7afc180e7060eb8e4de408c2552af24f45a9b493984d65de5250479ebb`:

```text
feature = scbv_mismatch_max
direction = protected_low
amplitude = 1/16
q_lo = 0.15502600238558073
q_hi = 1.8461993346107197
risk_scale = 0.3217840734843006
residual_threshold = 0.16344427817025572
```

Let `s0`, `u`, and `r` be the exact NEXT206 score, support, and residual
threshold. Let `R1` be the frozen anchor risk and `a1=1/16`:

```text
s1 = s0 + a1 * G * R1, active only where u AND finite(s0) AND s0 >= r
```

For each of the remaining 43 NEXT207 hypotheses, reuse its exact NEXT210
`q_lo`, `q_hi`, direction, and bounded risk `R2`; do not refit normalization.
For `a2 in {1/16, 1/8, 1/4, 1/2, 1}`:

```text
s2 = s1 + a2 * G * R2, active only where u AND finite(s0) AND s0 >= r
u2 = u
```

Missing secondary values keep `s1`. Include the unchanged anchor once and
exclude the anchor hypothesis from the secondary grid:

```text
1 + 43 * 5 = 216 candidates
```

No feature, direction, cutoff, amplitude, activation condition, or operating
gate may be changed after this plan is hashed.

## Task 1: NEXT212 contracts

**Files:**

- Create: `tests/test_next212_two_signal_risk_lift_search.py`

**Steps:**

1. Test exact anchored score construction, nonnegative secondary lift,
   missing-value behavior, and unchanged support.
2. Test deterministic 216-candidate construction and anchor exclusion.
3. Test exact virtual-term round trip and unique physical candidate keys.
4. Test the formal interface exposes discovery paths only and fails closed on
   missing input.
5. Observe missing-module RED, implement minimally, and require GREEN.

## Task 2: Formal NEXT212 search

**Files:**

- Create: `src/next212_two_signal_risk_lift_search.py`

**Steps:**

1. Verify complete NEXT211 provenance and the frozen anchor identity.
2. Reconstruct all 221 NEXT210 specs and the exact anchor score.
3. Build exactly 216 anchored candidates and run the unchanged evaluator once.
4. Publish atomically: catalogue, evaluation, frozen-candidate JSON,
   all-candidate Parquet, and hash-complete manifest.
5. Stop before validation even if an all-gate candidate is found.

## Task 3: Conditional NEXT213 diagnostic

If NEXT212 has no all-gate pass but has AUC+SAFE/non-BROAD candidates, freeze
their sorted key digest and reproduce their BROAD residuals without searching
a new formula. If none exist, close the two-signal branch. If an all-gate
candidate exists, stop before validation and report it for user review.

## Task 4: Standalone report and verification

Append verified NEXT210--NEXT213 results only to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Run targeted tests,
compilation, full pytest, `git diff --check`, formal hash checks, canonical-path
checks, report-fence checks, and CodeGraph status. Do not commit in the existing
dirty additive workspace.
