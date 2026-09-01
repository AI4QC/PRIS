# NEXT214 Forward-Stagewise Risk-Lift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task.

**Goal:** Determine whether a short, interpretable additive path of frozen
bounded x0 risks can materially close the remaining Pauling BROAD gap, using a
fully predeclared stopping rule rather than manually launching one stage at a
time.

**Architecture:** Verify NEXT213 and reconstruct its exact closest two-signal
score. At each depth from 3 through 8, add at most one remaining NEXT207-stable
hypothesis with one unchanged NEXT210 amplitude, run the unchanged evaluator,
and diagnose only AUC+SAFE/non-BROAD candidates. Select by the frozen BROAD
residual ordering. Continue only on strict normalized-shortfall improvement;
stop on an all-gate pass, no eligible candidate, no strict improvement, or
depth 8.

**Tech Stack:** Python 3.11, NumPy, pandas, PyArrow, pytest, and existing
NEXT125/NEXT164/NEXT210--NEXT213 helpers.

## Hard boundary

- Executable inputs remain composition plus initial unrelaxed geometry only.
- No DFT calculation/value, learned energy/force/stress proxy, model or proxy
  potential, relaxed structure, trajectory, or physical relaxation may enter.
- Discovery outcomes remain offline labels used only by search/evaluation.
- Validation and replication remain unopened even if discovery passes.
- Additive files only; do not edit canonical manuscripts/reports.

## Frozen starting point

NEXT213's exact closest two-signal key has SHA-256
`927b4473b0720df41473a9def0bacaf75a6d69ae8be26ae14e700ad12834e895`.
It contains:

```text
1/16 * scbv_mismatch_max__protected_low
1/16 * nm_site_max__protected_low
```

Its normalized shortfall is `0.2725415699844472`; all six failures are SCIGEN
`protected_kept` constraints.

Formal NEXT213 identities are:

- design: `215ce456a9b02bf525eececff9958537ea0b2100b27da26432f4300700bcb6e0`
- source: `4ed1d506456eee6619fd7d85974ae37c5229051f372dcfd6c5071bb80d4a9846`
- manifest: `9dec88549c84fdf8b8d64042fa50b1c0a2646e29db7e3784468a53c7d329c2df`
- diagnostic: `e4cf2851a54bde52ce43f3444bccc0775efb9bad8cccef1ce61cc6ce2925d05d`
- table: `6157821669f00fa5da5cbaa9a21c790e5afd131014cbee2dace98dc7d9e009f2`

## Frozen search loop

The 44 audited hypotheses, their directions, their NEXT210 `q_lo/q_hi`, the
risk scale, activation threshold, missing policy, and amplitude grid
`{1/16,1/8,1/4,1/2,1}` are immutable. The two starting hypotheses are removed
from the proposal pool.

For depth `d = 3..8`:

1. Build the unchanged current path once plus every remaining
   hypothesis/amplitude proposal.
2. Add the proposal's nonnegative bounded risk to the current score only on the
   original NEXT206 activation mask; support is unchanged.
3. Run the unchanged dual-source AUC/SAFE/BROAD evaluator.
4. If any all-gate candidate exists, select by the existing evaluator ordering
   and stop before validation.
5. Otherwise diagnose exact BROAD residuals for AUC+SAFE/non-BROAD candidates.
6. Select by `(failed_constraint_count, normalized_shortfall_sum,
   candidate_key)`.
7. Accept only if failed-constraint count decreases, or it is unchanged and
   normalized shortfall decreases by more than `1e-12`. The unchanged path
   cannot count as an improvement.
8. Remove the accepted hypothesis and continue; otherwise stop.

Maximum accepted terms: 8. No pruning, beam search, refitting, new feature,
amplitude interpolation, or manual override is allowed.

## Tasks

1. Create `tests/test_next214_forward_stagewise_risk_lift.py`; observe RED for
   path-score composition, proposal construction, and strict stopping logic.
2. Create `src/next214_forward_stagewise_risk_lift.py`; verify NEXT213,
   reconstruct the start, execute the exact loop, and publish per-depth and
   accepted-path evidence atomically.
3. Run formally into
   `$PRIS_ARCHIVE/next214_forward_stagewise_risk_lift_v1/`.
4. Append verified NEXT210--NEXT214 evidence to the standalone report and run
   targeted, compilation, full-suite, hash, boundary, report-fence, git, and
   CodeGraph verification. Do not commit in the dirty additive workspace.
