# NEXT17 strict-relaxation group-gap design

Date: 2026-08-02  
Status: frozen before NEXT17 feature execution  
Scope: additive code, tests, external feature artifacts, and a new report only after success

## Motivation

NEXT16 used full-cell MatterSim relaxation with `fmax=0.05 eV/Angstrom` and at
most 64 prediction steps. On the 1,988-row ELEMENTA cohort, 1,888 rows stopped
after the first prediction. The resulting within-composition score was almost
identical to the existing x0 MatterSim score (correlation 0.994; mean absolute
gap change 0.00037 eV/atom). NEXT16 therefore did not meaningfully test whether
deeper pre-relaxation improves the ranking.

NEXT7 already falsified fixed-cell eight-step relaxation as a material quality
improvement. NEXT17 tests the remaining bounded mechanism: stricter full-cell
relaxation. It does not modify or replace NEXT7, NEXT15, or NEXT16.

## Frozen label-free execution

- Input: the existing geometry-only NEXT16 ELEMENTA v2 cohort (400 complete
  reduced-composition groups, 1,988 structures).
- Model: MatterSim 1.2.3, frozen 5M checkpoint.
- Optimizer: FIRE with `FrechetCellFilter`.
- Convergence: `fmax=0.005 eV/Angstrom`.
- Maximum: 64 prediction steps and atom budget 512.
- Support: every structure must produce a finite, physically valid final
  snapshot. If any row in a composition group is unsupported, the whole group
  is unsupported for relative scoring.
- No ELEMENTA endpoint label, endpoint hash, MP hull table, Pauling decision, or
  threshold is read by the feature execution.

For a supported composition group `g`, define

```text
R64s(i) = E_MatterSim_strict_relaxed(i) / N_i
          - min_j_in_g E_MatterSim_strict_relaxed(j) / N_j
```

This is a cohort-relative screening formula. It applies when a generator or
structure search supplies multiple candidates of the same composition. It is
not a single-structure universal law.

## Development threshold and safety gates

The 400 groups have historically visible ELEMENTA labels and are development
data. After the label-free feature artifact is sealed, a separate evaluator may
scan only the finite catalog

```text
T = {0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12} eV/atom.
```

For each threshold, reject iff `R64s >= T`; unsupported groups abstain. A
candidate is eligible only if all of the following hold:

1. DFT group-minimum recall Wilson 95% lower bound >= 0.95.
2. DFT valuable-item (`regret <= 0.05 eV/atom`) recall Wilson lower bound >= 0.95.
3. Reject precision above the group minimum Wilson lower bound >= 0.95.
4. DFT savings Wilson lower bound >= 0.10.
5. No complete supported composition group is fully rejected.

Among eligible thresholds, choose maximum DFT savings, then the larger threshold
as the deterministic tie-break. The resulting threshold is development-selected
and must be frozen before any additional cohort is scored.

## Promotion condition

NEXT17 is promoted only if strict relaxation materially improves over the x0
score on the same 400 groups under paired composition bootstrap:

- DFT savings difference lower bound > 0 at matched group-minimum and valuable
  safety gates; and
- high-regret rejection recall is non-inferior by no more than 0.05; and
- the strict run changes the within-group score by a nontrivial amount rather
  than reproducing x0 numerically.

If it fails, preserve the negative artifact and stop this branch. If it passes,
freeze the formula, threshold, source hashes, and evaluator before selecting an
additional set of complete groups. Because all current ELEMENTA labels have
historically been opened elsewhere, that later result is still retrospective,
not a fresh lockbox.

## Reporting boundary

Do not modify existing reports, papers, README, notes, tex, or preregistration.
Write a standalone NEXT17 report only after a successful promotion or later
validation; otherwise report the negative result in the working handoff only.
