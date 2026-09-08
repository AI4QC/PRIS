# MatterSim few-step pre-relaxation screening design (frozen)

Date: 2026-08-01  
Status: frozen before any next7 test-partition trajectory is generated  
Scope: adds only `next7` code, tests, outputs and a standalone report; every existing script,
result, report, paper and README is preserved

## 1. The problem and the evidence boundary

The existing MatterSim x0 same-composition energy difference is the strongest pre-DFT screening
baseline in this repository, but on the historical ELEMENTA test set it has not yet reached both
30% DFT count savings and a 95% lower bound on exact/near-min retention. This stage tests a
limited, falsifiable new hypothesis: local geometric error in x0 distorts the single-point energy
ranking, and a fixed small number of force-driven coordinate updates can improve that ranking at
very little cost.

All four ELEMENTA partitions currently count as discovery. Even if the new results improve on the
baseline, they may only be written up as retrospective mechanism evidence; they may not be called
blind or confirmatory after a new split. Genuine confirmation requires the formula, thresholds,
code, checkpoint and decisions all to be sealed first, and must then be done on a physically
isolated new generator batch, or on a new composition holdout that has never entered the existing
queue, with labels computed for every candidate.

## 2. Three routes, and the choice among them

1. `x0 energy + force diagnostics`: no atom moves, the cheapest option, and a control group that
   must always be reported.
2. `fixed-cell few-step FIRE`: the cell is fixed and only atomic positions are updated;
   recommended as this stage's main experiment.
3. `cell + position relaxation`: closer to a full relaxation, but it introduces stress, volume
   collapse and model extrapolation. It does not enter the candidate pool at this stage, and may
   only be pre-registered separately once route 2 shows a clear gain.

Route 2 is adopted. It uses the same 5M checkpoint of MatterSim 1.2.3 and only the species, cell,
PBC and x0 coordinates. Batched inference is combined with a per-structure ASE FIRE state, which
avoids the throughput bottleneck of a per-structure calculator; the `BatchRelaxer` in the
installed package is not modified.

## 3. The frozen optimisation trajectory

```text
cell=fixed
snapshots={0,2,4,8}
optimizer=ASE FIRE
dt=0.05
dtmax=0.20
maxstep=0.05 Angstrom
Nmin=5
finc=1.1
fdec=0.5
astart=0.1
fa=0.99
early_stop=false
```

A "step" is strictly one FIRE coordinate update. Reaching x8 therefore takes 9 energy/force
evaluations from x0 to x8 and 8 coordinate updates. Each structure starts from the same x0 and
snapshots are taken along one continuous trajectory; independent 2-, 4- and 8-step jobs are never
chained into 2, 6 and 14 steps. The cell does not change, and coordinates may be wrapped by the
periodic boundary but the minimum-image displacement must not change.

## 4. Label-free features and a limited set of formulas

Each snapshot stores only: total energy, energy per atom, `Fmax`, `Frms`, the Frobenius norm of
the stress, the largest principal stress, the minimum-image RMS and max displacement relative to
x0, the shortest pairwise distance, the single-step energy change, run errors, the number of
force evaluations, GPU time and peak GPU memory. The DFT energy, forces, stress and ionic
end-state fields in the raw extxyz must be cleared when the ASE `Atoms` object is constructed.

Only the following six same-composition badness scores, with no fitted weights, are permitted:

```text
S0     = gap(E0/N)
S2     = gap(E2/N)
S4     = gap(E4/N)
S8     = gap(E8/N)
Sbest4 = gap(min(E0,E2,E4)/N)
Sbest8 = gap(min(E0,E2,E4,E8)/N)
```

`gap(x_i)=x_i-min_j(x_j)`, with `j` running only over supported candidates of the same
composition. Sweeping over a continuous step count or continuous linear weights is forbidden, as
is using the candidate suffix, the sid or the `rk` order as a score. `rk` is used only to group
candidates whose x0 stoichiometry matches, never as a model input.

## 5. The fail-open support domain

Any of the following must `ABSTAIN` and go to DFT rather than being automatically REJECTed:

- not a genuine `ionic_step=0`, a parse/model/optimisation failure, or any stored quantity
  non-finite;
- fewer than 2 supported candidates of the same composition;
- the shortest pairwise distance at any step falls into the dangerously short contact region
  relative to x0;
- a single coordinate update exceeds the enforced `0.05 Angstrom`, or the cumulative maximum
  displacement at x8 exceeds `0.40 Angstrom`;
- the energy rises by more than `0.02 eV/atom` between adjacent stored snapshots;
- `Fmax > 20 eV/Angstrom` at any stored snapshot.

The engineering check for short contacts is fixed as: abstain if x0's existing
`geom_min_pair_ratio < 0.45`, or if the shortest absolute distance along the few-step trajectory
is non-finite or non-positive. Forces, displacements and short contacts control only the support
domain and never enter a weighted score.

## 6. Partitions and selection

Generate trajectories first for `search_calibration`, `formula_selection` and
`threshold_calibration` only:

1. `search_calibration` only establishes conformal thresholds for the six formulas;
2. `formula_selection`, with the safety gates fixed, chooses one formula by DFT count savings,
   cost and fewer steps;
3. `threshold_calibration` freezes the final threshold for the chosen formula;
4. write `FROZEN_PROTOCOL.json`, containing SHA-256 hashes of the code, inputs and checkpoint;
5. only once the frozen file exists may the historical `test` trajectories be generated and their
   labels read.

The primary track is fixed at `protected=valuable (delta_E <= 0.05 eV/atom)`, `within_group=max`,
`alpha=0.01`. The secondary track is fixed at the historically comparable
`protected=near_min (1 meV/atom)`, `within_group=min`, `alpha=0.035`, and may be read only as a
mechanistic comparison, never as a deployment conclusion.

## 7. Improvement gates

On the current ELEMENTA, a "credible retrospective improvement" may be claimed only when a paired
composition bootstrap satisfies all of: savings increase over step0 by at least 3 percentage
points with a 95% CI lower bound above 0; the 95% CI lower bound of the valuable-recall
difference is not below -0.005; and the increase in abstention is at most 1 percentage point.
Otherwise, only a directional or negative result is reported.

Genuine success still requires a physically isolated new confirmation batch to reach all of:
one-sided 95% lower bounds of at least 0.99 for valuable/stable recall, at least 0.99 for
exact/near-min retention, at least 0.95 for valuable-all group retention, at least 0.30 for DFT
savings, and a p95 regret no greater than 0.025 eV/atom -- passing separately for every
generator. The MLIP GPU cost must additionally be below 10% of the DFT cost saved.

## 8. Artefact isolation

New code goes in `src/next7_*`, tests in `tests/test_next7_*`, outputs in
`outputs/20260801_mattersim_fewstep/`, and the report in a new `reports/2026-08-01-*.md`. No
`next6` file, existing report, paper or normative document is modified.
