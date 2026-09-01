# Better Laws and Formulas: Additive Search Design

**Date:** 2026-07-31

**Experiment ID:** `np-next-20260731`
**Status:** frozen before the new descriptor matrices or candidate outcomes are inspected

## Scope boundary

This experiment is additive. It must not modify or overwrite:

- `PREREG.md`;
- `README.md` or `src/README.md`;
- `paper/FACTS.md`, `paper/data/`, `paper/si_data/`, or figures;
- `notes/`;
- `tex/`;
- any existing script or external feature-store artifact.

New code lives in new files. Row-level feature caches are written outside the repository.
Only aggregate, identifier-free results and a new report may be added to `outputs/` and
`reports/`. The sealed lockbox is not opened or read.

## Evidence state that constrains this design

The current reproducible rule baselines on calibration are:

| set | real satisfaction | perturbed exclusion |
|---|---:|---:|
| L1 | 0.991882 | 0.289037 |
| L1' | 0.989428 | 0.383721 |
| L2 | 0.957901 | 0.612126 |
| L3 | 0.917123 | 0.700443 |

The archived seven-term F2 formula is not a reproducible executable baseline. Its
fold-specific standardisation was not retained, and the current `formula2.py` does not
contain the seven-term selection path. It also scores pairs rather than groups equally,
uses a non-group-aware inner CV, and chooses abstention fractions on evaluation scores.
This experiment therefore compares against a newly documented, reproducible refit and
reports the archival F2 numbers only as non-reproduced historical context.

The old calibration split has been inspected repeatedly, and the repository discloses
implicit lockbox contamination in an earlier full-sample quantile fit. It remains useful
for exact historical comparisons but cannot support a fresh claim of unbiased discovery.
The primary new evidence therefore comes from deterministic, group-preserving splits
inside the non-lockbox development data. Calibration is a secondary comparability check.

## Alternatives considered

### A. Repeat the old scalar-threshold sweep

This is cheap and maximally comparable, but it cannot represent bond-strength
distribution, directional imbalance, or smooth neighbour weights. It is unlikely to
move the frontier except by exploiting the same perturbation fingerprints.

### B. Unrestricted symbolic regression or a shallow decision tree

These can improve in-distribution accuracy. The repository already shows that a certified
depth-three tree dominates L2 in distribution and then loses that advantage when a
perturbation mechanism is held out. Unrestricted search also makes multiplicity and
interpretability harder to control.

### C. Physics-frozen descriptors plus the existing loop, with nested validation

This is the selected approach. The candidate vocabulary is frozen from primary
literature before outcomes are computed. It adds only local, interpretable descriptors,
then reuses the same propose -> featurise -> search -> test -> refute loop. Law search
stays threshold/guard/band based; formula search stays sparse and linear.

## Frozen descriptor vocabulary

### P1: bond-valence local triplet

Using the committed `bvparm2020.cif` parameters and formal oxidation states only:

\[
s_{ij}=\exp[(R_{0,ij}-d_{ij})/B]
\]

\[
\delta_i={\sum_j s_{ij}-|V_i|\over |V_i|},\qquad
N_i^{BV}=\exp\left(-\sum_j p_{ij}\log p_{ij}\right)
\]

\[
A_i^{BV}={\left\|\sum_j s_{ij}\hat r_{ij}\right\|\over\sum_j s_{ij}},
\qquad p_{ij}={s_{ij}\over\sum_j s_{ij}}.
\]

The structure-level vocabulary is fixed to charge-sign-separated mean, q95, and maximum
for absolute relative valence mismatch, \(N_i^{BV}\), and \(A_i^{BV}\), plus parameter
coverage. No bond-valence parameter is fitted on NeoPauling labels. `BVAnalyzer` remains
banned.

### P2: optional solid-angle weighting

If P1 does not meet the success gate, the same pass may add Voronoi solid-angle entropy
coordination and the solid-angle fraction assigned to same-charge neighbours. This stage
is reported separately because it is slower and changes neighbour semantics.

### P5: strict iterative Hoppe diagnostics

The pass may also compute strict iterative ECoN/MEFIR diagnostics and their difference
from the existing nearest-distance approximation. These are secondary because the
repository already contains approximate ECoN/MEFIR.

P3 Hawthorne network residuals and full P4 CSM are not implemented in this experiment.
Their full algorithms are materially larger, and partial substitutes must not be labelled
as the published quantities.

## Law-search protocol

1. Load the same deterministic S1-S5 lineage used by `phys_bad.parquet`.
2. Hard-fail on null splits or `lockbox`.
3. Fit all numeric thresholds on original `discovery` rows only.
4. Enumerate:
   - one-sided scalar thresholds;
   - two-sided bands on one feature;
   - `if guard then threshold/band` rules, including composition-only `fi`/`dchi`
     guards and the existing structural guard vocabulary.
5. Search conjunctions with a Pareto beam. Retain candidates by pooled exclusion,
   exclusion per unit satisfaction cost, and worst-perturbation exclusion.
6. Report S1-S5 separately, feature coverage, worst anion satisfaction, and rule count.
7. Run leave-one-perturbation-out as a diagnostic. Report all five signed changes and
   their mean absolute change; do not use cancellation in the signed mean as robustness.
8. Apply fixed candidates to the 295 DFT-relaxed false-positive structures when every
   required feature is available. Unknown is not pass.

### Law success gate

A new set is called **promising**, not confirmed, only if all are true:

- at a matched real-satisfaction operating point it raises exclusion by at least 0.02,
  or raises the minimum S1-S5 exclusion by at least 0.03;
- the direction is the same in deterministic discovery sub-splits and in the historical
  calibration comparison;
- real and perturbed descriptor coverage are each at least 0.90;
- worst-anion satisfaction is no worse than the matched baseline by more than 0.01;
- DFT-relaxed false-positive loss is no worse than 0.03 where measurable;
- no lockbox or unknown-split row enters fitting or evaluation.

The label **confirmed** is reserved for a future genuinely untouched source/temporal
holdout or an authorised lockbox opening.

## Formula-search protocol

1. Restrict to non-lockbox experimental ranking rows.
2. Form pairs only within reduced formula.
3. Assign whole reduced-formula groups to deterministic folds.
4. Standardise and impute using training-fold statistics only; save those statistics.
5. Give every formula group total weight one, independent of its pair count.
6. Select at most seven terms by inner grouped validation from:
   - the archival seven terms;
   - the existing high-coverage physical features;
   - P1/P2/P5 descriptors if available;
   - dimensionally valid depth-one transforms fixed in code.
7. Evaluate once in the outer fold. Select abstention thresholds on inner-fold scores,
   then apply the numeric threshold to the outer fold.
8. Report full coverage, fixed 30% and 10% commitment targets, tie/commitment counts, and
   energy-gap strata including \(|\Delta E|\ge 50\) meV/atom.

### Formula success gate

A formula is **promising** only if:

- group-equal outer accuracy improves the reproducible sparse-linear refit by at least
  0.02 at full coverage or at a fixed commitment target;
- the improvement is not caused by a single composition or by pair weighting;
- the same sign is obtained in every outer fold or all but one fold with the remaining
  fold within 0.01 of the baseline;
- it uses at most seven nonzero terms and all fold statistics are saved;
- the energy-gap-stratified table and bootstrap interval are reported even when they
  weaken the headline.

## Outputs and stopping rule

The run writes only:

- new source and tests;
- identifier-free JSON/CSV aggregates under
  `outputs/20260731_better_laws_formulas/`;
- a new Markdown report under `reports/`.

If neither success gate is met, the report is a negative-result report. Existing reports
and the paper remain unchanged until the user reviews the new report and explicitly asks
for integration.
