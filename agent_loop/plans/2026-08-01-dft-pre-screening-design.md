# Pre-DFT crystal structure screening: the unrelaxed-input confirmation plan (frozen)

Date: 2026-08-01  
Status: frozen before any joint WBM feature-label search  
Scope: adds only scripts, tests, outputs and a standalone report; no existing script, existing
result, the paper, the README or any formal report is modified

## 1. Objective and immutable boundaries

The objective is not to keep raising the "pairwise win rate on DFT-relaxed structures" but to
find an interpretable criterion that safely reduces the number of full DFT relaxations while
reading only the raw candidate structure `x0`.

The criterion may read only:

- elements and stoichiometry;
- the cell and coordinates of `x0`;
- element tables, bond-valence parameters and formula constants frozen before any test result is
  seen.

It may not read: DFT final-state coordinates, final volume, the final-state neighbour graph, DFT
energies, forces, stresses or ionic step counts, nor any feature derived from those. When the
oxidation state has no unique reliable solution, a parameter is missing, or a neighbour
computation fails, it must `ABSTAIN` and send the structure to DFT; an unknown may not be counted
as a correct rejection.

The output has three states: `KEEP / REJECT / ABSTAIN`. In the deployment statistics, `ABSTAIN`
consumes DFT budget just as `KEEP` does.

## 2. The roles of the data

### 2.1 Discovery (may propose candidates only, and cannot confirm deployment capability)

- The existing `polymorph_rank2.parquet`, `elem_rank.parquet`, `alex_rank.parquet` and
  `lemat_rank.parquet`: their features come mainly from DFT-relaxed final states.
- `real_rank.parquet`: a safety audit on experimental structures.
- `synth_rank.parquet`: a positive-unlabelled synthesis-enrichment diagnostic; the absence of an
  ICSD entry may not be treated as a true negative.

These data have been inspected repeatedly and all count as discovery.

### 2.2 Initial-to-final transfer diagnostics

The ELEMENTA core originally contains 38,808,603 frames across 2,028,008 continuous DFT
relaxation trajectories. The rules are computed on the first frame of each trajectory only, with
the final-frame energy, convergence status and the initial-to-final structural drift as labels.
That data is used to:

- check whether candidates discovered on final states still hold on `x0`;
- measure same-composition energy regret;
- measure the rule's score and ranking flip rate from first frame to last.

ELEMENTA shares an origin with this repository's existing searches and therefore cannot by itself
carry the final external confirmation.

### 2.3 The unrelaxed external benchmark

The local Matbench Discovery/WBM contains 256,963 initial and DFT-relaxed structure pairs, with an
official `unique_prototype` subset of 215,488. The input is fixed as the initial extxyz and the
label is fixed as:

```text
stable := e_above_hull_mp2020_corrected_ppd_mp <= 0
```

Sensitivity analyses at `E_hull <= 0.05` and `<= 0.10 eV/atom` are also reported, but they may not
replace the primary label. `site_stats_fingerprint_init_final_norm_diff` is a structural-survival
outcome only and may not enter the features.

WBM's total row count and total positive count have already been read, so this round may not
claim a "never-read lockbox"; but once this document is frozen, the candidate family, the
directions, the primary label and the primary gates may not be changed on the basis of any joint
WBM feature-label result.

### 2.4 The final confirmation layer (outside this round)

If both WBM and ELEMENTA pass, freeze the formula and then run prospective DFT under a uniform
workflow on new candidates from at least two generators that took no part in the search. Only that
layer can support a strong deployment conclusion.

## 3. Frozen splits and units of independence

- Every split takes the reduced composition as its smallest unit; the same composition may not
  cross calibration and test.
- Different frames from the same prototype, the same generation lineage and the same DFT
  trajectory must stay on the same side.
- WBM uses the fixed mapping `sha256(reduced_formula)`: the first 20% of compositions are the
  outer calibration and the remaining 80% the test. The outer calibration is then split by the
  parity of an independent `sha256("stage:" + reduced_formula)` into `formula_selection` and
  `threshold_calibration`; the first selects the formula only and the second sets the rejection
  threshold and the risk upper bound only.
- All three parts physically write out their own labels, `x0` feature files and a SHA-256
  manifest first; the test is executed exactly once and writes an opening log.
- Confidence intervals use a whole-cluster bootstrap over compositions; structure pairs may not be
  treated as independent samples.

## 4. The frozen candidate law family

The search permits only the following non-negative combinations, whose physical directions are
fixed, and no free per-element fitting:

\[
\Phi(x;q,\lambda)=
a G_q(\lambda x)^2+b R_{\rm rep}(\lambda x;q)
+c \widehat E_{\rm Ewald}(\lambda x;q)+d P_{\rm pack}(\lambda x)
+\eta |\log\lambda|.
\]

- `G_q`: the per-atom-averaged bond-valence-sum mismatch; smaller is more plausible.
- `R_rep`: a softplus/Born-type repulsion for short-range overlap; smaller is more plausible.
- `Ewald`: a per-atom electrostatic surrogate under charge-balanced oxidation states; lower is
  more plausible.
- `P_pack`: a two-sided packing penalty for being too dense and too loose; smaller is more
  plausible.
- `q`: a finite set of charge-balanced oxidation-state assignments; abstain when there is no
  reliable solution.
- `lambda`: a scale envelope taken only over the pre-registered grid `[0.8, 1.2]`, penalising
  departure from 1.

Complexity increases in order:

1. static single-term rules;
2. scale-envelope single-term rules;
3. non-negative sparse combinations of at most three physical terms;
4. if the first three layers show a stable gain, evaluate analytic pseudo-forces and a fixed
   few-step surrogate-potential pre-relaxation.

The coefficients, repulsion exponents, scale grid and rejection thresholds may be chosen only
from pre-registered finite grids; report the whole Pareto frontier, and do not keep only the best
row.

## 5. Baselines

Compare under exactly the same coverage/abstention accounting:

1. no filter, and random rejection;
2. Pauling's original rules and their best single rule;
3. `bl_min`/short bonds, packing, volume per atom;
4. traditional BVS/GII;
5. the Ewald term alone;
6. the currently frozen sparse new formula.

The official Matbench Discovery MP-only and wide-data MLIP results serve as performance
references only; unless they are actually run locally, they may not be written up as
same-environment reproductions.

## 6. Primary metrics

The WBM primary task reports:

- stable recall and the false-negative rate;
- precision, F1 and DAF (precision / prevalence);
- the actual rejection fraction, i.e. the theoretical saving in DFT count;
- coverage, abstention and the number of feature failures;
- top-k precision (k = 10,000 and at a fixed budget fraction);
- the worst stratified value across major chemical families, element OOD and prototype batch;
- composition-cluster bootstrap 95% CIs.

The ELEMENTA same-composition task reports:

- group-min retention;
- false rejection of near-optimal candidates (25/50 meV/atom);
- median/p90/p95 regret of the lowest retained energy;
- rejection rate of high-energy candidates (>= 0.20 eV/atom);
- the rule's score and ranking flip rate between first and last frame;
- DFT savings weighted approximately by candidate count and ionic steps.

The pairwise win rate is kept only as a secondary diagnostic.

## 7. Pre-registered acceptance gates

### Effective pre-screening (all must hold)

- the one-sided 95% lower bound of stable recall >= 0.99;
- the one-sided 95% lower bound of the rejection fraction >= 0.30;
- the one-sided 95% lower bound of ELEMENTA group-min retention >= 0.95;
- ELEMENTA p95 energy regret <= 0.05 eV/atom;
- at the same missed-screening risk, DFT savings strictly exceed Pauling, short bonds, packing,
  BVS/GII and Ewald;
- every pre-registered independent source passes; a pooled average may not mask a source failure;
- every unknown counts as abstention.

### The high gate for approaching DFT (all must hold)

- at a saving of at least 50% of the full DFT, the stable-recall lower bound >= 0.99;
- the group-min retention lower bound >= 0.99;
- p95 regret <= 0.025 eV/atom;
- the retention point estimate >= 0.95 for every major chemical family and source.

If the gates are not cleared, the conclusion must be "an exploratory physical surrogate, a
negative result", and the definition of success may not be narrowed.

## 8. The evidence boundary

With full DFT relaxation as the label, one may not claim to "surpass DFT" on that same label. The
legitimate, testable statements are:

- making screening decisions close to DFT at very low missed-screening risk, and reducing the
  number of full relaxations;
- surpassing a coarse DFT single point on unrelaxed structures, or the existing non-DFT
  baselines;
- in future, with r2SCAN, phonons or experimental survival/synthesis as the outer label, testing
  whether the PBE decision is surpassed.

## 9. Artefact isolation

New code goes in `src/next6_*`, tests in `tests/test_next6_*`, outputs in a separate
`outputs/20260801_dft_prescreen/`, and a standalone report is written first. The old scripts, old
outputs and the paper all stay as they are.
