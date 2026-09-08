# PREREG-F3: a DFT-free structural score challenging DFT E_hull head-on at synthesizability ranking

Date frozen: 2026-08-14 (written before any fit on dev and before any contact with the holdout)
Status: **frozen**. Once written, this document may not be modified except by appending a
"revision log" section.

> **Translation note.** This file was translated into English on 2026-09-08. The frozen
> Chinese original is the version hashed as `a1b2f72e0e4f336546b95555e9ca79192ff7280542c81088b0a1d5714bbef299`
> in `agent_loop/frozen/20260814_f3_synth/PREREG_SHA256`, and remains recoverable from the git
> history. Nothing below has been changed apart from the language.

## 0. Motivation and prior evidence (all from published or archived numbers; no new computation)

On the paper's same-composition synthesizability ranking task (the S13 protocol), the published
baselines are:

| criterion | commit | group-equal acc | acc \| pairs e_hull gets wrong |
|---|---:|---:|---:|
| DFT E_hull | 0.9930 | **0.7501** | 0.0 (by definition) |
| vol_per_atom | 0.9863 | 0.6435 | 0.5463 |
| Shannon packing | 0.9879 | 0.6281 | 0.5630 |
| bl_min (rho_c) | 1.0000 | 0.5688 | 0.5959 |
| Pauling 5 | 0.2230 | 0.6553 | 0.5675 |

The key facts: (a) F2 was fitted against the e_hull label and then transferred to this task, so
**no score has ever been fitted directly against the synth label**; (b) on the pairs e_hull gets
wrong, the structural criteria reach 0.55-0.60, so there is genuine complementary signal; and
(c) R10 found that synthesised entries are more symmetric (median space group 87 vs 62), while
no symmetry feature is in the existing feature tables.

## 1. Data (frozen, and in existence before this pre-registration)

`features/synth_rank.parquet` (6,878 rows, 85 features). The task population is the compositions
that have both synth classes, reproducing the paper's convention: **1,508 groups / 6,758
structures / 18,920 pairs** (checked). Any new row must first pass the four-row, four-decimal
reproduction check in `rank_rulesets.py` before publication.

## 2. Splits (frozen)

```python
dev = zlib.crc32(f"{rk}|synthsplit20260814".encode()) % 10 < 6
```

- dev: 919 groups / 4,187 structures / 14,563 pairs (SiO2 falls in dev)
- holdout: 589 groups / 2,571 structures / 4,357 pairs

The holdout compositions may not be read at any step of fitting, feature selection, tuning or
inspecting single-feature accuracies.
Permitted contacts with the holdout: **1** (the one-shot evaluation of section 6).

## 3. Permitted features (frozen)

Allowed: the float features of synth_rank.parquet plus the new symmetry and classical-energy
features (section 4), excluding each of the following:

- labels and identity: `synth`, `e_hull`, `mp_id`, `rk`
- size confounders (extensive quantities): `nsites`, `n_sites`, `p2_n_bad_020`, `p2_sum_dev`,
  `p3_n_pairs`, `p3_n_face`, `p3_n_edge`, `p4_n_viol`
- the Ewald decomposition in total-quantity form: `ewald_real`, `ewald_recip`, `ewald_point`
  (`ewald_per_atom` is kept)
- the table-coverage pseudo-quantity: `bv_param_cov`

Any DFT quantity, machine-learning potential or relaxation trajectory is forbidden outright (the
execution boundary matches the paper's).
Composition-level features are always ties on a same-composition task: harmless, but not
selected.

## 4. New features (definitions frozen; computed without looking at the labels)

Rebuilt for all 6,878 structures from the MP snapshot and written to a **new file**,
`synth_rank_aug.parquet`:

Symmetry (SpacegroupAnalyzer/spglib, symprec in {0.01, 0.1}):
- `sg_num_001`, `sg_num_01`: international space-group number
- `csys_rank_001`: crystal-system rank (triclinic 1 ... cubic 7)
- `wyckoff_econ_001`, `wyckoff_econ_01`: inequivalent sites / sites (the continuous structural
  form of Pauling's fifth rule)

Classical Born terms (charges = composition-only integer balanced valences, radii =
`phys_law.shannon` by (element, oxidation state, CrystalNN CN); over all neighbour pairs with
d < 1.25 x r_sum):
- `rep9_ca_pa`, `rep9_aa_pa`, `rep9_cc_pa`: Sum (r_sum/d)^9 / N, classified by the pair's charge
  signs
- `repexp_ca_pa`, `repexp_aa_pa`, `repexp_cc_pa`: Sum exp((r_sum - d)/0.345 A) / N
- `strain2_ca_pa`: Sum ((d - r_sum)/r_sum)^2 / N (elastic strain of opposite-charge contacts)
- `density`: mass density in g/cm^3

## 5. Model class and selection procedure (frozen)

- Missing values: imputed with the dev median; standardisation: dev mean and variance. Both are
  frozen and then applied to the holdout.
- Paired samples: all within-group (synth=1, synth=0) pairs, X = f1 - f0; an antisymmetric
  logistic (no intercept), each pair weighted 1/(pairs in group) so that groups count equally,
  matching the evaluation metric.
- **F3 (the deliverable)**: greedy forward selection, scored by the group-equal accuracy on the
  validation folds of a 5-fold GroupKFold within dev (grouped by composition, using the same
  `rank_rulesets.evaluate` implementation); stopping when the best gain falls below 0.002 or at 8
  terms. Refit on all of dev, freeze the coefficients, record the SHA-256.
- **F3-full (an upper reference only)**: every permitted feature, with the L2 strength chosen by
  the same CV. The claim rests only on the sparse F3.
- Any diagnostic is allowed within dev; every number from dev is labelled as development.

## 6. The one-shot holdout evaluation and the success gates (frozen)

The frozen F3 is evaluated once over the holdout groups. Every paired comparison is made on the
groups where both criteria commit.
Clustered bootstrap: resample the holdout composition groups, B = 2000, seed 20260728, taking the
95% percentile interval of the **difference**.

- **G1 (the primary gate)**: the 95% lower bound of D = acc(F3) - acc(E_hull) is above 0
- **G2**: F3's commit rate is at least 0.99
- **G3**: top-1 lift(F3) is at least top-1 lift(E_hull) (point estimate)
- **G4**: acc(F3) exceeds the holdout accuracy of all three single quantities vol_per_atom,
  sh_pack and bl_min (point estimates; otherwise the combination is meaningless)
- **G5**: report the largest group's share and the number of groups; no claim is made under any
  convention dominated by a single group

Graded outcomes (wording frozen):
- G1-G4 all pass: "frozen DFT-free score surpasses DFT E_hull on held-out compositions of this
  task" (the label confounder, a preference in synthesis history, must still be stated)
- G1 fails but the point estimate of D is at least 0 and G2-G4 pass: "parity with DFT within CI"
  -- folded into the paper as a negative or neutral result, with no retuning and no retry
- Otherwise: a negative result, entered in the refutation ledger

After the holdout evaluation, any further change may only be confirmed as a **new pre-registered
chain** on new data.

## 7. Boundary statements (frozen)

- The label is "the MP entry carries an ICSD number", which is contaminated by a selection
  preference in synthesis history; positive-class contamination lowers the ceiling.
- The structures are MP's DFT-relaxed geometries; executing the criterion itself calls no DFT,
  consistent with the paper's T3 framework.
- This task is relative ranking within a composition, not an absolute plausibility judgement; its
  semantics differ from D1-D6 and the two may not be tabulated together.

(SHA-256 of this file at freeze time is recorded in
`outputs/20260814_f3_synth/PREREG_SHA256`.)

## Revision log

**Revision 1 (2026-08-14, before any contact with the holdout; found on the dev side)**: the
greedy search of the F2R chain, scored on committed-only accuracy, selected a degenerate solution
with low coverage and high accuracy (cn_cat_max, cov 0.458) -- exactly the "trade abstention for
accuracy" pattern diagnosed in the paper's section sec:pauling. The revision: the CV score of the
forward selection assigns -1 (a hard rejection) to any candidate set whose fold-averaged coverage
falls below 0.99. Gate G2 already required a final coverage of at least 0.99; this revision moves
the same constraint forward into the selection procedure, so that the single permitted holdout
contact is not wasted on a degenerate solution bound to fail. The same revision applies to
PREREG-F2R.
Neither the holdout nor calibration had been contacted at the time of this revision
(neither HOLDOUT_CONTACT.log nor CALIB_CONTACT.log exists).

**Revision 2 (2026-08-14, before any contact with the holdout)**: a secondary frozen model,
**F3H (hybrid)**, is added: an antisymmetric logistic fitted on dev over the two features
[s_F3, e_hull] (s_F3 being the frozen F3 score; the standardisation statistics come from dev),
with the coefficients frozen and evaluated alongside F3 within the **same single** holdout
contact. Secondary gate **G6**: the clustered-bootstrap 95% lower bound of
acc(F3H) - acc(E_hull) is above 0. G6 tests a claim different from and independent of G1:
"structural chemistry carries synthesizability information beyond DFT stability". F3H contains a
DFT quantity and may not be described as a DFT-free criterion; the wording of G1 is unchanged by
G6. The holdout had not been contacted at the time of this revision (HOLDOUT_CONTACT.log does not
exist).
