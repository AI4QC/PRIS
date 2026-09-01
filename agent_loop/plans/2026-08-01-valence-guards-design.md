# Valence-Confidence Guards: Additive Search Design (round np-next-20260802)

**Date:** 2026-08-01 (frozen before any candidate outcome of this round is inspected)

**Experiment ID:** `np-next-20260802`

**Status:** frozen. The guard vocabulary extension in §2, the unchanged searchable
descriptor vocabulary in §3, and the gates in §5 are fixed before the search runs.

## 1. Scope and sequential-round disclosure

Additive again: no existing script, document, output, or feature-store artifact is
modified, including the np-next-20260801 additions; new code lives in new `next2_*`
files. This is a sequential round: it is motivated by the diagnosed nitride-stratum
failure of np-next-20260801 (nitride satisfaction −0.0342 vs the −0.01 gate, 35×
enrichment of `p2vor_an_sa_like_fraction_max` violations in real nitrides) and by the
matching P3 residual tail (486 real infeasible structures enriched in N/P/Te). The
remedy — chemistry-aware guards — is pre-registered here before any outcome of this
round is seen. The success gates are **not** loosened; discovery remains the only
selection split and calibration remains a disclosed, adaptively reused diagnostic.

## 2. Frozen guard vocabulary extension

Three new composition-only guard columns, computed from reduced formula with the same
`oxi_state_guesses(max_sites=-10)` call used by `discriminate.guess_oxi`:

- `z_an_abs`: |anion formal valence| under the first guess (the assignment the
  descriptor pipeline actually used). Values in {1, 2, 3}; nitrides/phosphides = 3.
- `oxi_n_guesses`: number of composition oxidation-state assignments returned, capped
  at 20. A valence-ambiguity measure.
- `oxi_unique`: 1 iff exactly one assignment exists.

Perturbations S1–S5 preserve composition, so bad rows inherit their parent's values
(exactly like the existing `fi`/`dchi` composition guards).

The guarded-candidate guard vocabulary is frozen as the previous seven
(`mean_cn_cat`, `z_cat_max`, `cn_an_mean`, `n_el`, `cat_an_ratio`, `fi`, `dchi`)
plus these three, at the same quantiles (0.25/0.5/0.75, both directions).

## 3. Unchanged searchable vocabulary

Exactly the 61 frozen descriptor columns of np-next-20260801 (P1 18 + P2 18 + P3 7 +
P5 18). No new descriptor family, no re-featurization; the previous round's descriptor
caches are reused as-is. The new guard columns are **guard-only**: they are identical
for a structure and its perturbations, so their exclusion power is identically zero
and they are excluded from threshold/band candidate generation.

## 4. Analysis plan

1. Compute guard tables for the isolated real records, bad parents, and the 295
   DFT-relaxed false-positive structures (composition only).
2. Re-run the identical law loop (floors 0.99/0.98/0.95, min coverage 0.90, width 24,
   max 12 rules, paired-anion-guarded variant included) with the extended guard
   vocabulary on the physically isolated tables.
3. True LOKO refits at floor 0.98; 295 DFT-relaxed falsification, unknown fails
   closed.
4. Pre-registered targeted analysis (not search): the satisfaction/exclusion
   trade-off of the specific guarded forms
   `if fi > τ then p2vor_an_sa_like_fraction_max ≤ 0.9677` and
   `if z_an_abs ≤ 2 then p2vor_an_sa_like_fraction_max ≤ 0.9677`,
   including their per-anion satisfaction and per-kind exclusion.
5. The formula loop is **not** re-run: the new guard columns are constant within
   every reduced-formula group, so they carry zero within-group ranking signal.
   The previous round's formula result stands unchanged.

## 5. Frozen success gates

Identical to np-next-20260801: promising requires matched satisfaction (≥ baseline
−0.005), pooled exclusion +≥ 0.02 or minimum-kind +≥ 0.03, same direction in
discovery sub-splits and calibration, coverage ≥ 0.90 on every selected feature,
worst-anion drop ≤ 0.01, DFT-relaxed pass-rate drop ≤ 0.03, zero lockbox
materialization downstream of the isolation builder. `confirmed` remains reserved
for a genuinely untouched holdout.

## 6. Stopping rule

If no candidate passes every gate, the report is again a negative-result report, and
the guarded-trade-off analysis in §4.4 stands as the round's deliverable. Existing
reports and the paper remain unchanged regardless.
