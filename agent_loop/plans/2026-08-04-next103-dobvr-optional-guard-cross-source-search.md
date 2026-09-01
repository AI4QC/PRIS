# NEXT103 DOBVR Optional-Guard Cross-Source Search

**Status:** frozen before NEXT102 features are joined to either discovery endpoint table.

## Goal

Test whether the new discrete oxidation-state/bond-valence mechanism repairs the remaining cross-source SAFE-cell failures of existing no-DFT CVR-style laws without sacrificing their coverage. This is discovery-only model selection. Replication remains unopened unless every frozen gate passes.

## Inputs and boundary

- Existing SCIGEN and WyFormer discovery analytic features and discovery endpoints used by NEXT98b.
- Existing complete NEXT98b candidate table, restricted to the 67 formulas that already passed both source AUC gates.
- New NEXT102 SCIGEN and WyFormer discovery-only DOBVR feature tables.
- No validation or replication feature/endpoint path is an argument to the executable.
- The formula execution input remains one raw unrelaxed x0 structure plus frozen analytic tables. DFT outcomes are labels in this discovery search only; no DFT value enters the resulting formula.

## Why the DOBVR term is optional

NEXT102 label-free support is 2,332/13,470 and 1,884/5,232 for strict NEXT101, and 7,112/13,470 and 3,000/5,232 for expanded NEXT101b. Requiring every candidate term to be present would violate the frozen 0.90 coverage gate by construction.

For an optional DOBVR guard, its robust one-sided hinge is evaluated only when that feature is finite and its family status is supported. Otherwise its contribution is exactly zero and the old base score remains active. Overall support is the intersection of the old base terms only. This is fail-open `KEEP_BASE`, not imputation: an unsupported DOBVR feature can neither raise nor lower risk. Activation coverage is reported unconditionally and by source.

## Frozen new term catalogue

Every term uses a pooled, label-free discovery median and `(q90-q10)/2` scale, with either `log1p_nonnegative` or `asinh` as specified. Terms require at least 0.15 finite coverage in each source, at least eight unique transformed values, and nondegenerate scale.

High-direction terms:

- `dobvr_best_mismatch_rms`
- `dobvr_best_mismatch_q95`
- `dobvr_median_mismatch_rms`
- `dobvr_best_parameter_generic_fraction`
- `dobvr_assignment_log_count`
- `dobvrb_best_mismatch_rms`
- `dobvrb_best_mismatch_q95`
- `dobvrb_median_mismatch_rms`
- `dobvrb_best_parameter_generic_fraction`
- `dobvrb_assignment_log_count`
- `dobvrb_best_catalogue_tier`

Low-direction terms:

- `dobvr_runner_up_gap_rms`
- `dobvrb_runner_up_gap_rms`
- `dobvrb_core_assignment_fraction`
- `dobvrb_best_eneg_margin`

All are `log1p_nonnegative`; no endpoint label influences eligibility, center, scale, or direction.

## Frozen candidate grammar

1. Read the complete NEXT98b records and retain exactly the formulas with `passes_source_auc_gates == True` (expected 67 under formal hashes).
2. Reconstruct each base from its stored old term IDs and positive weights using the NEXT98 pooled label-free calibration catalogue.
3. Include the unchanged base once as a control.
4. Add exactly one eligible optional DOBVR guard with weight in `{0.25, 0.5, 1.0, 2.0, 4.0}`.
5. Do not search two-new-term combinations in NEXT103. That is a possible separately frozen extension only after this complete one-guard result is archived.
6. Deduplicate formulas by canonical JSON of base term IDs/weights plus optional term ID/weight.

Expected maximum is `67 * (1 + 15 * 5) = 5,092` candidates before catalogue exclusions.

## Evaluation and gates

- Formula groups use the existing deterministic five-fold assignment.
- Cells are both source aggregates plus five folds per source: 12 total.
- Source AUC gates: pooled >= 0.75, macro lattice >= 0.60, worst lattice >= 0.55, at least five evaluable lattice classes.
- SAFE gates in every cell: Wilson coverage lower >= 0.90, protected recall lower >= 0.90, severe rejection precision lower >= 0.80, savings lower >= 0.02.
- BROAD threshold must be below SAFE and strictly Pareto-dominate Pauling in every cell, with source-aggregate severe precision lower >= 0.45.
- A candidate passes only if both source AUC gates, all 12 SAFE cells, and all 12 BROAD dominance cells pass.

The selected record is ranked by: complete pass; SAFE availability; BROAD availability; both-source AUC pass; number of passing SAFE cells; worst-cell severe recall; worst-cell precision lower; worst-source pooled AUC; fewer terms; canonical identity.

## Decision

- If at least one candidate passes everything, freeze its formula and predictions before constructing or opening either replication endpoint.
- If none passes, do not open replication. Archive the complete search, nearest feasibility diagnostics, and an independent report. Any mixed-valence or two-guard extension requires a new preregistration and remains additive.

No old script, old result, paper file, canonical report, or replication artifact is modified by NEXT103.
