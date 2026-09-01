# NEXT106 CMVF Optional-Guard Cross-Source Search

Date: 2026-08-04

## Frozen purpose

Evaluate whether NEXT104 convex mixed-valence periodic-flow incompatibility adds enough discovery-only signal to the 67 NEXT98b formulas that already pass both-source AUC gates. This document is frozen before reading any CMVF-labelled endpoint join or CMVF outcome statistic.

The executable law remains strictly pre-DFT: one raw unrelaxed `x0`, frozen analytic element/oxidation-state data, Voronoi geometry, Brown generic bond-valence priors, and deterministic linear programming. DFT outcomes are offline discovery labels only.

## Inputs and isolation

- SCIGEN discovery endpoint and its already frozen NEXT85 analytic features;
- WyFormer discovery endpoint and its already frozen NEXT94 analytic features;
- NEXT105 core/expanded CMVF discovery features;
- exactly the 67 NEXT98b base formulas that pass the frozen AUC gates in both sources;
- NEXT98 term catalogue and unchanged Pauling decisions.

No validation output, replication endpoint, validation/replication geometry, relaxed structure, learned proxy, or same-composition alternative may be accepted by the CLI. Optional terms are calibrated from label-free CMVF values before endpoint tables are loaded or joined.

## Prespecified optional terms

All directions are fixed from the necessary-condition physics: larger incompatibility is riskier. No direction is learned from labels.

| term id | feature | direction | transform | support |
|---|---|---:|---|---|
| `cmvf_core_reallocation__high` | `cmvf_core_reallocation` | +1 | `log1p_nonnegative` | `cmvf_core_supported` |
| `cmvf_core_overload__high` | `cmvf_core_overload` | +1 | `log1p_nonnegative` | `cmvf_core_supported` |
| `cmvf_core_log_scale_mismatch__high` | `cmvf_core_log_scale_mismatch` | +1 | `log1p_nonnegative` | `cmvf_core_supported` |
| `cmvf_expanded_reallocation__high` | `cmvf_expanded_reallocation` | +1 | `log1p_nonnegative` | `cmvf_expanded_supported` |
| `cmvf_expanded_overload__high` | `cmvf_expanded_overload` | +1 | `log1p_nonnegative` | `cmvf_expanded_supported` |
| `cmvf_expanded_log_scale_mismatch__high` | `cmvf_expanded_log_scale_mismatch` | +1 | `log1p_nonnegative` | `cmvf_expanded_supported` |

Domain-width and sign-pattern-count features are provenance/ambiguity diagnostics only. They are not risk terms because broad oxidation-state domains and multiple chemical sign explanations can be legitimate transition-metal chemistry.

Each term is eligible only if, separately in SCIGEN and WyFormer, active coverage is at least 0.15 and the pooled active values contain at least eight unique transformed values. The center is the pooled label-free median; scale is the pooled transformed q90-minus-q10. A nonfinite or nonpositive scale excludes the term.

## Candidate grammar

For each of the 67 frozen bases, evaluate:

- the unchanged base control;
- the base plus exactly one eligible CMVF optional guard;
- optional weight in `{0.25, 0.5, 1.0, 2.0, 4.0}`.

No candidate may contain two CMVF guards, change an old term, refit an old center/scale, or alter the base support mask. If CMVF is inactive, its correction is exactly zero and the base score is preserved bit-for-bit.

## Unchanged gates

The score must pass all of the following without changing any threshold after search:

1. Both-source AUC gates: pooled extreme AUC, macro crystal-system AUC, worst crystal-system AUC, and evaluable-system count.
2. A single SAFE threshold must pass 12 cells: SCIGEN aggregate + five fixed reduced-formula folds and WyFormer aggregate + five fixed folds.
3. A BROAD threshold no greater than SAFE must Pareto-dominate Pauling in all 12 cells using the existing coverage, protected-kept, severe-rejected, severe-precision-lower, and savings-lower comparisons.
4. Formula, centers, scales, weights, and thresholds may not change after results are visible.

The same NEXT103 evaluator and gate constants are authoritative. The search result may rank a near-feasible diagnostic candidate, but a diagnostic is never a frozen law.

## Replication rule

If and only if at least one candidate passes every frozen gate, NEXT107 may freeze the selected formula, thresholds, and row-level scores before any replication endpoint is opened. Otherwise both replication endpoints remain physically unopened and the outcome is recorded in a new standalone report.
