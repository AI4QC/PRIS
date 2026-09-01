# NEXT136 Conjunctive BROAD Residual Diagnostic Freeze

Protocol: `2026-08-08-next136-conjunctive-broad-residual-diagnostic-v1`

## Purpose

Diagnose, without proposing or tuning a new formula, why the formally
published NEXT135 candidates that pass all 12 SAFE operating cells do not
strictly Pareto-dominate Pauling in all 12 BROAD cells.

## Frozen candidate set

- Source artifact: the formal NEXT135 candidate-search parquet.
- Selection: `passes_safe_all_cells == true`.
- Expected count: 119 candidates.
- The published NEXT135 safe threshold of each candidate is retained.
- No candidate, weight, feature, threshold, comparator, or gate is added or
  altered by this diagnostic.

## Frozen diagnostic

For each selected candidate:

1. reconstruct its exact discovery score from the frozen NEXT135 inputs;
2. reconstruct the 12 discovery cells and the Pauling baseline;
3. enumerate the already-defined threshold table;
4. call the frozen NEXT128 BROAD residual diagnostic using the candidate's
   published SAFE12 threshold as the lower threshold bound;
5. record the minimum failed-constraint count, normalized shortfall, closest
   threshold, and exact failed cell/components.

Summaries are grouped by conjunctive term count and conjunctive configuration.
The diagnostic may guide a later, separately frozen search grammar, but its
outputs cannot retrospectively change NEXT135.

## Boundaries

- Discovery endpoints are offline labels only.
- Internal validation and replication endpoints remain unopened.
- No DFT calculation or DFT value enters the executable formula.
- No learned energy/force/stress proxy is used.
- No physical relaxation is executed.
- Existing scripts, reports, and canonical manuscript files remain unchanged;
  NEXT136 is additive.
