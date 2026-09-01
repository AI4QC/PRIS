# NEXT159 mechanism-family BROAD residual diagnostic freeze

## Scope and boundary

NEXT158 produced exactly three candidates that pass both frozen cross-source
AUC gates and all 12 SAFE cells, while no candidate passes every BROAD cell.
NEXT159 is an additive diagnostic of those three already-published candidates.
It searches no new formula, changes no score, weight, threshold policy, gate,
or candidate ordering, and keeps validation and replication endpoints sealed.

The executable scores remain analytic pre-DFT laws. No DFT calculation or DFT
value is used by formula execution, no learned energy/force/stress proxy is
used, and no relaxation is performed. Discovery outcomes are used only as
offline diagnostic labels.

## Frozen population and procedure

- Input: the complete published NEXT158 candidate table.
- Include iff `passes_source_auc_gates == true` and
  `passes_safe_all_cells == true`.
- Expected count: 3.
- Candidate-key SHA-256 over newline-joined sorted canonical keys:
  `298e04881ce0b1135e0d909e4578e5ea20d771b96785c9be8a2332b4135dfcd3`.
- Reconstruct all 176 NEXT158 scores and verify the three selected candidates
  reproduce their published six AUCs, SAFE threshold, and four pass flags to
  absolute tolerance `1e-12`.
- For each selected candidate, hold its published SAFE threshold fixed as the
  lower bound and evaluate every already-generated threshold table with the
  unchanged NEXT128 BROAD residual diagnostic.
- Rank closest candidates lexicographically by failed-constraint count,
  normalized shortfall sum, best threshold, then canonical candidate key.
- Record every failing cell/component and its frequency at per-candidate
  optima.

No result-dependent follow-up family, feature, or grid is authorized by this
diagnostic. Any later branch must be independently motivated and frozen before
execution.

## Outputs

Atomically publish a manifest, JSON diagnostic, and per-candidate parquet under
`$PRIS_ARCHIVE/next159_mechanism_family_broad_residual_diagnostic_v1`.
Only the standalone research report may be updated after the run.
