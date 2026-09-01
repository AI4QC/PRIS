# NEXT164 interior-attenuation BROAD residual diagnostic freeze

## Scope and boundary

NEXT163 produced exactly 12 candidates that pass both frozen source-AUC gates
and SAFE12, with three candidates at each interior attenuation
`gamma in {0.01, 0.025, 0.05, 0.075}`. None passes every BROAD cell. NEXT164
diagnoses only these 12 already-published candidates; it searches no formula,
changes no score, grid, threshold policy, gate, or ordering, and opens no
validation or replication endpoint.

All executable scores remain analytic pre-DFT laws. No DFT calculation or DFT
value is used by formula execution, no learned energy/force/stress proxy is
used, and no relaxation is performed. Discovery outcomes are offline labels
only. No prior or canonical artifact is replaced.

## Frozen population and procedure

- Include a NEXT163 record iff `passes_source_auc_gates == true` and
  `passes_safe_all_cells == true`.
- Expected count: 12, exactly three at each frozen interior gamma.
- Candidate-key SHA-256 over newline-joined sorted canonical keys:
  `76bb23d466efd3b7eeef634ae49c77651b802ba631514e40d8262acefa77d0bc`.
- Reconstruct all 704 NEXT163 candidates and reproduce the selected 12 records'
  six AUCs, SAFE threshold, and four pass flags to absolute tolerance `1e-12`.
- For each selected candidate, keep its published SAFE threshold as the lower
  bound and run the unchanged NEXT128 exact BROAD residual diagnostic.
- Rank candidates by failed-constraint count, normalized shortfall sum, best
  threshold, then canonical candidate key.
- Summarize the minimum failed count and shortfall separately for each gamma,
  and record every failing cell/component frequency at per-candidate optima.
- Compare each gamma only against the already-published `gamma=0` NEXT159
  residual `(6 failures, normalized shortfall 0.868227030677262)`.

No result-dependent additional gamma or formula is authorized by this
diagnostic. Any later branch must be independently motivated and frozen before
execution.

## Outputs

Atomically publish a manifest, JSON diagnostic, and per-candidate parquet under
`$PRIS_ARCHIVE/next164_interior_attenuation_broad_residual_v1`.
Only the standalone research report may be updated after the run.
