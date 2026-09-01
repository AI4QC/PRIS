# NEXT141 Analytic-Field BROAD Residual Diagnostic Freeze

Protocol: `2026-08-08-next141-analytic-field-broad-residual-diagnostic-v1`

Diagnose all 33 formally published NEXT140 candidates with
`passes_safe_all_cells == true`. Reconstruct their exact scores and published
SAFE12 thresholds, enumerate the unchanged NEXT128 BROAD threshold residual,
and report failed components and normalized shortfall by analytic-field weight.

No formula, feature, weight, threshold, comparator, or gate may change. If no
positive-weight candidate reduces either the fixed six SCIGEN
`protected_kept` failures or the weight-0 shortfall `0.792624940220628`, the
low analytic-field protection branch is terminated.

Discovery endpoints remain offline labels only. Validation/replication remain
unopened. No DFT calculation/value, learned energy/force/stress proxy, or
physical relaxation is used. The analytic point-charge Ewald derivative is
recorded separately and is not DFT.
