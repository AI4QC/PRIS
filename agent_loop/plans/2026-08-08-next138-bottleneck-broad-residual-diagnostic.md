# NEXT138 Bottleneck BROAD Residual Diagnostic Freeze

Protocol: `2026-08-08-next138-bottleneck-broad-residual-diagnostic-v1`

## Purpose

Diagnose, without searching or tuning another formula, the exact BROAD
constraint residuals of all formally published NEXT137 candidates that pass
all 12 SAFE operating cells.

## Frozen candidate set and procedure

- Source: formal NEXT137 candidate-search parquet.
- Selection: `passes_safe_all_cells == true`.
- Expected count: 66 candidates.
- Preserve each candidate's published SAFE12 threshold.
- Reconstruct the exact NEXT137 score, the same 12 discovery cells, and the
  same Pauling baselines.
- Apply the frozen NEXT128 threshold-residual diagnostic at or above the
  candidate's SAFE12 threshold.
- Record exact failed cells/components, minimum failed-constraint count, and
  normalized shortfall; summarize by bottleneck term count/configuration.

This diagnostic cannot change NEXT137 and cannot authorize validation. If no
candidate reduces the fixed six SCIGEN `protected_kept` failures, the
coordination-by-compactness branch is terminated.

## Boundaries

- Discovery endpoints are offline labels only.
- Validation and replication endpoints remain unopened.
- No DFT calculation/value, learned energy/force/stress proxy, or physical
  relaxation is used.
- All work is additive; canonical manuscript/report files remain unchanged.
