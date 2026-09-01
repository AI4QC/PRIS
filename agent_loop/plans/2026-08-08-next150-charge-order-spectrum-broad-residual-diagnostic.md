# NEXT150 Charge-Order Spectrum BROAD Residual Diagnostic Freeze

Protocol: `2026-08-08-next150-charge-order-spectrum-broad-residual-diagnostic-v1`

Diagnose every formally published NEXT149 candidate with
`passes_safe_all_cells == true` (expected: 33 candidates at weights 0, 1, and
2). Reconstruct each exact score and published SAFE12 threshold, enumerate the
unchanged NEXT128 BROAD residual, and report failed components and normalized
shortfall by charge-order spectrum weight. The report must separately record
that all positive-weight SAFE candidates failed the frozen source-AUC gates.

No formula, feature, weight, threshold, comparator, or gate may change. This
diagnostic searches no new formula. After publication, the additive
charge-order spectrum protection branch is terminated.

Discovery endpoints remain offline labels only. Validation and replication
remain unopened. No DFT calculation or DFT value, learned energy/force/stress
proxy, or physical relaxation is used. The analytic formal-charge reciprocal
spectrum is recorded separately and is not DFT.
