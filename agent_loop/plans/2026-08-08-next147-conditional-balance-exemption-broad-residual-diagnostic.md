# NEXT147 Conditional Balance Exemption BROAD Residual Diagnostic Freeze

Protocol: `2026-08-08-next147-conditional-balance-exemption-broad-residual-diagnostic-v1`

Diagnose every formally published NEXT146 candidate with
`passes_safe_all_cells == true` (expected: 55 candidates: 11 baselines and 44
active candidates at exemption weight 0.1). Reconstruct the exact score and
published SAFE12 threshold, enumerate the unchanged NEXT128 BROAD threshold
residual, and report failed components and normalized shortfall by raw
Coulomb--steric residual cutoff.

No formula, feature, cutoff, weight, threshold, comparator, or gate may
change. This diagnostic searches no new formula. After publication, the
conditional balance exemption branch is terminated whether or not one cutoff
reduces the weight-zero residual.

Discovery endpoints remain offline labels only. Validation and replication
remain unopened. No DFT calculation or DFT value, learned energy/force/stress
proxy, or physical relaxation is used. Analytic formal-charge Coulomb and
steric vectors are recorded separately and are not DFT.
