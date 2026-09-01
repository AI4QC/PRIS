# NEXT145 Coulomb--Steric Balance BROAD Residual Diagnostic Freeze

Protocol: `2026-08-08-next145-coulomb-steric-balance-broad-residual-diagnostic-v1`

Diagnose every formally published NEXT144 candidate with
`passes_safe_all_cells == true` (expected: 22 candidates at weights 0 and
0.1). Reconstruct the exact score and published SAFE12 threshold for each
candidate, enumerate the unchanged NEXT128 BROAD threshold residual, and
report failed components and normalized shortfall by balance-protection
weight.

No formula, feature, weight, threshold, comparator, or gate may change. This
is a diagnostic only: it searches no new formula. After publication, the
standalone Coulomb--steric balance protection branch is terminated whether or
not weight 0.1 reduces the residual; any later use must be a separately frozen
new physical hypothesis rather than an extension of this weight search.

Discovery endpoints remain offline labels only. Validation and replication
remain unopened. No DFT calculation or DFT value, learned energy/force/stress
proxy, or physical relaxation is used. The analytic point-charge Coulomb and
steric vectors are recorded separately and are not DFT.
