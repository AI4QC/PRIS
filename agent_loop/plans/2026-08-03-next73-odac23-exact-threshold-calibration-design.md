# NEXT73 exact threshold calibration of the frozen NEXT72 term list

## Scope

NEXT72 passed six of seven discovery gates; its reject-precision lower bound
was 0.69649 versus 0.70 while the observed precision was 105/138=0.76087.  The
only remaining degree of freedom is the scalar threshold.  Freeze the exact
five NEXT72 terms, centers, scales, signs, and weights.  Do not add, remove, or
refit a feature.

## Frozen calibration

Recompute the fixed continuous score on robust discovery with the unchanged
domain and missing=KEEP policy.  Enumerate every unique supported score whose
resulting rejection fraction is between 0.02 and 0.30 inclusive.  At each exact
score boundary evaluate the unchanged coverage, protected recall, reject
precision, savings, pooled/macro/worst extreme AUC, Wilson lower bounds, gate
rank, and canonical threshold tie break.  The AUC diagnostics are invariant
because the term list is fixed.

No internal-validation, replication, official-validation, test, or OOD label
path is accepted.  If no exact threshold passes, the formula fails discovery.
If one passes, seal its exact threshold and artifact hash before any one-shot
internal validation.

## Boundary

Execution remains a five-term deterministic raw-x0 analytic sum and one scalar
comparison.  It uses no DFT calculation/value, relaxed geometry,
energy/force/stress model, proxy potential, physical relaxation, or
same-composition alternative.
