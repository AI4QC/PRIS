# NEXT68 sparse stable explicit x0 law

## Scope

The finite small-integer catalogue cannot meet the robust-endpoint precision
gate despite passing AUC gates.  NEXT68 permits numerical discovery of sparse
coefficients while preserving an explicit executable formula.  This is not an
energy/force/stress model: inputs remain the 181 deterministic NEXT65 x0
features and execution is a fixed weighted sum plus threshold.

## Frozen fitting procedure

Use robust discovery only.  Restrict fitting labels to protected (<=0.05 A) and
severe (>=0.20 A) rows; intermediate rows still count against threshold reject
precision.  Center every nonconstant candidate by the eligible discovery median
and scale by IQR.  Fit class-balanced L1 logistic regressions with deterministic
`liblinear`, intercept enabled, maximum 5000 iterations, tolerance `1e-8`, and
`C` in 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, or 1.0.

Assign five stability folds by
`SHA256("NEXT68-CV-v1\0" + material_id) mod 5`.  For each C, fit the full extreme
set and five leave-one-fold-out models.  A feature is stable only when its full
coefficient is nonzero and at least four of five fold coefficients are nonzero
with the same sign.  From stable features, retain the largest absolute full
coefficients for k=3, 5, or 8 (or all if fewer).  The intercept is absorbed into
the threshold; exact coefficient magnitudes become positive term weights and
coefficient signs become term directions.

Evaluate thresholds at the unchanged 0.02--0.30 rejection-fraction grid.  Apply
the same domain, missing=KEEP behavior, strata, seven gates, rank order, and
canonical tie-break as NEXT67.  Duplicate formulas count once.

## Firewall

No internal-validation, replication, official-validation, test, or OOD path is
accepted.  A discovery failure cannot advance.  A passing sparse formula is
sealed with exact centers, scales, signs, weights, and threshold before one-shot
internal validation.
