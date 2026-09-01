# NEXT71 precision-targeted stable explicit x0 law

## Motivation

NEXT68 reached pooled/macro/worst AUC 0.827/0.811/0.776 but rejected many
intermediate-displacement structures, giving reject-precision lower bound
0.568.  Its extreme-only loss never penalized those false rejects.  NEXT71
directly fits severe (`>=0.20 A`) versus every non-severe discovery row while
preserving extra protection for the `<=0.05 A` class.

## Frozen fitting catalogue

Use the 181 NEXT65 features plus the 33 label-free NEXT70 metal-donor
bond-valence features.  Retain nonconstant features with at least 95% finite
domain coverage and fit on rows complete across this fixed catalogue.  Center
by discovery median and scale by IQR.  Fit deterministic L2 logistic regression
with `liblinear`, balanced severe/non-severe class weights, intercept, maximum
5000 iterations, tolerance `1e-8`, C in 0.001, 0.01, 0.1, 1, or 10, and an
additional protected-row sample multiplier in 1, 2, 4, or 8.

Assign five folds by `SHA256("NEXT71-CV-v1\0" + material_id) mod 5`.  For each
hyperparameter pair fit one full and five leave-one-fold-out models.  Discard
the pair if any fit reaches 5000 iterations.  A coefficient is stable when at
least four fold fits have the same sign as the full fit.  Form explicit
candidate laws from the 3, 5, or 8 largest-absolute stable full coefficients;
the intercept is absorbed into the searched threshold.  Exact coefficient
magnitudes are weights and signs are directions.

Evaluate the unchanged 0.02--0.30 rejection-fraction thresholds with
missing=KEEP, the unchanged x0 domain gate, four source strata, seven frozen
gates, gate rank, and canonical tie break.  Duplicate formulas count once.  No
internal-validation, replication, official-validation, test, or OOD label path
is accepted.  A passing formula must be sealed before any one-shot validation.

## Boundary

The offline fit discovers coefficients only.  Execution is a fixed analytic
x0 weighted sum plus threshold.  It consumes no DFT calculation/value, relaxed
geometry, energy/force/stress model, proxy potential, physical relaxation, or
same-composition alternative.
