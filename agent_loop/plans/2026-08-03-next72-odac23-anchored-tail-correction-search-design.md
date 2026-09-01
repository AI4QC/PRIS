# NEXT72 anchored finite tail-correction search

## Purpose

Preserve the converged, finite-catalogue NEXT67 three-term physical axis and
search only small explicit corrections for its reject-tail precision failure.
NEXT68/NEXT71 fitted coefficients are diagnostic only and are not anchors.

## Frozen catalogue

Use the exact sealed NEXT67 terms with its old threshold removed.  Candidate
guards are all 181 NEXT65 features plus 33 label-free NEXT70 metal-donor
bond-valence features, excluding an anchor feature itself.  Retain a guard only
when it is nonconstant and finite on at least 95% of discovery-domain rows;
standardize it by discovery median and IQR.

First enumerate the anchor alone and anchor plus every guard with both signs and
weight 0.125, 0.25, 0.5, 1, 2, 4, or 8.  Rank each guard by its best complete
seven-gate rank.  Freeze the leading 24 guard identities, then enumerate every
two-guard pair with independent signs and weights 0.25, 0.5, 1, 2, or 4.  Each
formula has at most five explicit terms.  Duplicate term lists count once.

For every term list evaluate unchanged 0.02--0.30 rejection-fraction
thresholds, continuous-score pooled/macro/worst extreme AUC, coverage,
protected recall, reject precision, savings, missing=KEEP, domain gate, and
canonical rank/tie break.  No internal-validation, replication,
official-validation, test, or OOD label path is accepted.  A discovery pass is
sealed before one-shot validation.

## Boundary

Execution is deterministic x0 geometry/elemental-table arithmetic followed by
a maximum five-term standardized sum and threshold.  It performs no DFT
calculation, reads no DFT value, relaxed geometry, energy/force/stress model,
proxy potential, physical relaxation, or same-composition alternative.
