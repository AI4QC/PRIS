# NEXT62 robust-endpoint conjunctive law search

## Motivation

NEXT61's additive candidate nearly met the AUC gate but failed reject precision.
A physically implausible framework may require several simultaneous warning
conditions rather than compensation between them.  NEXT62 therefore searches
finite logical conjunctions.  All lockboxes remain unopened.

## Frozen catalogue

Use the same NEXT58 features, NEXT60 robust discovery endpoint, framework domain
gate, four strata, extreme-class AUC, Wilson gates, and ranking as NEXT61.  For
each feature, freeze discovery median/IQR scaling and the risk direction chosen
by mean evaluable-stratum AUC.  Rank features by worst, macro, then pooled AUC.

Directional per-feature cutoffs are their eligible discovery risk quantiles:
0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, and 0.975.

Search every one-term cutoff; every two-term conjunction among the top 20
features at all cutoff pairs; and every three-term conjunction among the top 12
features using cutoffs 0.70, 0.80, 0.90, 0.95, and 0.975.  A row is rejected only
if every term meets its cutoff.  Its continuous risk score is the minimum
normalized excess across terms, so score >= 0 exactly matches the conjunction.
Missing terms, unsupported geometry, or out-of-domain structures force `KEEP`.

Candidates use the unchanged seven advancement gates.  Rank by all-gates pass,
minimum normalized margin, worst/macro stratum AUC, precision/protection/savings
bounds, pooled AUC, fewer terms, then canonical JSON.  A discovery failure
cannot open internal validation.
