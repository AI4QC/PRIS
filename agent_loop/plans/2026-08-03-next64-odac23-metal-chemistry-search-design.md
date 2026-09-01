# NEXT64 metal-chemistry augmented robust discovery search

NEXT64 reuses the exact NEXT61 additive finite-search engine and NEXT60 robust
discovery endpoint.  The only change is the candidate catalogue: all 121
NEXT58 features plus the 48 frozen NEXT63 metal-chemistry features, for 169
explicit x0-only terms.  Domain gates, strata, median/IQR scaling, direction
selection, top-20 pairs, top-12 triples, weights, rejection-fraction thresholds,
seven advancement gates, rank order, and canonical tie-break are unchanged.

No validation, replication, official validation, test, or OOD label path is an
input.  Failure remains diagnostic.  A pass seals exactly one formula before
the internal-validation endpoint is opened once without refitting.
