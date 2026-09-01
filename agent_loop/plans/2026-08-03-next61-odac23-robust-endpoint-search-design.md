# NEXT61 robust-scaffold discovery search

NEXT61 is frozen before opening the NEXT60 discovery label values.  It uses the
already sealed NEXT58 121-feature x0 catalogue and reuses the exact NEXT59
search engine without modification: domain gate, extreme-class treatment,
four defective/OMS evaluation strata, median/IQR scaling, direction rule,
top-20 pair and top-12 triple shortlists, weight catalogue, 0.02--0.30 threshold
grid, Wilson gates, ranking, and canonical tie-break are unchanged.

The only endpoint substitution is the NEXT60 robust, common-translation-aligned
median framework p95 response across at least four adsorbate configurations per
scaffold condition.  Protected remains <= 0.05 angstrom and severe remains >=
0.20 angstrom.

Only `next60/.../discovery/robust_offline_labels.parquet` is an input.  Internal
validation, internal replication, official validation, test, and OOD paths are
not accepted.  A failed discovery formula remains diagnostic.  A passing
formula is sealed, hashed, and then evaluated exactly once on internal
validation without refitting.
