# NEXT57 ODAC23 discovery-only finite law search

## Firewall and endpoint

The search process accepts the sealed NEXT55 x0-only feature artifact and only
the NEXT56 `discovery/offline_labels.parquet` file.  Paths for internal
validation, internal replication, official validation, test, and OOD payloads
are not arguments and must not be opened.

The endpoint remains median framework p95 initial-to-PBE+D3-relaxed
displacement for the selected exact x0.  Protected is <= 0.05 angstrom and
severe is >= 0.20 angstrom.  Intermediate rows count against reject precision
but are excluded from AUC, so the ranking measures protected-versus-severe
separation without silently calling intermediate motion stable.

## Domain and feature catalogue

The rule is eligible only when all selected terms are finite, NEXT55 support is
true, periodic translation rank is at least one, and periodic framework
fraction is at least 0.5.  All other inputs force `KEEP`.

Candidate terms are exactly the 56 frozen NEXT55 features.  On discovery only,
each nonconstant feature is centered by its median and scaled by its IQR.  Its
sign is chosen so the mean evaluable stratum AUC points toward severe risk.
Strata are the four predeclared combinations of official `defective` and
`open_metal_site` flags; strata lacking either endpoint class are unevaluable,
not imputed.

Rank features by worst, then macro, then pooled directional AUC.  Search:

- every one-term formula;
- every two-term combination among the top 20 features, with second-term
  weights 0.5, 1, or 2;
- every equal-weight three-term combination among the top 12 features.

Thresholds are the inverted-CDF score quantiles for reject fractions 0.02
through 0.30 in steps of 0.01.  This finite catalogue, all tie-breaking, and
all gates are fixed before discovery labels are opened.

## Advancement and ranking

Use the frozen NEXT53 one-sided 95% Wilson gates: coverage >= 0.95, protected
recall >= 0.95, severe-rejection precision >= 0.70, savings >= 0.02, pooled
extreme-class AUC >= 0.75, macro stratum AUC >= 0.65, and worst evaluable
stratum AUC >= 0.55.

Candidates rank lexicographically by: all-gates pass; minimum normalized gate
margin; worst and macro stratum AUC; precision, protected-recall, and savings
lower bounds; pooled AUC; fewer terms; canonical formula JSON.  If no formula
passes, the highest-ranked failure is sealed for diagnosis only and cannot open
an internal lockbox.
