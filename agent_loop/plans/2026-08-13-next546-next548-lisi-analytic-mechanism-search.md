# NEXT546--NEXT548: bounded Li--Si analytic mechanism search

Date frozen: 2026-08-13

## Status and scope

NEXT545 is an opened development set after its preregistered MUPR confirmation
failed.  This branch may use its offline `dft_waste` endpoint to discover a new
hypothesis, but it cannot produce a scientific success claim.  Any discovered
formula must later be frozen and tested on a completely new source.

Every candidate input remains composition plus one raw, unrelaxed, fully
periodic x0.  No DFT value, relaxed/later geometry, trajectory, ML prediction,
potential, or coordinate relaxation may enter an executable formula.  All old
scripts/artifacts remain intact.

## Frozen development split

- discovery: `Li1Si1_02`, `Li2Si1_02` (100 rows)
- internal validation: `Li7Si2_03`, `Li15Si4_02` (100 rows)

The split is source-prefix disjoint and fixed before computing the expanded
feature bank.  The already opened NEXT545 endpoint definition is unchanged.

## Label-free feature bank

NEXT546 applies the frozen NEXT43 analytic x0 bank (74 descriptors), then adds
only the already frozen SSSP and PBAAA values from NEXT544 plus the following
coefficient-free primitive geometry quantities:

- volume per atom;
- covalent-sphere packing fraction
  `sum(4*pi*r_cov^3/3) / cell_volume`;
- cell metric anisotropy `max(singular_value(cell))/min(...)`;
- nearest-contact ratio mean, standard deviation, q10, q50, and q90 using the
  frozen covalent radii and periodic contacts.

These additions are dimensionless except volume per atom and contain no learned
or endpoint-dependent parameters.

## Bounded univariate screen

For every finite feature, both risk directions are tested by using its midrank
percentile and its complement.  A direction is searchable only with at least
80% support and at least 20 distinct values in both partitions.

For each direction report discovery, validation, combined, failure-only, and
high-final-energy-only AUC.  A feature enters pair search only when discovery
and validation AUC are both at least 0.58.  Candidates are ordered by
`min(discovery_auc, validation_auc)`, then combined AUC, then feature name.
After removing candidates with absolute Spearman correlation above 0.95 to an
earlier candidate, retain at most 16.

## Bounded coefficient-free formula search

For every retained pair `(u, v)`, test exactly four symmetric formulas:

```
mean        = (u + v) / 2
maximum     = max(u, v)
union       = 1 - (1-u)(1-v)
concurrence = min(u, v)
```

No coefficients, thresholds, powers, transforms, or third terms are fitted.
Every formula is evaluated without missing-value imputation on rows supporting
both terms.  It is internally eligible only if:

- support is at least 0.80 in both partitions;
- discovery and validation AUC are each at least 0.65;
- combined AUC is at least 0.67;
- top-15% precision is at least 0.50 in both partitions;
- it exceeds each of its two component AUCs by at least 0.02 in both
  partitions.

The winner maximizes minimum partition AUC, then minimum component margin, then
combined AUC, with a lexical tie break.  If no pair is eligible, no formula is
frozen and the branch stops.  The search grid may not be expanded after results.

## NEXT547 formula freeze

If an eligible pair exists, NEXT547 publishes its exact raw feature definitions,
risk directions, percentile normalization, symmetric formula, operating top-15%
rule, development hashes, and predictions for all 200 rows.  This is a
development candidate only.

## NEXT548 prospective source requirement

Before viewing any new endpoint, NEXT548 must choose a chemically broader public
x0-to-DFT source not used by NEXT31--NEXT547, freeze the source/sample IDs, and
score every x0.  The new-source success gates must be specified in a separate
design before endpoint access.  Li--Si performance alone can never authorize a
report or canonical edit.
