# NEXT555--NEXT556 HEA extreme-waste redesign

## Provenance and status

NEXT553 stopped before search because its development endpoint was degenerate:
1,107 positive versus 93 negative rows.  Validation endpoint fields and final
structures remain unopened.  Development-only diagnosis showed that the
standard `0.10 eV/atom` hull cutoff marked 1,049/1,200 rows, whereas the frozen
geometric-response clause marked 450/1,200 rows.

This document defines a new, explicitly development-calibrated hypothesis.  It
does not alter or rescue NEXT553.  Existing artifacts remain immutable.

## Revised frozen endpoint

Use the same four DFT-only offline endpoint quantities, but distinguish extreme
thermodynamic waste from ordinary HEA metastability:

- extreme energetic instability: `e_above_hull >= 0.40 eV/atom`;
- large geometric response: displacement p90 at least 0.25 A, cell log strain
  at least 0.08, or volume log change at least 0.10;
- `extreme_dft_waste`: either clause is true;
- continuous severity: maximum of energy/0.40, displacement/0.25,
  strain/0.08, and volume-change/0.10;
- protected: energy at most 0.10, displacement p90 at most 0.10 A, strain at
  most 0.04, and volume change at most 0.05.

These choices were made after seeing development endpoint distributions and
must be described as such.  They are now absolute and cannot be recalibrated
on validation.

## NEXT555 development search

Reuse without alteration the NEXT552 label-free features, full-cohort midrank
percentiles, searchable-direction mask, coefficient-free pair catalogue,
redundancy rule, and deterministic winner ordering.  Reuse the NEXT553 search
gates, with an additional requirement of at least 50 protected development
rows.  No new features, risk directions, weights, or formulas are allowed.

If no pair is eligible, validation remains sealed.  If a pair is eligible,
publish its exact formula and checksums before NEXT556.

## NEXT556 sealed validation

Only an eligible NEXT555 winner authorizes materialization of endpoint fields
for the 1,200 validation FIDs.  Apply the revised absolute endpoint and the
frozen formula without refitting.  Require at least 100 rows in each class and
50 protected rows, at least 95% score coverage, AUC at least 0.70 with
chemical-system cluster-bootstrap lower bound at least 0.65, AUC at least 0.62
in both ordered and SQS structures, Spearman severity at least 0.30 with lower
bound at least 0.20, top-15% lift at least 1.75, protected fraction at most 2%
in the top 15%, and AUC at least 0.02 above both frozen components.

Passing supports an HEA-domain, development-calibrated but chemically disjoint
confirmation.  It is not universal stability or a replacement for DFT.

