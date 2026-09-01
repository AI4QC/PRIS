# NEXT41 source-balanced contact-guard design

## Hypothesis

NEXT23 B+E measures directional and cell-scale inconsistency but can assign high
risk to OMat24 structures that nevertheless change little.  A genuinely unsafe
raw structure should also show an independent absolute short-contact burden.
Require B+E risk and a frozen covalent-contact guard jointly, rather than merely
raising the source-dependent B+E threshold.

## Development sources

- Exposed WBM NEXT23 blind cohort after its completed evaluation.
- Exposed, parent-disjoint OMat24 NEXT40 short-horizon cohort after its completed
  evaluation.

Both are development sources from this point forward.  Candidate acceptance
must satisfy the four NEXT23 gates in each source separately.  Pooled metrics
may be reported but may not rescue a source-specific failure.

## DFT-free features

Compute from each frozen step-0 geometry only:

- covalent-radius pair-ratio q01 and q05;
- contacts below ratio 0.85 per atom;
- squared overlap burden per atom;
- site-overlap q95 and maximum.

The element-radius table and periodic neighbor enumeration are frozen.  No
energy, force, stress, relaxed/later structure, MLIP, coordinate update, or
same-composition candidate is available to the feature builder.

## Search boundary

Search only conjunctions of B+E score and one contact burden.  Low pair-ratio
features are reversed so larger transformed values always mean higher risk.
All thresholds are development-only and any selected formula must be frozen
before a third source is opened.  If no candidate passes both sources, publish
a negative report and do not broaden the catalogue post hoc.

## Confirmation

A development pass is not a result.  Confirmation requires a third, previously
unseen trajectory source with predictions frozen before later geometry opens.
MPtrj, MatPES, or another official trajectory source must first pass provenance,
license, identity, overlap, and endpoint-balance audits.
