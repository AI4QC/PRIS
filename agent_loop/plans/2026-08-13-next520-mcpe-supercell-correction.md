# NEXT520 MCPE label-blind supercell correction certificate

This additive certificate records one implementation correction to the frozen
NEXT520 design.  The design, formula, feature direction, thresholds, atomic
table, source cohorts, and all outcome-isolation rules remain unchanged.

## Preserved first execution

The first engineering-probe result is preserved byte-for-byte at
`experiments/next520_mcpe_label_blind_engineering_probe_result.full.json`,
SHA-256
`eee6b4e2c4a39bf8c5908f01f55d62f54adfa7c63209d9492b846926a8ec1970`.
It passed support, closed-domain, and nondegeneracy gates, but reported an
invariance failure.  No prior feature table, discovery outcome, endpoint,
validation geometry, or replication geometry was opened during or after that
execution.

## Root cause established without labels

The inherited equivalence generator contains exactly five transformations:
rigid rotation, wrapped translation, site permutation, unimodular rebasing,
and exact `(2, 1, 1)` supercell replication.  Per-transformation diagnostics
on the first supported geometry from each frozen 80-row cohort gave exactly
zero error for the first four transformations.  Only exact replication
changed MCPE: `0.0037994576` for SCIGEN and `0.0216342009` for WyFormer.

The failure occurs before the MCPE kernel.  NEXT19's last-resort
`electronegativity_partition` normalizes the total positive and negative
charges to `+1` and `-1` over the entire supplied cell.  Exact replication
therefore divides every fallback site charge by the formula-unit
multiplicity; Ewald site potentials scale by the same factor.  For example,
the two-site fallback assignment `(-1,+1)` becomes
`(-0.5,-0.5,+0.5,+0.5)` after a twofold repeat.  Those are cell-normalized
weights, not intensive per-site formal charges, so feeding them directly to
the frozen chemical-potential formula violates the design's explicit exact
supercell-invariance requirement.

## Frozen-scope correction

Only when NEXT19 reports policy `electronegativity_partition`, multiply its
cell-normalized weights by the greatest common divisor of the elemental site
counts (the number of reduced-formula units in the supplied cell).  Integer
and fractional formal-valence assignments are left unchanged.  This uses only
composition, introduces no fitted parameter, and makes the same raw periodic
crystal representation-independent before Ewald evaluation.

The original result remains the immutable audit trail.  A corrected execution
must use a new output file and must cite both this certificate and the original
result hash.  It may authorize the novelty probe only if all originally frozen
engineering gates pass.  The correction is invalid if any outcome or later
geometry was opened before the corrected execution.
