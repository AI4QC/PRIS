# NEXT44 rich analytic feature design

## Motivation

NEXT43's best formula (`cov_q05 + sscp_load_q95`) generalized from discovery to internal validation and materially improved over NEXT23 B+E, but it still failed rejection-precision and savings gates. NEXT44 tests whether the remaining gap is caused by the deliberately compressed NEXT43 family summaries.

## Additive scope

NEXT44 preserves every NEXT42/NEXT43 script, artifact, formula, and report. It adds a second label-free table computed from the same sealed raw-x0 archive. No endpoint is read during feature generation.

## Added descriptor families

- the complete SIVR output, including scale, RMS/max mismatch, site maximum, hydrostatic response, stiffness, and soft/negative-mode fractions;
- the complete normalized Madelung decomposition and site tails;
- the complete scale-calibrated bond-valence mismatch, asymmetry, effective coordination, isolation, and parameter-source diagnostics;
- exact periodic nonbonded contact-pressure and coordination summaries;
- previously implemented molecular-packing controls, retained as fail-open diagnostics rather than assumed valid for all inorganic cells;
- always-defined cell, elemental-size, electronegativity, and periodic covalent-coordination summaries.

All quantities remain deterministic functions of one raw x0 plus frozen tables. No learned potential, energy/force/stress proxy, physical relaxation, or same-composition comparison is introduced.

## Search and gates

The NEXT44 table is joined with the sealed NEXT43 table only after both label-free manifests and hashes validate. The same NEXT43 deterministic discovery/validation split, formula catalogue, missing-value KEEP policy, and primary gates are reused unchanged. NEXT44 is development-only; unseen confirmation remains closed unless one formula passes every gate on both internal splits.
