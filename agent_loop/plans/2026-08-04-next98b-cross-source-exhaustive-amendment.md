# NEXT98b exhaustive cross-source catalogue amendment

Date: 2026-08-04

NEXT98's deterministic top-slice search found cross-source discriminative
scores but no SAFE threshold satisfying all two source aggregates and ten
source-by-reduced-formula folds. Before closing the sparse-sum family, NEXT98b
performs one finite exhaustive union over every unique formula and weight tuple
already present in the complete NEXT87 and NEXT95 discovery catalogues, plus
all eligible single terms.

All boundaries and gates in the NEXT98--NEXT100 design remain unchanged.
NEXT98b reads only the two discovery feature files, the two discovery endpoint
files, the pooled label-free term catalogue frozen by NEXT98, and the two prior
complete candidate catalogues. It does not read NEXT92 or NEXT97 validation
outputs and does not read either replication endpoint. Candidate ranking,
SAFE/BROAD threshold selection, per-source AUC gates, and all 12 cell gates are
identical to NEXT98. If no candidate passes, the entire at-most-three-term
nonnegative sparse-sum family is closed for these frozen term transforms and
both replication endpoints remain unopened.
