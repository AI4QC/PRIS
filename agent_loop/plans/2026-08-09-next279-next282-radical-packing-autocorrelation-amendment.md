# Pre-outcome amendment to NEXT279--NEXT282

**Status:** frozen during engineering review, before any formal NEXT279 feature
or discovery outcome was computed or opened.

The parent plan remains unchanged. This append-only amendment resolves two
internal wording contradictions without changing the eight features,
directions, quantiles, gates, search grid, diagnostic rank, or stopping rules.

1. "Reproduce NEXT267 support" means reproduce the exact power-cell geometry
   support and tiling result before applying graph-specific requirements.
   NEXT279 additionally abstains when the structure has fewer than two sites,
   has no directed active-facet contact incidence, has any empty labelled cell,
   or fails contact reciprocity. Its published feature support may therefore be
   a strict subset of NEXT267 support.
2. Uniform coordinate/cell scaling is removed from the required invariance
   tests. Neutral tabulated atomic radii are fixed physical lengths, so changing
   lattice lengths while keeping those radii fixed is a physical dilation, not
   a representation change. Rigid rotation, periodic translation, site
   permutation, unimodular cell rebasing, and integral supercell replication
   remain mandatory invariances.

No discovery label, endpoint, validation output, replication output, DFT
quantity, proxy-potential result, or relaxed structure informed this amendment.
