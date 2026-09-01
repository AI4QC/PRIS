# NEXT21 Normalized Madelung Consistency Design

NEXT21 is additive and preserves the negative NEXT19/NEXT20 artifacts.
It evaluates one raw structure independently and uses only an analytic Ewald
sum with the NEXT19 tabulated-composition charge assignment.  It does not read
DFT values, relaxed structures, forces, stresses, learned potentials, or any
same-composition candidate.

For a neutral charge assignment q_i, cell length L=V^(1/3), Coulomb constant
k_e, and Q2=sum_i q_i^2, define the dimensionless normalization

    C = L / (k_e Q2).

The total, real-space, reciprocal-space, point, and site-resolved Ewald terms
are multiplied by C.  The resulting values are invariant to uniform cell
scaling and to a global rescaling of all q_i.  The frozen descriptor set is:

- reduced total, real, reciprocal, and point terms;
- reduced site-term spread, extrema, and positive-site fraction;
- a dimensionless charge-concentration diagnostic.

All computations use the supplied x0 coordinates without modification.  The
same WBM-only finite monotone catalogue and gates used by NEXT20 will be used;
ELEMENTA is evaluated unchanged only if a WBM candidate passes.  Alexandria
remains unopened until a complete candidate is frozen.
