# NEXT20 Valence-Rigidity Analytic Law Design

## Status and evidence role

NEXT19 valence transport is preserved as a negative development result.  No
candidate passed its WBM gate, so its external endpoint was not opened.
NEXT20 is a separate additive branch.  It does not replace NEXT19 or any
canonical report, paper, script, or result.

WBM and ELEMENTA are historically exposed development sources.  Alexandria
remains unopened for NEXT20 and may be used once only after the analytic law,
all constants, and all gates are frozen.

## Hard scientific boundary

The executable law accepts exactly one raw, unrelaxed structure.  It may use
only its lattice, atomic coordinates, composition, tabulated elemental radii
and electronegativities, analytic neighbor graphs, and analytic linear
algebra.  It must not use DFT values, relaxed structures, forces, stresses,
machine-learned potentials, proxy energies, relaxation, or other candidates
of the same composition.  Missing support must fail open.

DFT-derived outcomes are permitted only as offline development labels and as
the frozen external evaluation endpoint.  They are never law inputs.

## Hypothesis

VTF tests whether formal valence can be transported through the neighbor
graph, but its WBM ranking was nearly nondiscriminating.  The missing signal
may be mechanical: an apparently charge-balanced graph can still contain
mutually inconsistent bond scales, unbalanced local prestress, or unstable
central-force modes.

NEXT20 defines a scale-invariant valence-rigidity (SIVR) descriptor.  For each
opposite-sign periodic edge k=(i,j), let d_k be its periodic displacement,
r_k=|d_k|, R_k=R_i+R_j its tabulated radius sum, w_k its analytic neighbor
weight, and

    x_k = log(r_k / R_k)
    mu  = weighted_median_k(x_k)
    e_k = x_k - mu.

The centering removes one uniform scale degree of freedom without changing or
relaxing the supplied structure.  `mu` is retained as a separate packing-scale
descriptor.

The edge residuals induce an analytic dimensionless prestress model

    phi_k = 1/2 w_k e_k^2.

Its site imbalance, cell stress, and central-force Hessian are computed at the
supplied coordinates.  For u_k=d_k/r_k, the dimensionless edge stiffness is

    K_k = w_k exp(-2 e_k) [ e_k I + (1 - 2 e_k) u_k u_k^T ].

The candidate feature set is fixed before reading any new endpoint:

- absolute scale offset `abs(mu)`;
- weighted RMS, q95, and maximum `|e_k|`;
- RMS and maximum normalized site imbalance;
- hydrostatic and deviatoric cell-prestress magnitudes;
- minimum non-translational Hessian eigenvalue;
- negative-mode and soft-mode fractions;
- graph coverage counts needed for auditable fail-open behavior.

Both unweighted and charge-product-weighted edge variants are permitted in the
finite development catalogue.  No other learned representation is permitted.

## Frozen development loop

1. Unit-test the pure analytic kernel with hand-checkable edge systems.
2. Build identifier-bearing SIVR features from the existing sanitized WBM and
   ELEMENTA geometry archives without opening their labels.
3. Freeze a finite catalogue of one-term and at most three-term monotone sums.
   Centers and scales are estimated on WBM only.
4. Select thresholds on WBM only, then apply the complete formula unchanged to
   ELEMENTA.
5. Require the predeclared absolute gates: coverage lower bound >= 0.90,
   stable/valuable recall lower bound >= 0.95, rejection-precision lower bound
   >= 0.90, and DFT-savings lower bound >= 0.10.  Group-safety is evaluated
   only on datasets with genuine multi-candidate groups; the mostly-singleton
   WBM subset cannot use an all-group-survival gate.
6. If and only if both development sources pass, freeze source hashes,
   formula, constants, threshold, and comparison procedure before touching
   Alexandria.
7. Write a standalone NEXT20 report.  Do not modify the paper or existing
   reports until the user confirms.

## Scientific claim boundary

Passing development gates creates a candidate, not a discovery claim.
Superiority to Pauling requires the frozen Alexandria endpoint plus paired
uncertainty estimates.  Failure is retained and reported as a negative result.
