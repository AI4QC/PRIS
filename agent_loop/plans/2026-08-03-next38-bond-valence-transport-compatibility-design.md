# NEXT38 Bond-Valence Transport Compatibility Design

## Prior-work audit and novelty

NEXT38 is additive and preserves every prior source, artifact, report, paper,
and unopened confirmation source. NEXT19 already asks whether a periodic
opposite-sign graph can carry globally conserved formal valence while staying
close to a geometric edge prior. NEXT22 already computes scale-calibrated
bond-valence sums, site mismatch, effective coordination, and bond-valence
vector asymmetry. NEXT37 projects radius mismatch through an affine periodic
rigidity matrix.

None of these computes the differential compatibility of the *global
bond-valence correction itself*. NEXT38 therefore combines three fixed ideas
without fitting an interatomic potential:

1. Brown/IUCr bond valence supplies a positive geometry-dependent edge prior;
2. global site-valence conservation supplies a unique minimum-norm edge
   correction;
3. the analytic Jacobian of the normalized bond-valence prior determines
   which part of that correction can be produced by an infinitesimal,
   geometrically coherent coordinate/cell change.

Bond-valence sums are widely used for structure validation
([Brown, 2009](https://doi.org/10.1021/cr900053k)), while the bond-strain index
compares observed and a-priori bond valences edge by edge
([Gagné and Hawthorne, 2017](https://journals.iucr.org/b/issues/2017/06/00/bm5096/)).
The projection step is motivated by affine periodic rigidity and is distinct
from treating either mismatch as an empirical energy.

## Hard execution boundary

Execution accepts one raw unrelaxed `x0`, elements/cell/coordinates, frozen
composition valences, frozen Brown/IUCr parameters with the already disclosed
generic fallbacks, a deterministic opposite-sign Voronoi graph, and
deterministic least-squares/SVD. It may copy four sealed geometry-only
comparators. It does not read DFT values, run electronic structure, use a
learned/fitted potential, read relaxed structures or trajectories, compare
same-composition candidates, or change coordinates/cell.

The least-squares systems are compatibility decompositions only. No coordinate
step is solved, retained, or applied; no iterative physical relaxation is run;
no energy, force, or stress proxy is emitted. DFT response is opened only as
an offline development endpoint after all code, constants, directions,
formulae, and hashes are sealed.

## Frozen bond-valence prior

Reuse the exact NEXT19 unweighted opposite-sign Voronoi multigraph. For edge
`e=(c,a,image)` from cation `c` to anion `a`, frozen bond-valence parameters
give

    b_e = exp((R0_e - d_e) / B_e).

Within each cation star, normalize the positive strengths to its formal supply
`q_c=|z_c|`:

    p_e = q_c b_e / sum_{k incident from c} b_k.

Thus every cation sum is satisfied exactly and absolute scale is removed; the
separate frozen steric comparator retains absolute short-contact information.

Let `C` be the unsigned site-edge incidence matrix, with one at both endpoints,
and let `a_i=|z_i|`. The site conservation deficit is

    y = a - C p.

The unique minimum-Euclidean-norm edge correction is

    t = C^T (C C^T)^+ y,

provided `C t = y` within the frozen numerical tolerance. This is the linear
global-valence analogue of transport, not a nonnegative final bond assignment;
negative values of `p+t` are recorded only as a diagnostic. The use of the
Moore-Penrose solution removes the non-uniqueness that an L1 transport optimum
could introduce under exact supercell replication.

## Frozen normalized-prior Jacobian

For atomic displacements plus six symmetric affine-strain columns, define the
edge-distance differential row

    D_e = [-n_e at c, +n_e at a,
           d_e nx^2, d_e ny^2, d_e nz^2,
           2 d_e ny nz, 2 d_e nx nz, 2 d_e nx ny].

The log-strength differential is `L_e=-D_e/B_e`. Because `p_e` is normalized
within each cation star, its exact analytic Jacobian row is

    J_e = p_e [L_e - sum_{k from c} (p_k/q_c) L_k].

This guarantees that every cation-star row sum of `J` is zero. With the SVD
rank tolerance

    tol = eps * max(J.shape) * sigma_max,

define

    g = P_col(J) t,
    h = t - g.

`g` is the part of the required global valence correction compatible with the
frozen bond-valence geometry to first order; `h` is the differential
incompatibility. This is a projection only, not a coordinate update.

## Frozen candidate features

All six eligible NEXT38 features are fixed as high-is-risk:

1. `bvtc_correction_rms = ||t|| / sqrt(E)`;
2. `bvtc_compatible_rms = ||g|| / sqrt(E)`;
3. `bvtc_compatible_q95`, the inverted-CDF q95 of `|g_e|`;
4. `bvtc_incompatible_rms = ||h|| / sqrt(E)`;
5. `bvtc_incompatible_fraction = ||h|| / ||t||`;
6. `bvtc_compatible_localization = E max(g_e^2) / sum(g_e^2)`.

When `||t||` or `||g||` is numerically zero, the corresponding fractions and
localization are exactly zero. Diagnostics only are site-deficit RMS/max,
Jacobian rank/rank fraction, negative corrected-edge fraction, and bond-valence
parameter-source fractions. Inverted-CDF quantiles and the edge-count factor
make exact graph replication invariant. No post-label direction changes are
allowed.

## Frozen catalogue

Four fixed comparators are copied from sealed NEXT32/NEXT37 artifacts:
`scbv_mismatch_q95`, rep12 steric vector RMS/max, and SIVR site imbalance.
The seven predeclared pairs are:

1. correction RMS + SCBV mismatch q95;
2. compatible RMS + steric vector RMS;
3. compatible q95 + steric vector max;
4. incompatible RMS + SCBV mismatch q95;
5. incompatible fraction + SIVR site imbalance;
6. compatible localization + steric vector max;
7. steric vector RMS + SIVR site imbalance (unchanged fixed comparator).

Thus 10 singles plus 7 pairs give 17 formulae and exactly 85 rows at rejection
fractions `{0.025,0.05,0.075,0.10,0.15}`. Terms use development median/IQR
robust-z and pairs use equal weights. No continuous parameters are fitted.

## Development and confirmation

Use the already exposed 4,096-row OMat24 `rattled-relax` development cohort,
unchanged severe/protected endpoint definitions, and unchanged six gates:
coverage LB 0.95, protected recall LB 0.98, severe precision LB 0.90, savings
LB 0.05, AUC 0.85, and precision-LB minus prevalence-UB 0.20.

If 0/85 pass, publish a standalone negative report and leave all confirmation
sources unopened. If a candidate passes all six, freeze every constant and
publish label-free predictions before opening exactly one predeclared
confirmation source.

## Claim boundary

The bond-valence model and its Jacobian are empirical geometric constraints,
not a real potential. Even a confirmed response-screening pass would not
establish formation energy, hull stability, kinetics, synthesizability, or a
general replacement of DFT.
