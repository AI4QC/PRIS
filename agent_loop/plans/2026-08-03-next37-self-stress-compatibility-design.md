# NEXT37 Self-Stress Compatibility Projection Design

## Prior-work audit and novelty

NEXT37 is additive and preserves every prior source, artifact, report, paper,
and unopened confirmation source. A direct source audit of NEXT20 shows that
SIVR already computes radius-normalized edge mismatch, site imbalance, cell
prestress, and a prestressed central-force Hessian. It does not construct the
affinely periodic rigidity matrix and does not project the mismatch tension
onto the rigidity-matrix column space or the self-stress co-kernel.

Periodic rigidity theory identifies self-stresses with the left nullspace of
the rigidity matrix and affine periodic flexes with its nullspace; see
[Power, *Crystal Frameworks, Symmetry and Affinely Periodic Flexes*](https://arxiv.org/abs/1103.1914).
Recent incompatible-flow formulations likewise connect frustration to graph
compatibility rather than to a fitted energy
([Phys. Rev. Lett. 134, 147401](https://doi.org/10.1103/PhysRevLett.134.147401)).

NEXT37 tests whether this exact decomposition adds information beyond the
local SIVR imbalance: a radius-mismatch pattern whose weighted tension lies
mostly in the generalized-load subspace should exert a coherent tendency to
move sites or strain the cell, while a component in the left nullspace is
self-balanced on the frozen analytic graph.

## Hard execution boundary

Execution accepts one raw unrelaxed `x0`, elements/cell/coordinates, frozen
composition valences and elemental radii, the deterministic opposite-sign
Voronoi graph, and deterministic SVD/linear algebra. It may copy four sealed
geometry-only comparators from NEXT36. It does not read DFT values, run an
electronic-structure calculation, use a learned/fitted potential, read relaxed
structures or trajectories, compare same-composition alternatives, or change
coordinates/cell.

The SVD computes an orthogonal projection only. No displacement vector is
retained or applied, no iterative minimization is run, and no energy, force,
or stress proxy is emitted. DFT response is an offline development endpoint
opened only after the complete feature and rule artifacts are sealed.

## Frozen edge residual and generalized rigidity matrix

Reuse the exact NEXT20 unweighted opposite-sign Voronoi graph. For edge `e`
from site `i` to periodic image of site `j`, let `d_e`, unit direction `n_e`,
radius sum `R_e`, and analytic neighbor weight `w_e` be fixed. Define

    x_e = log(d_e / R_e)
    mu  = weighted_inverted_cdf_median_e(x_e; w_e)
    r_e = x_e - mu
    t_e = sqrt(w_e) r_e.

The atomic columns of the row-weighted log-length rigidity matrix are

    A[e,i] = -sqrt(w_e) n_e / d_e
    A[e,j] = +sqrt(w_e) n_e / d_e.

Append six symmetric affine-strain columns

    sqrt(w_e) [nx^2, ny^2, nz^2, 2 ny nz, 2 nx nz, 2 nx ny].

Let `A_atom` be the atomic block and `A_full` the atomic-plus-affine matrix.
Using an SVD rank tolerance

    tol = eps * max(A.shape) * sigma_max,

define the orthogonal projections

    p_atom = P_col(A_atom) t
    p_full = P_col(A_full) t
    s      = t - p_full.

Then `s` lies in `null(A_full^T)` up to numerical precision and is the
self-balanced residual component. Column scaling, global coordinate scaling,
rotation, translation, atom permutation, and exact supercell replication do
not change the frozen candidate values.

## Frozen candidate features

All six eligible features are fixed as high-is-risk:

1. `sscp_load_fraction = ||p_full|| / ||t||`;
2. `sscp_load_rms = ||p_full|| / sqrt(sum w_e)`;
3. `sscp_load_q95`, the weighted q95 of
   `|p_full,e|/sqrt(w_e)` using `inverted_cdf`;
4. `sscp_atomic_load_fraction = ||p_atom|| / ||t||`;
5. `sscp_cell_load_fraction = sqrt(max(||p_full||^2-||p_atom||^2,0))/||t||`;
6. `sscp_load_localization = E max(p_full,e^2)/sum(p_full,e^2)`, whose
   edge-count factor makes exact graph replication invariant.

When `||t||=0`, all six risks are exactly zero. Diagnostics only are the
self-balanced fraction, co-kernel dimension fraction, matrix rank, and edge
count. The inverted-CDF center and q95 make exact edge replication invariant.
No post-label direction changes are allowed.

## Frozen catalogue

Four fixed comparators are copied from NEXT36: AEFI residual max, rep12 steric
vector RMS/max, and SIVR site imbalance. The seven predeclared pairs are:

1. load fraction + SIVR site;
2. load RMS + steric vector RMS;
3. load q95 + steric vector max;
4. atomic load fraction + SIVR site;
5. cell load fraction + AEFI residual max;
6. load localization + steric vector max;
7. steric vector RMS + SIVR site (unchanged fixed comparator).

Thus 10 singles plus 7 pairs give 17 formulas and exactly 85 rows at rejection
fractions `{0.025,0.05,0.075,0.10,0.15}`. Terms use development median/IQR
robust-z and pairs use equal weights. No continuous parameters are fitted.

## Development and confirmation

Use the already exposed 4,096-row OMat24 `rattled-relax` development cohort,
unchanged severe/protected endpoint definitions, and unchanged six gates:
coverage LB 0.95, protected recall LB 0.98, severe precision LB 0.90, savings
LB 0.05, AUC 0.85, and precision-LB minus prevalence-UB 0.20.

If 0/85 pass, publish a standalone negative report and leave all confirmation
sources unopened. If a candidate passes all six, freeze all constants and
label-free predictions before opening exactly one predeclared confirmation
source.

## Claim boundary

This finite analytic graph is not a real interatomic potential, and its
projection is not a physical relaxation. Even a confirmed response-screening
pass would not establish formation energy, hull stability, kinetics,
synthesizability, or general replacement of DFT.
