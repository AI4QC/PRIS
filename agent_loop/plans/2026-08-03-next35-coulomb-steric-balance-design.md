# NEXT35 Analytic Coulomb--Steric Balance Design

## Prior-work audit

NEXT35 is additive and preserves every earlier script, artifact, report, paper,
and confirmation source. Before selecting the mechanism, the repository was
audited for apparent alternatives. Coordination-shell gaps (P6), Voronoi
solid-angle entropy/like-sign fractions (P2), neighbor-method disagreement
(P8), Voronoi free volume/off-centering (P10), valence flow (NEXT19), analytic
rigidity (NEXT20), Madelung (NEXT21), and bond-valence equilibrium (NEXT22) have
already been implemented and evaluated. NEXT35 will not repackage them.

NEXT34 found that the magnitude of a composition point-charge Ewald field is
nearly nondiscriminating, while NEXT33 retained a modest short-range steric
vector signal. Adding their scalar robust-z scores diluted rather than improved
the best existing candidate. NEXT35 therefore tests a different, pre-frozen
mechanism: whether the two analytic vector fields can oppose one another at
the supplied geometry after one closed-form, nonnegative amplitude matching.

## Hard boundary

Execution accepts one raw unrelaxed `x0`, its elements/cell/coordinates, the
frozen composition-valence cascade, covalent radii, a classical point-charge
Ewald lattice sum, and deterministic algebra. It may reuse sealed geometry-only
NEXT34 columns. It does not use DFT values, electronic-structure calculations,
relaxed structures, trajectories, learned potentials/proxies, coordinate or
cell updates, or same-composition alternatives.

The one fitted scalar below is calculated independently for each supplied
structure from two analytic vector arrays. It is an amplitude normalization,
not a coordinate optimization, training fit, energy minimization, or relaxation.
DFT response remains an offline exposed-development label only after all feature
definitions, directions, candidates, fractions, and gates are frozen.

## Frozen vector fields

For the neutral analytic charges `z_i`, let `C_i` be the dimensionless Ewald
coordinate-derivative vector used by NEXT34:

    C_i = G_i ell^2 / (k_e q_rms^2).

For every unique periodic pair with covalent-radius ratio
`q_ij = d_ij/(r_i+r_j)`, define the short-range vector field

    w_ij = max(0, max(q_ij, 0.45)^(-12) - 1)
    S_i -= w_ij u_ij
    S_j += w_ij u_ij.

Both arrays sum to zero up to numerical precision. Their independent amplitude
scales are irrelevant to the balance descriptors.

## Closed-form balance

Flatten the two `(N,3)` arrays. The nonnegative repulsion amplitude that best
cancels the Coulomb array is

    lambda* = max(0, -<S,C>/<S,S>).

Set `R_i = C_i + lambda* S_i`. This does not move an atom. Frozen candidate
features are:

1. `acsb_opposition_deficit`: `(1+cos(C,S))/2`, zero for perfect opposition,
   one for aligned fields; one when exactly one global field is active and zero
   when both are zero;
2. `acsb_global_residual`: `||R|| / sqrt(||C||^2 + lambda*^2 ||S||^2)`, with
   the same zero/one conventions for both/one inactive fields;
3. `acsb_site_residual_rms`;
4. `acsb_site_residual_q95`;
5. `acsb_site_residual_max`, where each site residual is
   `||R_i||/(||C_i||+lambda*||S_i||)` and inactive sites are zero;
6. `acsb_site_direction_deficit_q95`, using `(1+cos(C_i,S_i))/2`, one for
   exactly one active site field and zero for neither;
7. `acsb_active_disagreement_fraction`, the fraction of sites in the union of
   active fields that either has only one active field or has non-opposing
   (`dot(C_i,S_i) >= 0`) directions.

All seven risk directions are frozen as high-is-risk. Quantiles use
`method="inverted_cdf"` for representation invariance.

Diagnostics not eligible as terms are `acsb_optimal_repulsion_scale`, global
field norms, and joint-active-site fraction.

## Frozen bounded catalogue

Single terms comprise the seven ACSB features plus four fixed comparators:

- `aefi_residual_max` high;
- `steric_rep12_vector_rms` high;
- `steric_rep12_vector_max` high;
- `sivr_site_imbalance_rms` high.

The seven mechanism pairs are frozen as:

1. global residual + steric vector RMS;
2. site residual q95 + steric vector max;
3. opposition deficit + AEFI residual max;
4. active disagreement fraction + SIVR site imbalance;
5. site direction deficit q95 + steric vector RMS;
6. global residual + SIVR site imbalance;
7. steric vector RMS + SIVR site imbalance (unchanged NEXT33 comparator).

Thus 18 formulas times the fixed rejection fractions
`{0.025,0.05,0.075,0.10,0.15}` give exactly 90 scan rows. Every term uses a
development median/IQR robust-z; pairs are equal-weight. No continuous weights,
extra transforms, or post-label direction changes are permitted.

## Development and confirmation

Use the same exposed 4,096-row OMat24 `rattled-relax` cohort and unchanged severe
and protected endpoints. Publish a hash-locked geometry-only feature artifact
before joining endpoints. Require the same six absolute gates as NEXT32--34:
coverage LB 0.95, protected recall LB 0.98, severe precision LB 0.90, savings
LB 0.05, AUC 0.85, and precision-LB minus prevalence-UB 0.20.

If 0/90 pass, stop without downloading or reading `rattled-300`, `rattled-500`,
or `rattled-1000`, and publish a standalone negative report. If a candidate
passes all six, freeze every constant and prediction before opening exactly one
predeclared confirmation source.

## Claim boundary

Even a development pass is not a discovery. An independent confirmation pass
is required for a narrow initial-response screening claim beyond operational
Pauling controls. No result here establishes formation energy, hull stability,
kinetics, synthesis, or general replacement of DFT.

