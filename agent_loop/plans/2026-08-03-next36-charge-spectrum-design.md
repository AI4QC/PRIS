# NEXT36 Weighted Charge-Spectrum Design

## Prior-work and literature audit

NEXT36 is additive. It preserves every prior source, artifact, report,
confirmation source, and canonical document. Repository audit found no existing
implementation of a scale-resolved charge-weighted structure factor. NEXT19
solves global valence transport on a local graph, NEXT21 aggregates normalized
Ewald terms, NEXT22 evaluates bond-valence capacity and vector asymmetry, and
NEXT34/NEXT35 evaluate point-charge derivative fields. None measures how formal
charge fluctuations are distributed across reciprocal length scales.

The physical motivation is weighted hyperuniformity, not a claim that one
finite crystal cell establishes a thermodynamic-limit class. Torquato and
Stillinger introduced structure-factor/local-variance order metrics for
long-wavelength fluctuations
([Phys. Rev. E 68, 041113](https://doi.org/10.1103/PhysRevE.68.041113)).
Torquato et al. generalized the framework to particles carrying scalar weights,
explicitly including charges, and showed that positional hyperuniformity need
not imply weighted hyperuniformity
([Phys. Rev. X 16, 011042](https://doi.org/10.1103/fr99-qh7h)). The older
Stillinger--Lovett screening analysis supplies the related premise that charge
fluctuations are suppressed at long wavelength in screened Coulomb systems.

NEXT36 therefore tests a narrow finite-cell hypothesis: among raw predicted
crystals with the same execution boundary, unusually large formal-charge
spectral weight at long dimensionless wavelengths is a risk marker for a large
initial first-principles response.

## Hard execution boundary

Execution accepts exactly one raw unrelaxed `x0`, its elements/cell/coordinates,
the frozen composition-valence cascade, reciprocal-lattice construction, and
deterministic complex arithmetic/linear algebra. It may copy four already
sealed geometry-only comparators from NEXT35. It does not use DFT values,
electronic-structure calculations, relaxed structures, trajectories, learned
potentials or energy/force/stress proxies, coordinate/cell updates, or
same-composition alternatives.

DFT force/stress response is joined only after the source, tests, feature
schema, directions, candidate catalogue, fractions, and gates are frozen and a
hash-locked label-free feature artifact has been published.

## Frozen reciprocal construction

For neutral analytic charges `z_j`, fractional coordinates `f_j`, and every
nonzero reciprocal integer vector `h`, define

    Z(h) = sum_j z_j exp(-2 pi i h dot f_j)
    Q2   = sum_j z_j^2
    ell  = (V/N)^(1/3)
    u_h  = |G_h| ell
    I_h  = |Z(h)|^2 / (N Q2).

Enumerate every reciprocal vector with `0 < u_h <= 18`. The cutoff is fixed
before labels. For an exact supercell containing `m` copies, forbidden
supercell reciprocal modes have zero amplitude, while allowed primitive modes
have both numerator and `N Q2` multiplied by `m^2`; sums of `I_h` are therefore
representation invariant up to numerical precision.

At the fixed dimensionless smoothing scales `tau={0.25,0.40,0.60}`, define

    H_tau = sum_h I_h exp[-(tau u_h)^2].

These are finite-cell Gaussian charge-spectrum order metrics. They are not
called a measured `k -> 0` limit and do not prove hyperuniformity.

At `tau=0.60`, also define the normalized tensor

    M = sum_h I_h exp[-(0.60 u_h)^2] ghat_h ghat_h^T / H_0.60

and its bounded deviator

    A = sqrt(3/2) ||M - I/3||_F.

The long-mode peak fraction is the largest weighted mode divided by `H_0.60`.

## Frozen feature directions

Six eligible terms are fixed as high-is-risk:

1. `csf_gaussian_t025`;
2. `csf_gaussian_t040`;
3. `csf_gaussian_t060`;
4. `csf_long_fraction = H_0.60 / H_0.25`;
5. `csf_long_peak_fraction`;
6. `csf_long_anisotropy = A`.

Diagnostics only are reciprocal-vector count, minimum dimensionless wave
number, and unweighted truncated intensity. No direction will be reversed after
labels.

## Frozen bounded catalogue

Four unchanged comparators are copied from the sealed NEXT35 artifact:

- `aefi_residual_max` high;
- `steric_rep12_vector_rms` high;
- `steric_rep12_vector_max` high;
- `sivr_site_imbalance_rms` high.

The seven predeclared pairs are:

1. `H_0.25 + steric vector max`;
2. `H_0.40 + steric vector RMS`;
3. `H_0.60 + SIVR site`;
4. `long fraction + SIVR site`;
5. `long anisotropy + steric vector RMS`;
6. `long peak fraction + AEFI residual max`;
7. `steric vector RMS + SIVR site` (unchanged NEXT33 comparator).

Thus 10 singles plus 7 pairs give 17 formulas. At the fixed rejection
fractions `{0.025,0.05,0.075,0.10,0.15}`, the exact scan has 85 rows. Every
term uses development median/IQR robust-z; pairs have equal weight. There is no
continuous fitting or candidate expansion.

## Development and confirmation gate

Use the already exposed 4,096-row OMat24 `rattled-relax` development cohort and
unchanged severe/protected endpoint definitions. Require the same six absolute
gates as NEXT32--35: coverage LB 0.95, protected recall LB 0.98, severe
precision LB 0.90, savings LB 0.05, AUC 0.85, and precision-LB minus
prevalence-UB 0.20.

If 0/85 pass, stop without downloading or reading `rattled-300`,
`rattled-500`, or `rattled-1000`, and publish a standalone negative report. If
a candidate passes all six, freeze all constants and predictions before opening
exactly one predeclared confirmation source.

## Claim boundary

Even a development pass would not prove thermodynamic hyperuniformity,
formation energy, hull stability, synthesis, or replacement of DFT. An
independent confirmation pass is required for a narrow initial-response
screening claim, and additional independent stability data are required for the
overall research goal.
