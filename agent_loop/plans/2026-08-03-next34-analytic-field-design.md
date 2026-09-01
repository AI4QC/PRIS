# NEXT34 Analytic Electrostatic-Field Imbalance Design

## Status and relationship to prior work

NEXT34 is an additive DFT-free branch. It preserves NEXT19 valence transport,
NEXT21 normalized Madelung, NEXT22 scale-calibrated bond valence, NEXT23,
NEXT32, NEXT33, every report, and every canonical document unchanged.

The repository audit changes the intended direction. NEXT19 already tests
global formal-valence flow feasibility, and NEXT22 already tests local
bond-valence capacity, effective coordination, and bond-valence-vector
asymmetry. Repeating those ideas under new names would not be a new mechanism.
NEXT34 instead tests a quantity not yet present in the analytic catalogue: the
dimensionless periodic electrostatic field imbalance at every site, computed
from a neutral composition-derived point-charge assignment, and its bounded
combination with the independent short-range steric imbalance from NEXT33.

## Hard execution boundary

The executable feature accepts one raw unrelaxed periodic structure `x0`. It
may use only element identities, cell, coordinates, the frozen repository
valence-assignment cascade, an analytic point-charge Ewald sum, fixed constants,
and deterministic linear algebra. The subsequent rule may also reuse the
already sealed NEXT33 geometry-only features.

It must not read or calculate DFT values, use a relaxed structure or trajectory,
call MatterSim/MLIP or any learned model, perform coordinate or cell relaxation,
or inspect another structure of the same composition. DFT forces and stresses
remain offline labels only after the complete feature schema, risk directions,
candidate catalogue, rejection fractions, and promotion gates are frozen.

The Ewald calculation is a classical analytic lattice sum over assigned point
charges. Its coordinate derivative is used to obtain the electrostatic field;
no electronic-structure calculation, fitted potential, learned proxy, or
coordinate update is performed. Neither Coulomb energy nor any energy-like
value is emitted as a feature.

## Hypothesis

NEXT33 shows that scalar overlap is incomplete and that directional steric
imbalance contributes a modest independent signal. Short-range repulsion alone
cannot describe long-range charge organization. The frozen NEXT34 hypothesis is:

> An unreasonable unrelaxed ionic or polar structure is more likely to show a
> large dimensionless periodic point-charge field at one or more sites; risk is
> stronger when this long-range imbalance co-occurs with a short-range steric
> vector imbalance.

This is deliberately narrower than a stability claim. A nonzero analytic
field in a real stable structure may be balanced by covalency, polarization,
or short-range interactions absent from the point-charge model. The absolute
promotion gates, rather than a mechanistic story, decide whether the descriptor
is useful enough to advance.

## Frozen field normalization

Let the valence cascade assign neutral site charges `z_i`, and define

    ell   = (V / N)^(1/3)
    q_rms = sqrt(mean_i z_i^2)
    k_e   = 14.3996454784255 eV Angstrom.

An Ewald sum over the supplied periodic geometry gives the analytic Coulomb
coordinate derivative vector `G_i` in eV/Angstrom. No coordinate is moved.
For each active site `|z_i| > 1e-12 max_j |z_j|`, define two dimensionless
vectors:

    e_i = G_i ell^2 / (k_e |z_i| q_rms)
    h_i = G_i ell^2 / (k_e q_rms^2).

Both are invariant to a common rescaling of all assigned charges; `ell` also
makes the magnitudes invariant to uniform scaling of cell and coordinates for
the pure Coulomb lattice. `e_i` is a per-unit-charge field residual, whereas
`h_i` retains the relative charge weighting of the site response.

The frozen output schema is:

- `aefi_field_rms`, `aefi_field_q95`, `aefi_field_max` from `|e_i|`;
- `aefi_residual_rms`, `aefi_residual_q95`, `aefi_residual_max` from `|h_i|`;
- `aefi_field_tensor_deviator`, the Frobenius deviator of
  `sum_i e_i e_i^T / sum_i |e_i|^2` (zero when the denominator is numerically
  zero);
- `aefi_active_site_fraction` as coverage diagnostics only.

The 95% quantiles use `method="inverted_cdf"` so primitive/supercell replication
does not change them. All seven candidate features have their risk direction
frozen as high-is-risk before development labels are joined.

## Frozen candidate catalogue

Each candidate is either one robust-z term or an equal-weight sum of two terms.
No continuous weights, learned transforms, interactions, or post-label direction
flips are permitted.

New single terms:

- the seven AEFI candidate features above, excluding active-site fraction.

Reused diagnostic/comparator single terms:

- NEXT33 `steric_rep12_vector_rms/q95/max`;
- NEXT33 `steric_overlap2_vector_rms`;
- NEXT33 `steric_rep12_tensor_deviator`;
- NEXT32 `sivr_site_imbalance_rms` and `sivr_edge_mismatch_q95`;
- NEXT32 `cov_q05` with low-is-risk direction.

Mechanism-linked pairs are frozen as:

1. field RMS + rep12 vector RMS;
2. field q95 + rep12 vector q95;
3. field max + rep12 vector max;
4. residual RMS + overlap2 vector RMS;
5. residual q95 + SIVR site imbalance;
6. residual max + rep12 vector max;
7. field tensor deviation + steric tensor deviation;
8. field RMS + low `cov_q05`;
9. residual RMS + SIVR site imbalance;
10. rep12 vector RMS + SIVR site imbalance (fixed NEXT33 comparator);
11. low `cov_q05` + SIVR edge mismatch (fixed NEXT32 comparator).

This gives 26 formulas and five fixed rejection fractions
`{0.025, 0.05, 0.075, 0.10, 0.15}`, or 130 development scan rows.

## Development and confirmation protocol

The exposed 4,096-row OMat24 `rattled-relax` cohort, endpoint definitions, and
Pauling controls remain identical to NEXT32/NEXT33. Feature extraction must
finish and publish a no-replace manifest while endpoint fields remain unopened.
Only then may the already exposed development labels be joined once.

Promotion requires all six unchanged gates:

- coverage one-sided 95% Wilson lower bound >= 0.95;
- protected-low-response recall lower bound >= 0.98;
- severe-response rejection precision lower bound >= 0.90;
- savings lower bound >= 0.05;
- ROC AUC >= 0.85;
- precision lower bound minus severe-prevalence upper bound >= 0.20.

If no candidate passes, stop without downloading or reading `rattled-300`,
`rattled-500`, or `rattled-1000`, publish a standalone negative report, and do
not flip an AEFI direction on the same cohort. If a candidate passes, freeze
its exact terms, robust-z constants, threshold, source hashes, and predictions
before opening exactly one predeclared confirmation source.

## Claim boundary

Passing development is only permission for confirmation. A discovery or a
claim beyond Pauling requires the frozen rule to pass independent confirmation
and the absolute safety/usefulness gates. Even success on initial DFT response
would not establish formation energy, hull stability, kinetics, synthesis, or
general replacement of DFT.

