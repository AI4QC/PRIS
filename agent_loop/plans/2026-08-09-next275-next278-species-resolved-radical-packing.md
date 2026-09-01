# NEXT275--NEXT278 Species-Resolved Radical-Packing Plan

**Status:** frozen before opening any NEXT275--NEXT278 feature outcome or
endpoint-conditioned result.

**Sequential-development disclosure:** NEXT267--NEXT274 discovery outcomes are
known. NEXT270 remains the best radical-packing BROAD residual at
`(5, 0.0955435292756307)`; the structure-level joint transforms in
NEXT271--NEXT274 did not improve it. This is a new prospective candidate space,
not independent confirmation.

## Motivation and boundary

The radical-plane literature shows that atomic size must be included when
partitioning multicomponent structures
([DOI 10.1016/j.intermet.2011.12.019](https://doi.org/10.1016/j.intermet.2011.12.019)).
The Soliquidy work further motivates species-restricted Voronoi summaries for
multicomponent systems
([DOI 10.1038/s41524-025-01529-1](https://doi.org/10.1038/s41524-025-01529-1)).
This plan does not reproduce that paper's fitted classifier and does not use
its enthalpy term. It tests only a deterministic geometric hypothesis:
legitimate differences between elements should be separated from anomalous
dispersion among sites of the same element.

Every executable quantity may use only composition and the initial, raw,
unrelaxed periodic geometry. It must not use a DFT calculation or
per-structure DFT value, learned energy/force/stress proxy, model or proxy
potential, relaxed structure, trajectory, or physical relaxation. Discovery
outcomes are offline labels only. Validation and replication outputs remain
sealed throughout NEXT275--NEXT278.

All work is additive. Existing scripts, outputs, reports, and canonical paper,
note, preregistration, and README files remain unchanged. Append to the
independent exploration report only after the branch is complete.

## Frozen geometric construction

Recompute the exact NEXT267 periodic radical cells with the same radius table,
Minkowski-reduced lattice, Wigner--Seitz support, half-space enumeration,
resource guards, tiling certificate, empty-cell policy, and `1e12` output grid.
For every nonempty site `i`, define

```text
v_i = radical_cell_volume_i / ((4*pi/3) * atomic_radius_i^3)
c_i = radical_cell_chebyshev_radius_i / atomic_radius_i.
```

For either positive site vector `z` (`v` or `c`), species groups `g`, group
size `n_g`, total nonempty size `N`, global mean `mu`, and group mean `mu_g`,
define

```text
W = sum_g sum_(i in g) (z_i - mu_g)^2
B = sum_g n_g (mu_g - mu)^2
T = W + B
within_cv = sqrt(W/N) / mu
between_cv = sqrt(B/N) / mu
within_variance_fraction = 0 if T == 0 else W/T
species_cv_g = 0 if n_g == 1 else
               sqrt(sum_(i in g)(z_i-mu_g)^2/n_g) / mu_g
weighted_species_cv = sum_g (n_g/N) * species_cv_g
max_species_cv = max_g species_cv_g.
```

Singleton species therefore contribute zero within-species variance. Uniform
supercell replication leaves every definition unchanged.

Materialize exactly these ten features in this order:

1. `prvs_volume_within_cv`
2. `prvs_volume_between_cv`
3. `prvs_volume_within_variance_fraction`
4. `prvs_volume_weighted_species_cv`
5. `prvs_volume_max_species_cv`
6. `prvs_chebyshev_within_cv`
7. `prvs_chebyshev_between_cv`
8. `prvs_chebyshev_within_variance_fraction`
9. `prvs_chebyshev_weighted_species_cv`
10. `prvs_chebyshev_max_species_cv`

No structure-level algebraic combinations may be added in this branch.
Structures unsupported by the unchanged NEXT267 geometry/resource certificate
must abstain.

## NEXT275: label-free full materialization

Implement tests before code. Tests must cover analytic one-species and
two-species decompositions; singleton behavior; exact `T = W + B`; positivity;
FCC, BCC, NaCl, diamond, and triclinic examples; rotation, translation, site
permutation, equivalent lattice-basis, and uniform-supercell replication
invariance; support/refusal guards; no endpoint interface; atomic publication;
and all boundary flags.

Materialize both full discovery sources using the same frozen raw-structure
inputs and per-structure resource guards as NEXT267. NEXT276 is authorized only
if all ten features are finite on every supported row, row identities are
unchanged, and the radical-cell tiling certificate remains valid.

## NEXT276: prospective feature audit

Reconstruct the exact NEXT224 rejected-extreme cohort and test both
`protected_low` and `protected_high` directions for the ten features: twenty
fixed hypotheses. Use the unchanged inverse-CDF quantiles, coverage gates,
source aggregate/macro/worst-fold AUC gates, and ranking policy from NEXT272.
Publish the exact eligible-set digest before any coefficient search.

NEXT277 is authorized only for directions passing every raw gate in both
discovery sources.

## NEXT277: bounded margin-local search

For every eligible direction, evaluate exactly the seven local-width fractions
and three nonnegative amplitude fractions used by NEXT273, plus one exact
NEXT224 reproduction control. Support, missingness, normalization population,
triangular term, folds, AUC, twelve SAFE cells, and BROAD gates remain
unchanged. No adaptive feature, direction, width, amplitude, coefficient, or
threshold may be added after results are visible.

Freeze a candidate only if it passes all cross-source discovery gates.
Otherwise NEXT278 is authorized only for exact new-candidate identities that
pass both source-AUC gates and all SAFE cells but fail BROAD.

## NEXT278: unchanged BROAD diagnostic

Exactly reproduce the authorized NEXT277 records and unchanged BROAD threshold
tables. Rank by

```text
(failed_constraint_count, normalized_shortfall_sum, candidate_key)
```

and compare with the frozen NEXT270 reference `(5, 0.0955435292756307)`.
NEXT278 performs no new formula search and opens no validation or replication
output. A discovery all-gate pass or strict residual improvement requires a new
preoutcome freeze before continuation; otherwise this branch closes.

## Verification and reporting

- Run focused NEXT275--NEXT278 tests and the complete repository suite.
- Verify every frozen input, executed-source, and published-output SHA-256.
- Confirm all no-DFT/no-proxy/no-relaxation and sealed-output flags.
- Check CodeGraph status after edits.
- Append a conservative evidence section to the independent report only.
