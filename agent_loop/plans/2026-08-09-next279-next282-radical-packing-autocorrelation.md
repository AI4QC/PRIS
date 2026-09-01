# NEXT279--NEXT282 Radical-Packing Autocorrelation Plan

**Status:** frozen before opening any NEXT279--NEXT282 feature or discovery
outcome.

## Motivation

NEXT267--NEXT270 established that radius-normalized periodic radical-cell
heterogeneity transfers across both discovery sources and currently gives the
best radical-packing BROAD residual. NEXT275--NEXT278 showed that directly
penalizing inter-species dispersion is weaker and harms protected SCIGEN
structures. The next question is therefore not whether cell values vary, but
whether their deviations spatially compensate across the exact radical
contact network.

This is motivated by two packing results frozen here before outcomes:

- Klatt and Torquato found that distributions of single-cell Voronoi
  functionals can be structurally insensitive, while two-cell correlation
  functions distinguish jammed packings from equilibrium hard-sphere liquids
  ([Phys. Rev. E 90, 052120](https://doi.org/10.1103/PhysRevE.90.052120)).
- Zhao et al. measured short-range correlations and the onset of
  anti-correlation between free Voronoi volumes in binary disc packings
  ([EPL 97, 34004](https://doi.org/10.1209/0295-5075/97/34004)).

The executable construction uses neither paper's experimental packing fraction
or outcome. It uses only composition, neutral tabulated atomic radii, and the
initial periodic coordinates.

## Immutable boundary

- No DFT calculation is executed.
- No per-structure DFT energy, force, stress, band, charge, relaxed geometry,
  trajectory, or convergence value enters an executable feature or formula.
- No learned energy/force/stress model, interatomic potential, proxy potential,
  or physical relaxation is permitted.
- SCIGEN and WyFormer discovery outcomes are offline labels only after NEXT279
  is fully materialized.
- Validation and replication geometry, endpoints, and outputs remain sealed.
- All prior scripts, outputs, report text, and canonical documents remain
  unchanged. Only additive plan, source, test, formal-output, and independent-
  report files may be created.

## NEXT279: periodic radical contact autocorrelation

NEXT279 must recompute the exact NEXT267 radius-weighted periodic power cells
and reproduce its support and volume-tiling certificates. The radius policy,
reduced-cell handling, neighbor-image guard, half-space tolerances, Qhull
settings, and quantization grid are unchanged.

### Contact incidences

Every active two-dimensional power-cell facet defines one directed contact
incidence. A plane is active only when at least three cell vertices lie on it
within the frozen NEXT267 plane tolerance and their centered singular spectrum
has rank at least two. Contacts retain periodic-image multiplicity. Lattice
Wigner--Seitz planes are labelled as contacts to a periodic image of the same
raw site. Degenerate coincident active planes retain all distinct quantized
generator displacements. The complete directed incidence multiset must be
reciprocal under `(i, j, d) -> (j, i, -d)`; otherwise the structure abstains.

### Site residuals

For every nonempty site cell define the same two radius-normalized values used
by NEXT267:

```text
v_i = V_i / ((4 pi / 3) r_i^3)
c_i = R_i / r_i.
```

For each family `y in {log(v), log(c)}`, define its centered residual
`x_i = y_i - mean(y)`. Let `E` be the directed contact-incidence multiset,
`N` the site count, and `W = |E|`. For positive residual variance freeze four
dimensionless autocorrelation summaries:

```text
Moran(y) = (N / W) * sum_(i,j in E) x_i x_j / sum_i x_i^2

Geary(y) = ((N - 1) / (2 W))
           * sum_(i,j in E) (x_i - x_j)^2 / sum_i x_i^2

AbsMoran(y) = Moran formula applied to
              a_i = |x_i| - mean(|x|)

ExtremeEdge(y) = mean_(i,j in E)
                 [|x_i| >= Q_0.75(|x|) and |x_j| >= Q_0.75(|x|)].
```

`Q_0.75` is the empirical inverted-CDF quantile. If the residual variance (or
the absolute-residual variance for `AbsMoran`) is zero, the corresponding
statistic is exactly zero. If `N = 1`, `W = 0`, or any cell is empty, the
structure abstains rather than inventing graph evidence.

This produces exactly eight features:

```text
prpa_volume_moran                 protected_low
prpa_volume_geary                 protected_high
prpa_volume_absolute_moran        protected_low
prpa_volume_extreme_edge_fraction protected_low
prpa_chebyshev_moran              protected_low
prpa_chebyshev_geary              protected_high
prpa_chebyshev_absolute_moran     protected_low
prpa_chebyshev_extreme_edge_fraction protected_low
```

The directions encode the preregistered compensation hypothesis: coherent
packing should avoid positive clustering of large same-sign or large-magnitude
cell residuals and may show stronger neighbor dissimilarity. No opposite
direction may be added after outcomes are visible.

Required engineering tests include analytic one-site abstention, reciprocal
contacts in at least two multi-site crystals, rigid rotation, uniform scale,
periodic translation, site permutation, reduced/replicated representation,
constant-residual zero handling, invalid-input failure, boundary-interface
exclusion, and an end-to-end atomic-output test.

## NEXT280: prospective raw-feature audit

Audit exactly the eight frozen directions in the unchanged NEXT224 rejected-
extreme cohort. Reuse the existing five formula-group folds, combined-
discovery inverted-CDF `1/16` and `15/16` quantiles, source aggregate/macro/
worst-fold AUC gates, minimum counts, and minimum cell coverage without
modification.

Only hypotheses passing every fixed raw gate in both discovery sources are
eligible. Freeze the sorted eligible identities and SHA-256 before any search.
If none is eligible, terminate the branch; NEXT281 and NEXT282 are not
authorized.

## NEXT281: bounded margin-local search

For every eligible direction, evaluate exactly the seven local-width fractions
and three nonnegative amplitude fractions used by NEXT269/NEXT273/NEXT277,
plus one exact NEXT224 reproduction control. Support, missingness,
normalization population, triangular term, folds, source-AUC gates, twelve
SAFE cells, and BROAD gates remain unchanged.

No adaptive feature, direction, width, amplitude, coefficient, threshold, or
algebraic combination may be added. A candidate may be frozen only if it
passes all cross-source discovery gates. Otherwise NEXT282 is authorized only
for the exact new-candidate identities that pass both source-AUC gates and all
SAFE cells but fail BROAD.

## NEXT282: unchanged BROAD diagnostic

Exactly reproduce the authorized NEXT281 records and unchanged BROAD threshold
tables. Rank by

```text
(failed_constraint_count, normalized_shortfall_sum, candidate_key)
```

and compare with the frozen NEXT270 reference `(5, 0.0955435292756307)`.
NEXT282 performs no new formula search and opens no validation or replication
output. A discovery all-gate pass or strict residual improvement requires a
new preoutcome freeze before continuation; otherwise this branch closes.

## Verification and reporting

- Run focused NEXT279--NEXT282 tests and the complete repository suite.
- Verify every frozen input, executed-source, and published-output SHA-256.
- Confirm all no-DFT/no-proxy/no-relaxation and sealed-output flags.
- Check CodeGraph status after edits.
- Append a conservative evidence section to the independent report only after
  the branch terminates or reaches its frozen stopping condition.
