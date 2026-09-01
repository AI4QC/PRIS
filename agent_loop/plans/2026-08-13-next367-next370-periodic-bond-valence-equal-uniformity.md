# NEXT367--NEXT370 periodic bond-valence equal-uniformity

Date: 2026-08-13

Status: frozen after the label-blind termination of NEXT363 and before
computing any NEXT367 feature value or opening any NEXT367 endpoint outcome.

## Scope and information boundary

The target remains an interpretable pre-DFT screen for raw generated or
theoretical periodic crystals. The executable may use element identities,
frozen tabulated oxidation/bond-valence data, and one initial raw unrelaxed
periodic geometry only. It must not execute or consume DFT, endpoint values,
energies, forces, stresses, their learned proxies, model potentials,
relaxation, trajectories, or later geometries. Discovery endpoints may be
opened only after a frozen label-blind probe and complete formal label-free
build pass. Validation and replication stay physically sealed.

All files are additive. Existing scripts and content remain unchanged, and
only the independent no-DFT report may be extended before user review.

## Existing-coverage audit and selected missing rule

The existing implementation was inspected before selection.

1. Scalar bond-valence sum mismatch and a global scale are already NEXT22.
2. Bond-valence vector sums are not missing: NEXT22 already exposes
   `scbv_vector_asymmetry_rms/max`, and NEXT70 independently implements the
   same local vector sum for metal--donor environments.
3. Entropy effective coordination number
   `exp(-sum(p log p))` is already present in the original local descriptors
   and NEXT22, but its raw magnitude changes with coordination number.
4. NEXT38 measures differential transport compatibility, NEXT101/104/119
   measure alternative or bounded valence-flow realizability, and NEXT307
   measures the graph-cycle component of the observed bond-valence field.
5. NEXT303 directionally balances uniform, inverse-square, and
   charge-inverse-square contact cages, not the equal-valence distribution of
   empirical bond strengths.

The retained missing quantity is the coordination-number-normalized entropy
form of Brown's equal-valence principle. It removes the degree carried by the
existing effective-CN feature and asks only how uniformly one site's observed
bond valence is distributed over its actual periodic bonds.

## Frozen graph, parameters, and formula

Use the unchanged NEXT19 formal-valence assignment, unchanged opposite-sign
periodic Voronoi graph, and unchanged NEXT38/NEXT307 frozen-fallback bond-
valence parameter resolver. For an incident periodic bond `j` at site `i`,

```text
s_ij = exp((R0_ij - R_ij) / B_ij),
p_ij = s_ij / sum_k s_ik,
d_i  = number of incident periodic bond images,
U_i  = exp(-sum_j p_ij log(p_ij)) / d_i
     = exp(-D_KL(p_i || Uniform(d_i))).
```

Thus `0 < U_i <= 1`; it is exactly one if and only if the observed bond
valences at that site are equal. Repeated periodic images are separate bonds,
as they are in the frozen periodic multigraph. Uniform scaling of all bond
valences cancels exactly.

The sole public feature is

```text
pbveu_equal_valence_uniformity_q10
    = inverted-CDF 0.10 quantile of {U_i over all sites},
```

quantized to `1e-10`. The sole direction is `protected_high`. The lower-tail
quantile is fixed to detect a materially nonuniform local environment without
letting one numerical worst site dominate. No alternative quantile, mean,
minimum, sign subgroup, parameter policy, graph, transformation, composite,
or direction is searched.

## Representation and analytic tests

A rigid Euclidean transformation or periodic translation preserves all bond
lengths. Site permutation and unimodular lattice rebasing only relabel the
periodic multigraph. An exact integer supercell copies every primitive site
star with the same incident bond-valence multiset. Consequently every `U_i`
and the empirical site distribution are unchanged.

Unit tests must cover equal and deliberately unequal analytic stars,
monotonic loss under a mean-preserving bond-valence transfer, uniform strength
scale, edge order, disjoint exact replication, rigid rotation, periodic
translation, site permutation, unimodular rebasing, explicit `2x1x1`
supercell, malformed input, and the geometry-only firewall.

## Label-blind sequential gates

The probe selects the same deterministic 80 discovery structures per source
and reads raw discovery geometry plus label-free controls only. In each source
it requires:

- support at least `72/80`;
- finite values in `(0,1]` and at least 20 distinct values at 10 decimals;
- maximum equivalent-representation error at most `1e-8`;
- maximum absolute Spearman correlation strictly below `0.90` against all
  numeric NEXT85/NEXT94 discovery base features plus all formal later
  label-free features through NEXT359.

For this and future branches, a control is eligible for the novelty maximum
only when it has at least `40/80` jointly finite probe rows and both variables
are nonconstant on those rows. The `40` threshold is frozen before any PBVEU
value is computed. It prevents a sparse three- or four-row control from
dominating the novelty gate; it does not reopen or rescue the already closed
NEXT363 branch. The probe records both eligible and sparse-skipped control
counts.

No endpoint is read during the probe. Any failure terminates the branch and
NEXT368--NEXT370 are not created.

Only a passing probe authorizes the all-row NEXT367 label-free build. Formal
coverage must be at least `0.90` independently in 13,470 SCIGEN and 5,232
WyFormer discovery rows. Only a passing immutable manifest authorizes
NEXT368, which reuses the unchanged NEXT224/NEXT268/NEXT324 audit: the frozen
rejected-extreme cohort, reduced-formula five-folds, inverted-CDF `1/16` and
`15/16`, coverage `0.90`, class count `20`, pooled AUC `0.55`, macro AUC
`0.53`, worst-fold AUC `0.50`, and the frozen `protected_high` direction.

Failure in either source gives zero eligible hypotheses and terminates the
branch. NEXT369/NEXT370 may exist only if NEXT368 explicitly authorizes the
predeclared formula-search stage. Nothing may be reversed, repaired, imputed,
or tuned after outcomes are opened.

## Decision log

- The mechanism is an empirical bond-valence law, not an energy, force,
  stress, potential, relaxation, or DFT surrogate.
- The vector-sum candidate was rejected before execution because it is already
  implemented by NEXT22 and NEXT70.
- Raw effective coordination was rejected because it is already implemented
  and conflates uniformity with degree.
- `exp(-D_KL)` is selected because it is dimensionless, bounded, scale
  invariant, and has no fitted coefficient.
- Parameter and geometry failures are abstentions and are never imputed.
- Geometry and label archives stay local and are never transmitted.
