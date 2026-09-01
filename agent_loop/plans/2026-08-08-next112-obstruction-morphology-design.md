# NEXT112 obstruction morphology certificate design

## Scope and boundary

NEXT112 is an additive, Brown-free and DFT-free extension of the frozen
NEXT109 convex mixed-valence obstruction certificate.  It consumes one raw,
unrelaxed structure together with the same analytic oxidation-state catalogue
and periodic Voronoi graph used by NEXT109.  It must not read energies,
forces, stresses, relaxed structures, trajectories, learned potentials,
same-composition alternatives, discovery labels, validation payloads, or
replication payloads.

NEXT109--NEXT111 and their reports remain byte-for-byte unchanged.  NEXT112
has a new protocol and source file.  NEXT113 will materialize its features
from physically split discovery geometries without opening endpoints.
NEXT114 will freeze every transform and candidate before opening the two
discovery endpoints.  Validation and replication remain closed unless every
frozen discovery gate passes.

## Why a new certificate

NEXT111 established a real but incomplete signal: the maximum connected-
component imbalance improved cross-source AUC, while global/slack terms could
repair all SAFE cells, but the same scalar tail could not do both.  The four
NEXT109 scalars erase where obstruction occurs, how broadly it is distributed,
and whether one charge side is forced to absorb it.  Reading a single LP slack
vector is not acceptable because the optimum can be non-unique and a solver's
chosen vector can change under an otherwise irrelevant site or edge ordering.

NEXT112 therefore exposes only graph aggregates or optimum values of new
secondary linear programs.  It never exposes an arbitrary primary optimizer.

## Frozen mathematics

For site charge-magnitude intervals `[l_i, u_i]`, opposite-sign incidence
matrix `B`, normalized non-negative edge flow `y`, inverse charge scale `r`,
and non-negative interval relaxation `s`, keep the NEXT109 primary problem:

```
D = min 0.5 sum_i s_i
    subject to B_i y - u_i r <= s_i
               l_i r - B_i y <= s_i
               sum_e y_e = 1
               y, r, s >= 0.
```

The sign pattern is selected with exactly the frozen NEXT109 lexicographic
rank `(D, global_gap, maximum_component_gap, unserved_site_fraction, pattern)`.
The new features cannot change which sign pattern is evaluated.

Let connected component `c` contain `n_c` sites and have the normalized charge
interval gap `g_c` already defined by NEXT109.  With `N` total sites, define:

```
component_gap_site_mean = sum_c (n_c/N) g_c
component_gap_site_rms  = sqrt(sum_c (n_c/N) g_c^2)
obstructed_site_fraction = sum_{c:g_c>tol} n_c/N.
```

For `D > tol`, restrict all secondary programs to the exact primary optimum
face `sum_i s_i = 2D`.  Define `t*` as the minimum possible `max_i s_i` on
that face and

```
effective_support = 2D / (N t*)
localized_slack_severity = D (1 - effective_support).
```

Let `a_min` and `a_max` be the minimum and maximum possible positive-site
slack fractions on the same optimum face.  Define

```
forced_side = 2 distance(0.5, [a_min, a_max])
side_slack_asymmetry = D forced_side
side_slack_flexibility = D (a_max - a_min).
```

Every published quantity is finite and in `[0, 1]`.  When `D <= tol`, the
three slack-morphology terms are exactly zero.  For an empty graph, the
component terms are one and the secondary terms are zero: obstruction is
maximally widespread but no normalized edge-flow optimum face exists.

## Invariance and interpretation

Component weighting is invariant to site/edge permutation, common charge-unit
scaling, and exact integer replication.  Under `k` identical graph copies,
`D` is unchanged, each optimal site slack and `t*` scale by `1/k`, and `N`
scales by `k`; therefore `2D/(N t*)` is unchanged.  Positive-side slack
fractions are also unchanged.  Secondary optimum values are canonical even
when the underlying optimizer is not.

The six additive feature names are frozen as:

1. `cmvom_component_gap_site_mean`
2. `cmvom_component_gap_site_rms`
3. `cmvom_obstructed_site_fraction`
4. `cmvom_localized_slack_severity`
5. `cmvom_side_slack_asymmetry`
6. `cmvom_side_slack_flexibility`

The first three distinguish rare from widespread disconnected imbalance.  The
fourth distinguishes a Hall-like local bottleneck from diffuse relaxation.
The last two distinguish a forced cation/anion failure from a flexible
allocation of the same total obstruction.  They are descriptors, not claims
of stability and not substitutes for later unseen-source validation.

## Failure and provenance policy

Invalid intervals and incorrectly oriented edges raise `ValueError`.
Technical graph, catalogue, primary-solver, or secondary-solver failures
abstain with an auditable reason.  No epsilon is added to zero-IQR terms.
NEXT113 records code, catalogue, input, environment, feature-table, and
manifest SHA-256 identities.  NEXT114 records the pre-endpoint freeze digest,
finite candidate universe, exact gates, selected candidate, and unopened
validation/replication flags.
