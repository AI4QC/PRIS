# NEXT115 Generalized-Hall Closure Certificate Design

**Date:** 2026-08-08

**Status:** frozen before production implementation

**Scientific boundary:** one raw, unrelaxed structure plus frozen analytic
element, geometry, bond-valence, electrostatic, and symmetry data.  The
executable certificate must not read DFT quantities, relaxed structures or
trajectories, learned energies/forces/stresses, MLIP outputs, or
same-composition alternatives.  DFT-derived outcomes may only be opened later
as discovery labels after the feature and search freezes are written.

## Motivation

NEXT112--114 retained only whole-structure aggregates of the optimum CMVO
face.  Those terms changed SAFE behavior, but stayed on the same AUC--SAFE
trade-off and produced no all-gate candidate.  NEXT115 therefore preserves a
strictly more local object: the most obstructed subset of one sign and the
maximum charge capacity of its graph neighborhood.

## Graph and exact obstruction

For one frozen sign pattern, let `P` be positive sites and `N` negative sites.
Every site `v` has an allowed charge-magnitude interval `[l_v, u_v]` with
`0 < l_v <= u_v`.  The bipartite contact graph has oriented edges `P -> N`.

For `S` contained in `P`, define

```
deficit_P(S) = sum(l_p for p in S) - sum(u_n for n in neighborhood(S)).
```

The maximum positive-side violation is the closure LP

```
maximize  sum(l_p * x_p) - sum(u_n * y_n)
subject to x_p <= y_n for every edge p -> n
           0 <= x_p, y_n <= 1.
```

The constraint matrix is the order-polytope matrix of a bipartite preorder.
Its vertices are incidence vectors of closed sets, so its optimum is exactly
`max_S deficit_P(S)`, including the empty subset with value zero.  Swapping the
two signs gives the negative-side violation.

The two families of inequalities are the generalized Hall conditions for a
nonnegative edge flow whose incident flow at every vertex lies in
`[l_v, u_v]`.  Necessity follows by summing incident flow over a subset.
Sufficiency follows from the feasible-circulation/min-cut theorem applied to
`source -> P -> N -> sink` with vertex lower/upper intervals: the only finite
obstructing cuts reduce to one of the two closure families.  The production
tests retain an independent direct-flow LP and exhaustive subset enumeration
as finite oracles rather than relying only on this argument.

## Canonical optimum-face morphology

For each direction, let the primary maximum deficit be `Delta`.  If
`Delta <= tolerance`, every term for that direction is exactly zero.  If it is
positive, secondary LPs are solved on the exact primary optimum face:

- minimize origin lower-charge support `sum(l_origin * x_origin)`;
- minimize and maximize selected origin-site count;
- minimize selected neighbor-site count.

Only optimum values are published; no arbitrary cut vector is exposed.  The
directional terms are:

1. `hcid_<side>_global_deficit = Delta / sum(l_origin)`;
2. `hcid_<side>_local_density = Delta / min_origin_lower_support`;
3. `hcid_<side>_origin_site_fraction_min`;
4. `hcid_<side>_origin_site_fraction_max`;
5. `hcid_<side>_neighbor_site_fraction_min`.

Positive and negative directions remain separate because the frozen
electronegativity orientation has physical meaning.  No label-dependent
selection or dominant-side tie-break is allowed.

## Required invariants

- Site permutations and edge-order permutations leave all values unchanged.
- Duplicate edges are removed deterministically.
- Common positive charge scaling leaves normalized terms unchanged.
- Exact integer replication of a structure leaves normalized terms unchanged.
- Degenerate primary optima return the same secondary optimum values.
- Values are finite and lie in `[0, 1]` up to numerical tolerance.
- Invalid intervals and incorrectly oriented edges fail closed.
- The structure wrapper uses exactly the NEXT109 sign-pattern ranking and
  preserves its technical abstentions.
- The implementation source contains no Brown parameter, DFT, relaxation,
  model, energy, force, stress, MatterSim, CHGNet, or MLIP dependency.

## Frozen development sequence

1. NEXT115 implements and tests the graph certificate and raw-structure
   wrapper additively.
2. NEXT116 builds the ten new directional terms for the already frozen SCIGEN
   and WyFormer discovery structures outside the repository.  No labels are
   opened during feature construction or distribution audit.
3. A label-free freeze may remove zero-IQR or nearly redundant terms and must
   define a finite NEXT117 candidate catalogue before discovery outcomes are
   read.
4. NEXT117 evaluates only the existing discovery endpoints and their frozen
   AUC, SAFE, and BROAD gates.  Validation and replication remain unopened
   unless one candidate passes every discovery gate.
5. Results are documented in a new standalone report.  Prior reports and all
   canonical paper/document paths remain untouched.

## Prototype evidence before freeze

A disposable implementation checked 5,000 seeded random bipartite interval
graphs with one to five vertices per side.  Closure LP optima matched exhaustive
subset enumeration to maximum absolute error `2.220446049250313e-15`; HiGHS
returned no fractional primary vertex; and the pair of zero-deficit conditions
matched direct interval-flow feasibility in every case.  This is a design
sanity check, not a discovery result.
