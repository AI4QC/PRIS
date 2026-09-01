# Equitable Bond-Class Transport Margin Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> script/result and all canonical documents. Keep validation and replication
> sealed.

**Goal:** Test Hawthorne's requirement that topologically equivalent bonds
share a-priori strength by imposing a canonical topology-only equitable bond
partition before exact positive formal-valence transport.

**Architecture:** NEXT450 defines one bounded transport margin on the
unchanged NEXT19 opposite-sign periodic multigraph after all edges in the same
equitable topological class are tied to one unknown. NEXT451--NEXT454 are
conditional full label-free build, cross-source discovery audit, bounded
formula search, and BROAD residual diagnostic.

## 1. Scientific question and distinction

Hawthorne's a-priori bond-strength construction uses a bond-strength table in
which symmetrically/topologically equivalent bonds share unknowns. NEXT425,
NEXT435, NEXT440 and NEXT445 allowed one unknown per translated contact edge;
they therefore tested charge conservation and path constraints but not the
additional equality of topologically indistinguishable bond variables.

NEXT450 uses a deterministic one-dimensional Weisfeiler--Lehman equitable
partition as a conservative, automorphism-invariant definition of
topological indistinguishability. This is not claimed to recover every exact
automorphism orbit: it can tie more edges, making the hypothesis strictly
testable and reproducible without a geometry tolerance. P3 Hawthorne collapses
to a simple graph and reports least-squares/NNLS residuals but does not enforce
periodic multiedge class equality. NEXT435 optimizes the same positive margin
without class ties. NEXT190 audits Euclidean symmetry-recovery features rather
than a formal-valence quotient transport.

Rejected before any NEXT450 value: exact automorphism enumeration, whose
runtime explodes on highly symmetric periodic multigraphs; space-group orbits,
which import Euclidean tolerance and are not Hawthorne's topological
equivalence; and a fitted relaxation of class equality.

## 2. Hard no-DFT boundary

The executable formula may read only deterministic NEXT19 formal valences and
one raw initial unrelaxed fully periodic geometry. Geometry is used only to
construct the unchanged opposite-sign NEXT19 Voronoi multigraph. It must not
run/read DFT or energy/force/stress, use a learned proxy/MLIP/potential, relax,
read a trajectory/later geometry/same-composition alternative, or access
validation/replication. Discovery outcomes may be offline labels only after a
successful frozen full label-free build. Canonical documents remain untouched.

## 3. Frozen EBCTM formula

Let `q_i` be neutral nonzero formal charges and orient each periodic edge from
cation to anion. Initialize every site color with

```text
(sign(q_i), |q_i|/max_j |q_j|).
```

Repeatedly replace it by the canonical rank of

```text
(current_color_i, sorted multiset of neighboring current colors),
```

retaining periodic edge multiplicity, until the equitable partition stops
refining. Define an edge class by its ordered endpoint color pair. All edges
in one class `k` share strength `y_k`.

Let `A_ik` be the number of class-`k` edges incident to site `i`,
`b_i=|q_i|`, `CN_i` its translated degree, and freeze the positive class
reference

```text
r_k = sqrt((b_c/CN_c)(b_a/CN_a))
```

for any member edge (equitable refinement makes it constant within class;
this is verified). Solve

```text
lambda* = max lambda
subject to A y = b,
           y_k >= lambda r_k,
           y_k >= 0, lambda >= 0,
EBCTM(x0) = round_1e-10(lambda*/(1+lambda*)).
```

The sole feature is `ebctm_equitable_bond_class_transport_margin`, direction
`protected_high`. A proven infeasible quotient, isolated charged site, or
absent graph is a supported zero. Other failures fail closed. Independently
verify equality, lower-bound, stable-partition and within-class reference
residuals at `1e-8`.

Positive charge scaling, edge order, disjoint replication, rigid motion, site
permutation, unimodular rebasing and exact supercells must leave the feature
unchanged within `1e-8`. No alternate colors, conductance, graph, cutoff,
regularizer, equality slack, reference, transform, direction or companion
feature is available.

## 4. Frozen blind and conditional gates

Use the unchanged deterministic 80+80 discovery probes, all 32 prior formal
families, and recomputed ZBVVG, BECNS, SSSP, OBS, P4BSS, APRBS, ECSLO, PVTM,
PCABP and PCABSM controls. Per source require support `>=72/80`, `[0,1]`, at
least 20 distinct values at `1e-10`, invariance error `<=1e-8`, and maximum
adequate absolute Spearman `<0.90` with at least 40 joint finite rows.

Only if all pass: NEXT451 requires full discovery coverage `>=0.95` in both
sources; NEXT452 applies the unchanged NEXT224/NEXT413 rejected-extreme,
five-fold and source AUC/coverage gates; NEXT453 reuses the frozen bounded
width/amplitude grid only after a two-source pass; NEXT454 runs only for an
AUC+SAFE12 candidate missing BROAD. Validation and replication remain sealed.

## 5. Test and artifact order

1. Add RED tests for stable colors, analytic quotient feasibility, a quotient
   obstruction, scaling/order/replication invariance and the geometry firewall.
2. Implement the pure quotient LP and periodic wrapper.
3. Run the frozen 80+80 label-blind probe with all prior controls.
4. Continue mechanically only if gates authorize it.
5. Append the independent report and run focused/full verification.
