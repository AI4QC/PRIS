# Path-Field Participation Uniformity Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> script/result and all canonical documents. Keep validation and replication
> sealed.

**Goal:** Measure whether Hawthorne's unique zero-loop-sum a-priori bond field
spreads formal charge broadly across the available contact topology rather
than concentrating it on a small subset of bonds.

**Architecture:** NEXT455 defines one normalized effective-participation
feature on the frozen NEXT440 field. NEXT456--NEXT459 are conditional full
label-free build, cross-source discovery audit, bounded formula search, and
BROAD residual diagnostic.

## 1. Scientific question and prior-work boundary

NEXT440 established the unique unit-conductance path field but its sign-only
statistic saturated. NEXT445 compared that field with endpoint characteristic
strength and was redundant with PVTM. NEXT315 PCGR already reports the same
graph-Poisson field's quadratic dissipation and q90/maximum voltage drop.
NEXT367 PBVEU measures uniformity of a distance-derived bond-valence field,
not the topology-only a-priori field. P3 Hawthorne reports residuals and rank,
not how many bonds effectively participate.

NEXT450's proposed equality of equitable bond classes is analytically
redundant with PVTM: averaging any feasible edge flow over every
charge-preserving graph automorphism retains site conservation, positivity
and its uniform relative floor. Therefore NEXT450 is stopped before code or
data and is not used to justify NEXT455.

NEXT455 prospectively asks a distinct topological question: even when the
path field is positive and charge conserving, is its absolute strength spread
over most available bonds? No result from NEXT455 is known at freeze time.

## 2. Hard no-DFT boundary

The executable formula may read only deterministic NEXT19 formal valences and
one raw initial unrelaxed fully periodic geometry. Geometry is used only to
construct the unchanged opposite-sign NEXT19 Voronoi multigraph. It must not
run/read DFT or energy/force/stress, use a learned proxy/MLIP/potential, relax,
read a trajectory/later geometry/same-composition alternative, or access
validation/replication. Discovery outcomes may be offline labels only after a
successful frozen full label-free build. Canonical documents remain untouched.

## 3. Frozen PFPU formula

Use the exact NEXT440 orientation and unit-conductance zero-loop-sum field

```text
L = B B^T,       phi = L^+ q,       s = B^T phi.
```

For `m` translated edges define `w_e=|s_e|`,
`p_e=w_e/sum_f w_f`, Shannon entropy `H=-sum_(p_e>0) p_e log p_e`, and freeze

```text
PFPU(x0) = round_1e-10(exp(H)/m).
```

The sole feature is `pfpu_path_field_participation_uniformity`, direction
`protected_high`. It lies in `(0,1]`, equals one only when absolute
path-constrained strength is identical on every edge, and approaches the
fraction of effectively participating bonds for concentrated fields. A
component-charge obstruction, isolated charged site or absent graph is a
supported zero. Other failures fail closed.

Positive charge scaling, edge order, disjoint replication, rigid motion, site
permutation, unimodular rebasing and exact supercells must leave the feature
unchanged within `1e-8`. No Rényi order, epsilon, sign split, quantile,
reference field, graph, cutoff, conductance, transform, direction or companion
feature is available.

## 4. Frozen blind and conditional gates

Use the unchanged deterministic 80+80 discovery probes, all 32 prior formal
families, and recomputed ZBVVG, BECNS, SSSP, OBS, P4BSS, APRBS, ECSLO, PVTM,
PCABP and PCABSM controls. PCGR and PBVEU are already in the formal prior
families. Per source require support `>=72/80`, `[0,1]`, at least 20 distinct
values at `1e-10`, invariance error `<=1e-8`, and maximum adequate absolute
Spearman `<0.90` with at least 40 joint finite rows.

Only if all pass: NEXT456 requires full discovery coverage `>=0.95` in both
sources; NEXT457 applies the unchanged NEXT224/NEXT413 rejected-extreme,
five-fold and source AUC/coverage gates; NEXT458 reuses the frozen bounded
width/amplitude grid only after a two-source pass; NEXT459 runs only for an
AUC+SAFE12 candidate missing BROAD. Validation and replication remain sealed.

## 5. Test and artifact order

1. Add RED analytic entropy, obstruction, invariance and firewall tests.
2. Implement the pure kernel and periodic wrapper by reusing the immutable
   NEXT440 path field.
3. Run the frozen 80+80 label-blind probe with every stated control.
4. Continue mechanically only if gates authorize it.
5. Append the independent report and run focused/full verification.
