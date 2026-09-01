# Path-Constrained Bond-Strength Matching Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> script/result and all canonical documents. Keep validation and replication
> sealed.

**Goal:** Measure how closely Hawthorne's unique path-constrained a-priori
bond field matches the characteristic formal-valence strength implied by its
two endpoints, using only composition and raw periodic contact topology.

**Architecture:** NEXT445 defines one bounded no-parameter matching feature on
the frozen NEXT440 field and unchanged NEXT19 graph. NEXT446--NEXT449 are
conditional full label-free build, cross-source discovery audit, bounded
formula search, and BROAD residual diagnostic.

## 1. Frozen motivation and distinction

NEXT440 showed that the unit-conductance, charge-conserving, zero-loop-sum
a-priori field is label-blind novel but its sign-only positivity statistic is
too saturated: the deterministic probes had only 16 SCIGEN and 8 WyFormer
values distinct at `1e-10`. This outcome authorizes no retuning of NEXT440.

Hawthorne's separate bond-strength-matching rule states that stable structures
form when cation Lewis acidity and anion Lewis basicity match. NEXT445 freezes
an independent hypothesis before computing any NEXT445 value: compare every
path-constrained edge strength with the geometric mean of the two endpoint
characteristic strengths. Unlike NEXT425/ECSLO, it uses no bond length and no
length ordering. Unlike PVTM, it evaluates the unique zero-loop-sum field
rather than optimizing over all positive transports. Unlike NEXT440, positive
but over- or under-loaded edges remain informative.

## 2. Hard no-DFT boundary

The executable formula may read only deterministic NEXT19 formal valences and
one raw initial unrelaxed fully periodic geometry. Geometry is used only to
construct the unchanged opposite-sign NEXT19 Voronoi multigraph. It must not
run or read DFT; use energy/force/stress, a learned proxy, MLIP or potential;
relax; read a trajectory, later geometry, same-composition alternative,
validation or replication. Discovery outcomes may be offline labels only
after a successful full label-free build. Canonical documents remain
untouched.

## 3. Frozen PCABSM formula

Use the exact NEXT440 orientation, unit-conductance Laplacian and unique field

```text
B s = q,              s in row(B^T).
```

For site `i`, let `a_i=|q_i|/CN_i`, where `CN_i` counts translated incident
opposite-sign contacts. For edge `e=(c,a,image)`, freeze

```text
r_e = sqrt(a_c a_a),
M = sum_e |s_e-r_e| / sum_e (|s_e|+r_e),
PCABSM(x0) = round_1e-10(1-M).
```

The sole feature is `pcabsm_path_constrained_bond_strength_matching`, with
direction `protected_high`. It is one only for exact endpoint matching;
negative path-field edges receive their maximal per-edge mismatch
automatically. A component-charge obstruction, isolated charged site or
absent graph is a supported physical zero. Other failures fail closed.

The normalization makes the feature invariant to positive global charge
scaling and exact disjoint replication. Edge order, rigid motions, site
permutation, unimodular rebasing and exact supercells must agree within
`1e-8`. There is no epsilon, fitted scale, alternate endpoint mean,
conductance, graph, cutoff, norm, direction, transform or companion feature.

## 4. Frozen blind and conditional gates

Use the unchanged deterministic 80+80 discovery probes, all 32 formal prior
families, and recomputed ZBVVG, BECNS, SSSP, OBS, P4BSS, APRBS, ECSLO, PVTM
and PCABP controls. Per source require support `>=72/80`, `[0,1]`, at least 20
distinct values at `1e-10`, invariance error `<=1e-8`, and maximum adequate
absolute Spearman `<0.90` with at least 40 joint finite rows.

Only if all pass: NEXT446 requires full discovery coverage `>=0.95` in both
sources; NEXT447 applies the unchanged NEXT224/NEXT413 rejected-extreme,
five-fold and source AUC/coverage gates; NEXT448 reuses the frozen bounded
width/amplitude grid only after a two-source pass; NEXT449 runs only for an
AUC+SAFE12 candidate missing BROAD. Validation and replication remain sealed.

## 5. Test and artifact order

1. Add RED analytic/invariance/firewall tests.
2. Implement the pure kernel and periodic wrapper by reusing the frozen
   NEXT440 field without changing it.
3. Run the frozen 80+80 label-blind probe, explicitly adding PCABP to novelty.
4. Continue mechanically only if gates authorize it.
5. Append the independent report and run focused/full verification.
