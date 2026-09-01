# CCLAB Conservative Domain-Extension Plan

> **Status:** post-coverage, label-blind extension. NEXT491 had already shown
> `0.961915` SCIGEN and `0.932339` WyFormer support, so this branch must never
> be presented as pre-specified evidence for CCLAB's outcome performance.

**Goal:** Preserve NEXT490 CCLAB exactly wherever every donating cation has a
published Appendix 3 state, while assigning a conservative lower-bound score
when an anion neighbourhood contains an element absent from that table.

## 1. Frozen CCLAB-CDE formula

Keep NEXT490's composition-only charge assignment, initial opposite-sign
periodic Voronoi graph, coordination-selected acidity intervals and anion
demand. For anion `j`:

- if every incident cation is in Appendix 3, compute NEXT490's projected
  received acidity `r_j` unchanged;
- if at least one incident cation is absent, set `r_j=0`, which makes that
  anion's normalized mismatch exactly one.

Aggregate with the unchanged formula

```text
D = sum_j |r_j-|q_j|| / sum_j (r_j+|q_j|),
CCLAB-CDE(x0) = round_1e-10(1-D).
```

Thus CCLAB-CDE is exactly equal to CCLAB on their shared applicability domain,
and missing public constants can only add the maximum local penalty. It does
not impute, interpolate or fit a Lewis acidity. Valence-inference failures,
zero charges, isolated sites and malformed geometry remain unsupported.

## 2. Frozen authorization gates

Before labels are opened, require on the unchanged 80+80 discovery probe:
support `>=72/80`, range `[0,1]`, at least 20 distinct values and invariance
error `<=1e-8` in each source. Also require exact equality within `1e-10` to
NEXT490 on every row where NEXT490 is supported, and no finite extended value
may exceed its own optimistic variant that simply ignores unknown cation
contacts.

Only then build the complete discovery tables. Require coverage `>=0.95` in
both sources. This extension inherits NEXT490's label-blind novelty certificate
but receives no new novelty claim because it was motivated by the observed
coverage failure. Only a passing coverage certificate may authorize NEXT497's
pre-written discovery outcome audit. Validation and replication stay sealed.

## 3. Hard boundary

Inputs remain composition plus one raw initial unrelaxed periodic geometry and
fixed public Appendix 3 constants. Do not run/read DFT calculations or values,
energy/force/stress, learned proxies, MLIPs/potentials, relaxation,
trajectories, later geometry, same-composition alternatives, validation or
replication. Outcome labels remain unopened until the full coverage gate has
passed and the NEXT497 audit protocol and code hashes have been frozen.
