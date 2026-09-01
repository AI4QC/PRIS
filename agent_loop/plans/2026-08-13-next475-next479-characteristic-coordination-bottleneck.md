# Characteristic Coordination Bottleneck Implementation Plan

> **For Codex:** Execute additively and test first. Preserve NEXT470/NEXT471
> and all earlier artifacts. Keep validation and replication sealed.

**Goal:** Test the local, worst-environment form of Hawthorne's experimental
characteristic-coordination prior while treating table incompleteness
conservatively instead of making the whole material unsupported.

**Architecture:** NEXT475 computes a per-cation compatibility with the nearest
characteristic CN printed for that element and takes the minimum across sites.
NEXT476--NEXT479 are conditional full label-free build, cross-source discovery
audit, bounded formula search, and BROAD residual diagnostic.

## 1. Scientific distinction and table policy

NEXT470 pooled all cation deviations into one ratio and failed the frozen full
WyFormer coverage gate (`0.932339 < 0.95`) because one element absent from
Appendix 3 invalidated the entire structure. NEXT475 instead encodes the local
rule that one severely incompatible coordination environment is sufficient to
limit structural plausibility. For each element retain the exact finite set of
characteristic CN values already transcribed from Appendix 3. Add only the
unambiguous H+ characteristic CN `2.03` printed in the omitted H row; neither
of the two starred H acidity values is used. For any other element absent from
the source table, define site compatibility as conservative zero. This is not
imputation: no characteristic CN is invented for that element.

The element sets and H constant are frozen before the new probe. No
interpolation, radius, oxidation-state magnitude, electronegativity magnitude,
element subset, fitted tolerance, quantile or fallback is allowed.

## 2. Hard no-DFT boundary

The formula reads only element identities, the sign of NEXT19's deterministic
composition charge assignment, one raw initial unrelaxed fully periodic
geometry, the unchanged opposite-sign Voronoi contact multigraph, and the
published characteristic CN constants. It must not run/read DFT or energy,
force, stress, a learned proxy/MLIP/potential, relaxation, trajectory, later
geometry, same-composition alternative, validation or replication.

## 3. Frozen CCCB formula

For positive site `c`, let `CN_c` be translated opposite-sign contact
multiplicity. If its element has a characteristic set `T_E`, choose the nearest
`t_c` and define

```text
k_c = 1 - |CN_c-t_c|/(CN_c+t_c)
    = 2 min(CN_c,t_c)/(CN_c+t_c).
```

For an element absent from the table define `k_c=0`. Then

```text
CCCB(x0) = round_1e-10(min_c k_c).
```

The sole feature is `cccb_characteristic_coordination_bottleneck`, direction
`protected_high`, bounded `[0,1]`. Malformed contacts, zero-charge sites, or an
isolated charged site are unsupported; the standard absent opposite-sign graph
is supported physical zero. Edge order, disjoint exact replication, rigid
motion, translation, site permutation, unimodular rebasing and exact
supercells must be invariant within `1e-8`.

## 4. Frozen ordered blind gates

Use the unchanged deterministic 80+80 discovery probes. First require support
`>=72/80` per source. Only after support passes, compare with all 32 prior
formal families, recomputed ZBVVG through PFPU, sparse CLAM/MV-CLAM, and
recomputed ECCC. Require `[0,1]`, at least 20 values distinct at `1e-10`,
invariance error `<=1e-8`, and maximum adequate absolute Spearman `<0.90` with
at least 40 joint finite rows.

Only if all pass: NEXT476 requires full discovery coverage `>=0.95` in both
sources; NEXT477 applies the unchanged NEXT224/NEXT413 rejected-extreme,
five-fold and source AUC/coverage gates; NEXT478 reuses the frozen bounded
width/amplitude grid only after a two-source pass; NEXT479 runs only for an
AUC+SAFE12 candidate missing BROAD. Validation and replication remain sealed.

## 5. Artifact order

1. Add RED kernel/invariance/firewall tests.
2. Implement the independent kernel and raw-periodic wrapper.
3. Run ordered support and full novelty probes.
4. Continue mechanically only if every gate authorizes it.
5. Append the independent report and run focused/full verification.
