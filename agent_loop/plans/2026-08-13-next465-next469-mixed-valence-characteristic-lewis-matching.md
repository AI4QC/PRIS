# Mixed-Valence Characteristic Lewis Matching Implementation Plan

> **For Codex:** Execute additively and test first. Preserve NEXT460 and every
> earlier artifact. Keep validation and replication sealed.

**Goal:** Test whether Hawthorne's fixed characteristic Lewis acidities remain
useful for composition-determined mixed-valence materials by applying the
parameter-free lever rule between explicitly tabulated oxidation states.

**Architecture:** NEXT465 defines one bounded anion-matching feature on the
unchanged NEXT19 opposite-sign periodic Voronoi multigraph and the unchanged
NEXT460 Appendix 3 asset. NEXT466--NEXT469 are conditional full label-free
build, cross-source discovery audit, bounded formula search, and BROAD residual
diagnostic.

## 1. Prospective distinction from NEXT460

NEXT460 requires every positive site charge to be an exactly tabulated positive
integer oxidation state and therefore failed blind support (`2/80` SCIGEN,
`14/80` WyFormer). The dominant failure was not a fitted endpoint: NEXT19's
pre-existing composition-only fallback represents mixed-valence populations by
one element-average fractional charge, for example Fe3O4 as Fe^(8/3)+.

For a fractional positive charge `z` of element `E`, let `z_lo < z < z_hi` be
the closest lower and upper oxidation states explicitly printed for `E` in the
frozen Hawthorne table. Define the mixed-valence characteristic acidity by the
lever rule

```text
l(E,z) = ((z_hi-z)/(z_hi-z_lo)) l(E,z_lo)
       + ((z-z_lo)/(z_hi-z_lo)) l(E,z_hi).
```

Exact tabulated integer states use their printed value. No extrapolation,
nearest-state substitution, free mixture weight, interpolation across elements,
or invented state is allowed. Integer states missing from the table remain
unsupported. The wrapper rejects NEXT19's electronegativity-partition policy;
only formal/composition-charge assignments can invoke the lever rule.

## 2. Hard no-DFT boundary

The formula reads only element identities, NEXT19's deterministic composition
charge assignment, one raw initial unrelaxed fully periodic geometry, and the
unchanged frozen Appendix 3 constants. It must not run/read DFT or energy,
force, stress, learned proxy, MLIP, potential, relaxation, trajectory, later
geometry, same-composition alternative, validation or replication. Outcomes may
be opened only after a successful full label-free build. Canonical documents
remain untouched.

## 3. Frozen MV-CLAM formula

For each cation use `l_c=l(E_c,q_c)` above. Retaining translated contact
multiplicity, for every anion define

```text
R_a = sum_(e=(c,a,image)) l_c,
b_a = |q_a|,
M = sum_a |R_a-b_a| / sum_a (R_a+b_a),
MVCLAM(x0) = round_1e-10(1-M).
```

The sole feature is `mvclam_mixed_valence_characteristic_lewis_matching`,
direction `protected_high`. Missing brackets, ambiguous H+, zero-charge sites,
isolated charged sites, malformed contacts and absent opposite-sign graphs are
unsupported except that the repository-standard absent-graph case remains a
supported physical zero. Edge order, disjoint exact replication, rigid motion,
translation, site permutation, unimodular rebasing and exact supercells must be
invariant within `1e-8`.

## 4. Frozen blind and conditional gates

Use the unchanged deterministic 80+80 probes. Evaluate the frozen support gate
first and stop immediately if either source has support `<72/80`; a failed
candidate cannot reach any later gate. Only after support passes, load every 32
prior formal families, recomputed ZBVVG through PFPU, and recomputed sparse
CLAM. Then require bounded `[0,1]`, at least 20 values distinct at `1e-10`,
invariance error `<=1e-8`, and maximum adequate absolute Spearman `<0.90` with
at least 40 joint finite rows. The ordered short circuit changes neither the
cohort nor any threshold and avoids needless feature access after a decisive
support failure.

Only if all pass: NEXT466 requires full discovery coverage `>=0.95` in both
sources; NEXT467 applies the unchanged NEXT224/NEXT413 rejected-extreme,
five-fold and source AUC/coverage gates; NEXT468 reuses the frozen bounded
width/amplitude grid only after a two-source pass; NEXT469 runs only for an
AUC+SAFE12 candidate missing BROAD. Validation and replication remain sealed.

## 5. Artifact order

1. Add RED lever-rule, kernel, invariance and boundary tests.
2. Implement the independent kernel/wrapper without changing NEXT460.
3. Run the frozen label-blind probe with every stated control.
4. Continue mechanically only if gates authorize it.
5. Append the independent report and run focused/full verification.
