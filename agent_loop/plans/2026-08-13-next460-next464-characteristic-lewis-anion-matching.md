# Characteristic Lewis Anion Matching Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> script/result and all canonical documents. Keep validation and replication
> sealed.

**Goal:** Test Hawthorne's bond-strength-matching rule with fixed experimental
characteristic Lewis acidities, so an unusual candidate coordination number
cannot make its own Pauling strength self-consistent by construction.

**Architecture:** NEXT460 defines one bounded local anion-matching feature on
the unchanged NEXT19 graph and a frozen CC-BY characteristic-acidity table
from Hawthorne 2026 Appendix 3. NEXT461--NEXT464 are conditional full
label-free build, cross-source discovery audit, bounded formula search, and
BROAD residual diagnostic.

## 1. Scientific question and distinction

Hawthorne defines cation characteristic Lewis acidity as formal charge divided
by the experimentally weighted grand-mean coordination number across ICSD ion
configurations. Appendix 3 tabulates these values for most cation valence
states. The paper states that this is the most probable prior when structural
details are unknown, while individual structures may adjust coordination to
optimize Lewis acid/base matching (<https://doi.org/10.1180/mgm.2026.10215>).

P9/P9c and Pauling/NEXT19 use the candidate's own current `|q|/CN`; NEXT19
then measures anion mismatch and optimizes reallocation. They cannot isolate
deviation from an element/oxidation state's empirical characteristic CN.
NEXT445 compares a topology-only solved path field with the same current-CN
reference. NEXT460 instead uses a fixed `(element, oxidation)->acidity` prior
that does not change when the candidate changes coordination.

The Appendix 3 constants derive from observed structures refined since 1975,
not DFT. They are used as a fixed public empirical lookup analogous to
electronegativity, covalent-radius and bond-valence tables already present in
the repository. The article is CC BY 4.0. H+ is omitted prospectively because
the source gives two starred characteristic acidities rather than one unique
value. Unlisted ion states are unsupported; no interpolation or nearest-state
fallback is allowed.

## 2. Hard no-DFT boundary

The executable formula may read only deterministic NEXT19 formal valences,
element identities, one raw initial unrelaxed fully periodic geometry, and the
frozen Appendix 3 lookup. Geometry is used only for the unchanged
opposite-sign NEXT19 Voronoi multigraph. It must not run/read DFT or
energy/force/stress, use a learned proxy/MLIP/potential, relax, read a
trajectory/later geometry/same-composition alternative, or access
validation/replication. Discovery outcomes may be offline labels only after a
successful frozen full label-free build. Canonical documents remain untouched.

## 3. Frozen CLAM formula

For cation site `c`, set `l_c` to its tabulated characteristic Lewis acidity.
For every anion `a`, retaining translated contact multiplicity, define

```text
R_a = sum_(e=(c,a,image)) l_c,
b_a = |q_a|,
M = sum_a |R_a-b_a| / sum_a (R_a+b_a),
CLAM(x0) = round_1e-10(1-M).
```

The sole feature is `clam_characteristic_lewis_anion_matching`, direction
`protected_high`. It lies in `[0,1]`, equals one exactly when every anion's
incident characteristic acidities sum to its formal charge, and penalizes
both under- and over-bonding. Missing lookup values, ambiguous H+, malformed
inputs, isolated charged sites or absent opposite-sign contacts are
unsupported; they are not physical zero values because the empirical prior is
unavailable or the local matching population is undefined.

Edge order, disjoint exact replication, rigid motion, translation, site
permutation, unimodular rebasing and exact supercells must leave the feature
unchanged within `1e-8`. No current-CN correction, fitted scale, element
subset, oxidation fallback, epsilon, sign split, quantile, graph, cutoff,
transform, direction or companion feature is available.

## 4. Frozen table checks

The asset records element, positive integer oxidation state, characteristic
coordination number, characteristic Lewis acidity, article DOI and CC-BY-4.0
attribution. After prospectively excluding ambiguous H+, require exactly 134
unique keys and finite positive values. Preserve printed values rather than
recomputing them; use `0.06 e` only as a transcription check because the
printed Cl3+ row differs from `oxidation/characteristic_CN` by `0.05 e`.
Freeze the asset SHA-256 into every probe/formal manifest.

## 5. Frozen blind and conditional gates

Use the unchanged deterministic 80+80 discovery probes, all 32 prior formal
families, and recomputed ZBVVG, BECNS, SSSP, OBS, P4BSS, APRBS, ECSLO, PVTM,
PCABP, PCABSM and PFPU controls. Per source require support `>=72/80`, `[0,1]`,
at least 20 distinct values at `1e-10`, invariance error `<=1e-8`, and maximum
adequate absolute Spearman `<0.90` with at least 40 joint finite rows.

Only if all pass: NEXT461 requires full discovery coverage `>=0.95` in both
sources; NEXT462 applies the unchanged NEXT224/NEXT413 rejected-extreme,
five-fold and source AUC/coverage gates; NEXT463 reuses the frozen bounded
width/amplitude grid only after a two-source pass; NEXT464 runs only for an
AUC+SAFE12 candidate missing BROAD. Validation and replication remain sealed.

## 6. Test and artifact order

1. Add the attributed frozen lookup and RED table/kernel/invariance/firewall
   tests.
2. Implement the pure local-matching kernel and periodic wrapper.
3. Run the frozen 80+80 label-blind probe with every stated control.
4. Continue mechanically only if gates authorize it.
5. Append the independent report and run focused/full verification.
