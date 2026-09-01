# Element Characteristic Coordination Compatibility Implementation Plan

> **For Codex:** Execute additively and test first. Preserve NEXT460/NEXT465
> and all earlier artifacts. Keep validation and replication sealed.

**Goal:** Test Hawthorne's experimental characteristic-coordination prior
without requiring an absolute oxidation-state scale, so it remains defined for
the composition-only electronegativity partition used by most SCIGEN probes.

**Architecture:** NEXT470 compares every cation's raw opposite-sign Voronoi
coordination number with the nearest characteristic CN explicitly tabulated for
that element in Hawthorne 2026 Appendix 3. NEXT471--NEXT474 are conditional full
label-free build, cross-source discovery audit, bounded formula search, and
BROAD residual diagnostic.

## 1. Scientific distinction

For each element `E`, freeze the finite set `T_E` of characteristic
coordination numbers printed for all oxidation states of `E` in Appendix 3.
This asks whether the candidate's local coordination is compatible with any
experimentally characteristic state of that element. It does not infer which
oxidation state is present and does not use the printed Lewis-acidity values.
It is distinct from generic CN means/spans and Beck effective-CN simplicity:
the reference set is element-specific and external to the candidate geometry.

H+ remains absent because its characteristic acidity is ambiguous, although
its printed CN is single-valued; retaining the same prospectively frozen
NEXT460 row universe prevents a post-failure asset change. A positive-site
element absent from the table is unsupported. No interpolation, radius,
electronegativity magnitude, element subset, fitted tolerance or fallback is
allowed.

## 2. Hard no-DFT boundary

The executable formula reads only element identities, the sign of NEXT19's
deterministic composition charge assignment, one raw initial unrelaxed fully
periodic geometry, the unchanged opposite-sign Voronoi contact multigraph, and
the unchanged frozen Appendix 3 table. It must not run/read DFT or energy,
force, stress, a learned proxy/MLIP/potential, relaxation, trajectory, later
geometry, same-composition alternative, validation or replication.

## 3. Frozen ECCC formula

For each positive site `c`, let `CN_c` be its translated opposite-sign contact
multiplicity and let

```text
t_c = argmin_(t in T_element(c)) |CN_c-t|,
D = sum_c |CN_c-t_c| / sum_c (CN_c+t_c),
ECCC(x0) = round_1e-10(1-D).
```

Only the distance matters if the nearest value ties. The sole feature is
`eccc_element_characteristic_coordination_compatibility`, direction
`protected_high`, bounded `[0,1]`. Malformed contacts, a missing element table,
or an isolated cation is unsupported; the repository-standard absent
opposite-sign graph is supported physical zero. Edge order, disjoint exact
replication, rigid motion, translation, site permutation, unimodular rebasing
and exact supercells must be invariant within `1e-8`.

## 4. Frozen ordered blind gates

Use the unchanged deterministic 80+80 discovery probes. First require support
`>=72/80` per source. Only if that passes, load all 32 prior formal families and
recomputed ZBVVG, BECNS, SSSP, OBS, P4BSS, APRBS, ECSLO, PVTM, PCABP, PCABSM,
PFPU, CLAM and MV-CLAM controls; require `[0,1]`, at least 20 values distinct at
`1e-10`, invariance error `<=1e-8`, and maximum adequate absolute Spearman
`<0.90` with at least 40 joint finite rows.

Only if all pass: NEXT471 requires full discovery coverage `>=0.95` in both
sources; NEXT472 applies the unchanged NEXT224/NEXT413 rejected-extreme,
five-fold and source AUC/coverage gates; NEXT473 reuses the frozen bounded
width/amplitude grid only after a two-source pass; NEXT474 runs only for an
AUC+SAFE12 candidate missing BROAD. Validation and replication remain sealed.

## 5. Artifact order

1. Add RED table/kernel/invariance/firewall tests.
2. Implement the independent kernel and raw-periodic wrapper.
3. Run support first, then the full novelty probe only if authorized.
4. Continue mechanically only if every gate authorizes it.
5. Append the independent report and run focused/full verification.
