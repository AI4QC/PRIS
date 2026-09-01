# Coordination-Conditioned Lewis-Acidity Balance Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> artifact. Keep validation and replication sealed.

**Goal:** Test whether Hawthorne's characteristic coordination numbers can
resolve oxidation-state ambiguity well enough to turn the Appendix 3 Lewis
acidities into a broad, local anion bond-strength balance on one raw initial
periodic geometry.

**Architecture:** NEXT490 uses the unchanged opposite-sign periodic Voronoi
multigraph. Each cation's observed contact coordination selects the nearest
published characteristic-CN state or tied state set for that element. The
selected Lewis acidity interval is delivered along each incident contact and
compared with the receiving anion's composition-inferred formal demand.
NEXT491--NEXT494 are conditional full label-free build, discovery outcome
audit, bounded search and BROAD diagnostic.

## 1. Frozen state selection

Use the already attributed 134-row non-H Appendix 3 table without changing a
value. For cation site `i`, let `c_i` be its translated opposite-sign contact
count and let `S_E={(q,t,a)}` contain the printed oxidation state, characteristic
coordination `t`, and Lewis acidity `a` for element `E`. Select all states
attaining

```text
d_i = min_(q,t,a in S_E) |c_i - t|.
```

The site acidity uncertainty interval is the minimum and maximum printed
`a` among those tied nearest states. This retains genuinely unresolved states
such as Mn at characteristic CN 4 instead of choosing one after seeing data.
It does not interpolate, extrapolate, fit, use electronegativity or use the
inferred cation charge magnitude to break ties. An element absent from the
public table is unsupported.

## 2. Frozen CCLAB formula

For anion `j`, sum the lower and upper cation acidities along every translated
incident edge to obtain `[L_j,U_j]`. Let `b_j=|q_j|` be its formal bond-strength
demand inferred from composition. Project the demand onto the closed interval,

```text
r_j = min(max(b_j,L_j),U_j),
D = sum_j |r_j-b_j| / sum_j (r_j+b_j),
CCLAB(x0) = round_1e-10(1-D).
```

The sole feature is
`cclab_coordination_conditioned_lewis_acidity_balance`, direction
`protected_high`, range `[0,1]`. Every site must have a nonzero, neutral formal
charge assignment and at least one opposite-sign contact. A valid structure
with no opposite-sign periodic neighbour receives supported physical zero;
malformed or isolated charged populations fail closed.

## 3. Hard no-DFT boundary and invariance

The executable reads only composition, one raw initial unrelaxed periodic
geometry, the deterministic composition-only formal-valence assignment, the
translated opposite-sign Voronoi topology and the fixed public Appendix 3
constants. It must not run or read DFT; energy, force or stress; learned
proxies; MLIPs or potentials; relaxation; trajectories; later geometry;
same-composition alternatives; validation; replication; or outcome labels.
Edge order, disjoint exact replication, rigid motion, translation, site
permutation, unimodular rebasing and exact supercells must be invariant within
`1e-8`.

## 4. Frozen ordered blind gates

Use the unchanged 80+80 discovery probes. Before opening any prior feature
table, require support `>=72/80`, `[0,1]`, at least 20 values distinct at
`1e-10`, and invariance error `<=1e-8` in each source. Only if all four
engineering gates pass, compare with all 32 prior formal families, recomputed
ZBVVG through PFPU, CLAM, MV-CLAM, ECCC, CCCB, SBCC and CACC, requiring maximum
adequate absolute Spearman `<0.90` with at least 40 joint finite rows. The
failed, all-zero CACC probe is retained as an explicit sparse control but
cannot authorize or veto this branch.

Only if all pass: NEXT491 requires full discovery coverage `>=0.95` per source;
NEXT492 applies the unchanged NEXT224/NEXT413 outcome gates; NEXT493/NEXT494
remain conditional. Validation and replication stay sealed.

## 5. Artifact order

1. Add RED state-selection, interval-balance, invariance and firewall tests.
2. Implement an independent pure core and raw-periodic wrapper.
3. Run the ordered engineering probe, then the full novelty probe only if
   mechanically authorized.
4. Continue only on frozen gate authorization.
5. Update the independent report and rerun complete verification.
