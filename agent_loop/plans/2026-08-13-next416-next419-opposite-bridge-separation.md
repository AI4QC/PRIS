# NEXT416--NEXT419 Opposite-Bridge Separation Design

**Goal:** Test one continuous, charge-sign-symmetric extension of the
edge/face-sharing crystal-chemical mechanism while preserving every existing
script and keeping validation and replication sealed.

**Strict boundary:** The executable descriptor may consume composition and one
raw, unrelaxed, fully periodic geometry only. It may infer deterministic charge
signs and construct the ordinary opposite-sign periodic Voronoi graph. It must
not read or execute DFT values or calculations, learned energies, forces or
stresses, model or proxy potentials, relaxation, trajectories, later geometry,
same-composition alternatives, outcomes, endpoints, validation or replication.

## Motivation and non-duplication

Pauling's third rule links shared edges and faces to shorter like-ion
separation and increased electrostatic repulsion. George et al. (2020) found
that known oxide coordination polyhedra are 63% corner-, 27% edge- and 10%
face-sharing, and that face sharing falls below 2% when cation coordination is
at most eight. Langer and Kohlmann (2026) likewise identify edge/face sharing
as a central ionic-structure rule.

The repository already contains the discrete Pauling-3 edge/face fraction and
several direction-tensor or shell-gap descriptors. NEXT416 must not recreate
those quantities. It instead asks a continuous, local triangle question on the
opposite-sign Voronoi graph: for two same-sign periodic neighbours joined
through a central opposite-sign site, how completely does the central ion
separate the like-charged pair?

## Frozen descriptor

Use NEXT19 only to infer charge signs and construct the ordinary periodic
opposite-sign Voronoi graph. For every directed cation-to-anion periodic edge,
construct its Cartesian vector. At each cation, collect its cation-to-anion
vectors. At each anion, align incident periodic images on that anion and
collect the corresponding anion-to-cation vectors. Thus both charge signs are
treated identically.

For every site `i` with at least two incident vectors and every unordered pair
`(v_ij,v_ik)`, define the bounded bridge separation

```text
b_ijk = ||v_ij-v_ik|| / (||v_ij||+||v_ik||),       0 <= b_ijk <= 1.
```

This is the triangle-inequality ratio for the two like-signed neighbours. It is
one only when the central opposite-sign ion lies on the straight segment
between them, and decreases as the like-signed neighbours approach the same
side of the center. For each eligible center use its minimum pair value,

```text
b_i = min_(j<k) b_ijk,
obs_opposite_bridge_separation_q10
    = round_1e-10(inverted-CDF q10_i b_i).
```

The only frozen direction is `protected_high`. There is no alternative graph,
pair weighting, charge-magnitude weighting, center-sign split, element subset,
angle cutoff, distance cutoff, quantile, transform, conjunction, exception or
second feature.

## Label-blind admission

Use the exact deterministic 80+80 discovery-x0 selection and all numeric
label-free controls used by NEXT411. Recompute ZBVVG, BECNS and SSSP as
additional controls. Before any outcome or endpoint is opened, both sources
must independently satisfy:

- support at least `72/80`;
- finite closed-domain values in `[0,1]`;
- at least 20 values unique after rounding to `1e-10`;
- maximum exact representation error at most `1e-10`;
- maximum absolute Spearman correlation strictly below `0.90` among controls
  with at least 40 jointly finite rows.

Failure closes the branch. No label, endpoint, formal build, alternate formula
or post-result repair is then authorized.

## Conditional stages

- **NEXT417:** If and only if all label-blind gates pass, build full physically
  isolated SCIGEN and WyFormer discovery feature tables. Require at least 0.95
  support in each source; do not read outcomes.
- **NEXT418:** If and only if NEXT417 passes, audit the single frozen
  `protected_high` hypothesis using the unchanged NEXT413/NEXT268 rejected-
  extreme cohort, reduced-formula folds, `(1/16,15/16)` inverted-CDF mapping
  and cross-source gates. Validation and replication remain sealed.
- **NEXT419:** If and only if NEXT418 authorizes search, mechanically reuse the
  established NEXT261 one-term triangular margin-local grammar. Do not add a
  new grid or operator after observing the audit.

## TDD and publication

1. Write failing kernel, monotonicity, malformed-input, rigid-motion,
   site-order, periodic-translation, unimodular-rebasing, supercell and
   geometry-firewall tests.
2. Implement NEXT416 and make only those tests pass.
3. Write failing probe tests for deterministic selection, novelty, gates,
   invariance and the absence of endpoint/validation/replication interfaces.
4. Run the label-blind probe and obey its authorization bit.
5. Publish every stage atomically with input/output/source SHA-256 identities.
6. Append the result to the independent report only. Do not edit canonical
   paper, README, notes, preregistration or prior content without user review.
