# Pauling-4 Bond-Strength Segregation Implementation Plan

> **For Codex:** Execute this plan additively with test-driven development.
> Do not modify or replace canonical paper/report sources, and do not open
> validation or replication data without a separately authorized freeze.

**Goal:** Test whether high electrostatic-valence cation bonds avoid meeting at
the same anion in one raw periodic geometry, as a continuous and fully DFT-free
form of Pauling's fourth rule.

**Architecture:** NEXT420 defines one scale-free graph statistic on the frozen
NEXT19 opposite-sign periodic Voronoi contacts, then subjects it to the same
80+80 label-blind support, invariance, nondegeneracy, and novelty gates used by
NEXT411 and NEXT416. NEXT421--NEXT424 are conditional: full label-free build,
cross-source offline discovery audit, the unchanged bounded margin-local
formula grammar, and the unchanged BROAD residual diagnostic. Each later stage
is forbidden unless the preceding manifest explicitly authorizes it.

**Tech stack:** Python 3.11, NumPy, pandas/Parquet, ASE, pymatgen, the existing
NEXT19/NEXT267/NEXT295 geometry and provenance guards, pytest.

## 1. Scientific question and alternatives

Pauling's fourth rule says that, in crystals with different cations, cations
with large valence and small coordination number tend not to share elements of
their coordination polyhedra. George et al.'s large-scale assessment
(<https://doi.org/10.1002/anie.202000829>) found that coordination number was
the stronger part of this rule and that a binary violation flag was not
universal. That result argues for testing a continuous association statistic,
not silently treating the historical Boolean rule as exact.

Three label-free candidates were compared before computing a NEXT420 value:

1. A new local charge-residual statistic was rejected because NEXT319 already
   measures one- and two-shell charge neutralization.
2. The maximum sum of two Pauling bond strengths at an anion was rejected
   because it is a one-sided subset of the existing Pauling-2 total bond-sum
   residual.
3. The selected statistic asks whether large bond-strength contact stubs meet
   the same anion more often than expected from the structure-wide stub mean.
   It preserves the chemical statement of rule 4 while separating pair
   association from total anion charge balance.

This branch is exploratory. Neither the cited paper nor this design asserts a
universal stability theorem. A feature that fails label-blind novelty or
cross-source discovery gates is retained as a negative result and is not tuned
afterward.

## 2. Frozen information boundary

The executable feature may read only element identities, the deterministic
NEXT19 formal-valence assignment, and one initial unrelaxed fully periodic
geometry. It may use the unchanged NEXT19 opposite-sign Voronoi contact graph.

It must not execute DFT; read a DFT energy, force, stress, hull value, label, or
relaxed geometry; use an ML energy/force/stress proxy or potential; perform a
relaxation; read a trajectory or same-composition alternative; or access
validation/replication data. Discovery outcomes are permitted only as offline
labels after a successful full label-free build. Canonical `paper/`, `tex/`,
`notes/`, `README.md`, and `PREREG.md` remain untouched pending user review.

## 3. Frozen P4BSS formula

Let `q_i` be the nonzero neutral formal charge of site `i`. On the NEXT19
periodic opposite-sign Voronoi graph, let `CN_i` be the number of translated
opposite-sign contacts incident to cation site `i`. Each cation contact stub
`e` receives the classical Pauling electrostatic bond strength

`s_e = |q_i| / CN_i`.

For every reference-cell anion, rebase all incident cation images to that
anion and enumerate every unordered pair of distinct incident contact stubs.
Let `P` be the resulting periodic population of co-anion stub pairs. Freeze

`E = (mean_e s_e)^2`,

`O = mean_(e,f in P) (s_e s_f)`,

`A = E / (E + O)`.

`E` is the product expected for two independently sampled contact stubs;
`O` is their observed product conditional on sharing an anion. If large
bond-strength cations avoid sharing anions, `O < E` and `A > 1/2`; clustering
gives `A < 1/2`; uniform bond strengths give exactly `A = 1/2`. Because both
moments are quadratic, global charge rescaling cancels. Repeating a cell
duplicates both populations and leaves the formula unchanged.

Freeze exactly one feature:

`p4bss_bond_strength_pair_avoidance = round_1e-10(A)`.

Its sole direction is `protected_high`. Require finite nonzero charges, exact
charge neutrality, both charge signs, at least one contact per site, exact
cation-anion edge orientation, and at least one anion with two incident
translated contacts. Collapse no distinct periodic image: two images are two
coordination polyhedra and therefore a valid sharing pair. Input duplicates,
malformed indices, zero strengths, or a value outside `[0,1]` fail closed.
There is no alternate quantile, species subset, cation threshold, anion charge
normalization, graph, cutoff, transform, direction, or companion feature.

## 4. Frozen label-blind gates

Use the unchanged deterministic 80-record discovery probe in each of SCIGEN
and WyFormer, selected only from `(natoms, chemical_system, material_id)`.
Open only discovery `x0` geometry, base label-free features, all 32 formal
prior feature families, and recomputed ZBVVG, BECNS, SSSP, and OBS controls.
No endpoint or outcome field may be read.

For each source require all of:

- support at least `72/80`;
- every finite value in `[0,1]`;
- at least 20 distinct values after rounding to `1e-10`;
- maximum error at most `1e-8` over rigid rotation, translation, site
  permutation, unimodular rebasing, and exact `2 x 1 x 1` supercell
  representation;
- maximum absolute Spearman correlation strictly below `0.90` against every
  adequate prior label-free control on at least 40 joint finite records.

If any gate fails, record `next421_formal_build_authorized=false` and
`p4bss_branch_terminated=true`; do not create NEXT421--NEXT424 artifacts.

## 5. Conditional full loop

Only after every label-blind gate passes:

1. NEXT421 processes all 13,470 SCIGEN and 5,232 WyFormer discovery `x0`
   records in physical label isolation, publishes complete tables plus failure
   counts, and requires at least `0.95` support in each source.
2. NEXT422 reuses the unchanged NEXT224/NEXT413 rejected-extreme cohorts,
   reduced-formula five-fold split, inverted-CDF normalization, and source
   AUC/coverage gates for this sole frozen `protected_high` feature.
3. NEXT423 is authorized only if NEXT422 passes both sources. It inherits the
   exact NEXT261/NEXT414 width grid `(1/64,1/32,1/16,1/8,1/4,1/2,1)` and
   amplitude grid `(1/4,1/2,1)`, with no new tuning.
4. NEXT424 runs only for candidates that pass source AUC and all SAFE12 cells
   but miss BROAD. It reuses the unchanged NEXT415 residual and may authorize
   continuation only for a strict improvement frozen before any later data.

Validation and replication remain sealed throughout this plan. Passing a
discovery stage means “candidate signal,” not “confirmed law.”

## 6. Test and artifact sequence

1. Add failing pure-kernel tests for the exact avoidance/clustering examples,
   uniform-strength midpoint, scale/order/replication invariance, malformed
   inputs, periodic crystal equivalences, and the geometry-only firewall.
2. Implement only the NEXT420 kernel and one-row wrapper; run the focused test.
3. Add failing probe tests for the unchanged prior universe, boundary-free
   signature, frozen execution hashes, and unchanged formal-root resolver.
4. Implement and run the 80+80 label-blind probe. Record hashes and stop or
   continue mechanically according to Section 4.
5. Append the result to the independent additive report
   `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.
6. Run focused tests and the complete repository suite, then verify output
   hashes, boundary flags, and absence of canonical-file edits.
