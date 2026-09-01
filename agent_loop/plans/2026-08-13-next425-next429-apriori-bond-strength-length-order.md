# A-Priori Bond-Strength Length-Order Implementation Plan

> **For Codex:** Execute this plan additively with test-driven development.
> Preserve all existing scripts and canonical documents. Validation and
> replication remain sealed unless a later, separately frozen discovery result
> authorizes them.

**Goal:** Test whether a charge-conserving bond-strength field derived from
formal charges and periodic bond topology predicts the local ordering of raw
bond lengths, without DFT, relaxation, learned potentials, or fitted bond
parameters.

**Architecture:** NEXT425 defines a unique maximum-entropy a-priori
bond-strength field on the frozen NEXT19 opposite-sign periodic Voronoi graph
and one scale-free local length-order protection statistic. It first faces the
same deterministic 80+80 label-blind support, invariance, nondegeneracy, and
novelty gates as NEXT420. NEXT426--NEXT429 are strictly conditional: isolated
full label-free construction, cross-source discovery audit, the unchanged
bounded formula grammar, and the unchanged BROAD residual diagnostic.

**Tech stack:** Python 3.11, NumPy, pandas/Parquet, ASE, pymatgen, the existing
NEXT19/NEXT267/NEXT295 geometry and provenance guards, pytest.

## 1. Scientific basis and prior-work audit

Hawthorne's a-priori bond-strength rule states that bond strengths can be
obtained from formal charges and bond topology by solving charge-conservation
constraints, without Euclidean distances, and reports a strong relation to
observed bond lengths
(<https://doi.org/10.1180/mgm.2026.10215>). The paper obtains unique values by
adding path constraints between equivalent ions. A generic raw generated cell
does not supply a trustworthy crystallographic or topological-equivalence
labelling, so this branch does **not** claim to reproduce Hawthorne's exact
path-equation construction. It freezes a canonical maximum-entropy completion
of the same charge-conservation polytope and tests the narrower ordering
hypothesis prospectively.

The repository audit found related but non-identical quantities:

1. `p3_hawthorne_features` tests simple-graph conservation residual, rank
   deficiency, and distance from equal Pauling cation stubs, but does not
   compare a topology-only conserved field with raw bond-length order and
   collapses periodic image multiplicity.
2. NEXT19 asks how far a geometry-dependent edge prior must be reallocated to
   conserve all site charges.
3. NEXT38 compares a geometry-dependent bond-valence prior with a minimum-norm
   conservation correction and its differential geometric compatibility.
4. NEXT307 projects a geometry-derived bond-valence field into an incidence
   cycle space, while NEXT347 measures allocation redistribution capacity.

NEXT425 reverses the information flow: its bond-strength field is calculated
without distances, and distances enter only afterward to test a local ordering
claim. This makes it complementary to NEXT19/38, not a replacement for them.

Alternatives rejected before any NEXT425 value is computed are: the raw
minimum-norm incidence solution, because it can assign negative strengths; a
bond-valence exponential comparison, because NEXT38 already uses it; and
global strength--length correlation, because atomic-size differences would
confound bonds that do not share a site.

## 2. Hard information boundary

The executable formula may read only element identities, the deterministic
NEXT19 formal-valence assignment, and one initial unrelaxed fully periodic
geometry. It may use the unchanged NEXT19 opposite-sign Voronoi contact graph.

It must not execute DFT; read DFT energy, force, stress, hull, or any outcome
label; use a learned energy/force/stress proxy, MLIP, or fitted potential;
relax coordinates or cell; read a trajectory, later geometry, or alternative
same-composition candidate; or access validation/replication data. Discovery
outcomes may be opened only offline after the complete label-free feature
artifact is frozen. Canonical `paper/`, `tex/`, `notes/`, `README.md`, and
`PREREG.md` remain untouched pending user review.

## 3. Frozen a-priori field

Let `q_i` be the nonzero neutral formal charge at site `i`. Each periodic edge
`e=(c,a,image)` in the NEXT19 graph joins a positive cation `c` to a negative
anion `a`. Let `b_i=|q_i|`. Among positive edge fields satisfying

`sum_(e incident to i) x_e = b_i`

at every reference-cell site, choose the unique maximum-entropy field

`x* = argmax_x -sum_e x_e log(x_e)`.

Equivalently, `x_e=exp(u_c+v_a)` on the frozen graph support. Solve these row
and column marginals by deterministic log-domain iterative proportional
fitting from a uniform edge measure. Freeze a relative maximum marginal
residual of `1e-10`, at most `20,000` alternating iterations, and fail closed
if a positive feasible solution is not reached. No distance, radius,
electronegativity, bond-valence table, outcome, or tunable decay enters this
field. Parallel periodic images remain distinct edges. Global charge rescaling
rescales `x*` but cannot change the final statistic.

Maximum entropy is a label-independent uniqueness convention, not a new
physical law and not part of Hawthorne's published path construction. The
empirical question is whether its topology-only ordering contains a robust
cross-source structure signal.

## 4. Frozen APRBS length-order formula

For every unordered pair of distinct edges `(e,f)` incident to the same
reference-cell site, define dimensionless contrasts

`Delta_x = (x_e-x_f)/(x_e+x_f)`,

`Delta_d = (d_e-d_f)/(d_e+d_f)`,

and `p_ef=Delta_x Delta_d`. Positive `p_ef` means the stronger a-priori bond is
longer and therefore violates the proposed order; negative `p_ef` means the
stronger bond is shorter. Enumerate cation and rebased-anion stars separately,
including translated contacts, and freeze

`W = sum_(site pairs) |p_ef|`,

`V = sum_(site pairs) max(p_ef,0)`,

`P = 1 - V/W` when `W > 1e-15`, otherwise `P=1/2`.

Freeze exactly one quantized feature:

`aprbs_length_order_protection = round_1e-10(P)`.

Its sole direction is `protected_high`. `P=1` means all informative local
orders agree, `P=0` means all disagree, and `P=1/2` is neutral when no pair has
both strength and length contrast. Pair enumeration makes the statistic
independent of edge order; normalized contrasts make it invariant to uniform
charge and length scaling; exact cell replication duplicates numerator and
denominator populations.

Require finite nonzero neutral charges with both signs, positive finite bond
lengths and strengths, exact cation--anion orientation, at least one contact
per charged site, finite marginal residual, and a final value in `[0,1]`.
Malformed inputs fail closed. There is no alternate entropy reference,
regularizer, cutoff, radius normalization, quantile, species subset,
direction, or companion feature.

## 5. Frozen label-blind gates

Use the unchanged deterministic 80-record discovery probe in each of SCIGEN
and WyFormer, selected only from `(natoms, chemical_system, material_id)`.
Open only discovery `x0`, base label-free features, all prior formal feature
families, and recomputed ZBVVG, BECNS, SSSP, OBS, and P4BSS controls. No
endpoint or outcome field may be read.

For each source require all of:

- support at least `72/80`;
- every finite value in `[0,1]`;
- at least 20 distinct values after rounding to `1e-10`;
- maximum error at most `1e-8` under rigid rotation, translation, site
  permutation, unimodular rebasing, and exact `2 x 1 x 1` supercell
  representation;
- maximum absolute Spearman correlation strictly below `0.90` against every
  adequate prior label-free control on at least 40 joint finite records.

If any gate fails, record `next426_formal_build_authorized=false` and
`aprbs_branch_terminated=true`; do not create NEXT426--NEXT429 artifacts.

## 6. Conditional full loop

Only after all label-blind gates pass:

1. NEXT426 processes all 13,470 SCIGEN and 5,232 WyFormer discovery `x0`
   records in physical label isolation, publishes complete tables and failure
   counts, and requires at least `0.95` support in each source.
2. NEXT427 reuses the unchanged NEXT224/NEXT413 rejected-extreme cohorts,
   reduced-formula five-fold split, inverted-CDF normalization, and source
   AUC/coverage gates for this sole frozen `protected_high` feature.
3. NEXT428 runs only if NEXT427 passes both sources and inherits exactly the
   NEXT261/NEXT414 width grid `(1/64,1/32,1/16,1/8,1/4,1/2,1)` and amplitude
   grid `(1/4,1/2,1)` with no new tuning.
4. NEXT429 runs only for candidates passing source AUC and every SAFE12 cell
   but missing BROAD. It reuses the unchanged NEXT415 residual diagnostic and
   may authorize continuation only for a strict pre-frozen improvement.

Validation and replication stay sealed. A discovery pass is a candidate
signal, never a confirmed law.

## 7. Test and artifact sequence

1. Add failing pure-kernel tests for analytically solvable transportation
   fields, agreeing/reversed/tied local orders, charge/length/edge-order
   invariance, malformed inputs, and infeasible disconnected marginals.
2. Implement the NEXT425 kernel and one-row wrapper; add periodic crystal
   equivalence and geometry-only firewall tests.
3. Add failing probe tests for the fixed prior universe, boundary-free
   signature, execution hashes, gate exactness, and formal-root resolver.
4. Run the 80+80 label-blind probe and stop or continue mechanically according
   to Section 5.
5. Append the result to
   `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.
6. Run focused and complete repository tests, then verify hashes, boundary
   flags, and absence of canonical-document edits.
