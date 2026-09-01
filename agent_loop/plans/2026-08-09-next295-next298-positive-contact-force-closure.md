# NEXT295--NEXT298 Positive Contact Force Closure Plan

> **For Codex:** use the already-loaded brainstorming, writing-plans,
> test-driven-development, executing-plans, systematic-debugging, and
> verification-before-completion workflows. The active shared checkout is
> intentional. Do not create a branch, commit, PR, or worktree.

**Status:** frozen before opening any NEXT295--NEXT298 discovery outcome.

**Goal:** Test whether a sign-sensitive, positive local contact-balance
certificate supplies a genuinely new, fully pre-DFT correction to the frozen
NEXT224 law.

**Architecture:** NEXT295 reconstructs the frozen NEXT19 formal-valence and
CrystalNN opposite-sign periodic contact graph from physically isolated
discovery geometry, then publishes thirteen label-free Positive Contact Force
Closure (PCFC) descriptors. NEXT296 audits exactly those thirteen directions
in the unchanged NEXT224 rejected-extreme cohort. NEXT297 searches the
unchanged margin-local grammar only if NEXT296 authorizes at least one
hypothesis. NEXT298 is an exact BROAD residual diagnostic only if NEXT297
authorizes an AUC+SAFE/non-BROAD identity set.

**Tech stack:** Python 3.11, ASE, pymatgen, NumPy, pandas, SciPy `linprog`,
pytest, Parquet, SHA-256, and the unchanged NEXT19/NEXT227/NEXT224
reconstruction and gate evaluators.

Date frozen: 2026-08-09 (America/Chicago).

## 1. Prior-mechanism audit and scientific motivation

This branch is additive and preserves every prior plan, script, test, output,
report section, and canonical artifact.

NEXT168, NEXT173, and NEXT179 summarize local contact directions using Gram
tensors of the form `sum_j w_j u_j u_j^T`. Those tensors are useful but
sign-blind: replacing any individual `u_j` by `-u_j` leaves the tensor
unchanged. Full rank or a large minimum eigenvalue therefore cannot determine
whether all actual contact directions lie in one hemisphere. The later
NEXT169/NEXT174/NEXT180 discovery audits also established prospectively useful
context: CrystalNN directional-rigidity features transferred across both
discovery sources, while the corresponding Voronoi hypotheses did not pass
both sources. This plan therefore freezes CrystalNN before opening the new
feature outcomes and does not search graph families post hoc.

For ideal hard-sphere contacts, local jamming in three dimensions requires at
least four contacts whose directions are not all contained in a hemisphere;
Donev, Torquato, Stillinger, and Connelly formulate the corresponding
geometric feasibility problem with linear programming (Journal of
Computational Physics 197, 139--166, 2004,
<https://doi.org/10.1016/j.jcp.2003.11.022>). This source motivates only the
hemisphere/positive-balance geometry. CrystalNN contacts at finite separation
are not literal hard-sphere constraints, so PCFC is not claimed to be a
rigorous jamming test, a force calculation, or an energy surrogate. The
nonnegative coefficients below are dimensionless dual geometric
certificates, not physical forces.

Three approaches were compared before outcomes:

1. **Selected:** a site-local nonnegative equilibrium LP on the frozen
   CrystalNN opposite-sign graph, evaluated with both uniform and frozen graph
   priors.
2. Power-cell face-normal closure. Rejected because the normals of a bounded
   convex power cell already satisfy a positive closure relation, making the
   proposed signal close to tautological on supported cells.
3. A global collective-jamming LP. Rejected for this first interpretable
   branch because its result is more sensitive to contact thresholds,
   supercell representation, and collective boundary modes.

Before this freeze, label-free engineering prototypes used ideal NaCl, CsCl,
ZnS, a regular tetrahedral direction set, and a deliberately one-sided
full-rank direction set. Ideal symmetric environments returned exact unit
closure/equilibrium; the one-sided full-rank set returned positive elementary
closure but zero equilibrium. Forty SCIGEN discovery geometries were inspected
only for support, finiteness, range, and runtime: 38 were supported and all
five site summaries were nondegenerate. On a distorted NaCl cell, rigid
rotation, periodic translation, site permutation, and exact `2 x 1 x 1`
replication changed any aggregate by at most `6.7e-16`. No endpoint or outcome
was opened for these checks.

## 2. Non-negotiable information boundary

Every executable feature and candidate formula receives only element
identities and one initial, raw, unrelaxed three-dimensional periodic geometry.
Reject an ASE `Atoms` object with a calculator, nonempty `info`, arrays other
than `numbers` and `positions`, nonfinite values, nonpositive cell volume, or
incomplete PBC.

The executable path must not read, infer, call, or compute:

- a DFT energy, force, stress, charge density, band value, hull value, or any
  other per-structure DFT result;
- a learned energy/force/stress model, MLIP, interatomic potential, proxy
  potential, or model-derived descriptor;
- a relaxed structure, trajectory, later coordinate/cell, physical
  relaxation, or same-composition alternative;
- a validation or replication geometry, endpoint, outcome, or output.

Frozen NEXT19 formal valences are composition-only discrete assignments.
Frozen CrystalNN weights are analytic graph-construction outputs from the raw
geometry; they are not learned quantities or potential energies. Discovery
outcomes are offline labels and may be joined only by NEXT296 after the
NEXT295 source, catalogue, directions, tolerances, input identities, and this
plan are frozen. Internal validation and replication remain physically sealed
even if a discovery candidate passes all gates; opening them requires a
separate future freeze. Unsupported rows abstain/fail open and can never
become automatic rejections.

Every formal manifest must publish exact false values for:

```text
dft_calculation_executed
dft_values_used_by_executable_formula
learned_energy_force_stress_proxy_used
model_or_proxy_potential_used
physical_relaxation_executed
opened_validation_outputs_used
scigen_replication_endpoint_opened
wyformer_replication_endpoint_opened
scientific_improvement_claim
```

## 3. NEXT295 frozen PCFC construction

### 3.1 Frozen periodic contact graph

Reuse NEXT19 without editing it. Infer its exact formal valences, require both
positive and negative sites and exact charge neutrality under the existing
tolerance, and call its existing `build_periodic_edge_geometry` with graph
kind exactly `crystalnn`. Keep only the resulting opposite-sign periodic
edges and their strictly positive finite `neighbor_weight`. Do not add a
distance threshold, neighbor family, oxidation-state alternative, radius,
or outcome-dependent fallback.

For every edge from site `a` to the periodic image of site `b`, add the unit
direction `u_ab` to site `a` and `-u_ab` to site `b`, carrying the same graph
weight. All crystallographic sites participate. A site with no incident edge
receives zeros and makes `locally_enclosed = 0`; a record is supported only if
the frozen valence/graph construction succeeds and at least one edge exists.

### 3.2 Directional closure

For a site with `k` incident directions, define two strictly positive priors:

```text
q_uniform_j = 1 / k
q_weighted_j = neighbor_weight_j / sum_l neighbor_weight_l.
```

For either prior `q`, define

```text
closure(q) = clip(1 - |sum_j q_j u_j|, 0, 1).
```

An empty direction set has closure zero. Require direction norms and all
weights to be finite and positive above `1e-12`; otherwise fail the record
closed. Permit pre-clip numerical drift outside `[0,1]` only up to `1e-12`.

### 3.3 Positive equilibrium fraction

For either prior `q`, solve exactly one linear program:

```text
equilibrium(q) = max alpha

subject to
    sum_j f_j u_j = 0
    sum_j f_j = 1
    f_j >= alpha q_j      for every j
    f_j >= 0
    0 <= alpha <= 1.
```

Use `scipy.optimize.linprog(..., method="highs")`, minimizing `-alpha`.
Before solving, if `k < 4` or the direction matrix has rank below three using
relative singular-value tolerance `1e-10`, return zero. A HiGHS infeasible
status returns zero; any other unsuccessful solver status fails the record
closed. Verify equality, normalization, bound, and floor residuals to absolute
tolerance `1e-9`. Permit final numerical drift outside `[0,1]` only up to
`1e-9`, then clip and round to the `1e10` output grid.

Define

```text
locally_enclosed = 1 if equilibrium(q_uniform) > 1e-9 else 0.
```

Positive equilibrium is a sign-sensitive origin-in-convex-hull margin. It is
not an inferred physical force magnitude and must never be described as an
energy or stability proof.

## 4. NEXT295 frozen feature catalogue

Across all sites, publish `min`, inverse-CDF `q10`, and arithmetic `mean` for
each of the four continuous site metrics, plus the enclosed-site fraction.
Quantiles use `numpy.quantile(..., method="inverted_cdf")`. The exact ordered
feature universe is:

```text
pcfc_uniform_closure_min
pcfc_uniform_closure_q10
pcfc_uniform_closure_mean
pcfc_weighted_closure_min
pcfc_weighted_closure_q10
pcfc_weighted_closure_mean
pcfc_uniform_equilibrium_min
pcfc_uniform_equilibrium_q10
pcfc_uniform_equilibrium_mean
pcfc_weighted_equilibrium_min
pcfc_weighted_equilibrium_q10
pcfc_weighted_equilibrium_mean
pcfc_locally_enclosed_fraction
```

All thirteen preregistered directions are `protected_high`: larger values may
only protect a NEXT224 rejection in the conditional grammar. Publish every one
of the exact `13,470` SCIGEN and `5,232` WyFormer discovery identities.
Supported rows require finite bounded features. Unsupported rows retain their
identifiers and NaN features. Formal publication requires source coverage at
least `0.90`, records exact support identities and failure reasons, and opens
no outcome.

The formal NEXT295 manifest must pin the frozen plan, NEXT19 source, NEXT168
geometry helper source, builder source/test, exact discovery geometry
manifests/metadata/archives, catalogue, feature tables, and all generated
hashes. Atomic publication is mandatory.

## 5. Task 1: NEXT295 TDD and label-free formal build

**Create only:**

- `tests/test_next295_positive_contact_force_closure.py`
- `src/next295_positive_contact_force_closure.py`
- external formal directory
  `$PRIS_ARCHIVE/next295_positive_contact_force_closure_v1`

Steps:

1. Write the focused test first and observe a missing-module RED result.
2. Test the exact schema/directions and direct analytic LP cases: regular
   tetrahedron, octahedral directions, a one-sided full-rank set, too few
   directions, and rank-deficient directions.
3. Test ideal NaCl, CsCl, and ZnS plus a distorted ionic cell under rotation,
   periodic translation, site permutation, and exact integral replication.
4. Test the exact geometry-only builder interface and fail-closed calculator,
   metadata, extra-array, PBC, malformed-geometry, missing-input, and
   existing-output cases.
5. Implement only this frozen specification, reusing immutable NEXT19 and
   NEXT168 helpers where exact.
6. Run a tiny nonformal geometry-only smoke without any endpoint path; inspect
   only support, finiteness, range, invariance, and runtime.
7. Pin all frozen input/source hashes, then run each physically isolated
   discovery geometry source exactly once and publish the formal outputs
   atomically.

## 6. NEXT296 prospective feature audit

Reconstruct the exact NEXT224 score/support and rejected-extreme cohort through
unchanged NEXT227 machinery. Normalize each PCFC feature using only finite
combined discovery geometry with inverse-CDF `1/16` and `15/16` cutoffs. Audit
exactly the thirteen feature names above and only `protected_high`. Opposite
directions, new quantiles, transformations, conjunctions, and post-outcome
direction changes are prohibited.

Use the unchanged reduced-formula folds, coverage/class-count gates, source
aggregate AUC, macro-fold AUC, and worst-fold AUC gates from NEXT268. A
hypothesis is eligible only if both discovery sources pass every raw gate.
Reporting rank cannot authorize a failing hypothesis. If zero hypotheses are
eligible, close the branch and do not create NEXT297/NEXT298.

## 7. Task 2: NEXT296 TDD and formal audit

**Create only:**

- `tests/test_next296_pcfc_feature_audit.py`
- `src/next296_pcfc_feature_audit.py`
- external formal directory
  `$PRIS_ARCHIVE/next296_pcfc_feature_audit_v1`

Write tests first for the exact hypothesis universe/directions, deterministic
reporting rank, source-prefixed identity alignment, source/input/source hashes,
no validation/replication interface, and missing-input failure. Then implement
by adapting the frozen NEXT268 audit structure without changing its gates. Run
the audit once and freeze the exact eligible identities and digest.

## 8. Conditional NEXT297 margin-local search

Create NEXT297 only if NEXT296 authorizes at least one exact hypothesis. Use
one unchanged NEXT224 reproduction control plus, for every eligible
hypothesis, the exact frozen NEXT269 grid:

```text
local_width_fraction in {1/64,1/32,1/16,1/8,1/4,1/2,1}
amplitude_fraction in {1/4,1/2,1}
h = local_width_fraction * NEXT214_REPAIR_WIDTH
w(s) = max(0, 1 - |s - NEXT224_THRESHOLD| / h)
score = max(0, s + amplitude_fraction*h*w(s)*(1 - 2*protection)).
```

The exact interval edge has zero weight; missing/unsupported/outside rows keep
the NEXT224 score and support. Search only frozen eligible identities. Require
both-source AUC, all twelve SAFE cells, BROAD, and every unchanged discovery
gate. A reporting leader is not a frozen law. If no all-gate candidate exists
but at least one new candidate passes AUC+SAFE and fails BROAD, authorize
NEXT298 only for that exact sorted identity population. Otherwise close.

**Conditional files only:**

- `tests/test_next297_pcfc_margin_local_search.py`
- `src/next297_pcfc_margin_local_search.py`
- `$PRIS_ARCHIVE/next297_pcfc_margin_local_search_v1`

## 9. Conditional NEXT298 exact BROAD diagnostic

Create NEXT298 only if NEXT297 explicitly authorizes it. Reproduce only the
authorized evaluator records and unchanged BROAD threshold tables. Rank by
`(failed_constraint_count, normalized_shortfall_sum, candidate_key)` and
compare with the frozen NEXT270 reference `(5, 0.0955435292756307)`. No new
feature, direction, threshold, coefficient, formula, or endpoint may be
introduced. A strict diagnostic improvement may justify only a new
pre-outcome freeze; it cannot open validation or establish a law.

**Conditional files only:**

- `tests/test_next298_pcfc_broad_diagnostic.py`
- `src/next298_pcfc_broad_diagnostic.py`
- `$PRIS_ARCHIVE/next298_pcfc_broad_diagnostic_v1`

## 10. Verification, stopping, and reporting

1. Run every new focused test and the complete repository suite.
2. Verify plan/source/test/formal input/output hashes and atomic publication.
3. Verify all no-DFT/no-proxy/no-relaxation/sealed-endpoint flags.
4. Verify `paper/`, `tex/`, `notes/`, `README.md`, and `PREREG.md` remain
   untouched by this branch.
5. Verify CodeGraph has no pending sync.
6. Append only the independent report
   `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.

Stop at the first failed authorization gate. Preserve all earlier files and
results. Do not claim replacement of or superiority to Pauling unless a
prospectively frozen candidate later passes separately authorized, sealed
validation and replication. The overall goal remains active after a negative
or discovery-only result.
