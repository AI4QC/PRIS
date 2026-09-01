# NEXT299--NEXT302 Minimal Opposite-Sign Periodic Cage Plan

> **For Codex:** use the already-loaded brainstorming, writing-plans,
> test-driven-development, executing-plans, systematic-debugging, and
> verification-before-completion workflows. The active shared checkout is
> intentional. Do not create a branch, commit, PR, or worktree.

**Status:** frozen before opening any NEXT299--NEXT302 feature outcome.

**Sequential-discovery disclosure:** the earlier NEXT296 discovery audit has
been seen. It showed that PCFC's AUC signal transferred but its frozen SCIGEN
fold coverage missed the gate. This new branch is therefore sequential
discovery, not independent confirmation. Its geometry, feature universe,
directions, tolerances, and gates are frozen below before computing or joining
any new MOSPC feature outcome. No validation or replication output may be
opened.

**Goal:** Test whether a dimension-complete, threshold-free nearest
opposite-sign periodic cage preserves PCFC's sign-sensitive positive-span
information while removing dependence on an occasionally unsupported
CrystalNN contact graph.

**Architecture:** NEXT299 uses frozen formal valences and exact periodic
geometry to retain the four nearest opposite-sign periodic images at every
site, including every symmetry tie at the fourth distance, and publishes
thirteen label-free Minimal Opposite-Sign Periodic Cage (MOSPC) descriptors.
NEXT300 audits exactly those directions in the unchanged NEXT224
rejected-extreme cohort. NEXT301 searches the unchanged margin-local grammar
only if NEXT300 authorizes at least one hypothesis. NEXT302 is an exact BROAD
residual diagnostic only if NEXT301 authorizes an AUC+SAFE/non-BROAD identity
set.

**Tech stack:** Python 3.11, ASE, pymatgen, NumPy, pandas, SciPy `linprog`,
pytest, Parquet, SHA-256, and unchanged NEXT19/NEXT267/NEXT295/NEXT268/NEXT227
reconstruction and gate evaluators.

Date frozen: 2026-08-09 (America/Chicago).

## 1. Scientific motivation and alternatives

NEXT295 established a new sign-sensitive invariant. In its frozen discovery
audit, weighted closure met the AUC quality thresholds in both sources, but
the CrystalNN-based feature support missed the fixed SCIGEN fold-coverage gate.
The proposed object is not a post-outcome threshold adjustment: it removes the
neighbor-classifier dependency entirely and defines the smallest
dimension-complete opposite-sign periodic cage in three dimensions.

For ideal hard-sphere contacts, a locally enclosing three-dimensional contact
set requires at least `d + 1 = 4` directions not contained in a hemisphere;
Donev, Torquato, Stillinger, and Connelly formulate the corresponding
geometric feasibility test by linear programming (Journal of Computational
Physics 197, 139--166, 2004,
<https://doi.org/10.1016/j.jcp.2003.11.022>). This source fixes the order four
from dimension, not from any crystal outcome. MOSPC is still not a hard-sphere
jamming proof: its retained opposite-sign images are an analytic ionic-cage
construction, and its nonnegative coefficients are dimensionless geometric
certificates rather than physical forces.

Three alternatives were compared before new outcomes:

1. **Selected:** retain the four nearest opposite-sign periodic images plus
   all exact fourth-distance ties, with no contact classifier or fitted
   distance cutoff.
2. Add a nearest-neighbor fallback only when CrystalNN fails. Rejected because
   it creates two graph semantics selected by an implementation failure and
   directly patches the previous branch instead of defining one law.
3. Use every opposite-sign site in a large periodic cutoff or a complete
   bipartite field. Rejected because the result becomes cell-size/nonlocal and
   needs a cutoff or convergence convention unrelated to local enclosure.
4. Replace CrystalNN only with ordinary Voronoi adjacency. Rejected because
   it still does not guarantee a dimension-complete opposite-sign cage and
   prior sign-blind Voronoi rigidity did not transfer across sources.

Literal repository search found no existing four-nearest opposite-sign
periodic positive-span mechanism. NEXT168/NEXT173/NEXT179 use graph contacts;
NEXT35 uses analytic vector-field balance; NEXT37 uses self-stress mismatch;
none selects the dimension-minimal periodic ionic cage defined here.

Before this freeze, label-free prototypes returned exact unit features for
NaCl, CsCl, and ZnS while retaining their full 6-, 8-, and 4-fold tied first
shells. A distorted NaCl cell was exactly invariant under rigid rotation,
periodic translation, site permutation, and `2 x 1 x 1` replication. On the
first forty SCIGEN discovery geometries, 38 were supported and two lacked a
formal opposite-sign assignment; supported closure and equilibrium-mean
features were nondegenerate. The maximum certified translation range was two.
With the exact four-direction analytic solve below, the smoke took `0.251 s`.
Across 200 random four-direction/prior cases, the analytic result differed
from the existing LP result by at most `4.74e-11`, within the `1e-10` output
grid. No endpoint or outcome was opened for these prototypes.

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
Discovery outcomes are offline labels and may be joined only by NEXT300 after
the NEXT299 source, catalogue, directions, tolerances, and inputs are frozen.
Unsupported rows abstain/fail open and can never become automatic rejections.
The discovery data are reused sequentially, so passing discovery cannot by
itself establish a law. Internal validation and replication remain physically
sealed and require a separate future authorization.

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

## 3. NEXT299 frozen periodic cage construction

### 3.1 Reduced representation and formal signs

After the strict public input guard, reuse NEXT267's immutable
Minkowski-reduced-cell transformation and wrapping. Convert that geometry to a
pymatgen `Structure`, reuse NEXT19's exact formal-valence inference, and require
at least one positive and one negative site. Do not call CrystalNN, Voronoi, a
learned graph, a radius table, or a distance threshold.

For site `i`, let `O_i` be all basis sites with formal sign opposite to `i`.
For each `j in O_i`, call pymatgen's lattice nearest-image routine once and
write the returned fractional displacement as

```text
delta_ij = frac_j + nearest_integer_image_ij - frac_i.
```

### 3.2 Certified fourth-nearest periodic image set

Let `L` be the row-vector cell matrix and `sigma_min(L)` its smallest singular
value. For integer ranges `R = 1, 2, ..., 8`, enumerate exactly

```text
v_ijT = (delta_ij + T) L
for every j in O_i and T in {-R,...,R}^3.
```

Require every distance to be finite and greater than `1e-12`. Let `d4(R)` be
the fourth-smallest enumerated distance and define

```text
tie_tolerance = 1e-8 * max(1, d4(R))
D_infinity = max_j ||delta_ij||_infinity
outside_lower_bound(R) = sigma_min(L) * max(0, R + 1 - D_infinity).
```

Any unenumerated integer offset has at least one component of magnitude
`R + 1`, so the singular-value inequality gives the stated lower bound on its
Cartesian distance. Stop at the first range satisfying

```text
outside_lower_bound(R) > d4(R) + tie_tolerance.
```

Failure to certify by `R = 8`, more than `2,000,000` enumerated candidates at
one site, a nonfinite/zero distance, or a nonpositive smallest singular value
fails the record closed.

After certification, retain every enumerated image satisfying

```text
distance <= d4(R) + tie_tolerance.
```

This includes all symmetry ties at the fourth distance. Require a retained
population between 4 and 256. Do not deduplicate equal directions belonging to
different physical periodic images.

### 3.3 Uniform and inverse-square priors

For retained unit directions `u_j` and distances `d_j`, freeze two positive
priors:

```text
q_uniform_j = 1 / k
raw_inverse_square_j = (min_l d_l / d_j)^2
q_inverse_square_j = raw_inverse_square_j / sum_l raw_inverse_square_l.
```

For either prior, define directional closure and positive equilibrium exactly
as in NEXT295:

```text
closure(q) = clip(1 - |sum_j q_j u_j|, 0, 1)

equilibrium(q) = max alpha
subject to sum_j f_j u_j = 0, sum_j f_j = 1,
           f_j >= alpha q_j, f_j >= 0, 0 <= alpha <= 1.
```

For `k > 4`, call the immutable NEXT295 HiGHS implementation and retain its
rank and `1e-9` certificate guards. For `k = 4`, solve the exact square system

```text
[u_1 u_2 u_3 u_4; 1 1 1 1] f = [0,0,0,1]
```

and verify the residual to `1e-9`. A singular system or a coefficient below
`-1e-9` has equilibrium zero. Otherwise clip tiny negative roundoff to zero
and compute exactly

```text
equilibrium(q) = clip(min_j(f_j / q_j), 0, 1).
```

This is algebraically the same frozen LP because a rank-three four-direction
set has at most one normalized balance. Round site metrics and final features
to the `1e10` output grid. Define `locally_enclosed = 1` exactly when uniform
equilibrium exceeds `1e-9`.

## 4. NEXT299 frozen feature catalogue

Across all crystallographic sites, publish `min`, inverse-CDF `q10`, and
arithmetic `mean` for each continuous site metric, plus enclosed-site fraction.
The exact ordered feature universe is:

```text
mospc_uniform_closure_min
mospc_uniform_closure_q10
mospc_uniform_closure_mean
mospc_inverse_square_closure_min
mospc_inverse_square_closure_q10
mospc_inverse_square_closure_mean
mospc_uniform_equilibrium_min
mospc_uniform_equilibrium_q10
mospc_uniform_equilibrium_mean
mospc_inverse_square_equilibrium_min
mospc_inverse_square_equilibrium_q10
mospc_inverse_square_equilibrium_mean
mospc_locally_enclosed_fraction
```

All thirteen directions are `protected_high`. Publish all exact `13,470`
SCIGEN and `5,232` WyFormer discovery identities. Supported rows require every
site to have a certified retained cage and every feature to be finite and in
`[0,1]`. Unsupported rows retain identifiers and NaN features. Formal
publication requires source coverage at least `0.97`, records exact support
identities/failure reasons, and opens no outcome.

## 5. NEXT299 TDD and formal label-free build

**Create only:**

- `tests/test_next299_minimal_opposite_sign_periodic_cage.py`
- `src/next299_minimal_opposite_sign_periodic_cage.py`
- `$PRIS_ARCHIVE/next299_minimal_opposite_sign_periodic_cage_v1`

Write the focused test first and observe a missing-module RED result. Test the
exact schema; analytic four-direction/LP equivalence; symmetric fourth-shell
ties; lower-bound certification and range guard; one-sided/degenerate cages;
NaCl, CsCl, ZnS; rotation, translation, site permutation, equivalent lattice
representation, and integral replication; strict geometry-only rejection;
and the discovery-only builder interface. Pin the plan, NEXT19, NEXT267,
NEXT295, source/test, and exact geometry input hashes. Build the two physically
isolated discovery feature tables once and publish atomically.

## 6. NEXT300 prospective feature audit

Reconstruct the exact NEXT224 score/support and rejected-extreme cohort through
unchanged NEXT227 machinery. Normalize each feature with only the finite
combined discovery population at inverse-CDF `1/16` and `15/16`. Audit exactly
the thirteen `protected_high` hypotheses above. Opposite directions,
alternative orders/shells, new quantiles, transformations, conjunctions, and
post-outcome changes are prohibited.

Use the unchanged reduced-formula folds, coverage/class-count gates, source
aggregate AUC, macro-fold AUC, and worst-fold AUC gates from NEXT268. A
hypothesis is eligible only if both sources pass every raw gate. If zero are
eligible, close the branch and do not create NEXT301/NEXT302.

**Create only:**

- `tests/test_next300_mospc_feature_audit.py`
- `src/next300_mospc_feature_audit.py`
- `$PRIS_ARCHIVE/next300_mospc_feature_audit_v1`

## 7. Conditional NEXT301 margin-local search

Create NEXT301 only if NEXT300 authorizes at least one exact hypothesis. Use
one unchanged NEXT224 reproduction control plus, for every eligible
hypothesis, the exact NEXT269 grid:

```text
local_width_fraction in {1/64,1/32,1/16,1/8,1/4,1/2,1}
amplitude_fraction in {1/4,1/2,1}
h = local_width_fraction * NEXT214_REPAIR_WIDTH
w(s) = max(0, 1 - |s - NEXT224_THRESHOLD| / h)
score = max(0, s + amplitude_fraction*h*w(s)*(1 - 2*protection)).
```

The interval edge has zero weight; missing/unsupported/outside rows retain the
NEXT224 score/support. Require both-source AUC, all twelve SAFE cells, BROAD,
and every unchanged gate. If no all-gate candidate exists but at least one new
candidate passes AUC+SAFE and fails BROAD, authorize NEXT302 only for that
exact sorted identity population. Otherwise close.

**Conditional files only:**

- `tests/test_next301_mospc_margin_local_search.py`
- `src/next301_mospc_margin_local_search.py`
- `$PRIS_ARCHIVE/next301_mospc_margin_local_search_v1`

## 8. Conditional NEXT302 exact BROAD diagnostic

Create NEXT302 only if NEXT301 explicitly authorizes it. Reproduce only the
authorized evaluator records and unchanged BROAD tables. Rank by
`(failed_constraint_count, normalized_shortfall_sum, candidate_key)` and
compare with the frozen NEXT270 reference `(5, 0.0955435292756307)`. Introduce
no new feature, direction, shell order, threshold, coefficient, formula, or
endpoint. A strict residual improvement can justify only another future
freeze; it cannot establish a law.

**Conditional files only:**

- `tests/test_next302_mospc_broad_diagnostic.py`
- `src/next302_mospc_broad_diagnostic.py`
- `$PRIS_ARCHIVE/next302_mospc_broad_diagnostic_v1`

## 9. Verification, stopping, and reporting

Run every focused test and the complete repository suite. Verify all frozen
input/output/source hashes, boundary flags, atomic publication, exact stopping
conditions, protected-file status, and CodeGraph synchronization. Append only
the independent report
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.

Stop at the first failed authorization gate. Preserve all prior content. Do
not claim replacement of or superiority to Pauling unless a prospectively
frozen candidate later passes separately authorized sealed validation and
replication. The overall goal remains active after a negative or discovery-
only result.
